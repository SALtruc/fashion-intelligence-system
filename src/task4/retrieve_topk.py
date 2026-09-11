"""Run the Task 4 visual-search model: given query images, return the Top-K similar items.

Task 4 has no column in `styles_prediction.csv` -- the brief asks for "the Top-K similar
fashion items", which is a ranking, not a label. This script is the runnable form of that
answer, and writes one row per (query, rank) pair.

Two ways in, because they answer different questions:

    # Encode images with the frozen ArcFace encoder, then retrieve. Needs the project
    # environment (torch + torchvision) and the images on disk.
    python src/task4/retrieve_topk.py --images datasets/test/images_test --top-k 10

    # Retrieve for the hold-out queries whose embeddings notebook 06 already saved. No
    # encoder, no images -- the same cosine ranking the benchmark scored, reproducible
    # from the artefacts alone.
    python src/task4/retrieve_topk.py --from-saved-queries --top-k 10

Both rank against `artifacts/task4/gallery_embeddings.npy`, the 33,968 catalogue items the
hold-out benchmark used -- not the smaller development gallery the training run left beside
its checkpoint.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The submitted encoder, its gallery and its preprocessing record sit directly in
# artifacts/task4/. The nested models/ and embeddings/ trees below are what the training
# notebooks write, one directory per candidate; they stay local and are searched only as
# a fallback for a checkout restored from Drive in the older shape.
TASK4 = ROOT / "artifacts" / "task4"


def _resolve(name, *fallbacks):
    for candidate in (TASK4 / name, *(TASK4 / f for f in fallbacks)):
        if candidate.exists():
            return candidate
    return TASK4 / name


CHECKPOINT_PATH = _resolve("arcface_best.pt", "models/arcface/best.pt")
CONFIG_PATH = _resolve("image_preprocessing.json", "configs/image_preprocessing.json")
GALLERY_IDS = _resolve("gallery_ids.npy", "embeddings/arcface/gallery_ids.npy")
GALLERY_EMBEDDINGS = _resolve("gallery_embeddings.npy", "embeddings/arcface/gallery_embeddings.npy")
QUERY_IDS = _resolve("query_ids.npy", "embeddings/arcface/query_ids.npy")
QUERY_EMBEDDINGS = _resolve("query_embeddings.npy", "embeddings/arcface/query_embeddings.npy")
DEFAULT_OUTPUT = ROOT / "predictions" / "task4" / "task4_retrieval_top10.csv"


def load_gallery():
    ids, embeddings = GALLERY_IDS, GALLERY_EMBEDDINGS
    missing = [p for p in (ids, embeddings) if not p.is_file()]
    if missing:
        raise SystemExit(
            "Gallery embeddings are absent: " + ", ".join(str(p) for p in missing)
            + "\nartifacts/ is gitignored; get it from the team Drive, or rebuild it with "
              "notebooks/task4/06_final_benchmark.ipynb."
        )
    gallery_ids = np.load(ids)
    gallery = np.load(embeddings).astype(np.float32)
    norms = np.linalg.norm(gallery, axis=1)
    if not np.allclose(norms, 1.0, atol=2e-3):
        raise SystemExit("Gallery embeddings are not L2-normalised; cosine would be wrong.")
    return gallery_ids, gallery


def saved_queries():
    ids = np.load(QUERY_IDS)
    queries = np.load(QUERY_EMBEDDINGS).astype(np.float32)
    return ids, queries


def encode_images(image_dir: Path, batch_size: int = 128):
    """Embed every .jpg under `image_dir` with the frozen encoder.

    Imported lazily: the saved-embeddings path above must keep working on a machine that
    has numpy but not torch.
    """
    import json

    import torch
    from PIL import Image

    from src.task4.models import ResNet18Encoder

    checkpoint_path = CHECKPOINT_PATH
    if not checkpoint_path.is_file():
        raise SystemExit(f"{checkpoint_path} is absent; get artifacts/task4 from the Drive.")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    height, width = config["resize"]["target_size"]
    fill = tuple(config["resize"]["padding_color_rgb"])
    mean = np.asarray(config["normalization"]["mean_rgb"], dtype=np.float32)
    std = np.asarray(config["normalization"]["std_rgb"], dtype=np.float32)

    def letterbox(image):
        """Scale to fit, pad the remainder. Stretching would change the aspect ratio the
        encoder was trained on, which is the one transform that must not drift."""
        image = image.convert("RGB")
        scale = min(width / image.width, height / image.height)
        resized = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.BILINEAR,
        )
        canvas = Image.new("RGB", (width, height), fill)
        canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
        return np.asarray(canvas, dtype=np.float32) / 255.0

    paths = sorted(Path(image_dir).glob("*.jpg"), key=lambda p: int(p.stem))
    if not paths:
        raise SystemExit(f"No .jpg files in {image_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18Encoder().to(device).eval()
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state.get("model_state_dict", state.get("state_dict", state)))

    out = np.zeros((len(paths), 512), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            chunk = paths[start:start + batch_size]
            batch = np.stack([letterbox(Image.open(p)) for p in chunk])
            batch = (batch - mean) / std
            tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
            out[start:start + len(chunk)] = model(tensor).cpu().numpy()
            print(f"  encoded {min(start + len(chunk), len(paths)):,}/{len(paths):,}", flush=True)
    return np.asarray([int(p.stem) for p in paths]), out


def retrieve(query_ids, queries, gallery_ids, gallery, top_k, chunk=256):
    """Top-K by cosine similarity. Both sides are L2-normalised, so a dot product is the
    cosine; chunked because 3,774 x 33,968 in one matrix is 500 MB of float32."""
    rows = []
    lookup = {int(g): i for i, g in enumerate(gallery_ids)}
    for start in range(0, len(queries), chunk):
        block = queries[start:start + chunk]
        scores = block @ gallery.T
        for offset, row in enumerate(scores):
            query_id = int(query_ids[start + offset])
            # A query that is itself in the gallery would otherwise rank first against
            # itself at similarity 1.0, which answers nothing.
            self_index = lookup.get(query_id)
            if self_index is not None:
                row[self_index] = -np.inf
            best = np.argpartition(-row, top_k)[:top_k]
            best = best[np.argsort(-row[best])]
            for rank, index in enumerate(best, start=1):
                rows.append((query_id, rank, int(gallery_ids[index]), float(row[index])))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", type=Path, help="directory of query .jpg files")
    source.add_argument("--from-saved-queries", action="store_true",
                        help="use the hold-out query embeddings notebook 06 saved")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")

    gallery_ids, gallery = load_gallery()
    print(f"gallery: {len(gallery_ids):,} items, {gallery.shape[1]} dimensions")

    if args.from_saved_queries:
        query_ids, queries = saved_queries()
        print(f"queries: {len(query_ids):,} hold-out embeddings (no encoding needed)")
    else:
        query_ids, queries = encode_images(args.images)
        print(f"queries: {len(query_ids):,} images encoded from {args.images}")

    if args.top_k > len(gallery_ids):
        parser.error(f"--top-k {args.top_k} exceeds the {len(gallery_ids):,}-item gallery")

    rows = retrieve(query_ids, queries, gallery_ids, gallery, args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        handle.write("query_id,rank,retrieved_id,cosine_similarity\n")
        for query_id, rank, retrieved_id, score in rows:
            handle.write(f"{query_id},{rank},{retrieved_id},{score:.6f}\n")
    print(f"wrote {len(rows):,} rows ({len(query_ids):,} queries x top-{args.top_k}) "
          f"-> {args.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
