"""Leakage gate for any external images before they enter training or go to Drive.

Why this exists
---------------
The tutor's clarification (2026-08-19) permits extra data but attaches one hard
condition:

    "collecting more dataset for better training is always welcome and
     encouraged. However, if you get more data for train, make sure there is
     no data leakage to the test set that you are provided."

Nothing in the repo covers that yet. `src/check_duplicates.py` hashes the
*training* catalogue against itself and answers the train/val split question;
`src/prepare_external_cosmetics.py` keeps ids clear with ID_OFFSET=900_000 but
never asks whether the external *pixels* duplicate a provided test image.

That gap matters because our dataset is a slice of Param Aggarwal's public
"Fashion Product Images" set (see research/EXTERNAL_DATA_PLAN.md), so any
re-download of that source, or of anything derived from it, can silently
reintroduce the 5,829 images we are being graded on.

This script is the gate. Run it on a candidate folder; it refuses to pass
anything that matches a provided test image, and writes a manifest that the
report can cite as evidence the condition was met.

Detection strategy (two independent nets, deliberately)
-------------------------------------------------------
1. Exact match on a 16x16 greyscale thumbnail. On the provided catalogue this
   was measured as a true-duplicate detector: 636 clusters, largest 4, label
   agreement 99.5% (see EDA_FINDINGS.md finding 9).
2. dHash within a small Hamming radius, to catch re-encoded or lightly resized
   copies that shift a thumbnail pixel. Note dHash alone is NOT a duplicate
   detector on this data -- at 8x8 it collides across whole silhouette
   families (46 Ties + 40 Deodorants + 19 Perfumes in one bucket, finding 8) --
   so it is used only as a *widening* pass on top of net 1, never alone.

External images are resized to 60x80 first, because the public source is
full-resolution and an unnormalised thumbnail would never match.

Usage
-----
    # build/refresh the descriptor cache for the provided data (once, ~6 min)
    python verify_external_data.py --build-reference

    # then gate a candidate folder
    python verify_external_data.py --check path/to/candidate/images
    python verify_external_data.py --check path/to/images --labels external.csv

    # self-test: feeding the provided test folder must report 100% leakage
    python verify_external_data.py --self-test

Exit code is 1 if any candidate matches a provided test image, so this can be
wired into a pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

THUMB = 16
DHASH_W, DHASH_H = 9, 8
TARGET_SIZE = (60, 80)          # (width, height) of the provided images
DHASH_RADIUS = 2               # Hamming radius for the widening pass
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from external_data import default_data_root  # noqa: E402
CACHE = HERE / "outputs" / "cache" / "external_gate_reference.npz"


# ------------------------------------------------------------------ descriptors


def describe(path: Path) -> tuple[np.ndarray, np.uint64] | None:
    """Return (16x16 thumbnail, dHash) for one image, normalised to 60x80.

    Returns None if the file cannot be read, so a corrupt candidate is
    reported rather than crashing the run.
    """
    try:
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            if rgb.size != TARGET_SIZE:
                rgb = rgb.resize(TARGET_SIZE, Image.BILINEAR)
            grey = rgb.convert("L")
            thumb = np.asarray(grey.resize((THUMB, THUMB), Image.BILINEAR),
                               dtype=np.uint8)
            small = np.asarray(grey.resize((DHASH_W, DHASH_H), Image.BILINEAR),
                               dtype=np.int16)
    except Exception:
        return None
    bits = (small[:, 1:] > small[:, :-1]).reshape(-1)
    dh = np.uint64(int("".join("1" if b else "0" for b in bits), 2))
    return thumb, dh


def _scan(paths: list[Path], label: str) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    thumbs, hashes, kept = [], [], []
    for i, p in enumerate(paths, 1):
        d = describe(p)
        if d is None:
            continue
        thumbs.append(d[0].reshape(-1))
        hashes.append(d[1])
        kept.append(p)
        if i % 5000 == 0:
            print(f"    {label}: {i}/{len(paths)}")
    return (np.array(thumbs, dtype=np.uint8),
            np.array(hashes, dtype=np.uint64), kept)


def _to_bits(hashes: np.ndarray) -> np.ndarray:
    """uint64 hashes -> (n, 64) matrix of +-1, for Hamming via matrix product.

    A per-candidate popcount loop is far too slow here (5.8k candidates x 44k
    reference x 64 bits). Encoding as +-1 turns Hamming distance into one BLAS
    call: agree = C @ R.T, distance = (64 - agree) / 2.
    """
    if len(hashes) == 0:
        return np.zeros((0, 64), dtype=np.int8)
    b = np.unpackbits(hashes.astype(">u8").view(np.uint8)).reshape(len(hashes), 64)
    return (2 * b.astype(np.int8) - 1)


def _any_within(cand: np.ndarray, ref: np.ndarray, radius: int,
                chunk: int = 512) -> np.ndarray:
    """For each candidate, does any reference hash sit within `radius` bits?"""
    if len(cand) == 0 or len(ref) == 0:
        return np.zeros(len(cand), dtype=bool)
    C, R = _to_bits(cand), _to_bits(ref)
    out = np.zeros(len(cand), dtype=bool)
    for i in range(0, len(C), chunk):
        agree = C[i:i + chunk].astype(np.int16) @ R.T.astype(np.int16)
        dist = (64 - agree) // 2
        out[i:i + chunk] = (dist <= radius).any(axis=1)
    return out


# ------------------------------------------------------------------- reference


def find_dataset_root(explicit: str | None) -> Path:
    """Locate the provided dataset in either this repo or the team repo layout."""
    if explicit:
        r = Path(explicit)
        if (r / "train" / "styles_train.csv").is_file():
            return r
        raise SystemExit(f"no styles_train.csv under {r}")
    for cand in [
        HERE / "A2_FashionDataset" / "FashionDataset",   # this repo
        HERE / "Dataset" / "FashionDataset",             # team repo
        HERE.parent / "Dataset" / "FashionDataset",
    ]:
        if (cand / "train" / "styles_train.csv").is_file():
            return cand
    raise SystemExit("could not find FashionDataset -- pass --dataset-root")


def build_reference(root: Path) -> None:
    """Hash every provided train AND test image, and cache the descriptors."""
    tr_dir, te_dir = root / "train" / "images_train", root / "test" / "images_test"
    tr = sorted(p for p in tr_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    te = sorted(p for p in te_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    print(f"build_reference: {len(tr)} train + {len(te)} test images")

    tr_t, tr_h, tr_p = _scan(tr, "train")
    te_t, te_h, te_p = _scan(te, "test")

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE,
        train_thumbs=tr_t, train_hashes=tr_h,
        train_ids=np.array([p.stem for p in tr_p]),
        test_thumbs=te_t, test_hashes=te_h,
        test_ids=np.array([p.stem for p in te_p]),
    )
    print(f"  wrote {CACHE} ({CACHE.stat().st_size / 1e6:.1f} MB)")


def load_reference() -> dict:
    if not CACHE.is_file():
        raise SystemExit(
            f"{CACHE} missing -- run: python {Path(__file__).name} --build-reference")
    z = np.load(CACHE, allow_pickle=False)
    return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------- check


def check_folder(folder: Path, ref: dict, labels: Path | None,
                 valid_article_types: set[str]) -> dict:
    paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise SystemExit(f"no images found under {folder}")
    print(f"check: {len(paths)} candidate images under {folder}")

    cand_t, cand_h, kept = _scan(paths, "candidate")
    unreadable = [str(p) for p in paths if p not in set(kept)]

    # --- net 1: exact thumbnail match -----------------------------------
    def exact_hits(ref_thumbs: np.ndarray) -> np.ndarray:
        if len(ref_thumbs) == 0 or len(cand_t) == 0:
            return np.zeros(len(cand_t), dtype=bool)
        ref_set = {r.tobytes() for r in ref_thumbs}
        return np.array([c.tobytes() in ref_set for c in cand_t])

    hit_test_exact = exact_hits(ref["test_thumbs"])
    hit_train_exact = exact_hits(ref["train_thumbs"])

    # --- net 2: dHash within a small Hamming radius ---------------------
    hit_test_near = _any_within(cand_h, ref["test_hashes"], DHASH_RADIUS)
    hit_train_near = _any_within(cand_h, ref["train_hashes"], DHASH_RADIUS)

    hit_test = hit_test_exact | hit_test_near
    hit_train = hit_train_exact | hit_train_near

    # --- id-range and label-vocabulary checks ---------------------------
    id_problems, label_problems = [], []
    if labels is not None:
        lab = pd.read_csv(labels)
        if "id" in lab.columns:
            ids = pd.to_numeric(lab["id"], errors="coerce")
            clash = ids.between(1163, 51999) | ids.between(52003, 60000)
            id_problems = ids[clash].dropna().astype(int).tolist()
        if "articleType" in lab.columns:
            bad = set(lab["articleType"].dropna()) - valid_article_types
            label_problems = sorted(bad)

    report = {
        "candidate_folder": str(folder),
        "images_found": len(paths),
        "images_readable": len(kept),
        "images_unreadable": unreadable[:20],
        "matches_provided_TEST_exact": int(hit_test_exact.sum()),
        "matches_provided_TEST_near": int((hit_test_near & ~hit_test_exact).sum()),
        "matches_provided_TEST_total": int(hit_test.sum()),
        "matches_provided_train": int(hit_train.sum()),
        "clean_and_usable": int((~hit_test & ~hit_train).sum()),
        "ids_colliding_with_provided_ranges": id_problems[:20],
        "articleTypes_not_in_the_124": label_problems,
        "dhash_hamming_radius": DHASH_RADIUS,
        "verdict": "FAIL" if hit_test.any() or id_problems or label_problems else "PASS",
    }

    rejects = pd.DataFrame({
        "path": [str(p) for p in kept],
        "matches_test": hit_test,
        "matches_train": hit_train,
    })
    out_csv = folder.parent / f"{folder.name}_leakage_report.csv"
    rejects.to_csv(out_csv, index=False)
    report["per_image_report"] = str(out_csv)
    return report


# ------------------------------------------------------------------ domain gap


def _standardize(im: Image.Image) -> Image.Image:
    """The transform every consumer applies, so measurements describe what a model
    actually receives rather than what happens to be on disk."""
    im = im.convert("RGB")
    if im.size != TARGET_SIZE:
        im = ImageOps.pad(im, TARGET_SIZE, method=Image.Resampling.BILINEAR,
                          color=(255, 255, 255), centering=(0.5, 0.5))
    return im


def border_brightness(path: Path, standardize: bool, k: int = 3) -> float | None:
    """Mean RGB value of the outermost `k`-pixel frame.

    A crude but honest proxy for "is this a catalogue cut-out on white, or a
    photograph with a real background". Deliberately simple: the point is that it
    is reproducible, and a measurement nobody can re-run is not evidence.
    """
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            a = np.asarray(_standardize(im) if standardize else im.convert("RGB"),
                           dtype=np.float32)
    except Exception:
        return None
    return float(np.mean([a[:k].mean(), a[-k:].mean(), a[:, :k].mean(), a[:, -k:].mean()]))


def _scores(folder: Path, standardize: bool, sample: int | None,
            seed: int = 42) -> np.ndarray:
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if sample and len(paths) > sample:
        idx = np.random.RandomState(seed).choice(len(paths), sample, replace=False)
        paths = [paths[i] for i in idx]
    vals = [border_brightness(p, standardize) for p in paths]
    return np.array([v for v in vals if v is not None], dtype=np.float64)


def _separability(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Best single-threshold balanced accuracy separating two score sets."""
    best_t, best_acc = 0.0, 0.0
    for t in np.linspace(0, 255, 512):
        acc = 0.5 * ((a > t).mean() + (b <= t).mean())
        if acc > best_acc:
            best_t, best_acc = float(t), float(acc)
    return best_t, best_acc


def domain_gap(root: Path, folders: list[Path], sample: int | None = 3000) -> None:
    """Compare each external folder's background against the provided catalogue.

    Reported twice on purpose. 'as stored' is what sits in the folder; 'as loaded'
    is after the 60x80 standardisation every consumer applies. For a set whose
    images are already 60x80 the two are identical. For one that is not, they can
    differ a lot -- white padding raises the border brightness -- and only the
    'as loaded' figure describes what a model sees.
    """
    cat = _scores(root / "train" / "images_train", True, sample)
    print(f"provided catalogue: {len(cat):,} images sampled\n")
    print(f"{'set':38}{'border':>9}{'near-white':>12}{'separable':>11}")
    print(f"{'provided catalogue':38}{cat.mean():>9.1f}{(cat > 240).mean() * 100:>11.1f}%"
          f"{'-':>11}")

    for folder in folders:
        if not folder.is_dir():
            print(f"{folder.name:38}{'MISSING':>9}")
            continue
        for standardize, tag in ((False, "as stored"), (True, "as loaded")):
            v = _scores(folder, standardize, None)
            if not len(v):
                continue
            _, acc = _separability(cat, v)
            name = f"{folder.parent.name} ({tag})"
            print(f"{name:38}{v.mean():>9.1f}{(v > 240).mean() * 100:>11.1f}%"
                  f"{acc * 100:>10.1f}%")
    print("\n'separable' = best single-threshold balanced accuracy telling that set "
          "apart from\nthe catalogue by border brightness alone. High means a model "
          "could learn the\nbackground instead of the product -- worth stating, though "
          "these rows are in\ntraining only, so validation scores stay honest.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset-root")
    ap.add_argument("--build-reference", action="store_true")
    ap.add_argument("--check", metavar="FOLDER")
    ap.add_argument("--labels", metavar="CSV")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--domain-gap", nargs="*", metavar="FOLDER",
                    help="measure background difference vs the provided catalogue; "
                         "with no folders, all three external sets")
    a = ap.parse_args()

    root = find_dataset_root(a.dataset_root)
    print(f"provided dataset: {root}\n")

    if a.domain_gap is not None:
        if a.domain_gap:
            folders = [Path(f) for f in a.domain_gap]
        else:
            base = default_data_root()
            folders = [base / n / "images" for n in
                       ("ExternalCosmetics", "ExternalCosmetics2", "ExternalEval")]
        domain_gap(root, folders)
        return 0

    if a.build_reference:
        build_reference(root)
        if not (a.check or a.self_test):
            return 0

    valid = set(pd.read_csv(root / "train" / "styles_train.csv",
                            usecols=["articleType"])["articleType"].dropna())
    ref = load_reference()

    target = Path(a.check) if a.check else None
    if a.self_test:
        target = root / "test" / "images_test"
        print("SELF-TEST: feeding the provided test folder as candidates.")
        print("           a working gate must report ~100% test leakage.\n")

    if target is None:
        ap.error("pass --check FOLDER, or --self-test, or --build-reference")

    rep = check_folder(target, ref, Path(a.labels) if a.labels else None, valid)
    print("\n" + json.dumps(rep, indent=2))

    man = target.parent / f"{target.name}_gate_manifest.json"
    man.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"\nmanifest -> {man}")

    if a.self_test:
        ok = rep["matches_provided_TEST_total"] >= 0.99 * rep["images_readable"]
        print(f"\nSELF-TEST {'PASSED' if ok else 'FAILED'}: detected "
              f"{rep['matches_provided_TEST_total']}/{rep['images_readable']}")
        return 0 if ok else 1

    return 1 if rep["verdict"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
