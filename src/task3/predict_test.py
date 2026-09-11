"""Predict `gender` and `usage` for the test set from the saved Task 3 checkpoint.

`finalise_task3.py` trains the ultimate judgement and then predicts, which is the right
thing the first time and the wrong thing every time after: reproducing the submission CSV
should not cost an hour of GPU and should not risk landing on a different model. This
script only loads `artifacts/task3/task3_gender_usage_C_weighted.pt` and runs it.

    python src/task3/predict_test.py

Everything the model needs travels inside the checkpoint -- class order per target, image
size, and the channel statistics fitted on the training rows. Nothing is re-derived here,
because a statistic recomputed on the test set would be fitted to the data being
predicted.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data_paths                       # noqa: E402
from src.task3.models import Net                 # noqa: E402

CHECKPOINT = ROOT / "artifacts" / "task3" / "task3_gender_usage_C_weighted.pt"
OUTPUT = ROOT / "predictions" / "task3" / "task3_gender_usage_nguyen.csv"


def load_batch(paths, size):
    """Decode to the catalogue's own size, padding on white rather than stretching.

    Identical to the transform the notebook trained through; padding colour matters
    because the catalogue background is white and a stretched item is a different shape
    from anything the model saw.
    """
    width, height = size
    out = np.zeros((len(paths), height, width, 3), dtype=np.uint8)
    for i, path in enumerate(paths):
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            if image.size != (width, height):
                image = ImageOps.pad(image, (width, height), method=Image.Resampling.BILINEAR,
                                     color=(255, 255, 255), centering=(0.5, 0.5))
            out[i] = np.asarray(image)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(
            f"{args.checkpoint} is absent. artifacts/ is gitignored -- get it from the team "
            "Drive, or rebuild it with src/task3/finalise_task3.py."
        )
    data_paths.check()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes, heads = ck["classes"], ck["heads"]
    size = tuple(ck["image_size"])
    mean = torch.tensor(ck["channel_mean"]).view(1, 3, 1, 1)
    std = torch.tensor(ck["channel_std"]).view(1, 3, 1, 1)
    targets = list(heads)
    print(f"{ck['design']}")
    print(f"targets {targets} | image {size} | mirror TTA {ck['use_tta']}")
    for target in targets:
        print(f"  {target:7} {len(classes[target])} classes, "
              f"validation macro-F1 {ck['val_macro_f1'][target]:.4f}")

    device = torch.device(args.device)
    model = Net(heads).to(device).eval()
    model.load_state_dict(ck["state_dict"], strict=True)

    template_path, image_dir = data_paths.test_template(), data_paths.test_images()
    with template_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns, rows = reader.fieldnames, list(reader)
    if columns != ["id", "gender", "articleType", "season", "usage"]:
        raise SystemExit(f"Unexpected template columns: {columns}")
    missing = [r["id"] for r in rows if not (image_dir / f"{r['id']}.jpg").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} test images are missing, first {missing[:3]}")
    print(f"\ntemplate {template_path} | {len(rows):,} rows")

    predictions = {t: [] for t in targets}
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            arrays = load_batch([image_dir / f"{r['id']}.jpg" for r in batch], size)
            x = torch.from_numpy(arrays).to(device).permute(0, 3, 1, 2).float() / 255.0
            x = (x - mean.to(device)) / std.to(device)
            logits = model(x)
            if ck["use_tta"]:
                # The checkpoint was selected with mirror TTA on validation, so scoring
                # without it here would not be the model that was chosen.
                flipped = model(torch.flip(x, dims=[3]))
                logits = {t: (logits[t].softmax(1) + flipped[t].softmax(1)) / 2
                          for t in targets}
            for t in targets:
                predictions[t].extend(logits[t].argmax(1).cpu().tolist())
            print(f"  {min(start + len(batch), len(rows)):,}/{len(rows):,}", flush=True)

    for row, *picks in zip(rows, *(predictions[t] for t in targets)):
        for t, index in zip(targets, picks):
            row[t] = classes[t][int(index)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows):,} rows -> {args.output.relative_to(ROOT)}")
    for t in targets:
        counts = {}
        for row in rows:
            counts[row[t]] = counts.get(row[t], 0) + 1
        print(f"  {t:7} " + "  ".join(f"{k} {v:,}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
