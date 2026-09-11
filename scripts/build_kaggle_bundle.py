#!/usr/bin/env python3
"""Build the complete Task 1 Kaggle dataset bundle (task1_kaggle_bundle.zip).

This script packages all data required by `01_task1_article_type_classification.ipynb`
when running on Kaggle with `KAGGLE = True`:
  1. `datasets/` (train & test images + CSV manifests)
  2. `preprocessed_datasets/` (train_manifest.csv)
  3. `data/external_task1/` (60 independent evaluation images + labels)
  4. `models/task1/` (verified tables, figures, metadata, and checkpoints)

Usage:
    python scripts/build_kaggle_bundle.py [--output task1_kaggle_bundle.zip]
"""

import argparse
import csv
import hashlib
from pathlib import Path
import sys
import time
import zipfile

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def find_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for p in (here, *here.parents):
        if (p / "preprocessed_datasets" / "train_manifest.csv").is_file():
            return p
    raise FileNotFoundError("Cannot locate repository root containing preprocessed_datasets/train_manifest.csv")

def main():
    parser = argparse.ArgumentParser(description="Package Task 1 Kaggle dataset bundle.")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Destination path for task1_kaggle_bundle.zip (default: repo root / task1_kaggle_bundle.zip)")
    parser.add_argument("--skip-checkpoint", action="store_true",
                        help="Skip embedding the 45 MB model checkpoint in the bundle.")
    args = parser.parse_args()

    root = find_repo_root()
    output_zip = args.output or (root / "task1_kaggle_bundle.zip")

    print("=" * 75)
    print("TASK 1 KAGGLE DATASET BUNDLE BUILDER")
    print(f"Repository Root: {root}")
    print(f"Target Archive : {output_zip}")
    print("=" * 75)

    # 1. Verify critical prerequisites
    manifest_path = root / "preprocessed_datasets" / "train_manifest.csv"
    test_csv_path = root / "datasets" / "test" / "styles_prediction.csv"
    train_img_dir = root / "datasets" / "train" / "images_train"
    test_img_dir = root / "datasets" / "test" / "images_test"

    for req in [manifest_path, test_csv_path, train_img_dir, test_img_dir]:
        if not req.exists():
            print(f"ERROR: Missing required data path: {req}", file=sys.stderr)
            sys.exit(1)

    print("\n[1/3] Validating image references against manifests...")
    with open(manifest_path, newline="", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))
    with open(test_csv_path, newline="", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))

    required_train_imgs = [train_img_dir / r["filename"] for r in train_rows if r.get("articleType")]
    missing_train = [p for p in required_train_imgs if not p.is_file()]
    if missing_train:
        print(f"ERROR: {len(missing_train)} train images missing! Sample: {missing_train[:3]}", file=sys.stderr)
        sys.exit(1)

    required_test_imgs = [test_img_dir / f"{r['id']}.jpg" for r in test_rows]
    missing_test = [p for p in required_test_imgs if not p.is_file()]
    if missing_test:
        print(f"ERROR: {len(missing_test)} test images missing! Sample: {missing_test[:3]}", file=sys.stderr)
        sys.exit(1)

    print(f"  [OK] Validated {len(required_train_imgs):,} training images")
    print(f"  [OK] Validated {len(required_test_imgs):,} test images")

    # 2. Collect files to include
    print("\n[2/3] Collecting files to archive...")
    files_to_pack = []

    # Datasets
    for p in (root / "datasets").rglob("*"):
        if p.is_file() and p.suffix.lower() in [".csv", ".jpg", ".jpeg", ".png"]:
            rel = p.relative_to(root)
            files_to_pack.append((p, rel.as_posix()))

    # Preprocessed datasets
    for p in (root / "preprocessed_datasets").rglob("*"):
        if p.is_file() and p.suffix.lower() in [".csv", ".json"]:
            rel = p.relative_to(root)
            files_to_pack.append((p, rel.as_posix()))

    # External evaluation data
    ext_dir = root / "data" / "external_task1"
    if ext_dir.is_dir():
        for p in ext_dir.rglob("*"):
            if p.is_file():
                rel = p.relative_to(root)
                files_to_pack.append((p, rel.as_posix()))

    # Models metadata and tables
    models_dir = root / "models" / "task1"
    if models_dir.is_dir():
        for p in models_dir.rglob("*"):
            if p.is_file():
                if p.suffix.lower() == ".pt" and args.skip_checkpoint:
                    continue
                # Keep tables, figures, jsons, and final checkpoints
                if p.suffix.lower() in [".csv", ".json", ".png", ".pt", ".joblib"]:
                    rel = p.relative_to(root)
                    files_to_pack.append((p, rel.as_posix()))

    total_uncompressed = sum(src.stat().st_size for src, _ in files_to_pack)
    print(f"  [OK] Total files identified: {len(files_to_pack):,}")
    print(f"  [OK] Total uncompressed size: {total_uncompressed / (1024 * 1024):.1f} MB")

    # 3. Create zip archive
    print(f"\n[3/3] Compressing into {output_zip.name}...")
    t0 = time.time()
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    # Use ZIP_DEFLATED for CSVs/JSONs, fast compression
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for idx, (src_file, arc_name) in enumerate(files_to_pack, 1):
            zf.write(src_file, arcname=arc_name)
            if idx % 5000 == 0 or idx == len(files_to_pack):
                elapsed = time.time() - t0
                pct = idx / len(files_to_pack) * 100
                rate = idx / elapsed if elapsed > 0 else 0
                print(f"  [{pct:5.1f}%] Packed {idx:,} / {len(files_to_pack):,} files ({rate:.0f} files/s)")

    duration = time.time() - t0
    final_size = output_zip.stat().st_size
    print(f"\nDone in {duration:.1f}s!")
    print(f"Archive Size: {final_size / (1024 * 1024):.1f} MB ({final_size / (1024**3):.2f} GB)")

    print("\nComputing SHA-256 checksum...")
    checksum = sha256_file(output_zip)
    print(f"SHA-256: {checksum}")

    # Write a small manifest text file next to the zip
    manifest_txt = output_zip.with_suffix(".sha256.txt")
    manifest_txt.write_text(f"{checksum}  {output_zip.name}\n", encoding="utf-8")

    print("\n" + "=" * 75)
    print("HOW TO RUN ON KAGGLE:")
    print("1. Go to Kaggle (https://www.kaggle.com/datasets)")
    print("2. Click '+ New Dataset'")
    print(f"3. Upload '{output_zip.name}' and give it a title (e.g., 'task1-dataset')")
    print("4. In your Kaggle Notebook:")
    print("   • Click '+ Add Input' and select your newly created dataset")
    print("   • Set Accelerator to 'GPU T4 x2' (or P100)")
    print("   • Turn Internet ON in Notebook Settings (for HuggingFace backbones)")
    print("   • Verify cell 1 has 'KAGGLE = True' and click 'Run All'")
    print("=" * 75)

if __name__ == "__main__":
    main()
