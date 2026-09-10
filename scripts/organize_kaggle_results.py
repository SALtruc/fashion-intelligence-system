"""Copy a Kaggle Task 1 download into the repository's canonical layout.

The downloaded result is treated as immutable evidence. Existing files are never
silently replaced: an identical file is skipped and a different file raises an
error. Run with --force only after reviewing a reported conflict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "results" / "task1_full_16ay9832"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def copy_checked(source: Path, destination: Path, copied: list[dict], force: bool) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    source_hash = digest(source)
    if destination.exists():
        if destination.is_file() and digest(destination) == source_hash:
            copied.append({"source": str(source.relative_to(ROOT)),
                           "destination": str(destination.relative_to(ROOT)),
                           "status": "already-identical", "sha256": source_hash})
            return
        if not force:
            raise FileExistsError(
                f"Destination differs from Kaggle source: {destination}. "
                "Use --force only after reviewing the conflict."
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if digest(destination) != source_hash:
        raise IOError(f"Copy verification failed: {destination}")
    copied.append({"source": str(source.relative_to(ROOT)),
                   "destination": str(destination.relative_to(ROOT)),
                   "status": "copied", "sha256": source_hash})


def organize(source: Path, force: bool = False) -> Path:
    source = source.resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    task_root = source / "models" / "task1"
    if not (task_root / "run.json").is_file():
        raise FileNotFoundError(f"Not a complete Task 1 run: {task_root / 'run.json'}")

    copied: list[dict] = []

    # The model artifact tree is the repository's canonical Task 1 output.
    for path in task_root.rglob("*"):
        if path.is_file():
            copy_checked(path, ROOT / "models" / "task1" / path.relative_to(task_root), copied, force)

    # Make the submission prediction and exact split membership easy to find.
    copy_checked(task_root / "predictions" / "task1_predictions.csv",
                 ROOT / "predictions" / "task1" / "task1_predictions.csv", copied, force)
    for split in ("fit", "tuning", "reporting"):
        copy_checked(task_root / "tables" / f"split_{split}.csv",
                     ROOT / "splits" / "task1" / f"{split}.csv", copied, force)

    # Figures belong with other generated outputs, while retaining the task namespace.
    for path in (task_root / "figures").glob("*.png"):
        copy_checked(path, ROOT / "outputs" / "figures" / "task1" / path.name, copied, force)

    # Root-level Kaggle files are provenance/telemetry, not models or submission data.
    provenance = ROOT / "artifacts" / "task1" / "kaggle-full-16ay9832"
    for path in ("task1_ddp_runtime.py", "task1_ddp.log", "phase.json",
                 "phase_durations.jsonl", "gpu_telemetry.csv", "gpu_telemetry_summary.json"):
        copy_checked(source / path, provenance / path, copied, force)
    (provenance / "README.md").write_text(
        "# Kaggle Task 1 full run\n\n"
        "This folder contains the root-level runtime and telemetry files from the Kaggle download. "
        "The verified model tree is under `models/task1/`. A source download is not modified by this script; " +
        "the large ZIP is not copied.\n",
    )

    manifest = {
        "source": str(source.relative_to(ROOT)),
        "source_preserved": True,
        "canonical_model_root": "models/task1",
        "canonical_prediction": "predictions/task1/task1_predictions.csv",
        "canonical_splits": "splits/task1/{fit,tuning,reporting}.csv",
        "canonical_figures": "outputs/figures/task1",
        "provenance": "artifacts/task1/kaggle-full-16ay9832",
        "files": copied,
    }
    manifest_path = ROOT / "artifacts" / "task1" / "kaggle-full-16ay9832" / "ORGANIZATION_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--force", action="store_true",
                        help="replace differing destination files after review")
    args = parser.parse_args()
    manifest = organize(args.source, args.force)
    print(f"Organized {len(json.loads(manifest.read_text(encoding='utf-8'))['files'])} files.")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()


