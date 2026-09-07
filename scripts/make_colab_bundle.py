#!/usr/bin/env python
"""Pack the data a hosted session needs into one zip, in the flat layout it unpacks to.

Used by both targets, differently: on Colab you drag this file into each session and the
staging cell unpacks it; on Kaggle you upload it once as a Dataset, Kaggle extracts it itself,
and every later session mounts the result read-only. The archive is the same either way.

A Colab runtime starts empty and every session pays the staging cost again, so the shape of
this bundle is the thing that decides whether starting a session takes one minute or forty.
Three choices follow from that:

  * **One archive, not a folder of loose files.** The training set is 38,612 files, and the
    browser upload path opens a request per file. Dragging one 590 MB archive into the session
    and unpacking it onto the runtime's local disk is about a minute.
  * **Stored, not deflated.** JPEG and .npz are already compressed. Deflate spends minutes of
    CPU on both ends to save a percent or two.
  * **The flat layout, baked in.** `notebooks/Task1/dataset1/` is rewritten to `dataset1/` as
    it is added, so the archive already holds the layout the Colab notebooks expect and the
    unpack is a plain extract with nothing to move afterwards.

What is deliberately left out:

  * `datasets/train/styles_train.csv`. No Task 1 notebook opens it -- only notebook 00 does,
    and the manifest that notebook writes is tracked, so the EDA run is not repeated per
    session.
  * Everything under `models/`. Checkpoints are the *output* of a session: each one leaves as
    its own `checkpoints_<job>_<arm>.zip`, dragged into whichever session needs it next.
  * The audit and quarantine directories under the external collections. The evaluation
    notebook reads the CSV and the images; the audit trail is provenance for the report and
    is not needed on a runtime.

    python scripts/make_colab_bundle.py                  # everything, ~590 MB
    python scripts/make_colab_bundle.py --no-test        # worker sessions only, ~504 MB
    python scripts/make_colab_bundle.py --check          # verify an existing bundle
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLAB_DIR = ROOT / "task1-collab"
DEFAULT_OUTPUT = ROOT / "task1_colab_data.zip"

COMBINE_NAME = "01_task1_article_type.ipynb"


def entries(include_test=True, include_external=True):
    """(source path, name inside the archive) for everything the bundle carries.

    The archive name is the *flat* location, which is why the external collections change
    prefix here rather than on the runtime.
    """
    items: list[tuple[Path, str]] = []

    def add_file(source, name):
        items.append((source, name))

    def add_tree(source_dir, prefix, pattern="*"):
        for path in sorted(source_dir.rglob(pattern)):
            if path.is_file():
                add_file(path, f"{prefix}/{path.relative_to(source_dir).as_posix()}")

    # --- Always: what every worker session reads -----------------------------------------
    add_file(ROOT / "src" / "preprocessing.py", "src/preprocessing.py")
    add_file(ROOT / "preprocessed_datasets" / "train_manifest.csv",
             "preprocessed_datasets/train_manifest.csv")
    add_tree(ROOT / "datasets" / "train" / "images_train", "datasets/train/images_train",
             "*.jpg")

    # The arms split. Small, and carrying it always means a session can switch to the
    # enriched arm by editing one string rather than by re-staging.
    arms = ROOT / "preprocessed_datasets" / "task1_dataset1_arms"
    if arms.is_dir():
        add_tree(arms, "preprocessed_datasets/task1_dataset1_arms")

    # --- The test set: the combine session's prediction run only --------------------------
    if include_test:
        add_file(ROOT / "datasets" / "test" / "styles_prediction.csv",
                 "datasets/test/styles_prediction.csv")
        add_tree(ROOT / "datasets" / "test" / "images_test", "datasets/test/images_test",
                 "*.jpg")

    # --- The external collections: the enriched arm and the evaluation notebook -----------
    if include_external:
        for collection, csv_name in (("dataset1", "external_cosmetics.csv"),
                                     ("dataset2", "external_cosmetics2.csv")):
            source = ROOT / "notebooks" / "Task1" / collection
            if not source.is_dir():
                continue
            if (source / csv_name).is_file():
                add_file(source / csv_name, f"{collection}/{csv_name}")
            add_tree(source / "images", f"{collection}/images", "*.jpg")

    # The evaluation notebook lifts the SmallResNet class straight out of the combine
    # notebook's source, so a copy has to exist on the runtime's filesystem. The notebook open
    # in the browser tab is not one: Colab holds it server-side, not under /content. Carrying
    # it in the bundle is what makes the evaluation session self-contained.
    combine = COLAB_DIR / COMBINE_NAME
    if combine.is_file():
        add_file(combine, COMBINE_NAME)

    return items


def build(output, items):
    total = 0
    # ZIP_STORED: JPEGs and .npz files are already compressed.
    with zipfile.ZipFile(output, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
        for source, name in items:
            archive.write(source, name)
            total += source.stat().st_size
    return total


def summarise(items):
    """Bytes and file count per top-level group, which is what the README quotes."""
    groups: dict[str, list[int]] = {}
    for source, name in items:
        key = "/".join(name.split("/")[:2]) if "/" in name else name
        entry = groups.setdefault(key, [0, 0])
        entry[0] += 1
        entry[1] += source.stat().st_size
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"where to write the archive (default {DEFAULT_OUTPUT.name})")
    parser.add_argument("--no-test", action="store_true",
                        help="omit datasets/test/; worker sessions do not read it")
    parser.add_argument("--no-external", action="store_true",
                        help="omit dataset1/ and dataset2/; supplied-arm workers do not read them")
    parser.add_argument("--check", action="store_true",
                        help="list what an existing archive holds, write nothing")
    arguments = parser.parse_args()

    if arguments.check:
        if not arguments.output.is_file():
            print(f"{arguments.output} does not exist.")
            return 1
        with zipfile.ZipFile(arguments.output) as archive:
            names = archive.namelist()
        groups: dict[str, int] = {}
        for name in names:
            key = "/".join(name.split("/")[:2]) if "/" in name else name
            groups[key] = groups.get(key, 0) + 1
        print(f"{arguments.output.name}  "
              f"({arguments.output.stat().st_size / 1e6:.0f} MB, {len(names):,} files)\n")
        for key in sorted(groups):
            print(f"  {key:<44} {groups[key]:>7,} files")
        missing = [required for required in
                   ("src/preprocessing.py", "preprocessed_datasets/train_manifest.csv")
                   if required not in names]
        if missing:
            print("\n  MISSING:", ", ".join(missing))
            return 1
        return 0

    items = entries(include_test=not arguments.no_test,
                    include_external=not arguments.no_external)

    absent = [str(source.relative_to(ROOT)) for source, _ in items if not source.is_file()]
    if absent:
        raise FileNotFoundError("Not found in this checkout:\n  " + "\n  ".join(absent))

    print(f"Packing {len(items):,} files into {arguments.output} ...")
    total = build(arguments.output, items)

    print(f"\nWrote {arguments.output} "
          f"({arguments.output.stat().st_size / 1e6:.0f} MB on disk, "
          f"{total / 1e6:.0f} MB of content)\n")
    for key, (count, size) in sorted(summarise(items).items()):
        print(f"  {key:<44} {count:>7,} files  {size / 1e6:>7.1f} MB")

    print("\nUpload it into each Colab session: open the file browser in the left "
          "sidebar and drag this file onto /content, then Runtime -> Run all.")
    print("Every session unpacks it once, and it is deleted with the runtime -- so keep "
          "this local copy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
