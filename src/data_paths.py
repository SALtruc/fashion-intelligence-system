"""Where the course data lives, answered in one place.

Every task used to resolve the dataset its own way -- Task 1 walked up to
``datasets/``, Task 3 only ever looked at Colab mounts, and Task 4 expected
``preprocessed_datasets/train/images_train``.
Three layouts meant a checkout could satisfy at most one of them, so "clone and
run" failed for whichever task the machine was not set up for.

The documented layout is the one in the README, and it is what a fresh checkout
should aim at::

    datasets/train/styles_train.csv
    datasets/train/images_train/
    datasets/test/styles_prediction.csv
    datasets/test/images_test/

The course archive unpacks to ``datasets/FashionDataset/{train,test}`` and the
Colab bundle carries ``preprocessed_datasets/{train,test}``, so both are accepted
where they are already in place: renaming a 38k-image directory to satisfy a path
constant is not a useful thing to ask of a marker. ``A2_DATA_ROOT`` overrides
everything, which is how a machine keeping the data on another drive joins in.

The lookups are deliberately not cached. A Colab session that mounts Drive after
the first import should see the mount on the next call rather than a stale miss.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = ["repo_root", "data_roots", "raw_train_table", "train_table",
           "train_images", "test_template", "test_images", "describe", "check"]

# Relative to a data root. First hit wins, so the documented layout is listed first.
_TRAIN_TABLE = ("train/styles_train.csv", "styles_train.csv")
_TRAIN_IMAGES = ("train/images_train", "images_train")
_TEST_TEMPLATE = ("test/styles_prediction.csv", "styles_prediction.csv")
_TEST_IMAGES = ("test/images_test", "images_test")


def repo_root(start: Path | None = None) -> Path:
    """The nearest ancestor holding pyproject.toml, else this file's repository."""
    here = Path(start or Path.cwd()).resolve()
    for base in (here, *here.parents):
        if (base / "pyproject.toml").is_file():
            return base
    return Path(__file__).resolve().parents[1]


def data_roots() -> list[Path]:
    """Candidate data roots, most specific first."""
    root = repo_root()
    candidates = [
        os.environ.get("A2_DATA_ROOT"),
        root / "datasets",
        root / "datasets" / "FashionDataset",
        root / "datasets" / "A2_Fashion",
        root / "preprocessed_datasets",
        # Colab: the bundle unzips flat into /content, or into a named folder.
        Path("/content"),
        Path("/content/preprocessed_datasets"),
        Path("/content/ColabDataset"),
        Path("/content/ColabDataset/preprocessed_datasets"),
        Path("/content/drive/MyDrive/ColabDataset"),
    ]
    # Kaggle attaches datasets read-only under /kaggle/input/<name>, one directory
    # per attached dataset, so the names cannot be listed ahead of time.
    kaggle = Path("/kaggle/input")
    if kaggle.is_dir():
        candidates.extend(sorted(p for p in kaggle.iterdir() if p.is_dir()))
    return [Path(c) for c in candidates if c is not None]


def _find(relatives: tuple[str, ...], default_relative: str) -> Path:
    """First existing <root>/<relative>, else the documented location.

    Returning a path that does not exist rather than raising keeps ``import
    src.task4.config`` working in a checkout with no data; call `check()` when a
    missing dataset should be reported up front.
    """
    for root in data_roots():
        for relative in relatives:
            candidate = root / relative
            if candidate.exists():
                return candidate
    return repo_root() / "datasets" / default_relative


def raw_train_table() -> Path:
    """The course's own styles_train.csv, before any auditing.

    Only the EDA notebook should want this: it is the input the manifest is built
    from, so reading the manifest there would be circular.
    """
    return _find(_TRAIN_TABLE, "train/styles_train.csv")


def train_table() -> Path:
    """The labelled training table.

    The audited manifest is preferred over the raw CSV wherever it exists: it is
    the 37,847-row table the EDA notebook writes, it is tracked in git, and every
    task that reads it lands on the same rows. The raw file is the fallback for a
    checkout that has not run the EDA notebook yet.
    """
    manifest = repo_root() / "preprocessed_datasets" / "train_manifest.csv"
    if manifest.is_file():
        return manifest
    return raw_train_table()


def train_images() -> Path:
    return _find(_TRAIN_IMAGES, "train/images_train")


def test_template() -> Path:
    return _find(_TEST_TEMPLATE, "test/styles_prediction.csv")


def test_images() -> Path:
    return _find(_TEST_IMAGES, "test/images_test")


def describe() -> dict[str, tuple[Path, bool]]:
    """Every resolved path with whether it is actually there."""
    return {name: (path, path.exists()) for name, path in (
        ("train table", train_table()),
        ("train images", train_images()),
        ("test template", test_template()),
        ("test images", test_images()),
    )}


def check(*, need_test: bool = True) -> None:
    """Raise once, naming everything that is missing and everywhere that was tried."""
    resolved = describe()
    if not need_test:
        resolved = {k: v for k, v in resolved.items() if not k.startswith("test")}
    missing = [name for name, (_, ok) in resolved.items() if not ok]
    if not missing:
        return
    tried = "\n  ".join(str(root) for root in data_roots())
    raise FileNotFoundError(
        "Course data not found: " + ", ".join(missing) + ".\n\n"
        "Expected, relative to the repository root:\n"
        "  datasets/train/styles_train.csv\n"
        "  datasets/train/images_train/\n"
        "  datasets/test/styles_prediction.csv\n"
        "  datasets/test/images_test/\n\n"
        "Data roots tried:\n  " + tried + "\n\n"
        "Set A2_DATA_ROOT to point somewhere else. See the README's dataset section."
    )
