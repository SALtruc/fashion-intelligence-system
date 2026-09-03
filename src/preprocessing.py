"""Shared data access for the COSC2753 A2 task notebooks.

Every decision encoded here is justified in 00_eda_and_preprocessing.ipynb.
Nothing in this module makes a modelling choice: no architecture, no hyperparameter,
no metric. It loads the audited manifest, applies the one deterministic image
transform, splits a task frame, and fits normalisation constants on training rows.

Sections referenced in the docstrings are sections of that notebook.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

# Anchored to this file rather than to the caller's working directory, so a notebook in
# notebooks/ and a script run from the repository root both resolve to the same data.
_REPO_ROOT = Path(__file__).resolve().parent.parent

MANIFEST = _REPO_ROOT / "preprocessed_datasets" / "train_manifest.csv"
TRAIN_IMAGE_DIR = _REPO_ROOT / "datasets" / "train" / "images_train"
TEST_IMAGE_DIR = _REPO_ROOT / "datasets" / "test" / "images_test"

# The catalogue's modal image size, established in Section 3.1. The notebook derives this
# from the data and asserts that the value it derived matches the constant below, so the
# two cannot drift apart.
IMAGE_TARGET_SIZE = (60, 80)      # width, height
IMAGE_PAD_RGB = (255, 255, 255)   # catalogue background, used when padding

RANDOM_STATE = 42

TARGETS = ["articleType", "season", "gender", "usage"]


def load_manifest(target=None, image_dir=TRAIN_IMAGE_DIR):
    """Read the audited manifest, optionally keeping only rows labelled for one target.

    Filtering on one target rather than dropping incomplete rows globally is the rule set
    out in Section 3.3: a row missing `usage` is still valid for `articleType`, and a
    global filter would impose one task's label gaps on another.

    Args:
        target: column name to filter on, or None for the full retrieval pool (Task 4).
        image_dir: directory holding the JPEGs named in the `filename` column.

    Returns:
        The manifest with a `path` column added, filtered to rows whose target is present.
    """
    frame = pd.read_csv(MANIFEST)

    if target is not None:
        if target not in frame.columns:
            raise KeyError(f"{target!r} is not a manifest column. Available: {list(frame.columns)}")
        frame = frame.loc[frame[target].notna()].reset_index(drop=True)

    # str rather than Path: a Path survives PIL but writes a machine-local absolute path
    # if a split is ever exported to CSV.
    frame["path"] = frame["filename"].map(lambda name: str(Path(image_dir) / name))
    return frame


def standardize_image(image, target_size=None):
    """Deterministic Task 1-4 input transform. Augmentation after this, normalisation last.

    Converts to RGB and pads to the target size, preserving the portrait aspect ratio.
    Padding rather than cropping or stretching is argued in Section 3.1: the portrait
    shape carries class cues at both the top and bottom of a product, and 38,595 of
    38,612 images are already at the target size and so pass through unresampled.

    target_size resolves at call time rather than at definition time, so overriding
    IMAGE_TARGET_SIZE takes effect without reimporting the module.

    Args:
        image: an open PIL image.
        target_size: (width, height), defaulting to IMAGE_TARGET_SIZE.

    Returns:
        A PIL image in RGB mode at exactly target_size.
    """
    target_size = tuple(target_size or IMAGE_TARGET_SIZE)
    image = ImageOps.exif_transpose(image).convert("RGB")
    if image.size == target_size:
        return image  # already the target size: no resampling at all
    return ImageOps.pad(
        image, target_size, method=Image.Resampling.BILINEAR,
        color=IMAGE_PAD_RGB, centering=(0.5, 0.5),
    )


def load_image_array(path, target_size=None, scale=True):
    """Load one image through the standard transform.

    Args:
        path: file location.
        target_size: passed to standardize_image.
        scale: divide by 255 so values land in [0, 1]. Normalisation constants from
            compute_normalisation are fitted on that scale.

    Returns:
        An array of shape (height, width, 3), float32 if scaled, uint8 otherwise.
    """
    with Image.open(path) as image:
        pixels = np.asarray(standardize_image(image, target_size))
    return pixels.astype(np.float32) / 255.0 if scale else pixels


def make_split(frame, target, validation_share=0.2, random_state=RANDOM_STATE):
    """Split one task frame into training and validation rows.

    Whole `group_id` values stay on one side, so byte identical images cannot appear in
    both. In practice Section 3.2 already collapsed every multi-row group, so the
    manifest has one row per group and the grouping here is a verified invariant rather
    than an active control; the assertion below is what verifies it. It matters only if
    the duplicate handling upstream ever changes.

    Classes represented by fewer than two groups go entirely to training, since a class
    with one example cannot also be evaluated (Section 5, rule 4). The remainder is
    stratified on the class label. Because those classes are held out of the stratified
    draw, the achieved validation share is slightly below `validation_share`.

    Args:
        frame: output of load_manifest(target), or any frame with `group_id` and target.
        target: the label column to stratify on.
        validation_share: fraction of the splittable groups held out.
        random_state: seed for the draw.

    Returns:
        A tuple of (training frame, validation frame).
    """
    if target not in frame.columns:
        raise KeyError(f"{target!r} is not a column of the frame passed in.")

    frame = frame.loc[frame[target].notna()].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No rows carry a {target!r} label.")

    assert frame["group_id"].is_unique, (
        "Multi-row group_id values found. Section 3.2 of the preprocessing notebook "
        "should have collapsed or resolved every duplicate group; re-run it before splitting."
    )

    # value_counts().index[0] is safe here because the NaN rows were dropped above.
    group_label = frame.groupby("group_id")[target].agg(
        lambda values: values.value_counts().index[0]
    )
    groups_per_class = group_label.value_counts()
    single_group_classes = groups_per_class[groups_per_class < 2].index

    splittable = group_label[~group_label.isin(single_group_classes)]
    if splittable.empty:
        raise ValueError(
            f"Every {target!r} class has fewer than two groups, so no validation set "
            "can be drawn."
        )

    n_validation = int(round(len(splittable) * validation_share))
    n_classes = splittable.nunique()
    if n_validation < n_classes:
        raise ValueError(
            f"validation_share={validation_share} yields {n_validation} validation groups "
            f"for {n_classes} classes, which is too few to stratify. Raise it, or drop "
            "stratification for this target."
        )

    _, validation_groups = train_test_split(
        splittable.index,
        test_size=validation_share,
        stratify=splittable.values,
        random_state=random_state,
    )

    is_validation = frame["group_id"].isin(set(validation_groups))
    return (
        frame.loc[~is_validation].reset_index(drop=True),
        frame.loc[is_validation].reset_index(drop=True),
    )


def describe_split(training, validation, target):
    """Summarise a split, including the classes validation cannot score.

    Classes held entirely in training, and classes whose two or three groups all landed
    in training by chance, contribute a zero to any macro average. Reporting the count
    is what makes that visible rather than letting it silently depress the headline.

    Returns:
        A one-row DataFrame.
    """
    training_classes = set(training[target].unique())
    validation_classes = set(validation[target].unique())
    total = len(training) + len(validation)

    return pd.DataFrame([{
        "Target": target,
        "Training rows": len(training),
        "Validation rows": len(validation),
        "Achieved validation share %": len(validation) / total * 100,
        "Classes in training": len(training_classes),
        "Classes in validation": len(validation_classes),
        "Classes absent from validation": len(training_classes - validation_classes),
    }])


def compute_normalisation(frame, target_size=None, sample=None, random_state=RANDOM_STATE):
    """Per-channel mean and standard deviation over the frame's images, in [0, 1].

    Section 5, rule 5: fit this on the training frame from make_split only, then apply
    the result unchanged to validation and to the test set. Recompute whenever
    target_size changes, because white padding shifts the RGB distribution.

    Computed in one streaming pass from running sums, so memory does not scale with the
    number of images.

    Args:
        frame: rows to fit on, carrying a `path` column.
        target_size: passed to standardize_image; must match what training will use.
        sample: fit on this many rows instead of all of them, for a quick estimate.
        random_state: seed for that sample.

    Returns:
        A tuple of (mean, std), each an array of three floats in channel order R, G, B.
    """
    rows = frame if sample is None else frame.sample(
        n=min(sample, len(frame)), random_state=random_state
    )
    if rows.empty:
        raise ValueError("No rows to fit normalisation constants on.")

    total = np.zeros(3, dtype=np.float64)
    total_squared = np.zeros(3, dtype=np.float64)
    n_pixels = 0

    for path in rows["path"]:
        flat = load_image_array(path, target_size).reshape(-1, 3).astype(np.float64)
        total += flat.sum(axis=0)
        total_squared += (flat ** 2).sum(axis=0)
        n_pixels += len(flat)

    mean = total / n_pixels
    variance = np.maximum(total_squared / n_pixels - mean ** 2, 0.0)  # guard float error
    return mean, np.sqrt(variance)