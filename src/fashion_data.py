"""Shared data access for the COSC2753 A2 task notebooks.

Every decision encoded here is justified in 00_eda_and_preprocessing.ipynb.
Nothing in this module makes a modelling choice.
"""
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

MANIFEST = Path("../preprocessed_datasets/train_manifest.csv")
TRAIN_IMAGE_DIR = Path("../datasets/train/images_train")
IMAGE_TARGET_SIZE = (60, 80)
IMAGE_PAD_RGB = (255, 255, 255)
RANDOM_STATE = 42


def load_manifest(target=None):
    """Read the audited manifest, optionally keeping only rows labelled for one target.

    Args:
        target: column name to filter on, or None for the full retrieval pool.

    Returns:
        The manifest with a `path` column added, filtered to rows whose target is present.
    """
    frame = pd.read_csv(MANIFEST)
    if target is not None:
        frame = frame.loc[frame[target].notna()].reset_index(drop=True)
    frame["path"] = frame["filename"].map(lambda name: TRAIN_IMAGE_DIR / name)
    return frame


def standardize_image(image, target_size=IMAGE_TARGET_SIZE):
    """Deterministic Task 1-4 input transform. Augmentation after this, normalisation last."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    if image.size == tuple(target_size):
        return image
    return ImageOps.pad(
        image, target_size, method=Image.Resampling.BILINEAR,
        color=IMAGE_PAD_RGB, centering=(0.5, 0.5),
    )

def make_split(frame, target, validation_share=0.2, random_state=RANDOM_STATE):
    """Split one task frame into training and validation rows.

    Whole `group_id` values stay on one side, so the byte identical images found in
    Section 1.6 cannot appear in both. Classes represented by fewer than two groups go
    entirely to training, since a class with one example cannot also be evaluated. The
    remainder is stratified on the class label.

    Returns:
        A tuple of (training frame, validation frame).
    """
    group_label = frame.groupby("group_id")[target].agg(lambda values: values.value_counts().index[0])
    groups_per_class = group_label.value_counts()
    single_group_classes = groups_per_class[groups_per_class < 2].index

    splittable = group_label[~group_label.isin(single_group_classes)]
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