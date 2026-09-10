import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_metric_learning.samplers import MPerClassSampler
from torch.utils.data import DataLoader, Dataset

from src.task4.config import (
    EVAL_BATCH_SIZE,
    IMAGE_DIR,
    IMAGES_PER_CLASS,
    LABEL_ID_COLUMN,
    METRIC_BATCH_SIZE,
    NUM_WORKERS,
    PREFETCH_FACTOR,
    SEED,
)
from src.task4.training import PIN_MEMORY

ImageTransform = Callable[[Image.Image], Any]


class FashionImageDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        transform: ImageTransform,
        image_dir: str | Path,
    ):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.image_dir = Path(image_dir)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image_path = self.image_dir / f"{int(row['id'])}.jpg"
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            output = self.transform(image)

        label = row.get(LABEL_ID_COLUMN, pd.NA)
        label = -1 if pd.isna(label) else int(label)
        return {"image": output, "label": label, "id": int(row["id"])}


def metric_sampler(labels: pd.Series) -> MPerClassSampler:
    """Create metric-learning batches with a fixed number of examples per class."""
    labels_array = labels.to_numpy(dtype=np.int64)

    samples_per_epoch = max(
        METRIC_BATCH_SIZE,
        (len(labels_array) // METRIC_BATCH_SIZE) * METRIC_BATCH_SIZE,
    )

    return MPerClassSampler(
        labels,
        m=IMAGES_PER_CLASS,
        batch_size=METRIC_BATCH_SIZE,
        length_before_new_iter=samples_per_epoch,
    )


def seed_worker(worker_id: int):
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _loader_options(
    generator: torch.Generator | None = None, persistent_workers: bool = False
):
    options = {
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "worker_init_fn": seed_worker,
        "persistent_workers": persistent_workers and NUM_WORKERS > 0,
    }
    if generator is not None:
        options["generator"] = generator
    if NUM_WORKERS > 0:
        options["prefetch_factor"] = PREFETCH_FACTOR
    return options


def make_two_view_transform(transform: ImageTransform) -> ImageTransform:
    def apply(image: Image.Image):
        return transform(image), transform(image)

    return apply


def make_training_loader(
    frame: pd.DataFrame,
    transform: ImageTransform,
    batch_size: int,
    image_dir: str | Path = IMAGE_DIR,
):
    dataset = FashionImageDataset(frame, transform, image_dir)
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        **_loader_options(generator, persistent_workers=True),
    )


def make_metric_training_loader(
    frame: pd.DataFrame,
    transform: ImageTransform,
    image_dir: str | Path = IMAGE_DIR,
):
    dataset = FashionImageDataset(frame, transform, image_dir)
    sampler = metric_sampler(frame[LABEL_ID_COLUMN])

    return DataLoader(
        dataset,
        batch_size=METRIC_BATCH_SIZE,
        sampler=sampler,
        **_loader_options(persistent_workers=True),
    )


def make_evaluation_loader(
    frame: pd.DataFrame,
    transform: ImageTransform,
    image_dir: str | Path = IMAGE_DIR,
):
    ordered_frame = frame.sort_values("id").reset_index(drop=True)
    dataset = FashionImageDataset(ordered_frame, transform, image_dir)
    generator = torch.Generator().manual_seed(SEED)

    return DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        **_loader_options(generator, persistent_workers=False),
    )


def make_metric_validation_loss_loader(
    frame: pd.DataFrame,
    transform: ImageTransform,
    image_dir: str | Path = IMAGE_DIR,
):
    """Build a P-K validation loader for a metric-learning loss."""
    eligible_frame = frame.groupby(LABEL_ID_COLUMN).filter(
        lambda group: len(group) >= IMAGES_PER_CLASS
    )
    dataset = FashionImageDataset(eligible_frame, transform, image_dir)
    sampler = metric_sampler(eligible_frame[LABEL_ID_COLUMN])

    return DataLoader(
        dataset,
        batch_size=METRIC_BATCH_SIZE,
        sampler=sampler,
        **_loader_options(persistent_workers=False),
    )
