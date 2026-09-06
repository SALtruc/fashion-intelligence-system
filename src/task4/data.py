import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_metric_learning.samplers import MPerClassSampler
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose

from src.task4.config import (
    CAE_BATCH_SIZE,
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


class FashionImageDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_dir: str | Path,
        transform: Compose,
    ):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

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


def standard_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    persistent_workers: bool = False,
):
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        **_loader_options(generator, persistent_workers),
    )


def make_two_view_transform(transform: Compose):
    def apply(image: Image.Image):
        return transform(image), transform(image)

    return apply


def make_training_loader(
    model_name: str,
    frame: pd.DataFrame,
    cae_transform: Compose,
    metric_train_transform: Compose,
    image_dir: str | Path = IMAGE_DIR,
):
    transform = cae_transform if model_name == "cae" else metric_train_transform
    if model_name == "supcon":
        transform = make_two_view_transform(transform)

    dataset = FashionImageDataset(frame, image_dir, transform)
    if model_name == "cae":
        return standard_loader(
            dataset,
            CAE_BATCH_SIZE,
            shuffle=True,
            persistent_workers=True,
        )

    sampler = metric_sampler(frame[LABEL_ID_COLUMN])

    return DataLoader(
        dataset,
        batch_size=METRIC_BATCH_SIZE,
        sampler=sampler,
        **_loader_options(persistent_workers=True),
    )


def make_evaluation_loader(
    model_name: str,
    frame: pd.DataFrame,
    cae_transform: Compose,
    metric_eval_transform: Compose,
    image_dir: str | Path = IMAGE_DIR,
):
    transform = cae_transform if model_name == "cae" else metric_eval_transform
    ordered_frame = frame.sort_values("id").reset_index(drop=True)
    dataset = FashionImageDataset(ordered_frame, image_dir, transform)
    return standard_loader(dataset, EVAL_BATCH_SIZE, shuffle=False)


def make_validation_loss_loader(
    model_name: str,
    frame: pd.DataFrame,
    cae_transform: Compose,
    metric_eval_transform: Compose,
    image_dir: str | Path = IMAGE_DIR,
):
    """Build a P-K validation loader for the model's optimization loss."""
    if model_name == "cae":
        return make_evaluation_loader(
            model_name, frame, cae_transform, metric_eval_transform, image_dir
        )

    eligible_frame = frame.groupby(LABEL_ID_COLUMN).filter(
        lambda group: len(group) >= IMAGES_PER_CLASS
    )

    transform = metric_eval_transform
    if model_name == "supcon":
        transform = make_two_view_transform(transform)

    dataset = FashionImageDataset(eligible_frame, image_dir, transform)
    sampler = metric_sampler(eligible_frame[LABEL_ID_COLUMN])

    return DataLoader(
        dataset,
        batch_size=METRIC_BATCH_SIZE,
        sampler=sampler,
        **_loader_options(),
    )


def make_model_loaders(
    model_name: str,
    train_df: pd.DataFrame,
    tuning_gallery_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cae_transform: Compose,
    metric_train_transform: Compose,
    metric_eval_transform: Compose,
    image_dir: str | Path = IMAGE_DIR,
):
    return {
        "train": make_training_loader(
            model_name, train_df, cae_transform, metric_train_transform, image_dir
        ),
        "gallery": make_evaluation_loader(
            model_name,
            tuning_gallery_df,
            cae_transform,
            metric_eval_transform,
            image_dir,
        ),
        "query": make_evaluation_loader(
            model_name, val_df, cae_transform, metric_eval_transform, image_dir
        ),
    }
