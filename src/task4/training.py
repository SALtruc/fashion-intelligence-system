import math
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.task4.config import (
    SEED,
)


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"
PIN_MEMORY = AMP_ENABLED


def compute_training_loss(
    model_name: str, model: nn.Module, loss_function: nn.Module, batch: dict[str, Any]
):
    labels = batch["label"].to(DEVICE, non_blocking=True)

    if model_name == "supcon":
        first_view, second_view = batch["image"]
        images = torch.cat([first_view, second_view], dim=0).to(
            DEVICE, non_blocking=True
        )
        embeddings = model(images)
        return loss_function(embeddings, labels.repeat(2))

    images = batch["image"].to(DEVICE, non_blocking=True)
    if model_name == "cae":
        reconstruction, _ = model(images)
        return loss_function(reconstruction, images)
    if model_name == "arcface":
        return model.arcface_loss(model(images), labels)

    embeddings = model(images)
    return loss_function(embeddings, labels)


def train_one_epoch(
    model_name: str,
    model: nn.Module,
    loss_function: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
):
    model.train()

    if loader.batch_sampler is not None and hasattr(loader.batch_sampler, "set_epoch"):
        loader.batch_sampler.set_epoch(epoch)

    running_loss = 0.0
    example_count = 0

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):
            loss = compute_training_loss(model_name, model, loss_function, batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = len(batch["label"])
        running_loss += loss.detach().item() * batch_size
        example_count += batch_size

    return running_loss / max(1, example_count)


class EarlyStopping:
    def __init__(self, patience: int, minimum_delta: float):
        self.patience = patience
        self.minimum_delta = minimum_delta
        self.best_score = -math.inf
        self.bad_epochs = 0

    def update(self, score: float):
        improved = score > self.best_score + self.minimum_delta
        self.best_score = score if improved else self.best_score
        self.bad_epochs = 0 if improved else self.bad_epochs + 1
        return improved

    @property
    def should_stop(self):
        return self.bad_epochs >= self.patience


def cpu_state_dict(model: nn.Module):
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def new_optimizer_and_scheduler(model: nn.Module, parameters: dict[str, Any]):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=parameters["learning_rate"],
        weight_decay=parameters["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    return optimizer, scheduler
