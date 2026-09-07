"""Small, shared utilities for the independent Task 2 notebooks."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog
from sklearn.metrics import accuracy_score, f1_score, recall_score


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK2_MODEL_DIR = REPO_ROOT / "models" / "task2"
TASK2_CHECKPOINT_DIR = TASK2_MODEL_DIR / "checkpoints"
TASK2_OUTPUT_DIR = REPO_ROOT / "outputs" / "task2"
TASK2_FIGURE_DIR = TASK2_OUTPUT_DIR / "figures"
TASK2_SPLIT_PATH = REPO_ROOT / "splits" / "task2_season_split.csv"
TASK2_PREDICTION_PATH = REPO_ROOT / "predictions" / "task2_season_predictions.csv"
TASK2_PREPROCESSED_DIR = REPO_ROOT / "preprocessed_datasets" / "task2"
TASK2_METADATA_PATH = TASK2_PREPROCESSED_DIR / "task2_metadata.json"
TASK2_BASELINE_SCORES_PATH = TASK2_PREPROCESSED_DIR / "task2_baseline_validation_scores.npz"

FEATURE_CONFIG = {
    "hue_bins": 12,
    "saturation_bins": 8,
    "value_bins": 8,
    "hog_orientations": 9,
    "hog_pixels_per_cell": (8, 8),
    "hog_cells_per_block": (2, 2),
    "foreground_threshold": 0.95,
}


def ensure_task2_directories() -> None:
    """Create result directories, but never fabricate the prepared-data input folder."""
    for path in (
        TASK2_MODEL_DIR,
        TASK2_CHECKPOINT_DIR,
        TASK2_OUTPUT_DIR,
        TASK2_FIGURE_DIR,
        TASK2_SPLIT_PATH.parent,
        TASK2_PREDICTION_PATH.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)


def deterministic_json(value) -> str:
    """Encode checkpoint identity metadata deterministically."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def calculate_run_fingerprint(*, target, classes, train_ids, validation_ids,
                              random_state, image_target_size,
                              normalisation_mean, normalisation_std,
                              feature_version) -> str:
    """Identify only the shared Task 2 data and preprocessing contract."""
    identity = {
        "target": target,
        "classes": list(classes),
        "train_ids": [str(value) for value in train_ids],
        "validation_ids": [str(value) for value in validation_ids],
        "random_state": int(random_state),
        "image_target_size": list(image_target_size),
        "normalisation_mean": np.asarray(normalisation_mean).tolist(),
        "normalisation_std": np.asarray(normalisation_std).tolist(),
        "feature_version": feature_version,
    }
    return hashlib.sha1(deterministic_json(identity).encode()).hexdigest()[:12]


def model_checkpoint_path(name: str) -> Path:
    slug = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")
    return TASK2_CHECKPOINT_DIR / f"model_{slug}.pt"


def epoch_checkpoint_path(name: str) -> Path:
    slug = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")
    return TASK2_CHECKPOINT_DIR / f"epoch_{slug}.pt"


def _validate_candidate(blob, *, fingerprint, model_config=None, classes=None,
                        validation_ids=None, path=None) -> None:
    label = str(path or "checkpoint")
    if blob.get("fingerprint") != fingerprint:
        raise ValueError(f"{label} has an incompatible run fingerprint")
    if model_config is not None and blob.get("model_config") != model_config:
        raise ValueError(f"{label} has an incompatible model configuration")
    if classes is not None and list(blob.get("classes", [])) != list(classes):
        raise ValueError(f"{label} has an incompatible class ordering")
    if validation_ids is not None and not np.array_equal(
        np.asarray(blob.get("validation_ids", []), dtype=str),
        np.asarray(validation_ids, dtype=str),
    ):
        raise ValueError(f"{label} has an incompatible validation-ID ordering")


def evaluate_predictions(y_true, y_pred, scores, name: str) -> dict:
    """Return the common Task 2 metrics as one serialisable result row."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(y_true)
    result = {
        "Model": name,
        "Top-1 accuracy": accuracy_score(y_true, y_pred),
        "Macro-F1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "Balanced accuracy": recall_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "Weighted F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if scores is None:
        result["Top-2 accuracy"] = np.nan
    else:
        scores = np.asarray(scores)
        top = np.argpartition(scores, -min(2, scores.shape[1]), axis=1)[:, -2:]
        result["Top-2 accuracy"] = float(
            np.mean([truth in choices for truth, choices in zip(y_true, top)])
        )
    return result


def per_class_table(y_true, y_pred, classes) -> pd.DataFrame:
    """Return precision, recall and F1 for every represented season."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rows = []
    for index in np.unique(y_true):
        true_binary = y_true == index
        pred_binary = y_pred == index
        rows.append({
            "Season": classes[index],
            "Support": int(true_binary.sum()),
            "Precision": float((true_binary & pred_binary).sum() / max(pred_binary.sum(), 1)),
            "Recall": float((true_binary & pred_binary).sum() / max(true_binary.sum(), 1)),
            "F1": f1_score(true_binary, pred_binary, zero_division=0),
        })
    return pd.DataFrame(rows)


def load_task2_prepared_data():
    """Load Notebook 1's self-contained metadata and preprocessed arrays."""
    if not TASK2_PREPROCESSED_DIR.is_dir():
        raise FileNotFoundError(
            "Required Task 2 prepared-data folder was not found: "
            f"{TASK2_PREPROCESSED_DIR}. Run notebooks/task2/01_task2_setup.ipynb "
            "in this repository first, or copy the complete prepared-data folder here."
        )

    required = [
        TASK2_METADATA_PATH,
        TASK2_PREPROCESSED_DIR / "task2_deep_learning_train_images.npy",
        TASK2_PREPROCESSED_DIR / "task2_deep_learning_validation_images.npy",
        TASK2_PREPROCESSED_DIR / "task2_random_forest_train_features.npy",
        TASK2_PREPROCESSED_DIR / "task2_random_forest_validation_features.npy",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run notebooks/task2/01_task2_setup.ipynb first. Missing: " + ", ".join(missing)
        )

    config = json.loads(TASK2_METADATA_PATH.read_text())
    classes = config["classes"]
    train_frame = pd.DataFrame({"id": config["train_ids"]})
    validation_frame = pd.DataFrame({"id": config["validation_ids"]})
    return {
        "config": config,
        "classes": classes,
        "train_frame": train_frame,
        "validation_frame": validation_frame,
        "y_train": np.asarray(config["y_train"], dtype=np.int64),
        "y_validation": np.asarray(config["y_validation"], dtype=np.int64),
        "train_images": np.load(
            TASK2_PREPROCESSED_DIR / "task2_deep_learning_train_images.npy", mmap_mode="r"
        ),
        "validation_images": np.load(
            TASK2_PREPROCESSED_DIR / "task2_deep_learning_validation_images.npy", mmap_mode="r"
        ),
        "train_features": np.load(
            TASK2_PREPROCESSED_DIR / "task2_random_forest_train_features.npy", mmap_mode="r"
        ),
        "validation_features": np.load(
            TASK2_PREPROCESSED_DIR / "task2_random_forest_validation_features.npy", mmap_mode="r"
        ),
    }


def prepared_namespace():
    """Load Notebook 1 outputs and expose the commonly used derived values."""
    data = load_task2_prepared_data()
    classes = data["classes"]
    y_train = data["y_train"]
    y_validation = data["y_validation"]
    return SimpleNamespace(
        **data,
        target=data["config"]["target"],
        class_to_index={label: index for index, label in enumerate(classes)},
        n_classes=len(classes),
        y_val=y_validation,
        x_train=data["train_images"],
        x_val=data["validation_images"],
        x_train_features=data["train_features"],
        x_val_features=data["validation_features"],
        norm_mean=np.asarray(data["config"]["normalisation_mean"], dtype=np.float32),
        norm_std=np.asarray(data["config"]["normalisation_std"], dtype=np.float32),
        train_support=pd.Series(np.bincount(y_train, minlength=len(classes)), index=classes),
        validation_support=pd.Series(
            np.bincount(y_validation, minlength=len(classes)), index=classes
        ),
        scoreable=np.flatnonzero(np.bincount(y_validation, minlength=len(classes)) > 0),
        fingerprint=data["config"]["fingerprint"],
        validation_ids=data["validation_frame"]["id"].astype(str).to_numpy(),
    )


def result_frame(y_true, y_pred, scores, name: str) -> pd.DataFrame:
    """Return the standard metrics in the one-row format used by the notebooks."""
    return pd.DataFrame([evaluate_predictions(y_true, y_pred, scores, name)])


def export_model_results(
    group: str, name: str, prepared, logits, history: pd.DataFrame | None = None,
    *, scores_are_logits: bool = False,
) -> Path:
    """Write one model's metrics, history, and validation predictions together."""
    output_dir = TASK2_OUTPUT_DIR / group
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = np.asarray(logits).argmax(axis=1)
    result_frame(prepared.y_val, predictions, logits, name).to_json(
        output_dir / "results.json", orient="records", indent=2
    )
    if history is not None:
        history.to_csv(output_dir / "training_history.csv", index=False)
    export_validation_predictions(
        output_dir / "validation_predictions.csv", prepared.validation_frame,
        prepared.y_val, predictions, logits, prepared.classes,
        scores_are_logits=scores_are_logits,
    )
    return output_dir


def neural_training_config(
    *, quick_run: bool = True, random_state: int = 42,
    resume: bool = False, allow_cpu: bool = False,
) -> dict:
    """Return the single shared CNN recipe, with only run controls exposed per notebook."""
    return {
        "random_state": random_state,
        "quick_run": quick_run,
        "resume": resume,
        "allow_cpu": True if quick_run else allow_cpu,
        "epochs": 3 if quick_run else 30,
        "warmup_epochs": 1 if quick_run else 3,
        "batch_size": 128,
        "patience": 6,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "label_smoothing": 0.05,
        "history_metrics": ["loss", "accuracy"],
        "flip_probability": 0.5,
        "rotation_degrees": 8.0,
        "translate_fraction": 0.06,
        "jitter_strength": 0.05,
        "use_amp": True,
        "channels_last": True,
        "cache_on_device": True,
        "use_compile": False,
        "deterministic": False,
        "keep_epoch_checkpoints": False,
    }


class NeuralTrainer:
    """Shared augmentation, batching, training, and checkpoint recovery for Task 2 CNNs."""

    def __init__(self, prepared, config: dict):
        self.data = prepared
        self.cfg = dict(config)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        if self.device.type == "cpu" and not self.cfg["allow_cpu"]:
            raise RuntimeError(
                "No supported GPU is available. Install a CUDA-enabled PyTorch build, use "
                "Apple Silicon with an MPS-enabled PyTorch build, or set ALLOW_CPU = True "
                "(preferably with QUICK_RUN = True)."
            )
        if self.device.type == "cuda":
            try:
                (torch.zeros(8, device=self.device) + 1).sum().item()
            except RuntimeError as error:
                raise RuntimeError(f"CUDA is visible but cannot execute a kernel: {error}") from error
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        elif self.device.type == "mps":
            print("Apple Metal GPU detected; using the PyTorch MPS backend.")
        self.amp_enabled = self.cfg["use_amp"] and self.device.type == "cuda"
        self.amp_dtype = (torch.bfloat16 if self.amp_enabled and torch.cuda.is_bf16_supported()
                          else torch.float16)
        self.channels_last = self.cfg["channels_last"] and self.device.type == "cuda"
        self.cache_on_device = self.cfg["cache_on_device"] and self.device.type == "cuda"
        self.use_compile = self.cfg["use_compile"] and hasattr(torch, "compile")
        torch.backends.cudnn.benchmark = not self.cfg["deterministic"] and self.device.type == "cuda"
        self._set_seed(self.cfg["random_state"])
        self.mean = torch.tensor(prepared.norm_mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(prepared.norm_std, device=self.device).view(1, 3, 1, 1)
        self.luma = torch.tensor([.299, .587, .114], device=self.device).view(1, 3, 1, 1)
        self.train_loader = self.BatchStream(
            self, prepared.x_train, prepared.y_train, self.cfg["batch_size"], True, True
        )
        self.val_loader = self.BatchStream(
            self, prepared.x_val, prepared.y_val, 512, False, False
        )
        self.fingerprint = prepared.fingerprint
        print(f"Device: {self.device} | mixed precision: {self.amp_enabled} "
              f"| run fingerprint: {self.fingerprint}")

    @staticmethod
    def _set_seed(seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _capture_rng(self):
        state = {"python": random.getstate(), "numpy": np.random.get_state(),
                 "torch": torch.get_rng_state()}
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng(self, state):
        random.setstate(state["python"]); np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])

    def augment(self, x):
        n, _, height, width = x.shape
        flip = torch.rand(n, device=x.device) < self.cfg["flip_probability"]
        x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)
        angle = (torch.rand(n, device=x.device) * 2 - 1) * math.radians(self.cfg["rotation_degrees"])
        shift = self.cfg["translate_fraction"] * 2
        shift_x = (torch.rand(n, device=x.device) * 2 - 1) * shift
        shift_y = (torch.rand(n, device=x.device) * 2 - 1) * shift
        cos, sin = torch.cos(angle), torch.sin(angle)
        theta = torch.zeros(n, 2, 3, device=x.device)
        theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin * height / width, shift_x
        theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin * width / height, cos, shift_y
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(x - 1, grid, mode="bilinear", padding_mode="zeros",
                          align_corners=False) + 1
        strength = self.cfg["jitter_strength"]
        factor = lambda: 1 + (torch.rand(n, 1, 1, 1, device=x.device) * 2 - 1) * strength
        x = x * factor()
        grey = (x * self.luma).sum(1, keepdim=True)
        x = (x - grey) * factor() + grey
        x = (x - grey.mean((2, 3), keepdim=True)) * factor() + grey.mean((2, 3), keepdim=True)
        return x.clamp_(0, 1)

    class BatchStream:
        def __init__(self, owner, images, labels, batch_size, augment, shuffle):
            self.owner, self.batch_size, self.augment, self.shuffle = owner, batch_size, augment, shuffle
            # Prepared arrays are commonly read through a read-only NumPy memmap.
            # ascontiguousarray() may return that same non-writable buffer, which
            # PyTorch rejects because tensors are assumed to have writable storage.
            self.images = torch.from_numpy(np.array(images, copy=True, order="C"))
            if owner.cache_on_device:
                self.images = self.images.to(owner.device)
            # Labels may also be a read-only pandas/NumPy view.  Make the small
            # one-dimensional array explicitly writable before sharing its storage
            # with Torch; otherwise torch.as_tensor emits a non-writable-buffer
            # warning and any accidental in-place write would be undefined.
            writable_labels = np.array(labels, dtype=np.int64, copy=True, order="C")
            self.labels = torch.from_numpy(writable_labels).to(owner.device)

        def __len__(self):
            return math.ceil(len(self.labels) / self.batch_size)

        def __iter__(self):
            order = (torch.randperm(len(self.labels), device=self.images.device) if self.shuffle
                     else torch.arange(len(self.labels), device=self.images.device))
            for start in range(0, len(order), self.batch_size):
                index = order[start:start + self.batch_size]
                x = self.images[index].to(self.owner.device).permute(0, 3, 1, 2).float().div_(255)
                if self.augment:
                    x = self.owner.augment(x)
                x = (x - self.owner.mean) / self.owner.std
                if self.owner.channels_last:
                    x = x.contiguous(memory_format=torch.channels_last)
                yield x, self.labels[index.to(self.labels.device)]

    def _path(self, name, checkpoint=False):
        return epoch_checkpoint_path(name) if checkpoint else model_checkpoint_path(name)

    @staticmethod
    def _state(model):
        return {k.replace("_orig_mod.", ""): v.detach().cpu().clone()
                for k, v in model.state_dict().items()}

    @staticmethod
    def _load_state(model, state):
        if any(k.startswith("_orig_mod.") for k in model.state_dict()):
            state = {f"_orig_mod.{k}": v for k, v in state.items()}
        model.load_state_dict(state)

    @staticmethod
    def _load(path):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    @staticmethod
    def _save(payload, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary); os.replace(temporary, path)

    def _prepare(self, model):
        model = model.to(self.device)
        if self.channels_last:
            model = model.to(memory_format=torch.channels_last)
        if self.use_compile:
            model = torch.compile(model)
        return model

    def _epoch(self, model, criterion, optimiser=None, scaler=None):
        training = optimiser is not None
        model.train(training); collected = []; loss_sum = torch.zeros((), device=self.device)
        correct = torch.zeros((), dtype=torch.long, device=self.device)
        loader = self.train_loader if training else self.val_loader
        with torch.set_grad_enabled(training):
            for images, labels in loader:
                with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                    logits = model(images); loss = criterion(logits, labels)
                if training:
                    optimiser.zero_grad(set_to_none=True)
                    if scaler.is_enabled():
                        scaler.scale(loss).backward(); scaler.step(optimiser); scaler.update()
                    else:
                        loss.backward(); optimiser.step()
                else:
                    collected.append(logits.detach().float())
                loss_sum += loss.detach().float() * len(labels)
                correct += (logits.detach().argmax(dim=1) == labels).sum()
        logits = torch.cat(collected).cpu().numpy() if collected else None
        sample_count = len(loader.labels)
        return (loss_sum / sample_count).item(), (correct.float() / sample_count).item(), logits

    def train_or_restore(self, name, build, label, model_config):
        model = self._prepare(build())
        final_path, checkpoint = self._path(name), self._path(name, True)
        if final_path.exists():
            blob = self._load(final_path)
            try:
                _validate_candidate(
                    blob, fingerprint=self.fingerprint, model_config=model_config,
                    classes=self.data.classes, validation_ids=self.data.validation_ids,
                    path=final_path,
                )
                self._load_state(model, blob["state_dict"])
                history = pd.DataFrame(blob["history"])
                print(f"{label}: restored finished model; training skipped")
                return model, history, blob["val_logits"]
            except ValueError as error:
                print(f"{label}: {error}; training a compatible checkpoint")
        criterion = nn.CrossEntropyLoss(label_smoothing=self.cfg["label_smoothing"])
        optimiser = torch.optim.AdamW(model.parameters(), lr=self.cfg["learning_rate"],
                                      weight_decay=self.cfg["weight_decay"])
        def schedule(epoch):
            warmup = self.cfg["warmup_epochs"]
            if epoch < warmup:
                return (epoch + 1) / max(warmup, 1)
            progress = (epoch - warmup) / max(self.cfg["epochs"] - warmup, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, schedule)
        enabled = self.amp_enabled and self.amp_dtype == torch.float16
        try:
            scaler = torch.amp.GradScaler(self.device.type, enabled=enabled)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=enabled)
        history, best, first = [], {"macro_f1": -1., "epoch": -1}, 0
        if self.cfg["resume"] and checkpoint.exists():
            blob = self._load(checkpoint)
            try:
                _validate_candidate(blob, fingerprint=self.fingerprint,
                                    model_config=model_config, path=checkpoint)
                self._load_state(model, blob["model"]); optimiser.load_state_dict(blob["optimiser"])
                scheduler.load_state_dict(blob["scheduler"])
                if blob.get("scaler") is not None:
                    scaler.load_state_dict(blob["scaler"])
                self._restore_rng(blob["rng"]); history, best, first = blob["history"], blob["best"], blob["epoch"]
                print(
                    f"{label}: resumed after epoch {first}/{self.cfg['epochs']} "
                    f"(best macro-F1 {best['macro_f1']:.4f} at epoch {best['epoch']})",
                    flush=True,
                )
            except ValueError as error:
                print(f"{label}: {error}; starting from epoch 1")
        start_time = time.time()
        for epoch in range(first, self.cfg["epochs"]):
            epoch_time = time.time()
            epoch_number = epoch + 1
            print(
                f"[{label}] starting epoch {epoch_number}/{self.cfg['epochs']} "
                f"({len(self.train_loader)} train + {len(self.val_loader)} validation batches)",
                flush=True,
            )
            train_loss, train_accuracy, _ = self._epoch(model, criterion, optimiser, scaler)
            val_loss, val_accuracy, logits = self._epoch(model, criterion)
            scheduler.step()
            pred = logits.argmax(1)
            macro = f1_score(self.data.y_val, pred, labels=self.data.scoreable,
                             average="macro", zero_division=0)
            history.append({"epoch": epoch_number, "train loss": train_loss, "val loss": val_loss,
                            "train accuracy": train_accuracy, "val accuracy": val_accuracy,
                            "val macro-F1": macro, "lr": optimiser.param_groups[0]["lr"],
                            "seconds": time.time() - epoch_time})
            if macro > best["macro_f1"]:
                best = {"macro_f1": macro, "epoch": epoch_number,
                        "state": self._state(model), "logits": logits}
            stopping = epoch_number - best["epoch"] >= self.cfg["patience"]
            print(
                f"[{label}] finished epoch {epoch_number}/{self.cfg['epochs']} | "
                f"train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
                f"train accuracy {train_accuracy:.4f} | val accuracy {val_accuracy:.4f} | "
                f"macro-F1 {macro:.4f} | "
                f"best {best['macro_f1']:.4f} (epoch {best['epoch']}) | "
                f"lr {optimiser.param_groups[0]['lr']:.2e} | {history[-1]['seconds']:.0f}s",
                flush=True,
            )
            if not stopping:
                self._save({"fingerprint": self.fingerprint, "model_config": model_config,
                            "epoch": epoch_number,
                            "model": self._state(model), "optimiser": optimiser.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                            "rng": self._capture_rng(), "history": history, "best": best}, checkpoint)
            if stopping:
                print(
                    f"[{label}] early stopping after epoch {epoch_number}; "
                    f"no macro-F1 improvement for {self.cfg['patience']} epochs",
                    flush=True,
                )
                break
        self._load_state(model, best["state"])
        history_frame = pd.DataFrame(history)
        payload = {"name": name, "fingerprint": self.fingerprint,
                   "model_config": model_config,
                   "state_dict": self._state(model), "val_logits": best["logits"],
                   "validation_ids": self.data.validation_ids.tolist(),
                   "history": history, "classes": self.data.classes,
                   "normalisation_mean": self.data.norm_mean.tolist(),
                   "normalisation_std": self.data.norm_std.tolist(),
                   "image_target_size": self.data.config["image_target_size"]}
        self._save(payload, final_path)
        if checkpoint.exists() and not self.cfg["keep_epoch_checkpoints"]:
            checkpoint.unlink()
        print(f"{label}: best macro-F1 {best['macro_f1']:.4f} in {time.time()-start_time:.0f}s")
        return model, history_frame, best["logits"]

    @staticmethod
    def plot_history(history, title):
        required = {"epoch", "train loss", "val loss", "train accuracy", "val accuracy"}
        missing = required.difference(history.columns)
        if missing:
            raise ValueError(
                "Training history is missing " + ", ".join(sorted(missing))
                + ". Retrain the model with the current training configuration."
            )
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
        axes[0].plot(history["epoch"], history["train loss"], label="Train")
        axes[0].plot(history["epoch"], history["val loss"], label="Validation")
        axes[1].plot(history["epoch"], history["train accuracy"], label="Train")
        axes[1].plot(history["epoch"], history["val accuracy"], label="Validation")
        axes[0].set(title="Training and validation loss", xlabel="Epoch", ylabel="Loss")
        axes[1].set(title="Training and validation accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend()
        fig.suptitle(title); plt.tight_layout(); plt.show()


def export_validation_predictions(
    path: Path,
    validation_frame: pd.DataFrame,
    y_true,
    y_pred,
    scores,
    classes,
    *,
    scores_are_logits: bool = False,
) -> None:
    """Export readable per-image validation labels and comparable class probabilities."""
    scores = np.asarray(scores)
    if scores_are_logits:
        scores = torch.softmax(torch.from_numpy(scores), dim=1).numpy()
    output = pd.DataFrame({
        "id": validation_frame["id"].astype(str).to_numpy(),
        "true_index": np.asarray(y_true, dtype=int),
        "predicted_index": np.asarray(y_pred, dtype=int),
        "true_season": [classes[index] for index in y_true],
        "predicted_season": [classes[index] for index in y_pred],
    })
    for index, label in enumerate(classes):
        output[f"score_{label}"] = scores[:, index]
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)


def load_validation_predictions(path: Path, classes) -> pd.DataFrame:
    """Load and validate one model's exported validation predictions."""
    frame = pd.read_csv(path, dtype={"id": str})
    required = {"id", "true_index", "predicted_index", *[f"score_{c}" for c in classes]}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def random_forest_paths():
    return (
        TASK2_CHECKPOINT_DIR / "model_random_forest.joblib",
        TASK2_CHECKPOINT_DIR / "model_random_forest_scores.npz",
    )


def save_random_forest_checkpoint(model, scores, prepared, model_config):
    """Atomically bank an RF estimator and its aligned validation evidence."""
    import joblib

    model_path, scores_path = random_forest_paths()
    model_tmp = model_path.with_suffix(model_path.suffix + ".tmp")
    scores_tmp = scores_path.with_suffix(scores_path.suffix + ".tmp")
    joblib.dump(model, model_tmp)
    with scores_tmp.open("wb") as handle:
        np.savez_compressed(
            handle, fingerprint=prepared.fingerprint,
            model_config_json=deterministic_json(model_config),
            validation_ids=np.asarray(prepared.validation_ids, dtype=str),
            true_indices=prepared.y_val, scores=np.asarray(scores),
            classes=np.asarray(prepared.classes),
        )
    os.replace(model_tmp, model_path)
    os.replace(scores_tmp, scores_path)
    return model_path, scores_path


def restore_random_forest_checkpoint(prepared, model_config):
    """Return a compatible RF estimator and validation scores, or ``None``."""
    import joblib

    model_path, scores_path = random_forest_paths()
    if not (model_path.exists() and scores_path.exists()):
        return None
    with np.load(scores_path, allow_pickle=False) as saved:
        blob = {key: saved[key] for key in saved.files}
    actual_config = str(np.asarray(blob.get("model_config_json", "")).item())
    metadata = {
        "fingerprint": str(np.asarray(blob.get("fingerprint", "")).item()),
        "model_config": json.loads(actual_config) if actual_config else None,
        "classes": np.asarray(blob.get("classes", [])).astype(str).tolist(),
        "validation_ids": np.asarray(blob.get("validation_ids", [])).astype(str).tolist(),
    }
    try:
        _validate_candidate(
            metadata, fingerprint=prepared.fingerprint, model_config=model_config,
            classes=prepared.classes, validation_ids=prepared.validation_ids,
            path=scores_path,
        )
    except ValueError as error:
        print(f"Random Forest: {error}; refitting")
        return None
    scores = np.asarray(blob["scores"])
    if scores.shape != (len(prepared.y_val), len(prepared.classes)) or not np.isfinite(scores).all():
        print("Random Forest: score checkpoint has invalid dimensions or values; refitting")
        return None
    return joblib.load(model_path), scores


def load_checkpoint_candidate(name, *, fingerprint, classes, validation_ids):
    """Normalize one completed neural checkpoint for final analysis."""
    path = model_checkpoint_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run its model notebook first.")
    blob = NeuralTrainer._load(path)
    _validate_candidate(blob, fingerprint=fingerprint, classes=classes,
                        validation_ids=validation_ids, path=path)
    scores = np.asarray(blob["val_logits"])
    if scores.shape != (len(validation_ids), len(classes)) or not np.isfinite(scores).all():
        raise ValueError(f"{path} contains invalid validation logits")
    return {
        "name": blob["name"], "scores": scores,
        "validation_ids": np.asarray(blob["validation_ids"], dtype=str),
        "classes": list(blob["classes"]), "history": blob.get("history", []),
        "model_config": blob["model_config"], "checkpoint_path": path,
        "scores_are_logits": True, "blob": blob,
    }


def extract_visual_features(image_uint8) -> np.ndarray:
    """Create the colour, shape and foreground features used by Random Forest."""
    rgb = image_uint8.astype(np.float32) / 255.0
    hsv = rgb2hsv(rgb)
    gray = rgb2gray(rgb)
    values = []
    for image in (rgb, hsv):
        for channel in range(3):
            x = image[:, :, channel]
            values.extend([x.mean(), x.std(), *np.percentile(x, [25, 50, 75])])
    for channel, key in enumerate(("hue_bins", "saturation_bins", "value_bins")):
        histogram, _ = np.histogram(hsv[:, :, channel], bins=FEATURE_CONFIG[key], range=(0, 1))
        values.extend(histogram / max(histogram.sum(), 1))
    values.extend(hog(
        gray,
        orientations=FEATURE_CONFIG["hog_orientations"],
        pixels_per_cell=FEATURE_CONFIG["hog_pixels_per_cell"],
        cells_per_block=FEATURE_CONFIG["hog_cells_per_block"],
        block_norm="L2-Hys",
        feature_vector=True,
    ))
    foreground = np.any(rgb < FEATURE_CONFIG["foreground_threshold"], axis=2)
    height, width = foreground.shape
    values.extend([
        foreground.mean(), foreground[: height // 2].mean(), foreground[height // 2 :].mean(),
        foreground[:, : width // 2].mean(), foreground[:, width // 2 :].mean(),
    ])
    return np.asarray(values, dtype=np.float32)
