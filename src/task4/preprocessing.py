import copy
import json
from collections.abc import Iterable

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from src.task4.config import CONFIG_DIR, IMAGE_DIR, RESNET_INPUT_SIZE


class LetterboxResize:
    """Resize to fit within a canvas, padding instead of stretching or cropping."""

    def __init__(
        self, size: tuple[int, int], fill: tuple[int, int, int] = (255, 255, 255)
    ):
        self.size = tuple(size)
        self.fill = fill

    def __call__(self, image: Image.Image):
        return ImageOps.pad(
            image.convert("RGB"),
            self.size,
            method=Image.Resampling.BILINEAR,
            color=self.fill,
            centering=(0.5, 0.5),
        )


letterbox_to_resnet = LetterboxResize(RESNET_INPUT_SIZE)


def compute_rgb_mean_std(record_ids: Iterable[int]):
    """Calculate per-channel RGB statistics using training images only."""
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sum_sq = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0

    for record_id in record_ids:
        path = IMAGE_DIR / f"{int(record_id)}.jpg"

        if not path.exists():
            raise FileNotFoundError(f"Missing image: {path}")

        with Image.open(path) as image:
            tensor = transforms.ToTensor()(letterbox_to_resnet(image)).to(torch.float64)

        channel_sum += tensor.sum(dim=(1, 2))
        channel_sum_sq += (tensor**2).sum(dim=(1, 2))
        pixel_count += tensor.shape[1] * tensor.shape[2]

    mean = channel_sum / pixel_count
    std = torch.sqrt(channel_sum_sq / pixel_count - mean**2)
    return mean.float().tolist(), std.float().tolist()


image_preprocessing_path = CONFIG_DIR / "image_preprocessing.json"
if image_preprocessing_path.exists():
    with image_preprocessing_path.open(encoding="utf-8") as file:
        image_preprocessing_config = json.load(file)

    train_mean = image_preprocessing_config["normalization"]["mean_rgb"]
    train_std = image_preprocessing_config["normalization"]["std_rgb"]

    metric_train_transform = transforms.Compose(
        [
            letterbox_to_resnet,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=5,
                translate=(0.03, 0.03),
                scale=(0.95, 1.05),
                fill=(255, 255, 255),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=train_mean, std=train_std),
        ]
    )

    metric_eval_transform = transforms.Compose(
        [
            letterbox_to_resnet,
            transforms.ToTensor(),
            transforms.Normalize(mean=train_mean, std=train_std),
        ]
    )

    metric_preprocessing_config = copy.deepcopy(image_preprocessing_config)
    cae_preprocessing_config = copy.deepcopy(image_preprocessing_config)
    cae_preprocessing_config.pop("normalization", None)
else:
    image_preprocessing_config = None
    train_mean = None
    train_std = None
    metric_train_transform = None
    metric_eval_transform = None
    metric_preprocessing_config = None
    cae_preprocessing_config = None

cae_transform = transforms.Compose(
    [
        letterbox_to_resnet,
        transforms.ToTensor(),
    ]
)
