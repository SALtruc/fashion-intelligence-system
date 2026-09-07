from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from src.task4.training import AMP_ENABLED, DEVICE


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image in the [0, 1] range to HSV without OpenCV."""
    maximum = rgb.max(axis=-1)
    minimum = rgb.min(axis=-1)
    chroma = maximum - minimum

    hue = np.zeros_like(maximum)
    non_zero = chroma > 1e-8
    red = non_zero & (maximum == rgb[..., 0])
    green = non_zero & (maximum == rgb[..., 1])
    blue = non_zero & (maximum == rgb[..., 2])
    hue[red] = ((rgb[..., 1][red] - rgb[..., 2][red]) / chroma[red]) % 6
    hue[green] = (rgb[..., 2][green] - rgb[..., 0][green]) / chroma[green] + 2
    hue[blue] = (rgb[..., 0][blue] - rgb[..., 1][blue]) / chroma[blue] + 4
    hue /= 6

    saturation = np.divide(
        chroma, maximum, out=np.zeros_like(chroma), where=maximum > 1e-8
    )
    return np.stack((hue, saturation, maximum), axis=-1)


def color_descriptor(
    image: Image.Image | np.ndarray,
    hue_bins: int = 12,
    saturation_bins: int = 3,
    value_bins: int = 3,
    white_background_value: float = 0.92,
    white_background_saturation: float = 0.12,
) -> np.ndarray:
    """Return a unit-normalized HSV histogram for a product image.

    Near-white, low-saturation pixels are excluded to reduce the influence of
    catalogue backgrounds.  When that would discard an almost entirely white
    item, the central image area is used as a safe fallback.
    """
    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    else:
        rgb = np.asarray(image, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("image must have shape (height, width, 3)")
        if rgb.max(initial=0.0) > 1.0:
            rgb = rgb / 255.0
        rgb = np.clip(rgb, 0.0, 1.0)

    hsv = _rgb_to_hsv(rgb)
    saturation, value = hsv[..., 1], hsv[..., 2]
    foreground = ~(
        (value >= white_background_value)
        & (saturation <= white_background_saturation)
    )

    # White garments can be visually indistinguishable from a white backdrop.
    # A centre crop is a better fallback than an empty descriptor in that case.
    if foreground.sum() < max(32, int(foreground.size * 0.01)):
        height, width = foreground.shape
        margin_h, margin_w = height // 8, width // 8
        foreground = np.zeros_like(foreground, dtype=bool)
        foreground[margin_h : height - margin_h, margin_w : width - margin_w] = True

    histogram, _ = np.histogramdd(
        hsv[foreground],
        bins=(hue_bins, saturation_bins, value_bins),
        range=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    )
    descriptor = histogram.astype("float32", copy=False).ravel()
    norm = np.linalg.norm(descriptor)
    return descriptor / norm if norm > 0 else descriptor


def extract_color_descriptors(image_paths: Iterable[str | Path]) -> np.ndarray:
    """Extract colour descriptors for query or gallery images in the given order."""
    descriptors = []
    for path in image_paths:
        with Image.open(path) as image:
            descriptors.append(color_descriptor(image))
    if not descriptors:
        raise ValueError("image_paths must contain at least one image")
    return np.stack(descriptors).astype("float32", copy=False)


def color_similarities(
    query_descriptor: np.ndarray, gallery_descriptors: np.ndarray
) -> np.ndarray:
    """Cosine colour similarities for already-normalized HSV descriptors."""
    query = np.asarray(query_descriptor, dtype="float32")
    gallery = np.asarray(gallery_descriptors, dtype="float32")
    if query.ndim != 1 or gallery.ndim != 2 or query.shape[0] != gallery.shape[1]:
        raise ValueError("colour descriptor dimensions do not match")
    return np.clip(gallery @ query, 0.0, 1.0)


def retrieval_embeddings(model: nn.Module, images: torch.Tensor):
    outputs = model(images)
    embeddings = outputs[1] if isinstance(outputs, tuple) else outputs
    return F.normalize(embeddings, dim=1)


def extract_embeddings(model: nn.Module, loader: DataLoader):
    model.eval()
    embeddings, identifiers, labels = [], [], []

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(DEVICE, non_blocking=True)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):
                batch_embeddings = retrieval_embeddings(model, images)
            embeddings.append(batch_embeddings.cpu())
            identifiers.append(batch["id"].cpu())
            labels.append(batch["label"].cpu())

    return (
        torch.cat(embeddings).numpy().astype("float32"),
        torch.cat(identifiers).numpy().astype("int64"),
        torch.cat(labels).numpy().astype("int64"),
    )


def build_faiss_index(gallery_embeddings: np.ndarray):
    embeddings = np.ascontiguousarray(gallery_embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    if DEVICE.type == "cuda":
        try:
            if faiss.get_num_gpus() > 0:
                resources = faiss.StandardGpuResources()
                return faiss.index_cpu_to_gpu(resources, 0, index)

        except RuntimeError:
            pass

    return index


def search_cosine(
    query_embeddings: np.ndarray, gallery_embeddings: np.ndarray, maximum_k: int = 10
):
    index = build_faiss_index(gallery_embeddings)
    search_k = min(maximum_k, len(gallery_embeddings))
    scores, indices = index.search(
        np.ascontiguousarray(query_embeddings, dtype="float32"),
        search_k,
    )
    return scores, indices, index


def precision_k(
    ranked_indices: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    k: int,
) -> float:
    effective_k = min(k, ranked_indices.shape[1])
    ranked_labels = gallery_labels[ranked_indices[:, :effective_k]]
    relevant = ranked_labels == query_labels[:, None]
    return float(relevant.mean(axis=1).mean())


def recall_k(
    ranked_indices: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    k: int,
) -> float:
    effective_k = min(k, ranked_indices.shape[1])
    ranked_labels = gallery_labels[ranked_indices[:, :effective_k]]
    relevant = ranked_labels == query_labels[:, None]
    gallery_counts = Counter(gallery_labels.tolist())
    relevant_counts = np.array([gallery_counts[int(label)] for label in query_labels])
    recalls = relevant.sum(axis=1) / np.maximum(relevant_counts, 1)
    return float(recalls.mean())


def mean_average_precision_at_k(
    ranked_indices: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    k: int,
) -> float:
    gallery_counts = Counter(gallery_labels.tolist())
    effective_k = min(k, ranked_indices.shape[1])
    average_precisions = []

    for row, query_label in zip(ranked_indices[:, :effective_k], query_labels):
        relevant = (gallery_labels[row] == query_label).astype(np.float32)
        cumulative_precision = np.cumsum(relevant) / np.arange(1, effective_k + 1)
        denominator = min(gallery_counts[int(query_label)], effective_k)
        average_precision = float((cumulative_precision * relevant).sum())
        average_precisions.append(average_precision / max(1, denominator))

    return float(np.mean(average_precisions))


def retrieval_metrics(
    ranked_indices: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    metrics = {}
    for k in ks:
        metrics[f"Precision@{k}"] = precision_k(
            ranked_indices, query_labels, gallery_labels, k
        )
        metrics[f"Recall@{k}"] = recall_k(
            ranked_indices, query_labels, gallery_labels, k
        )

    metrics["mAP@10"] = mean_average_precision_at_k(
        ranked_indices, query_labels, gallery_labels, 10
    )
    return metrics


def evaluate_retrieval(
    model: nn.Module,
    gallery_loader: DataLoader,
    query_loader: DataLoader,
) -> dict[str, float]:
    gallery_embeddings, _, gallery_labels = extract_embeddings(model, gallery_loader)
    query_embeddings, _, query_labels = extract_embeddings(model, query_loader)
    _, ranked_indices, _ = search_cosine(
        query_embeddings, gallery_embeddings, maximum_k=10
    )

    return retrieval_metrics(ranked_indices, query_labels, gallery_labels)


def gallery_reciprocal_sets(
    gallery_embeddings: np.ndarray, index: faiss.Index, k1: int
):
    search_k = min(k1 + 1, len(gallery_embeddings))
    similarities, neighbours = index.search(gallery_embeddings, search_k)
    base_sets = []

    for gallery_index, row in enumerate(neighbours):
        forward = [int(item) for item in row if item != gallery_index][:k1]
        reciprocal = {
            candidate
            for candidate in forward
            if gallery_index in neighbours[candidate, 1:search_k]
        }
        base_sets.append(reciprocal)

    expanded_sets = []
    half_size = max(1, k1 // 2)
    for reciprocal in base_sets:
        expanded = set(reciprocal)
        for candidate in list(reciprocal):
            candidate_set = set(sorted(base_sets[candidate])[:half_size])
            overlap = len(candidate_set & reciprocal)
            if candidate_set and overlap >= (2 * len(candidate_set) / 3):
                expanded.update(candidate_set)
        expanded_sets.append(expanded)

    thresholds = similarities[:, -1]
    return expanded_sets, thresholds


def k_reciprocal_rerank(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    index: faiss.Index,
    k1: int = 20,
    k2: int = 6,
    blend: float = 0.3,
    maximum_k: int = 10,
    candidate_depth: int = 100,
    query_color_descriptors: np.ndarray | None = None,
    gallery_color_descriptors: np.ndarray | None = None,
    color_weight: float = 0.15,
):
    """Rerank visual candidates with k-reciprocal and optional image colour.

    Colour descriptors are image-derived HSV histograms, so query images do
    not need labels or metadata.  They only affect the candidate pool returned
    by FAISS; retrieval remains driven primarily by the learned embedding.
    """
    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend must be between 0 and 1")
    if not 0.0 <= color_weight <= 1.0:
        raise ValueError("color_weight must be between 0 and 1")

    use_color = (
        query_color_descriptors is not None or gallery_color_descriptors is not None
    )
    if use_color and (
        query_color_descriptors is None or gallery_color_descriptors is None
    ):
        raise ValueError(
            "query_color_descriptors and gallery_color_descriptors must be provided together"
        )
    if use_color:
        query_color_descriptors = np.asarray(
            query_color_descriptors, dtype="float32"
        )
        gallery_color_descriptors = np.asarray(
            gallery_color_descriptors, dtype="float32"
        )
        if query_color_descriptors.ndim != 2:
            raise ValueError(
                "query_color_descriptors must have shape (queries, features)"
            )
        if query_color_descriptors.shape[0] != len(query_embeddings):
            raise ValueError(
                "query embeddings and colour descriptors must have equal length"
            )
        if gallery_color_descriptors.shape[0] != len(gallery_embeddings):
            raise ValueError(
                "gallery embeddings and colour descriptors must have equal length"
            )
        if query_color_descriptors.shape[1] != gallery_color_descriptors.shape[1]:
            raise ValueError("query and gallery colour descriptor dimensions do not match")

    search_depth = min(candidate_depth, len(gallery_embeddings))
    initial_scores, candidates = index.search(query_embeddings, search_depth)
    gallery_sets, gallery_thresholds = gallery_reciprocal_sets(
        gallery_embeddings, index, k1
    )
    reranked_rows = []

    for query_index, (scores, row) in enumerate(zip(initial_scores, candidates)):
        forward_count = min(k1, len(row))
        forward = row[:forward_count]
        query_set = {
            int(candidate)
            for score, candidate in zip(scores[:forward_count], forward)
            if score >= gallery_thresholds[candidate]
        }
        for candidate in row[: min(k2, len(row))]:
            query_set.update(gallery_sets[int(candidate)])

        combined_distances = []
        if use_color:
            candidate_color_similarities = color_similarities(
                query_color_descriptors[query_index], gallery_color_descriptors[row]
            )
        for score, candidate in zip(scores, row):
            gallery_set = gallery_sets[int(candidate)]
            union = query_set | gallery_set
            intersection = query_set & gallery_set
            jaccard = 1.0 - len(intersection) / max(1, len(union))
            original_distance = 1.0 - float(score)
            non_color_distance = blend * original_distance + (1.0 - blend) * jaccard
            if use_color:
                color_similarity = candidate_color_similarities[len(combined_distances)]
                distance = (1.0 - color_weight) * non_color_distance + color_weight * (
                    1.0 - float(color_similarity)
                )
            else:
                distance = non_color_distance
            combined_distances.append(distance)

        ordering = np.argsort(combined_distances)[:maximum_k]
        reranked_rows.append(row[ordering])

    return np.asarray(reranked_rows, dtype=np.int64)
