from collections import Counter

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from src.task4.training import AMP_ENABLED, DEVICE


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
):
    search_depth = min(candidate_depth, len(gallery_embeddings))
    initial_scores, candidates = index.search(query_embeddings, search_depth)
    gallery_sets, gallery_thresholds = gallery_reciprocal_sets(
        gallery_embeddings, index, k1
    )
    reranked_rows = []

    for scores, row in zip(initial_scores, candidates):
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
        for score, candidate in zip(scores, row):
            gallery_set = gallery_sets[int(candidate)]
            union = query_set | gallery_set
            intersection = query_set & gallery_set
            jaccard = 1.0 - len(intersection) / max(1, len(union))
            original_distance = 1.0 - float(score)
            distance = blend * original_distance + (1.0 - blend) * jaccard
            combined_distances.append(distance)

        ordering = np.argsort(combined_distances)[:maximum_k]
        reranked_rows.append(row[ordering])

    return np.asarray(reranked_rows, dtype=np.int64)
