"""Evaluation metrics (numpy only). Definitions are documented in docs/ml.md.

Retrieval, per query q with relevant gallery set R_q (|R_q| > 0) and ranked top-K list T_q(K):
  Recall@K(q)    = |R_q ∩ T_q(K)| / |R_q|        (fraction of ALL relevant items retrieved)
  HitRate@K(q)   = 1 if |R_q ∩ T_q(K)| > 0 else 0 (at least one relevant item retrieved)
  Precision@K(q) = |R_q ∩ T_q(K)| / K
Each is averaged over queries with |R_q| > 0 (macro over queries); queries with no relevant
gallery item are counted separately and excluded from these averages.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RetrievalTask:
    """Evaluation interface: queries, gallery, relevance and the pipeline that produced scores."""

    query_ids: list[str]
    gallery_ids: list[str]
    relevance: np.ndarray  # bool [n_query, n_gallery]
    excluded: np.ndarray  # bool [n_query, n_gallery]: self/near-duplicate pairs never ranked
    description: dict  # dataset, split, relevance definition, pipeline/fusion version


def rank_gallery(scores: np.ndarray, excluded: np.ndarray, k: int) -> np.ndarray:
    """Top-k gallery indices per query; ties broken by gallery index (stable)."""
    masked = np.where(excluded, -np.inf, scores)
    order = np.argsort(-masked, axis=1, kind="stable")
    return order[:, :k]


def retrieval_metrics(task: RetrievalTask, scores: np.ndarray, ks: list[int]) -> dict:
    if scores.shape != task.relevance.shape:
        raise ValueError("scores and relevance shapes differ")
    relevance = task.relevance & ~task.excluded
    totals = relevance.sum(axis=1)
    has_relevant = totals > 0
    n_eligible = int((~task.excluded).sum(axis=1).min()) if len(task.query_ids) else 0
    top = rank_gallery(scores, task.excluded, max(ks))
    result = {
        "queries": len(task.query_ids),
        "queries_with_relevant": int(has_relevant.sum()),
        "queries_without_relevant": int((~has_relevant).sum()),
        "mean_relevant_per_query": float(totals[has_relevant].mean()) if has_relevant.any() else 0,
        "gallery_size": len(task.gallery_ids),
        "at_k": {},
    }
    for k in sorted(set(ks)):
        if k > n_eligible:
            continue
        hits = np.take_along_axis(relevance, top[:, :k], axis=1).sum(axis=1)
        valid = has_relevant
        result["at_k"][str(k)] = {
            "recall": float((hits[valid] / totals[valid]).mean()) if valid.any() else None,
            "hit_rate": float((hits[valid] > 0).mean()) if valid.any() else None,
            "precision": float((hits[valid] / k).mean()) if valid.any() else None,
        }
    return result


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    n = len(classes)
    confusion = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        confusion[t, p] += 1
    support = confusion.sum(axis=1)
    per_class = {
        classes[i]: (float(confusion[i, i] / support[i]) if support[i] else None) for i in range(n)
    }
    recalls = [v for v in per_class.values() if v is not None]
    return {
        "samples": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else None,
        "macro_recall": float(np.mean(recalls)) if recalls else None,
        "per_class_recall": per_class,
        "confusion_matrix": confusion.tolist(),
        "classes": classes,
    }


def kl_divergence(target: np.ndarray, predicted: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise KL(target || predicted) with 0*log0 = 0."""
    t = np.asarray(target, dtype=np.float64)
    p = np.clip(np.asarray(predicted, dtype=np.float64), eps, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(t > 0, t * (np.log(np.clip(t, eps, None)) - np.log(p)), 0.0)
    return terms.sum(axis=1)


def histogram_metrics(target: np.ndarray, predicted: np.ndarray) -> dict:
    kl = kl_divergence(target, predicted)
    mae = np.abs(target - predicted).mean(axis=1)
    tv = 0.5 * np.abs(target - predicted).sum(axis=1)
    return {
        "kl_mean": float(kl.mean()),
        "kl_median": float(np.median(kl)),
        "mae_per_bin_mean": float(mae.mean()),
        "total_variation_mean": float(tv.mean()),
    }


def histogram_intersection_matrix(queries: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    out = np.empty((len(queries), len(gallery)), dtype=np.float64)
    for i, q in enumerate(queries):
        out[i] = np.minimum(q[None, :], gallery).sum(axis=1)
    return out


def bhattacharyya_matrix(queries: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(queries, 0, None)) @ np.sqrt(np.clip(gallery, 0, None)).T
