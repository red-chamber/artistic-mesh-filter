from __future__ import annotations

from typing import Any

import numpy as np


def precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if len(labels) == 0:
        return float("nan")
    actual_k = min(k, len(labels))
    indices = np.argsort(-scores, kind="stable")[:actual_k]
    return float(np.asarray(labels)[indices].mean())


def binary_metrics(labels: np.ndarray, scores: np.ndarray, k: int = 50) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores >= 0.5).astype(np.int64)
    metrics: dict[str, Any] = {
        f"precision_at_{k}": precision_at_k(labels, scores, k),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "sample_count": int(len(labels)),
        "positive_count": int(labels.sum()),
    }
    if len(np.unique(labels)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
        metrics["average_precision"] = float(average_precision_score(labels, scores))
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
    return metrics
