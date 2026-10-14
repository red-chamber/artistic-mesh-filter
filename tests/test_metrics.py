import numpy as np

from mesh_filter.evaluation.metrics import precision_at_k


def test_precision_at_k_uses_highest_scores() -> None:
    labels = np.array([0, 1, 1, 0])
    scores = np.array([0.2, 0.9, 0.8, 0.1])
    assert precision_at_k(labels, scores, 2) == 1.0
