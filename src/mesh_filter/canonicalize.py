from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NormalizationTransform:
    center: tuple[float, float, float]
    scale: float


def normalize_vertices(
    vertices: np.ndarray,
    *,
    center: bool = True,
    unit_longest_extent: bool = True,
) -> tuple[np.ndarray, NormalizationTransform]:
    """Return a normalized copy without modifying topology or the input array."""
    values = np.asarray(vertices, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0:
        raise ValueError("vertices must have non-empty shape (N, 3)")
    if not np.isfinite(values).all():
        raise ValueError("vertices contain NaN or Inf")

    minimum = values.min(axis=0)
    maximum = values.max(axis=0)
    midpoint = (minimum + maximum) / 2.0 if center else np.zeros(3)
    normalized = values - midpoint
    longest_extent = float((maximum - minimum).max())
    if unit_longest_extent:
        if longest_extent <= 0:
            raise ValueError("mesh has zero bounding-box extent")
        scale = 1.0 / longest_extent
        normalized = normalized * scale
    else:
        scale = 1.0

    transform = NormalizationTransform(
        center=tuple(float(value) for value in midpoint),
        scale=scale,
    )
    return normalized, transform
