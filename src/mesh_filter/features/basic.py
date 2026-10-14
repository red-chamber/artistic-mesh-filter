from __future__ import annotations

import math
from typing import Any

import numpy as np


class GeometryError(ValueError):
    """Raised when arrays cannot represent a valid indexed triangle mesh."""


def _component_count(vertex_count: int, faces: np.ndarray) -> int:
    if vertex_count == 0:
        return 0
    parent = np.arange(vertex_count, dtype=np.int64)

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    used = np.unique(faces.reshape(-1))
    for a, b, c in faces:
        union(int(a), int(b))
        union(int(b), int(c))
    return len({find(int(index)) for index in used})


def compute_basic_features(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    zero_area_epsilon: float = 1.0e-12,
    equilateral_threshold: float = 0.90,
    sliver_threshold: float = 0.05,
    small_angle_threshold_deg: float = 5.0,
) -> dict[str, Any]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise GeometryError("vertices must have shape (N, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise GeometryError("faces must have shape (M, 3)")
    if len(vertices) == 0 or len(faces) == 0:
        raise GeometryError("mesh must contain vertices and triangle faces")
    if not np.isfinite(vertices).all():
        raise GeometryError("vertices contain NaN or Inf")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise GeometryError("face index is out of bounds")

    triangles = vertices[faces]
    side_a = np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=1)
    side_b = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)
    side_c = np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=1)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    zero_area = areas <= zero_area_epsilon

    length_sq_sum = side_a**2 + side_b**2 + side_c**2
    equilateral = np.divide(
        4.0 * math.sqrt(3.0) * areas,
        length_sq_sum,
        out=np.zeros_like(areas),
        where=length_sq_sum > 0,
    )

    def angle(opposite: np.ndarray, adjacent_1: np.ndarray, adjacent_2: np.ndarray) -> np.ndarray:
        denominator = 2.0 * adjacent_1 * adjacent_2
        cosine = np.divide(
            adjacent_1**2 + adjacent_2**2 - opposite**2,
            denominator,
            out=np.ones_like(opposite),
            where=denominator > 0,
        )
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    min_angles = np.minimum.reduce(
        [
            angle(side_a, side_b, side_c),
            angle(side_b, side_a, side_c),
            angle(side_c, side_a, side_b),
        ]
    )

    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
    canonical_faces = np.sort(faces, axis=1)
    _, face_counts = np.unique(canonical_faces, axis=0, return_counts=True)

    nonzero_areas = areas[~zero_area]
    area_mean = float(nonzero_areas.mean()) if len(nonzero_areas) else 0.0
    area_cv = (
        float(nonzero_areas.std(ddof=0) / area_mean)
        if len(nonzero_areas) and area_mean > 0
        else 0.0
    )

    return {
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "edge_count": int(len(unique_edges)),
        "component_count": int(_component_count(len(vertices), faces)),
        "zero_area_face_count": int(zero_area.sum()),
        "zero_area_face_ratio": float(zero_area.mean()),
        "duplicate_face_count": int(np.maximum(face_counts - 1, 0).sum()),
        "boundary_edge_count": int((edge_counts == 1).sum()),
        "nonmanifold_edge_count": int((edge_counts > 2).sum()),
        "equilateral_ratio": float((equilateral > equilateral_threshold).mean()),
        "sliver_ratio": float((equilateral < sliver_threshold).mean()),
        "small_angle_ratio": float((min_angles < small_angle_threshold_deg).mean()),
        "area_coefficient_of_variation": area_cv,
    }
