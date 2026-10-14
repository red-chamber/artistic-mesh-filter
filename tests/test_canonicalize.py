import numpy as np

from mesh_filter.canonicalize import normalize_vertices


def test_normalize_vertices_centers_and_scales() -> None:
    vertices = np.array([[-2.0, 0.0, 0.0], [2.0, 2.0, 1.0]])
    normalized, transform = normalize_vertices(vertices)

    assert np.allclose((normalized.min(axis=0) + normalized.max(axis=0)) / 2.0, 0.0)
    assert np.isclose((normalized.max(axis=0) - normalized.min(axis=0)).max(), 1.0)
    assert transform.scale == 0.25
