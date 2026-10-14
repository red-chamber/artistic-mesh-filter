import numpy as np
import pytest

from mesh_filter.features.basic import GeometryError, compute_basic_features


def test_equilateral_triangle() -> None:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3.0) / 2.0, 0.0]]
    )
    faces = np.array([[0, 1, 2]])

    result = compute_basic_features(vertices, faces)

    assert result["vertex_count"] == 3
    assert result["face_count"] == 1
    assert result["edge_count"] == 3
    assert result["boundary_edge_count"] == 3
    assert result["component_count"] == 1
    assert result["equilateral_ratio"] == pytest.approx(1.0)
    assert result["sliver_ratio"] == pytest.approx(0.0)


def test_disconnected_components() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [10.0, 1.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    assert compute_basic_features(vertices, faces)["component_count"] == 2


def test_out_of_bounds_index_is_rejected() -> None:
    vertices = np.zeros((3, 3))
    faces = np.array([[0, 1, 3]])
    with pytest.raises(GeometryError, match="out of bounds"):
        compute_basic_features(vertices, faces)
