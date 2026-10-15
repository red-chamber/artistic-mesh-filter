import numpy as np
import pytest

from mesh_filter.config import load_project_config
from mesh_filter.features.artistic import COLUMNS, compute_artistic_features
from mesh_filter.features.basic import compute_basic_features
from mesh_filter.inspection import load_artistic_features
from mesh_filter.manifest import build_manifest, read_csv
from mesh_filter.validate import validate_manifest


def _grid(size: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """A flat triangulated square grid: the hand-made-looking end of the spectrum."""
    xs, ys = np.meshgrid(np.arange(size), np.arange(size))
    vertices = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(size * size)]).astype(float)
    faces = []
    for row in range(size - 1):
        for column in range(size - 1):
            corner = row * size + column
            faces.append([corner, corner + 1, corner + size + 1])
            faces.append([corner, corner + size + 1, corner + size])
    return vertices, np.array(faces, dtype=np.int64)


def _subdivide(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = list(vertices)
    cache: dict[tuple[int, int], int] = {}
    new_faces = []

    def midpoint(left: int, right: int) -> int:
        key = (min(left, right), max(left, right))
        if key not in cache:
            middle = (points[left] + points[right]) / 2.0
            points.append(middle / np.linalg.norm(middle))  # keep it on the sphere
            cache[key] = len(points) - 1
        return cache[key]

    for a, b, c in faces:
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        new_faces += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
    return np.array(points), np.array(new_faces, dtype=np.int64)


def _octahedron() -> tuple[np.ndarray, np.ndarray]:
    """Eight equilateral triangles on a closed, curved shell.

    Every edge has the same length, so this is the tessellation the criteria call
    bad — and unlike a spherified tetrahedron it has no distorted corners.
    """
    vertices = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    faces = np.array(
        [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def test_equilateral_ratio_is_the_bad_signal() -> None:
    """The direction correction, as an assertion.

    A near-equilateral tessellation is what automatic remeshing produces and the
    criteria call it bad; a hand-made grid of right triangles must not trip it.
    """
    grid_vertices, grid_faces = _grid()
    shell_vertices, shell_faces = _octahedron()

    grid = compute_basic_features(grid_vertices, grid_faces)
    shell = compute_basic_features(shell_vertices, shell_faces)

    assert grid["equilateral_ratio"] == 0.0
    assert shell["equilateral_ratio"] == pytest.approx(1.0)


def test_flat_mesh_vertices_are_all_redundant() -> None:
    """On a plane every vertex can be removed for free, so QEM must be ~0."""
    vertices, faces = _grid()

    features = compute_artistic_features(vertices, faces)

    assert features["qem_redundancy_median"] == pytest.approx(0.0, abs=1e-12)


def test_curved_mesh_vertices_carry_geometry() -> None:
    shell_vertices, shell_faces = _octahedron()
    grid_vertices, grid_faces = _grid()

    shell = compute_artistic_features(shell_vertices, shell_faces)
    grid = compute_artistic_features(grid_vertices, grid_faces)

    assert shell["qem_redundancy_median"] > grid["qem_redundancy_median"]
    assert shell["qem_redundancy_median"] > 1e-3


def test_flat_mesh_reports_no_measurable_curvature_correlation() -> None:
    """Constant curvature makes the correlation undefined, not zero."""
    vertices, faces = _grid()

    features = compute_artistic_features(vertices, faces)

    assert features["curvature_density_correlation"] == ""


def test_tiny_mesh_reports_no_measurable_curvature_correlation() -> None:
    """From a handful of vertices a rank correlation is noise, so leave it blank."""
    vertices, faces = _octahedron()  # six vertices

    features = compute_artistic_features(vertices, faces)

    assert features["curvature_density_correlation"] == ""
    assert features["qem_redundancy_median"] != ""


def test_degree_statistics_are_concentrated_on_a_grid() -> None:
    vertices, faces = _grid(size=10)

    features = compute_artistic_features(vertices, faces)

    # Interior grid vertices all have degree six, so one value dominates.
    assert features["vertex_degree_mode_share"] > 0.5
    assert 0.0 < features["vertex_degree_entropy"] < 0.8


def test_features_are_invariant_to_scale_rotation_and_translation() -> None:
    """Nothing here may learn the scale gap between the two sources."""
    vertices, faces = _grid()
    angle = 0.7
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    moved = (vertices @ rotation.T) * 1.0e6 + np.array([500.0, -300.0, 90.0])

    original = compute_artistic_features(vertices, faces)
    transformed = compute_artistic_features(moved, faces)

    for column in COLUMNS:
        expected, actual = original[column], transformed[column]
        if expected == "" or actual == "":
            assert expected == actual, column  # "not measurable" must stay not measurable
        else:
            assert actual == pytest.approx(expected, abs=1e-9), column


def test_direction_columns_stay_in_range_on_both_shapes() -> None:
    """No ordering claim is made here.

    An earlier version of this test asserted that a flat grid is more anisotropic
    than a closed shell. It is not: the two score 0.49 and 0.50. Any vertex whose
    incident edge directions come in balanced opposite pairs drives this measure
    towards zero regardless of how the mesh was made, so the column has no
    established direction yet — the 200 human labels have to decide that.
    """
    grid_vertices, grid_faces = _grid()
    shell_vertices, shell_faces = _octahedron()

    grid = compute_artistic_features(grid_vertices, grid_faces)
    shell = compute_artistic_features(shell_vertices, shell_faces)

    for features in (grid, shell):
        assert 0.0 <= features["local_edge_direction_anisotropy_mean"] <= 1.0
        assert 0.0 <= features["neighboring_direction_consistency_mean"] <= 1.0
        assert 0.0 <= features["vertex_degree_mode_share"] <= 1.0
        assert 0.0 <= features["vertex_degree_entropy"] <= 1.0


def test_disabled_columns_are_not_computed() -> None:
    vertices, faces = _grid()

    features = compute_artistic_features(vertices, faces, {"qem_redundancy_median"})

    assert set(features) == {"qem_redundancy_median"}


def test_load_switches_from_the_config(make_config, tmp_path) -> None:
    config_path = make_config(
        sources={},
        outputs={},
        extra={"features": {"qem_redundancy_median": True, "vertex_degree_entropy": False}},
    )
    del tmp_path

    assert load_artistic_features(config_path) == frozenset({"qem_redundancy_median"})


def test_validate_writes_the_artistic_columns(make_config, source_tree, valid_mesh: str) -> None:
    """Wiring check: the switch in the config must reach the feature table."""
    positive = source_tree("positive", {"a.obj": valid_mesh})
    negative = source_tree("negative", {"b.obj": valid_mesh})
    artistic_path = positive.parent.parent / "artistic.yaml"
    artistic_path.write_text(
        "features:\n  qem_redundancy_median: true\n", encoding="utf-8"
    )
    config_path = make_config(
        sources={
            "toys4k": {
                "root": positive.as_posix(),
                "label": 1,
                "label_origin": "synthetic_fixture",
                "selection_rule": "test_only",
            },
            "objaverse": {
                "root": negative.as_posix(),
                "label": 0,
                "label_origin": "synthetic_fixture",
                "selection_rule": "test_only",
            },
        },
        outputs={"manifest": "data/assets.csv", "features": "data/features.csv"},
        extra={"formats": {"extensions": [".obj"]}},
    )
    config = load_project_config(config_path)
    build_manifest(config)

    output, _ = validate_manifest(config, None, artistic_path)

    rows = read_csv(output)
    assert "qem_redundancy_median" in rows[0]
    assert "artistic_status" in rows[0]
    assert {row["artistic_status"] for row in rows} == {"ok"}
    assert all(row["qem_redundancy_median"] != "" for row in rows)


def test_too_small_mesh_records_a_status_instead_of_failing(
    make_config, source_tree, valid_mesh: str
) -> None:
    """A one-triangle mesh parses fine but cannot support artistic features."""
    positive = source_tree("positive", {"a.obj": "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"})
    negative = source_tree("negative", {"b.obj": valid_mesh})
    artistic_path = positive.parent.parent / "artistic.yaml"
    artistic_path.write_text("features:\n  qem_redundancy_median: true\n", encoding="utf-8")
    config_path = make_config(
        sources={
            "toys4k": {
                "root": positive.as_posix(),
                "label": 1,
                "label_origin": "synthetic_fixture",
                "selection_rule": "test_only",
            },
            "objaverse": {
                "root": negative.as_posix(),
                "label": 0,
                "label_origin": "synthetic_fixture",
                "selection_rule": "test_only",
            },
        },
        outputs={"manifest": "data/assets.csv", "features": "data/features.csv"},
        extra={"formats": {"extensions": [".obj"]}},
    )
    config = load_project_config(config_path)
    build_manifest(config)

    output, counts = validate_manifest(config, None, artistic_path)

    rows = {row["source_dataset"]: row for row in read_csv(output)}
    assert counts["parsed"] == 2
    assert rows["toys4k"]["artistic_status"].startswith("failed:")
    assert rows["toys4k"]["qem_redundancy_median"] == ""
    assert rows["objaverse"]["artistic_status"] == "ok"
