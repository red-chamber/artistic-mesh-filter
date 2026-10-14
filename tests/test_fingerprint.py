import numpy as np
import pytest

from mesh_filter.config import ConfigError, load_project_config
from mesh_filter.deduplicate import mark_duplicates
from mesh_filter.fingerprint import (
    bucket_key,
    candidate_rows,
    geometric_signature,
    signatures_match,
    write_fingerprints,
)
from mesh_filter.manifest import build_manifest, read_csv
from mesh_filter.validate import validate_manifest

CUBE_VERTICES = np.array(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ]
)
CUBE_FACES = np.array(
    [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ]
)
TETRAHEDRON_VERTICES = np.array(
    [[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]]
)
TETRAHEDRON_FACES = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])

OUTPUTS = {
    "manifest": "data/assets.csv",
    "features": "data/features.csv",
    "deduplicated_manifest": "data/assets_dedup.csv",
    "fingerprints": "data/fingerprints.csv",
}


def _obj_text(vertices: np.ndarray, faces: np.ndarray) -> str:
    lines = [f"v {x} {y} {z}" for x, y, z in vertices]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    return "\n".join(lines) + "\n"


def _reexport(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The same shape, rotated, mirrored, scaled, moved and vertex-reordered.

    Reversing the vertex order is an involution, so the new face indices are the
    reversed old ones; without that remap this would silently be a different mesh.
    """
    rotation = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mirror = np.diag([-1.0, 1.0, 1.0])
    moved = (vertices @ (mirror @ rotation).T) * 3.5 + np.array([10.0, -2.0, 5.0])
    order = np.arange(len(vertices))[::-1]
    return moved[order], order[faces]


def _config(make_config, positive, negative, **extra):
    sources = {
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
    }
    return load_project_config(
        make_config(
            sources=sources,
            outputs=OUTPUTS,
            extra={
                "formats": {"extensions": [".obj"]},
                "runtime": {"hash_files": True, "continue_on_error": True},
                "deduplicate": {"near_duplicate": True, "tolerance": 0.001},
                **extra,
            },
        )
    )


def test_signature_survives_re_export() -> None:
    """Rotation, mirroring, scale, translation and vertex order must not matter."""
    original = geometric_signature(CUBE_VERTICES, CUBE_FACES)
    moved_vertices, moved_faces = _reexport(CUBE_VERTICES, CUBE_FACES)
    reexported = geometric_signature(moved_vertices, moved_faces)

    assert signatures_match(
        {
            **original,
            "vertex_count": 8,
            "face_count": 12,
            "component_count": 1,
        },
        {
            **reexported,
            "vertex_count": 8,
            "face_count": 12,
            "component_count": 1,
        },
        0.001,
    )


def test_signature_separates_different_shapes() -> None:
    cube = geometric_signature(CUBE_VERTICES, CUBE_FACES)
    tetrahedron = geometric_signature(TETRAHEDRON_VERTICES, TETRAHEDRON_FACES)

    assert not signatures_match(
        {**cube, "vertex_count": 8, "face_count": 12, "component_count": 1},
        {**tetrahedron, "vertex_count": 4, "face_count": 4, "component_count": 1},
        0.001,
    )


def test_bucket_key_is_stable_for_equal_shapes() -> None:
    left = geometric_signature(CUBE_VERTICES, CUBE_FACES)
    moved_vertices, moved_faces = _reexport(CUBE_VERTICES, CUBE_FACES)
    right = geometric_signature(moved_vertices, moved_faces)

    assert bucket_key(8, 12, 1, left["eigenvalue_ratios"]) == bucket_key(
        8, 12, 1, right["eigenvalue_ratios"]
    )


def test_only_assets_sharing_counts_are_parsed(make_config, source_tree) -> None:
    """An asset alone in its (vertex, face, component) bucket cannot be a duplicate."""
    positive = source_tree("positive", {"cube.obj": _obj_text(CUBE_VERTICES, CUBE_FACES)})
    negative = source_tree(
        "negative", {"tetra.obj": _obj_text(TETRAHEDRON_VERTICES, TETRAHEDRON_FACES)}
    )
    config = _config(make_config, positive, negative)
    build_manifest(config)
    validate_manifest(config, None)

    candidates, stats = candidate_rows(config)

    assert candidates == []
    assert stats["candidate_assets"] == 0
    assert stats["singleton_buckets_skipped"] == 2


def test_re_exported_model_is_merged_across_sources(make_config, source_tree) -> None:
    """This is the leak near-duplicate detection exists to prevent."""
    positive = source_tree("positive", {"cube.obj": _obj_text(CUBE_VERTICES, CUBE_FACES)})
    moved_vertices, moved_faces = _reexport(CUBE_VERTICES, CUBE_FACES)
    negative = source_tree("negative", {"copy.obj": _obj_text(moved_vertices, moved_faces)})
    config = _config(make_config, positive, negative)
    build_manifest(config)
    validate_manifest(config, None)

    output, report = mark_duplicates(config)

    groups = {row["asset_id"]: row["duplicate_group_id"] for row in read_csv(output)}
    assert len(set(groups.values())) == 1
    assert next(iter(groups.values())).startswith("dup-")
    assert report["groups_merged_by_near_duplicate"] == 1
    assert report["linked_pairs"] == 1
    assert report["assets_in_duplicate_groups"] == 2


def test_distinct_shapes_stay_in_separate_groups(make_config, source_tree) -> None:
    positive = source_tree("positive", {"cube.obj": _obj_text(CUBE_VERTICES, CUBE_FACES)})
    negative = source_tree(
        "negative", {"tetra.obj": _obj_text(TETRAHEDRON_VERTICES, TETRAHEDRON_FACES)}
    )
    config = _config(make_config, positive, negative)
    build_manifest(config)
    validate_manifest(config, None)

    output, report = mark_duplicates(config)

    groups = {row["duplicate_group_id"] for row in read_csv(output)}
    assert len(groups) == 2
    assert report["groups_merged_by_near_duplicate"] == 0
    assert report["duplicate_groups"] == 0


def test_identical_files_are_grouped_without_fingerprints(make_config, source_tree) -> None:
    text = _obj_text(CUBE_VERTICES, CUBE_FACES)
    positive = source_tree("positive", {"cube.obj": text})
    negative = source_tree("negative", {"same.obj": text})
    config = _config(make_config, positive, negative, deduplicate={"near_duplicate": False})
    build_manifest(config)
    validate_manifest(config, None)

    output, report = mark_duplicates(config)

    groups = {row["duplicate_group_id"] for row in read_csv(output)}
    assert len(groups) == 1
    assert next(iter(groups)).startswith("exact-")
    assert report["near_duplicate"] is False
    assert report["assets_in_duplicate_groups"] == 2


def test_fingerprint_table_records_the_candidates(make_config, source_tree) -> None:
    positive = source_tree("positive", {"cube.obj": _obj_text(CUBE_VERTICES, CUBE_FACES)})
    negative = source_tree("negative", {"same.obj": _obj_text(CUBE_VERTICES, CUBE_FACES)})
    config = _config(make_config, positive, negative)
    build_manifest(config)
    validate_manifest(config, None)

    output, summary = write_fingerprints(config)

    rows = read_csv(output)
    assert summary["fingerprinted"] == 2
    assert all(len(row["radial_quantiles"].split(";")) == 64 for row in rows)
    assert len({row["bucket_key"] for row in rows}) == 1


def test_missing_features_table_is_reported(make_config, source_tree) -> None:
    positive = source_tree("positive", {"cube.obj": _obj_text(CUBE_VERTICES, CUBE_FACES)})
    negative = source_tree("negative", {"cube.obj": _obj_text(CUBE_VERTICES, CUBE_FACES)})
    config = _config(make_config, positive, negative)
    build_manifest(config)

    with pytest.raises(ConfigError, match="validate"):
        mark_duplicates(config)
