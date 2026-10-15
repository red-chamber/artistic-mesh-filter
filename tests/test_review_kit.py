import os
from pathlib import Path

import pytest

from mesh_filter.config import load_project_config
from mesh_filter.evaluation.review_kit import _remap_to_used_vertices, prepare_review
from mesh_filter.manifest import build_manifest, read_csv, write_csv

OUTPUTS = {
    "manifest": "data/assets.csv",
    "audit_gold": "data/labels/audit_gold.csv",
    "review_dir": "review",
}

SHEET_FIELDS = ["anonymous_name", "asset_id", "human_label", "human_confidence",
                "human_notes", "reviewer", "reviewed_at"]

MESH_WITH_ORPHAN_VERTEX = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 9 9 9\nf 1 2 3\n"


def _config(make_config, positive: Path, negative: Path):
    return load_project_config(
        make_config(
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
            outputs=OUTPUTS,
            extra={"formats": {"extensions": [".obj"]}},
        )
    )


def _write_sheet(config, mapping: dict[str, str]) -> None:
    rows = [
        {"anonymous_name": name, "asset_id": asset_id, "human_label": ""}
        for name, asset_id in mapping.items()
    ]
    write_csv(config.output_path("audit_gold"), rows, SHEET_FIELDS)


def _asset_ids(config) -> dict[str, str]:
    """Map source_dataset/relpath to asset_id."""
    return {
        f"{row['source_dataset']}/{row['raw_relpath']}": row["asset_id"]
        for row in read_csv(config.output_path("manifest"))
    }


def test_review_files_are_minimal_and_anonymous(
    make_config, source_tree, valid_mesh: str, second_mesh: str
) -> None:
    positive = source_tree("positive", {"Cube_Model_A.obj": valid_mesh})
    negative = source_tree("negative", {"objaverse_secret_name.obj": second_mesh})
    config = _config(make_config, positive, negative)
    build_manifest(config)
    ids = _asset_ids(config)
    _write_sheet(
        config,
        {
            "A0001": ids["toys4k/Cube_Model_A.obj"],
            "A0002": ids["objaverse/objaverse_secret_name.obj"],
        },
    )

    output, summary = prepare_review(config)

    assert summary["written"] == 2
    names = sorted(path.name for path in output.glob("*.obj"))
    assert names == ["A0001.obj", "A0002.obj"]

    for name in names:
        text = (output / name).read_text(encoding="ascii")
        lines = text.strip().splitlines()
        assert all(line.startswith("v ") or line.startswith("f ") for line in lines)
        # No comments, no normals, no UVs: nothing that could name the source.
        assert "#" not in text
        assert "vt " not in text and "vn " not in text
        assert "Cube_Model_A" not in text and "objaverse_secret_name" not in text


def test_orphan_vertices_are_dropped(make_config, source_tree, valid_mesh: str) -> None:
    """How many unused vertices an exporter leaves behind is itself a fingerprint."""
    positive = source_tree("positive", {"a.obj": MESH_WITH_ORPHAN_VERTEX})
    negative = source_tree("negative", {"b.obj": valid_mesh})
    config = _config(make_config, positive, negative)
    build_manifest(config)
    ids = _asset_ids(config)
    _write_sheet(config, {"A0001": ids["toys4k/a.obj"], "A0002": ids["objaverse/b.obj"]})

    output, _ = prepare_review(config)

    lines = (output / "A0001.obj").read_text(encoding="ascii").strip().splitlines()
    assert sum(1 for line in lines if line.startswith("v ")) == 3
    assert sum(1 for line in lines if line.startswith("f ")) == 1


def test_every_file_gets_the_same_timestamp(
    make_config, source_tree, valid_mesh: str, second_mesh: str
) -> None:
    positive = source_tree("positive", {"a.obj": valid_mesh})
    negative = source_tree("negative", {"b.obj": second_mesh})
    config = _config(make_config, positive, negative)
    build_manifest(config)
    ids = _asset_ids(config)
    _write_sheet(config, {"A0001": ids["toys4k/a.obj"], "A0002": ids["objaverse/b.obj"]})

    output, _ = prepare_review(config)

    stamps = {os.stat(output / f"A000{i}.obj").st_mtime for i in (1, 2)}
    assert len(stamps) == 1


def test_remapping_preserves_the_triangles() -> None:
    """Dropping orphan vertices must not move, drop or rewire a single face."""
    import numpy as np

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [9.0, 9.0, 9.0],  # orphan
            [1.0, 1.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [1, 4, 2]])

    new_vertices, new_faces = _remap_to_used_vertices(vertices, faces)

    assert len(new_faces) == len(faces)
    for old_face, new_face in zip(faces, new_faces, strict=True):
        assert {tuple(np.round(old_vertex, 9)) for old_vertex in vertices[old_face]} == {
            tuple(np.round(new_vertex, 9)) for new_vertex in new_vertices[new_face]
        }


def test_stale_files_from_an_earlier_sample_are_removed(
    make_config, source_tree, valid_mesh: str
) -> None:
    positive = source_tree("positive", {"a.obj": valid_mesh})
    negative = source_tree("negative", {"b.obj": valid_mesh})
    config = _config(make_config, positive, negative)
    build_manifest(config)
    ids = _asset_ids(config)
    review_dir = config.output_path("review_dir")
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "A0999.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="ascii")
    _write_sheet(config, {"A0001": ids["toys4k/a.obj"]})

    output, summary = prepare_review(config)

    assert summary["stale_files_removed"] == 1
    assert not (output / "A0999.obj").exists()


def test_missing_asset_is_reported_not_skipped_silently(
    make_config, source_tree, valid_mesh: str
) -> None:
    positive = source_tree("positive", {"a.obj": valid_mesh})
    negative = source_tree("negative", {"b.obj": valid_mesh})
    config = _config(make_config, positive, negative)
    build_manifest(config)
    _write_sheet(config, {"A0001": "deadbeefdeadbeefdead"})

    _, summary = prepare_review(config)

    assert summary["written"] == 0
    assert summary["missing_count"] == 1
    assert summary["missing_from_manifest"] == ["deadbeefdeadbeefdead"]


def test_limit_writes_only_the_requested_prefix(
    make_config, source_tree, valid_mesh: str, second_mesh: str
) -> None:
    positive = source_tree("positive", {"a.obj": valid_mesh})
    negative = source_tree("negative", {"b.obj": second_mesh})
    config = _config(make_config, positive, negative)
    build_manifest(config)
    ids = _asset_ids(config)
    _write_sheet(config, {"A0001": ids["toys4k/a.obj"], "A0002": ids["objaverse/b.obj"]})

    output, summary = prepare_review(config, limit=1)

    assert summary["written"] == 1
    assert [path.name for path in output.glob("*.obj")] == ["A0001.obj"]


def test_protocol_and_missing_sheet(make_config, source_tree, valid_mesh: str) -> None:
    positive = source_tree("positive", {"a.obj": valid_mesh})
    negative = source_tree("negative", {"b.obj": valid_mesh})
    config = _config(make_config, positive, negative)

    with pytest.raises(FileNotFoundError, match="build-audit-gold"):
        prepare_review(config)

    build_manifest(config)
    ids = _asset_ids(config)
    _write_sheet(config, {"A0001": ids["toys4k/a.obj"]})

    output, summary = prepare_review(config)

    assert Path(summary["protocol"]).exists()
    assert "human_label" in (output / "REVIEW_PROTOCOL.md").read_text(encoding="utf-8")
