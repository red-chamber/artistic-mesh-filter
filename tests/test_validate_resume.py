from pathlib import Path

import pytest

from mesh_filter.config import load_project_config
from mesh_filter.manifest import MANIFEST_FIELDS, build_manifest, read_csv, write_csv
from mesh_filter.validate import validate_manifest


def _config(make_config, positive: Path, negative: Path, resume: bool):
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
        outputs={
            "manifest": "data/manifest.csv",
            "features": "data/features.csv",
        },
        extra={
            "formats": {"extensions": [".obj"]},
            "runtime": {"hash_files": True, "continue_on_error": True, "resume": resume},
        },
    )
    return load_project_config(config_path)


def test_resume_reuses_parsed_rows(
    make_config, source_tree, valid_mesh: str, second_mesh: str
) -> None:
    positive = source_tree("positive", {"a.obj": valid_mesh, "b.obj": second_mesh})
    negative = source_tree("negative", {"c.obj": valid_mesh})

    fresh = _config(make_config, positive, negative, resume=False)
    build_manifest(fresh)
    output, first = validate_manifest(fresh, None)
    first_text = output.read_text(encoding="utf-8")
    assert first["parsed"] == 3
    assert first["resumed"] == 0

    resumed = _config(make_config, positive, negative, resume=True)
    output, second = validate_manifest(resumed, None)

    assert second["parsed"] == 3
    assert second["resumed"] == 3
    assert output.read_text(encoding="utf-8") == first_text


def test_resume_processes_only_new_assets(
    make_config, source_tree, valid_mesh: str, second_mesh: str
) -> None:
    positive = source_tree("positive", {"a.obj": valid_mesh})
    negative = source_tree("negative", {"c.obj": valid_mesh})

    config = _config(make_config, positive, negative, resume=False)
    build_manifest(config)
    validate_manifest(config, None)

    (positive / "b.obj").write_text(second_mesh, encoding="utf-8")
    resumed = _config(make_config, positive, negative, resume=True)
    build_manifest(resumed)
    _, counts = validate_manifest(resumed, None)

    assert counts["total"] == 3
    assert counts["resumed"] == 2
    assert counts["parsed"] == 3


def _crash_config(make_config, positive: Path, negative: Path, flush_every: int):
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
        outputs={"manifest": "data/manifest.csv", "features": "data/features.csv"},
        extra={
            "formats": {"extensions": [".obj"]},
            "runtime": {
                "hash_files": True,
                "continue_on_error": False,
                "resume": False,
                "flush_every": flush_every,
            },
        },
    )
    return load_project_config(config_path)


def test_interrupted_run_leaves_a_resumable_table(
    make_config, source_tree, valid_mesh: str, second_mesh: str, invalid_mesh: str
) -> None:
    """A long run that dies must leave its progress on disk, not nothing."""
    positive = source_tree(
        "positive", {"a.obj": valid_mesh, "b.obj": second_mesh, "c.obj": invalid_mesh}
    )
    negative = source_tree("negative", {"d.obj": valid_mesh})
    config = _crash_config(make_config, positive, negative, flush_every=2)
    build_manifest(config)

    with pytest.raises(Exception):  # noqa: B017 - any parse error must propagate here
        validate_manifest(config, None)

    rows = read_csv(config.output_path("features"))
    assert len(rows) == 2
    assert all(row["parse_status"] == "parsed" for row in rows)


def test_resume_continues_from_a_partial_table(
    make_config, source_tree, valid_mesh: str, second_mesh: str, invalid_mesh: str
) -> None:
    """The point of flushing early: the next run only redoes what is missing."""
    positive = source_tree(
        "positive", {"a.obj": valid_mesh, "b.obj": second_mesh, "c.obj": invalid_mesh}
    )
    negative = source_tree("negative", {"d.obj": valid_mesh})

    crashed = _crash_config(make_config, positive, negative, flush_every=2)
    build_manifest(crashed)
    with pytest.raises(Exception):  # noqa: B017
        validate_manifest(crashed, None)

    resumed = _config(make_config, positive, negative, resume=True)
    _, counts = validate_manifest(resumed, None)

    assert counts["resumed"] == 2
    assert counts["total"] == 4
    assert counts["parsed"] == 3
    assert counts["failed"] == 1


def test_resume_drops_rows_that_left_the_manifest(
    make_config, source_tree, valid_mesh: str, second_mesh: str
) -> None:
    positive = source_tree("positive", {"a.obj": valid_mesh, "b.obj": second_mesh})
    negative = source_tree("negative", {"c.obj": valid_mesh})

    config = _config(make_config, positive, negative, resume=False)
    manifest_path, _ = build_manifest(config)
    validate_manifest(config, None)

    kept = read_csv(manifest_path)[:2]
    write_csv(manifest_path, kept, MANIFEST_FIELDS)

    resumed = _config(make_config, positive, negative, resume=True)
    output, counts = validate_manifest(resumed, None)

    assert counts["total"] == 2
    assert len(read_csv(output)) == 2
