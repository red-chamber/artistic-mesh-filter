"""The calibration kit: sampling, the blank form, and determinism."""

import pytest

from mesh_filter.config import load_project_config
from mesh_filter.evaluation.screening_kit import plan_screening_review
from mesh_filter.manifest import write_csv
from mesh_filter.screening import screening_paths

FEATURE_COLUMNS = ["asset_id", "face_count", "vertex_count"]
SCREEN_COLUMNS = [
    "asset_id",
    "source_dataset",
    "label",
    "verdict",
    "flags_passed",
    "flags_failed",
    "flags_unknown",
    "failed_rules",
    "unknown_rules",
]


def _row(asset_id: str, source: str, verdict: str, failed: str = "") -> dict[str, str]:
    return {
        "asset_id": asset_id,
        "source_dataset": source,
        "label": "1" if source == "toys4k" else "0",
        "verdict": verdict,
        "flags_passed": "5",
        "flags_failed": "0",
        "flags_unknown": "0",
        "failed_rules": failed,
        "unknown_rules": "",
    }


def _corpus() -> list[dict[str, str]]:
    return [
        _row("c1", "toys4k", "clean"),
        _row("c2", "objaverse", "clean"),
        _row("c3", "toys4k", "clean"),
        _row("c4", "objaverse", "clean"),
        _row("f1", "toys4k", "flagged", "redundant_vertices"),
        _row("f2", "objaverse", "flagged", "redundant_vertices"),
        _row("f3", "toys4k", "flagged", "degree_concentration"),
        _row("f4", "objaverse", "flagged", "degree_concentration"),
        _row("r1", "toys4k", "rejected", "narrow_faces"),
        _row("r2", "objaverse", "rejected", "narrow_faces"),
        _row("r3", "toys4k", "rejected", "face_count_range"),
        _row("r4", "objaverse", "rejected", "face_count_range"),
    ]


def _config_with_tables(make_config, screening_rows):
    config_path = make_config(
        sources={},
        outputs={
            "manifest": "data/manifests/assets.csv",
            "features": "data/manifests/features.csv",
        },
    )
    config = load_project_config(config_path)
    features = [
        {"asset_id": row["asset_id"], "face_count": "1200", "vertex_count": "600"}
        for row in screening_rows
    ]
    write_csv(config.output_path("features"), features, FEATURE_COLUMNS)
    table_path, _ = screening_paths(config)
    write_csv(table_path, screening_rows, SCREEN_COLUMNS)
    return config


def test_plan_covers_every_level_both_sources_and_the_decision_rules(make_config) -> None:
    config = _config_with_tables(make_config, _corpus())

    sheet, key, report = plan_screening_review(config, per_level=3)

    assert report["levels"] == {"accept": 3, "flag": 3, "reject": 3}
    assert report["available"] == {"accept": 4, "flag": 4, "reject": 4}
    labels = [row["machine_label"] for row in sheet]
    assert labels.count("accept") == 3
    assert labels.count("flag") == 3
    assert labels.count("reject") == 3

    assert {row["source_dataset"] for row in key} == {"toys4k", "objaverse"}
    reasons = {entry["reason"] for entry in report["strata"] if entry["sampled"]}
    assert {"narrow_faces", "face_count_range", "redundant_vertices"} <= reasons

    key_by_name = {row["anonymous_name"]: row for row in key}
    for row in sheet:
        assert key_by_name[row["anonymous_name"]]["machine_label"] == row["machine_label"]
        assert key_by_name[row["anonymous_name"]]["asset_id"] == row["asset_id"]


def test_the_sheet_is_a_blank_form(make_config) -> None:
    """The machine's label is shown on purpose; the judgement columns are empty."""
    config = _config_with_tables(make_config, _corpus())

    sheet, _, _ = plan_screening_review(config, per_level=2)

    names = [row["anonymous_name"] for row in sheet]
    assert names == [f"A{position:04d}" for position in range(1, len(sheet) + 1)]
    assert all(row["my_label"] == "" for row in sheet)
    assert all(row["note"] == "" for row in sheet)


def test_the_sample_is_deterministic_for_a_seed(make_config) -> None:
    config = _config_with_tables(make_config, _corpus())

    first_sheet, _, _ = plan_screening_review(config, per_level=3)
    second_sheet, _, _ = plan_screening_review(config, per_level=3)

    def _fingerprint(rows):
        return [
            (row["anonymous_name"], row["asset_id"], row["machine_label"]) for row in rows
        ]

    assert _fingerprint(first_sheet) == _fingerprint(second_sheet)


def test_plan_refuses_without_a_screening_table(make_config) -> None:
    config_path = make_config(
        sources={},
        outputs={
            "manifest": "data/manifests/assets.csv",
            "features": "data/manifests/features.csv",
        },
    )
    config = load_project_config(config_path)

    with pytest.raises(FileNotFoundError):
        plan_screening_review(config)
