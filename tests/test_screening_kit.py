"""The calibration kit: stratified sampling, the blank form, determinism."""

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


def _corpus() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Four assets per level, spread over four face-count bins and both sources."""
    entries = [
        ("c1", "toys4k", "clean", 100),
        ("c2", "objaverse", "clean", 700),
        ("c3", "toys4k", "clean", 3000),
        ("c4", "objaverse", "clean", 9000),
        ("f1", "toys4k", "flagged", 100),
        ("f2", "objaverse", "flagged", 700),
        ("f3", "toys4k", "flagged", 3000),
        ("f4", "objaverse", "flagged", 9000),
        ("r1", "toys4k", "rejected", 100),
        ("r2", "objaverse", "rejected", 700),
        ("r3", "toys4k", "rejected", 3000),
        ("r4", "objaverse", "rejected", 9000),
    ]
    screening = [_row(asset_id, source, verdict) for asset_id, source, verdict, _ in entries]
    features = [
        {"asset_id": asset_id, "face_count": str(faces), "vertex_count": "600"}
        for asset_id, _, _, faces in entries
    ]
    return screening, features


def _config_with_tables(make_config, screening_rows, feature_rows):
    config_path = make_config(
        sources={},
        outputs={
            "manifest": "data/manifests/assets.csv",
            "features": "data/manifests/features.csv",
        },
    )
    config = load_project_config(config_path)
    write_csv(config.output_path("features"), feature_rows, FEATURE_COLUMNS)
    table_path, _ = screening_paths(config)
    write_csv(table_path, screening_rows, SCREEN_COLUMNS)
    return config


def test_plan_covers_every_level_and_spreads_over_face_bins(make_config) -> None:
    config = _config_with_tables(make_config, *_corpus())

    sheet, key, report = plan_screening_review(config, per_level=3)

    assert report["levels"] == {"accept": 3, "flag": 3, "reject": 3}
    assert report["available"] == {"accept": 4, "flag": 4, "reject": 4}
    labels = [row["machine_label"] for row in sheet]
    assert labels.count("accept") == 3
    assert labels.count("flag") == 3
    assert labels.count("reject") == 3

    assert {row["source_dataset"] for row in key} == {"toys4k", "objaverse"}
    sampled_bins = {
        entry["face_bin"]
        for entry in report["strata"]
        if entry["sampled"] and entry["level"] == "reject"
    }
    assert len(sampled_bins) >= 2, "a level must not collapse onto one face-count bin"

    key_by_name = {row["anonymous_name"]: row for row in key}
    for row in sheet:
        assert key_by_name[row["anonymous_name"]]["machine_label"] == row["machine_label"]
        assert key_by_name[row["anonymous_name"]]["asset_id"] == row["asset_id"]


def test_a_small_budget_is_not_eaten_by_the_first_strata(make_config) -> None:
    """Regression: the first cells alphabetically are all one source.

    A plain round robin hands the entire budget to the first N cells, and cells
    sort by source name first — so a budget smaller than the number of cells
    used to sample one source only. The remainder is now drawn in a shuffled
    order.
    """
    screening, features = [], []
    for position, count in enumerate((100, 400, 700, 1500, 3000, 6000, 9000, 18000)):
        for source in ("toys4k", "objaverse"):
            asset_id = f"{source[:2]}{position}"
            screening.append(_row(asset_id, source, "clean"))
            features.append(
                {"asset_id": asset_id, "face_count": str(count), "vertex_count": "600"}
            )
    config = _config_with_tables(make_config, screening, features)

    _, key, report = plan_screening_review(config, per_level=4)

    assert {row["source_dataset"] for row in key} == {"toys4k", "objaverse"}
    assert len([entry for entry in report["strata"] if entry["sampled"]]) == 4


def test_the_sheet_is_a_blank_form(make_config) -> None:
    """The machine's label is shown on purpose; the judgement columns are empty."""
    config = _config_with_tables(make_config, *_corpus())

    sheet, _, _ = plan_screening_review(config, per_level=2)

    names = [row["anonymous_name"] for row in sheet]
    assert names == [f"A{position:04d}" for position in range(1, len(sheet) + 1)]
    assert all(row["my_label"] == "" for row in sheet)
    assert all(row["note"] == "" for row in sheet)


def test_the_sample_is_deterministic_for_a_seed(make_config) -> None:
    config = _config_with_tables(make_config, *_corpus())

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
