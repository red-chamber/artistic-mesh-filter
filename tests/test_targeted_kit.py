"""The targeted kit: arms, quotas, exclusions, and the ranked tail."""

import pytest

from mesh_filter.config import load_project_config
from mesh_filter.evaluation.targeted_kit import TargetArm, plan_targeted_review
from mesh_filter.manifest import write_csv
from mesh_filter.screening import screening_paths

FEATURE_COLUMNS = [
    "asset_id",
    "face_count",
    "vertex_count",
    "redundant_face_share",
    "vertex_degree_mode_share",
    "equilateral_median",
]
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


def _screening_row(asset_id: str, verdict: str = "clean") -> dict[str, str]:
    return {
        "asset_id": asset_id,
        "source_dataset": "toys4k",
        "label": "1",
        "verdict": verdict,
        "flags_passed": "5",
        "flags_failed": "0",
        "flags_unknown": "0",
        "failed_rules": "",
        "unknown_rules": "",
    }


def _feature_row(asset_id: str, **overrides: object) -> dict[str, str]:
    row = {
        "asset_id": asset_id,
        "face_count": "1200",
        "vertex_count": "800",
        "redundant_face_share": "0.01",
        "vertex_degree_mode_share": "0.5",
        "equilateral_median": "0.7",
    }
    row.update({key: str(value) for key, value in overrides.items()})
    return row


def _corpus() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """20 clean assets with face counts 100..2000 and everything else uniform."""
    screening = [_screening_row(f"a{index:02d}") for index in range(20)]
    features = [
        _feature_row(f"a{index:02d}", face_count=100 * (index + 1))
        for index in range(20)
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


def test_arms_fill_quotas_exclude_seen_ids_and_never_repeat(make_config) -> None:
    config = _config_with_tables(make_config, *_corpus())
    arms = [
        TargetArm("low", 8, lambda s, f: float(f["face_count"]) <= 1000),
        TargetArm("high", 4, lambda s, f: float(f["face_count"]) > 1000),
    ]

    sheet, key, report = plan_targeted_review(config, arms, exclude_ids={"a19"})

    # a00..a09 are eligible for `low` and a10..a18 for `high` (a19 excluded).
    assert report["sampled"] == 12
    assert [(entry["drawn"], entry["available"]) for entry in report["arms"]] == [
        (8, 10),
        (4, 9),
    ]
    ids = [row["asset_id"] for row in sheet]
    assert len(ids) == len(set(ids))
    assert "a19" not in ids, "an already-judged asset must never be drawn again"
    assert {row["asset_id"] for row in key if row["arm"] == "high"} <= {
        f"a{index:02d}" for index in range(10, 19)
    }


def test_a_ranked_arm_draws_from_the_top_of_its_pool(make_config) -> None:
    """`rank` favours the tail without hand-picking: top 3*count, then sample."""
    config = _config_with_tables(make_config, *_corpus())
    arm = TargetArm(
        "top", 3, lambda s, f: True, rank=lambda s, f: -float(f["face_count"])
    )

    _, key, report = plan_targeted_review(config, [arm], seed=5)

    # The head is the nine highest face counts: a11..a19.
    assert report["arms"][0]["drawn"] == 3
    assert {row["asset_id"] for row in key} <= {f"a{index:02d}" for index in range(11, 20)}


def test_an_underfilled_arm_is_reported_not_backfilled(make_config) -> None:
    config = _config_with_tables(make_config, *_corpus())
    arm = TargetArm("only_a00", 5, lambda s, f: f["asset_id"] == "a00")

    _, key, report = plan_targeted_review(config, [arm])

    assert report["sampled"] == 1
    assert report["arms"][0]["available"] == 1
    assert report["arms"][0]["drawn"] == 1


def test_sheet_is_blank_and_the_key_records_the_arm(make_config) -> None:
    config = _config_with_tables(make_config, *_corpus())
    arms = [TargetArm("q", 3, lambda s, f: True)]

    sheet, key, _ = plan_targeted_review(config, arms)

    names = [row["anonymous_name"] for row in sheet]
    assert names == [f"A{position:04d}" for position in range(1, len(sheet) + 1)]
    assert all(row["my_label"] == "" and row["note"] == "" for row in sheet)
    assert {row["arm"] for row in key} == {"q"}
    key_by_name = {row["anonymous_name"]: row for row in key}
    for row in sheet:
        assert key_by_name[row["anonymous_name"]]["asset_id"] == row["asset_id"]
        assert key_by_name[row["anonymous_name"]]["machine_label"] == row["machine_label"]


def test_the_sample_is_deterministic_for_a_seed(make_config) -> None:
    config = _config_with_tables(make_config, *_corpus())
    arms = [TargetArm("q", 6, lambda s, f: True)]

    first, _, _ = plan_targeted_review(config, arms, seed=9)
    second, _, _ = plan_targeted_review(config, arms, seed=9)

    assert [(row["anonymous_name"], row["asset_id"]) for row in first] == [
        (row["anonymous_name"], row["asset_id"]) for row in second
    ]


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
        plan_targeted_review(config, [TargetArm("q", 1, lambda s, f: True)])
