from pathlib import Path

import pytest

from mesh_filter.config import load_project_config
from mesh_filter.manifest import read_csv, write_csv
from mesh_filter.split import _assign_groups, create_splits

ROW_FIELDS = ["asset_id", "source_dataset", "label", "duplicate_group_id"]


def _rows(count_per_label: int) -> list[dict[str, str]]:
    rows = []
    for label in (0, 1):
        for index in range(count_per_label):
            asset_id = f"l{label}n{index}"
            rows.append(
                {
                    "asset_id": asset_id,
                    "source_dataset": "toys4k" if label else "objaverse",
                    "label": str(label),
                    "duplicate_group_id": asset_id,
                }
            )
    return rows


def _config(make_config, train: float, val: float, test: float) -> Path:
    return make_config(
        sources={},
        outputs={"deduplicated_manifest": "in.csv", "split_root": "out/splits"},
        extra={
            "split": {
                "seed": 42,
                "train_ratio": train,
                "val_ratio": val,
                "test_ratio": test,
            }
        },
    )


def test_coverage_keeps_every_split_non_empty(make_config, tmp_path: Path) -> None:
    """A four-asset smoke sample must still exercise all three splits."""
    rows = _rows(2)
    write_csv(tmp_path / "in.csv", rows, ROW_FIELDS)
    config_path = _config(make_config, 0.5, 0.25, 0.25)

    _, counts = create_splits(load_project_config(config_path), None)

    assert counts["train"] > 0
    assert counts["val"] > 0
    assert counts["test"] > 0
    assert sum(counts.values()) == len(rows)


def test_assignment_is_deterministic(make_config, tmp_path: Path) -> None:
    write_csv(tmp_path / "in.csv", _rows(8), ROW_FIELDS)
    config = load_project_config(_config(make_config, 0.7, 0.15, 0.15))

    first, _ = create_splits(config, None)
    first_rows = (first / "all.csv").read_text(encoding="utf-8")
    second, _ = create_splits(config, None)

    assert (second / "all.csv").read_text(encoding="utf-8") == first_rows


def test_group_with_conflicting_labels_is_never_assigned_a_split(
    make_config, tmp_path: Path
) -> None:
    """The guard inside the assignment step must stay, even though create_splits
    now filters conflicting groups out before reaching it."""
    rows = _rows(8)
    rows[0]["duplicate_group_id"] = "shared-group"
    rows[-1]["duplicate_group_id"] = "shared-group"
    write_csv(tmp_path / "in.csv", rows, ROW_FIELDS)

    with pytest.raises(ValueError):
        _assign_groups(
            {
                "shared-group": [rows[0], rows[-1]],
                "single": [rows[1]],
            },
            {"train": 1.0, "val": 0.0, "test": 0.0},
            42,
        )


def test_ratios_must_sum_to_one(make_config, tmp_path: Path) -> None:
    write_csv(tmp_path / "in.csv", _rows(4), ROW_FIELDS)

    with pytest.raises(ValueError):
        create_splits(load_project_config(_config(make_config, 0.9, 0.2, 0.2)), None)


def test_label_conflicting_group_is_excluded_and_recorded(make_config, tmp_path: Path) -> None:
    """One geometry carrying both labels is unusable, and must not be guessed at."""
    rows = _rows(6)
    rows[0]["duplicate_group_id"] = "cross-source"  # label 0
    rows[-1]["duplicate_group_id"] = "cross-source"  # label 1
    write_csv(tmp_path / "in.csv", rows, ROW_FIELDS)

    output, counts = create_splits(
        load_project_config(_config(make_config, 0.7, 0.15, 0.15)), None
    )

    assert counts["conflicting_groups"] == 1
    assert counts["assets_excluded_as_conflicting"] == 2
    assert sum(counts[name] for name in ("train", "val", "test")) == len(rows) - 2

    split_ids = {row["asset_id"] for row in read_csv(output / "all.csv")}
    assert not split_ids & {"l0n0", "l1n5"}

    conflicts = read_csv(output / "conflicts.csv")
    assert {row["asset_id"] for row in conflicts} == {"l0n0", "l1n5"}
    assert {row["label"] for row in conflicts} == {"0", "1"}
