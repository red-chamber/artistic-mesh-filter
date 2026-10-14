from pathlib import Path

import pytest

from mesh_filter.config import ConfigError, load_project_config
from mesh_filter.evaluation.matched_test import build_matched_test
from mesh_filter.manifest import read_csv

OUTPUTS = {
    "features": "data/features.csv",
    "split_root": "data/splits/v1",
    "matched_test": "out/matched_test.csv",
}


def _config(make_config, outputs: dict | None = None, extra: dict | None = None):
    return load_project_config(
        make_config(
            sources={},
            outputs=outputs or OUTPUTS,
            extra={"matched_test": {"pool": ["test"], **(extra or {})}},
        )
    )


def _balanced_specs(bin_faces: int = 100, count: int = 3, split: str = "test") -> list[tuple]:
    specs = []
    for index in range(count):
        specs.append((f"p{index}", 1, split, bin_faces + index, 50 + index))
        specs.append((f"n{index}", 0, split, bin_faces + index, 52 + index))
    return specs


def test_pairs_one_to_one_within_a_bin(make_config, write_eval_tables, tmp_path: Path) -> None:
    write_eval_tables(_balanced_specs())

    output, summary = build_matched_test(_config(make_config))

    rows = read_csv(output)
    assert summary["pairs"] == 3
    assert summary["assets"] == 6
    assert sum(1 for row in rows if row["label"] == "1") == 3
    assert sum(1 for row in rows if row["label"] == "0") == 3
    assert len({row["face_count_bin"] for row in rows}) == 1


def test_pairing_keeps_vertex_counts_close(make_config, write_eval_tables) -> None:
    """Sorting both labels by vertex count before pairing is the matching."""
    write_eval_tables(
        [
            ("p_low", 1, "test", 200, 10),
            ("p_high", 1, "test", 200, 1000),
            ("n_low", 0, "test", 200, 12),
            ("n_high", 0, "test", 200, 990),
        ]
    )

    _, summary = build_matched_test(_config(make_config))

    assert summary["bins"][0]["vertex_count_gap_max"] == 10


def test_single_label_bin_is_dropped_and_reported(make_config, write_eval_tables) -> None:
    """A bin only positives can reach must be recorded, never silently balanced."""
    write_eval_tables(
        _balanced_specs(bin_faces=100, count=2)
        + [
            ("p_dense_0", 1, "test", 9000, 5000),
            ("p_dense_1", 1, "test", 9500, 5100),
        ]
    )

    output, summary = build_matched_test(_config(make_config))

    dropped = {entry["bin"]: entry for entry in summary["dropped_bins"]}
    assert "[5000, 10000)" in dropped
    assert dropped["[5000, 10000)"]["reason"] == "single_label_bin"
    assert dropped["[5000, 10000)"]["label_0"] == 0
    assert "p_dense_0" not in {row["asset_id"] for row in read_csv(output)}


def test_pool_excludes_training_rows(make_config, write_eval_tables) -> None:
    write_eval_tables(_balanced_specs(split="train") + _balanced_specs(bin_faces=300, split="test"))

    output, summary = build_matched_test(_config(make_config))

    assert summary["pool_size"] == 6
    assert all(row["split"] == "test" for row in read_csv(output))


def test_disjoint_face_counts_cannot_be_matched(make_config, write_eval_tables) -> None:
    """If the sources do not overlap in polygon count, say so instead of guessing."""
    write_eval_tables(
        [("p0", 1, "test", 100, 50), ("p1", 1, "test", 120, 60)]
        + [("n0", 0, "test", 9000, 5000), ("n1", 0, "test", 9500, 5100)]
    )

    with pytest.raises(ConfigError, match="both labels"):
        build_matched_test(_config(make_config))


def test_output_is_deterministic(make_config, write_eval_tables, tmp_path: Path) -> None:
    write_eval_tables(_balanced_specs(count=6))
    config = _config(make_config)
    first, _ = build_matched_test(config)
    first_text = first.read_text(encoding="utf-8")

    second, _ = build_matched_test(config)

    assert second.read_text(encoding="utf-8") == first_text


def test_max_per_bin_caps_the_pair_count(make_config, write_eval_tables) -> None:
    write_eval_tables(_balanced_specs(count=10))

    _, summary = build_matched_test(_config(make_config), max_per_bin_per_label=4)

    assert summary["pairs"] == 4
