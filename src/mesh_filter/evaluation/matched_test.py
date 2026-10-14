"""Face-count-matched test set.

A single overall test score cannot distinguish "the model learned artistic
topology" from "the model learned that this class has fewer faces", because the
negative set was screened to fewer than 20000 faces. This module builds a test
set in which both labels are drawn from the same face-count bins, so any
remaining separation has to come from something other than polygon count.

The set is a subset of the frozen pool (by default the `test` split), not a new
split: it must never contribute training samples.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Sequence

from mesh_filter.config import ConfigError, ProjectConfig
from mesh_filter.evaluation.candidates import (
    bin_index,
    bin_label,
    face_count_edges,
    load_evaluation_rows,
    population_summary,
)
from mesh_filter.manifest import write_csv
from mesh_filter.sampling import deterministic_sample

BLOCK = "matched_test"

MATCHED_TEST_FIELDS = [
    "asset_id",
    "source_dataset",
    "label",
    "label_origin",
    "split",
    "face_count",
    "vertex_count",
    "component_count",
    "face_count_bin",
    "match_pair_id",
]


def _sorted_by_vertex_count(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (row["vertex_count"], row["asset_id"]))


def build_matched_test(
    config: ProjectConfig,
    pool: Sequence[str] | None = None,
    max_per_bin_per_label: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Pair positive and negative assets inside the same face-count bin.

    Pairing is greedy and deterministic: inside a bin both labels are sorted by
    vertex count, evenly subsampled to the smaller side, then zipped in order.
    That also makes vertex counts comparable within a pair, which is reported as
    a gap so the matching can be checked rather than assumed.
    """
    settings = config.values.get(BLOCK, {})
    edges = face_count_edges(config, BLOCK)
    pool_names = tuple(pool or settings.get("pool", ["test"]))
    cap = int(
        max_per_bin_per_label
        if max_per_bin_per_label is not None
        else settings.get("max_per_bin_per_label", 200)
    )
    min_pairs = int(settings.get("min_pairs_per_bin", 1))

    rows, skipped = load_evaluation_rows(config)
    pool_rows = [row for row in rows if row["split"] in pool_names]
    if not pool_rows:
        raise ConfigError(
            f"No parsed asset in the requested pool {list(pool_names)}; "
            f"available splits: {sorted({row['split'] for row in rows})}"
        )

    buckets: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: {0: [], 1: []})
    for row in pool_rows:
        buckets[bin_index(row["face_count"], edges)][row["label"]].append(row)

    selected: list[dict[str, Any]] = []
    bins_report: list[dict[str, Any]] = []
    for index in sorted(buckets):
        # Variable names follow the label they hold: label 0 is the negative
        # source, label 1 the positive one. Mixing these up would mislabel the
        # report while still producing a valid pairing.
        label_zero = _sorted_by_vertex_count(buckets[index][0])
        label_one = _sorted_by_vertex_count(buckets[index][1])
        label_text = bin_label(edges, index)
        entry: dict[str, Any] = {
            "bin": label_text,
            "label_0": len(label_zero),
            "label_1": len(label_one),
            "pairs": 0,
            "dropped": False,
        }

        if not label_zero or not label_one:
            entry["dropped"] = True
            entry["reason"] = "single_label_bin"
            bins_report.append(entry)
            continue

        take = min(len(label_zero), len(label_one), cap)
        if take < min_pairs:
            entry["dropped"] = True
            entry["reason"] = f"fewer_than_min_pairs_per_bin({min_pairs})"
            bins_report.append(entry)
            continue

        chosen_zero = deterministic_sample(label_zero, take)
        chosen_one = deterministic_sample(label_one, take)
        gaps: list[int] = []
        for pair_number, (negative, positive) in enumerate(
            zip(chosen_zero, chosen_one, strict=True)
        ):
            pair_id = f"{index:02d}-{pair_number:04d}"
            gaps.append(abs(positive["vertex_count"] - negative["vertex_count"]))
            for row in (positive, negative):
                selected.append(
                    {
                        **row,
                        "face_count_bin": label_text,
                        "match_pair_id": pair_id,
                    }
                )

        entry["pairs"] = take
        entry["vertex_count_gap_median"] = median(gaps)
        entry["vertex_count_gap_max"] = max(gaps)
        bins_report.append(entry)

    if not selected:
        raise ConfigError(
            "No face-count bin contains both labels. The two sources do not overlap "
            "in polygon count, so a matched test set cannot be built from this pool. "
            f"bins={bins_report}"
        )

    output = config.output_path("matched_test")
    write_csv(output, selected, MATCHED_TEST_FIELDS)

    summary: dict[str, Any] = {
        "output": str(output),
        "pool": list(pool_names),
        "pool_size": len(pool_rows),
        "skipped": skipped,
        "pairs": len(selected) // 2,
        "assets": len(selected),
        "max_per_bin_per_label": cap,
        "bins": bins_report,
        "dropped_bins": [entry for entry in bins_report if entry["dropped"]],
        "population": {
            "pool": population_summary(pool_rows, edges),
            "selected": population_summary(selected, edges),
        },
    }
    return output, summary
