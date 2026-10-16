"""Targeted calibration kits: each round's sample is aimed at a question.

`build_screening_review` re-tests the whole system — every level, every face
bin, both sources. That is the right design when the question is "how far off
is the machine overall", and the wrong one when the question is "is this
specific corner real": a uniform draw spends most of its budget re-confirming
what the last round already settled.

This kit carries the rounds in between. Each arm names one question and draws
only from the assets that can answer it — a corner of the corpus the rules are
silent on, the tails of a column that is computed but unused, or the two sides
of a threshold that is about to move. An asset is drawn at most once (the
first arm that wants it wins), the picks inside an arm are random for a seed,
and an arm may rank its pool and draw from the top when the question is about
the extreme tail itself.

The output has the same shape as the stratified kit — a correction sheet, a
blind key one directory up, anonymised meshes — plus the arm name in the key,
so the round can be scored arm by arm. The arm definitions live in the round's
script, not here: they encode what the last round taught, and that changes
every time.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from mesh_filter.config import ProjectConfig
from mesh_filter.evaluation.review_kit import prepare_review
from mesh_filter.evaluation.screening_kit import LEVELS, VERDICT_TO_LABEL
from mesh_filter.manifest import read_csv, write_csv
from mesh_filter.screening import screening_paths

SHEET_FIELDS = ["anonymous_name", "asset_id", "machine_label", "my_label", "note"]

# The feature values that could explain each pick, so scoring the round needs
# no joins against the feature table.
KEY_FEATURE_FIELDS = [
    "face_count",
    "redundant_face_share",
    "vertex_degree_mode_share",
    "equilateral_median",
    "vertex_degree_entropy",
    "one_ring_edge_length_cv_median",
    "local_edge_direction_anisotropy_mean",
    "neighboring_direction_consistency_mean",
]

KEY_FIELDS = [
    "anonymous_name",
    "asset_id",
    "source_dataset",
    "label",
    "machine_label",
    "arm",
    "failed_rules",
    "unknown_rules",
    "flags_passed",
    *KEY_FEATURE_FIELDS,
]

Row = dict[str, str]
# (screening row, feature row) -> wants this asset?
Predicate = Callable[[Row, Row], bool]
# (screening row, feature row) -> sort key; lower is drawn first.
Rank = Callable[[Row, Row], float]

# How many top-ranked candidates an arm keeps before sampling: a ranked arm
# takes the best-scoring pool of `3 * count`, then draws randomly from it, so
# the arm favours the tail without being a hand-picked list.
RANK_POOL_MULTIPLIER = 3


@dataclass(frozen=True)
class TargetArm:
    """One question, one pool, one quota."""

    name: str
    count: int
    predicate: Predicate
    rank: Rank | None = None


def _draw(
    eligible: list[tuple[Row, Row]], arm: TargetArm, rng: random.Random
) -> list[tuple[Row, Row]]:
    ordered = sorted(eligible, key=lambda pair: pair[0]["asset_id"])
    if arm.rank is not None:
        def rank_key(pair: tuple[Row, Row]) -> tuple[float, str]:
            value = arm.rank(pair[0], pair[1])
            if math.isnan(value):
                value = math.inf
            return (value, pair[0]["asset_id"])

        head = sorted(ordered, key=rank_key)[: max(RANK_POOL_MULTIPLIER * arm.count, arm.count)]
        return rng.sample(head, min(arm.count, len(head)))
    return rng.sample(ordered, min(arm.count, len(ordered)))


def plan_targeted_review(
    config: ProjectConfig,
    arms: list[TargetArm],
    exclude_ids: Iterable[str] = (),
    seed: int = 13,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Pick the sample. Returns (sheet rows, key rows, report). No files touched."""
    table_path, _ = screening_paths(config)
    if not table_path.exists():
        raise FileNotFoundError(
            f"Screening table not found: {table_path}. Run `screen` first."
        )
    rows = read_csv(table_path)
    features = {row["asset_id"]: row for row in read_csv(config.output_path("features"))}
    excluded = {value for value in exclude_ids if value}

    picked: list[tuple[str, Row]] = []
    taken: set[str] = set()
    arm_report: list[dict[str, Any]] = []
    for arm in arms:
        skipped = excluded | taken
        eligible = [
            (row, features.get(row["asset_id"], {}))
            for row in rows
            if row.get("asset_id") not in skipped
            and row.get("verdict") in VERDICT_TO_LABEL
            and arm.predicate(row, features.get(row["asset_id"], {}))
        ]
        chosen = _draw(eligible, arm, random.Random(f"{seed}:targeted:{arm.name}"))
        taken.update(row["asset_id"] for row, _ in chosen)
        picked.extend((arm.name, row) for row, _ in chosen)
        arm_report.append(
            {
                "arm": arm.name,
                "count": arm.count,
                "available": len(eligible),
                "drawn": len(chosen),
            }
        )

    random.Random(f"{seed}:targeted:anonymise").shuffle(picked)

    sheet_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for position, (arm_name, row) in enumerate(picked, start=1):
        anonymous_name = f"A{position:04d}"
        label = VERDICT_TO_LABEL[row["verdict"]]
        sheet_rows.append(
            {
                "anonymous_name": anonymous_name,
                "asset_id": row["asset_id"],
                "machine_label": label,
                "my_label": "",
                "note": "",
            }
        )
        feature = features.get(row["asset_id"], {})
        key_rows.append(
            {
                "anonymous_name": anonymous_name,
                "asset_id": row["asset_id"],
                "source_dataset": row.get("source_dataset", ""),
                "label": row.get("label", ""),
                "machine_label": label,
                "arm": arm_name,
                "failed_rules": row.get("failed_rules", ""),
                "unknown_rules": row.get("unknown_rules", ""),
                "flags_passed": row.get("flags_passed", ""),
                **{name: feature.get(name, "") for name in KEY_FEATURE_FIELDS},
            }
        )

    report: dict[str, Any] = {
        "seed": seed,
        "sampled": len(picked),
        "excluded_assets": len(excluded),
        "levels": {
            level: sum(1 for _, row in picked if VERDICT_TO_LABEL[row["verdict"]] == level)
            for level in LEVELS
        },
        "arms": arm_report,
    }
    return sheet_rows, key_rows, report


def build_targeted_review(
    config: ProjectConfig,
    arms: list[TargetArm],
    exclude_ids: Iterable[str] = (),
    notes: str = "",
    seed: int = 13,
) -> tuple[Path, dict[str, Any]]:
    """Write the sheet, the key, the notes and the anonymised meshes."""
    sheet_rows, key_rows, report = plan_targeted_review(config, arms, exclude_ids, seed)
    if not sheet_rows:
        raise FileNotFoundError("Nothing to review; every arm came back empty.")

    root = screening_paths(config)[0].parent
    review_dir = root / "review"
    sheet_path = review_dir / "screening_review.csv"
    key_path = root / "screening_review_key.csv"
    review_dir.mkdir(parents=True, exist_ok=True)
    write_csv(sheet_path, sheet_rows, SHEET_FIELDS)
    write_csv(key_path, key_rows, KEY_FIELDS)
    if notes:
        (review_dir / "SCREENING_REVIEW_NOTES.md").write_text(notes, encoding="utf-8")

    _, kit = prepare_review(config, None, sheet_path=sheet_path, output_dir=review_dir)
    report.update(
        {
            "output": str(sheet_path),
            "key_output": str(key_path),
            "review_dir": str(review_dir),
            "kit": kit,
        }
    )
    return sheet_path, report
