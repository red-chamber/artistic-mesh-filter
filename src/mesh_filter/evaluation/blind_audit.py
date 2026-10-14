"""Blind audit sample for human review.

The only evaluation that is independent of the proxy label is a human judging
topology quality without knowing where a model came from. This module draws a
stratified sample and writes two files:

- the blind sheet, which is what a reviewer opens: an anonymous name, the asset
  id needed to find the rendered views, and empty label columns;
- the key, which maps anonymous names back to source and proxy label.

The key must not be opened by whoever is reviewing. Splitting the two files is
the whole mechanism: a single file containing the source column would make the
"blind" part decorative.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

from mesh_filter.config import ConfigError, ProjectConfig
from mesh_filter.evaluation.candidates import (
    bin_index,
    bin_label,
    face_count_edges,
    load_evaluation_rows,
)
from mesh_filter.manifest import write_csv

BLOCK = "audit_gold"

# The reviewer's vocabulary, as defined in docs/LABELING_GUIDE.md.
REVIEW_LABELS = ("accept", "reject", "uncertain", "invalid")

BLIND_FIELDS = [
    "anonymous_name",
    "asset_id",
    "human_label",
    "human_confidence",
    "human_notes",
    "reviewer",
    "reviewed_at",
]

KEY_FIELDS = [
    "anonymous_name",
    "asset_id",
    "source_dataset",
    "label",
    "label_origin",
    "split",
    "face_count",
    "vertex_count",
    "component_count",
    "face_count_bin",
    "stratum",
]


def _allocate(sizes: dict[str, int], size: int, min_per_stratum: int) -> dict[str, int]:
    """Split a sample budget across strata.

    Three passes, in this order, because the order is the policy:

    1. coverage: every stratum gets one sample for as long as the budget lasts,
       largest strata first, so the ones we cannot reach are the rarest;
    2. floor: strata below `min_per_stratum` are topped up, because a stratum
       sampled zero or one time has no usable error estimate;
    3. proportional fill of whatever is left, so the sample still reflects the
       composition of the data.

    A budget smaller than the number of strata cannot cover all of them; that is
    reported through the returned quotas rather than hidden.
    """
    keys = sorted(sizes)
    total = sum(sizes.values())
    if total == 0:
        return {}
    if size >= total:
        return {key: sizes[key] for key in keys}
    if size <= 0:
        return dict.fromkeys(keys, 0)

    quota = dict.fromkeys(keys, 0)
    by_size = sorted(keys, key=lambda key: (-sizes[key], key))
    for key in by_size[:size]:
        quota[key] = 1
    remaining = size - sum(quota.values())

    for key in by_size:
        if remaining <= 0:
            break
        want = min(min_per_stratum, sizes[key]) - quota[key]
        if want <= 0:
            continue
        addition = min(want, remaining, sizes[key] - quota[key])
        quota[key] += addition
        remaining -= addition

    while remaining > 0:
        order = sorted(
            keys,
            key=lambda key: (size * sizes[key] / total - quota[key], key),
            reverse=True,
        )
        progressed = False
        for key in order:
            if remaining <= 0:
                break
            if quota[key] < sizes[key]:
                quota[key] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return quota


def _stratify(
    rows: Sequence[dict[str, Any]],
    edges: Sequence[int],
    grouping: str,
) -> dict[str, list[dict[str, Any]]]:
    strata: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        bin_text = bin_label(edges, bin_index(row["face_count"], edges))
        row["face_count_bin"] = bin_text
        key = bin_text if grouping == "face_count_bin" else f"label{row['label']}:{bin_text}"
        strata.setdefault(key, []).append(row)
    return strata


def build_audit_gold(
    config: ProjectConfig,
    size: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    settings = config.values.get(BLOCK, {})
    requested = int(size if size is not None else settings.get("size", 200))
    seed = int(settings.get("seed", 7))
    min_per_stratum = int(settings.get("min_per_stratum", 5))
    grouping = str(settings.get("stratify_by", "label_and_face_count_bin"))
    if grouping not in {"label_and_face_count_bin", "face_count_bin"}:
        raise ConfigError(
            f"{BLOCK}.stratify_by must be 'label_and_face_count_bin' or 'face_count_bin'"
        )

    edges = face_count_edges(config, BLOCK)
    rows, skipped = load_evaluation_rows(config)
    strata = _stratify(rows, edges, grouping)
    sizes = {key: len(items) for key, items in strata.items()}
    quota = _allocate(sizes, requested, min_per_stratum)

    picked: list[tuple[str, dict[str, Any]]] = []
    strata_report: list[dict[str, Any]] = []
    for key in sorted(strata):
        items = sorted(strata[key], key=lambda row: row["asset_id"])
        take = min(quota.get(key, 0), len(items))
        chosen = random.Random(f"{seed}:audit:{key}").sample(items, take)
        strata_report.append({"stratum": key, "available": len(items), "sampled": take})
        picked.extend((key, row) for row in chosen)

    if not picked:
        raise ConfigError("Nothing to sample; the parsed population is empty")

    random.Random(f"{seed}:anonymise").shuffle(picked)
    blind_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for position, (key, row) in enumerate(picked, start=1):
        anonymous_name = f"A{position:04d}"
        blind_rows.append({"anonymous_name": anonymous_name, "asset_id": row["asset_id"]})
        key_rows.append({**row, "anonymous_name": anonymous_name, "stratum": key})

    blind_output = config.output_path("audit_gold")
    key_output = config.output_path("audit_gold_key")
    write_csv(blind_output, blind_rows, BLIND_FIELDS)
    write_csv(key_output, key_rows, KEY_FIELDS)

    return blind_output, {
        "output": str(blind_output),
        "key_output": str(key_output),
        "requested": requested,
        "sampled": len(blind_rows),
        "stratify_by": grouping,
        "seed": seed,
        "min_per_stratum": min_per_stratum,
        "skipped": skipped,
        "population": len(rows),
        "strata": strata_report,
        "review_labels": list(REVIEW_LABELS),
    }
