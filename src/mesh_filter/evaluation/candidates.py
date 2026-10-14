"""Shared plumbing for building evaluation sets.

Both the face-count-matched test set and the blind audit sample start from the
same thing: the frozen split table joined with the parsed feature table, binned
by face count. Keeping that join in one place means the two sets can never
disagree about which assets exist or what their face count is.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from mesh_filter.config import ConfigError, ProjectConfig
from mesh_filter.manifest import read_csv

DEFAULT_FACE_COUNT_EDGES: tuple[int, ...] = (
    0,
    250,
    500,
    1000,
    2000,
    5000,
    10000,
    20000,
    50000,
)

COUNT_FIELDS = ("face_count", "vertex_count", "component_count")


def _as_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def face_count_edges(config: ProjectConfig, block: str) -> tuple[int, ...]:
    configured = config.values.get(block, {}).get("face_count_edges")
    edges = tuple(int(edge) for edge in configured) if configured else DEFAULT_FACE_COUNT_EDGES
    if len(edges) < 2 or list(edges) != sorted(edges) or len(set(edges)) != len(edges):
        raise ConfigError(f"{block}.face_count_edges must be at least two increasing values")
    return edges


def bin_index(face_count: int, edges: Sequence[int]) -> int:
    index = 0
    for position, edge in enumerate(edges):
        if face_count >= edge:
            index = position
        else:
            break
    return index


def bin_label(edges: Sequence[int], index: int) -> str:
    lower = edges[index]
    if index + 1 < len(edges):
        return f"[{lower}, {edges[index + 1]})"
    return f"[{lower}, +inf)"


def load_evaluation_rows(
    config: ProjectConfig,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Join the frozen split table with the parsed feature table.

    Only assets with `parse_status = parsed` carry a usable face count, and an
    asset without a face count cannot be matched or stratified. Those are dropped
    here, once, and their counts are returned so callers can report the loss
    instead of silently discarding samples.
    """
    split_root = config.output_path("split_root")
    split_path = split_root / "all.csv"
    if not split_path.exists():
        raise ConfigError(f"Split table not found: {split_path}. Run `split` first.")

    features = {row["asset_id"]: row for row in read_csv(config.output_path("features"))}
    rows: list[dict[str, Any]] = []
    skipped = {"unparsed": 0, "incomplete": 0}
    for split_row in read_csv(split_path):
        feature = features.get(split_row.get("asset_id", ""))
        if feature is None or feature.get("parse_status") != "parsed":
            skipped["unparsed"] += 1
            continue
        counts = {field: _as_int(feature.get(field)) for field in COUNT_FIELDS}
        if counts["face_count"] is None or counts["vertex_count"] is None:
            skipped["incomplete"] += 1
            continue
        rows.append(
            {
                "asset_id": split_row["asset_id"],
                "source_dataset": split_row.get("source_dataset", ""),
                "label": int(split_row["label"]),
                "label_origin": split_row.get("label_origin", ""),
                "split": split_row.get("split", ""),
                **counts,
            }
        )
    if not rows:
        raise ConfigError(
            "No parsed asset with a face count was found. "
            f"unparsed={skipped['unparsed']} incomplete={skipped['incomplete']}"
        )
    return rows, skipped


def population_summary(rows: Iterable[dict[str, Any]], edges: Sequence[int]) -> dict[str, Any]:
    """Describe a row population, per label and per face-count bin."""
    materialized = list(rows)
    summary: dict[str, Any] = {
        "size": len(materialized),
        "by_label": {},
        "by_bin": {},
        "face_count": {},
        "vertex_count": {},
        "component_count": {},
    }
    for label in sorted({row["label"] for row in materialized}):
        summary["by_label"][str(label)] = sum(1 for row in materialized if row["label"] == label)
    for index in range(len(edges)):
        label_text = bin_label(edges, index)
        bucket = [row for row in materialized if bin_index(row["face_count"], edges) == index]
        if bucket:
            summary["by_bin"][label_text] = {
                "total": len(bucket),
                "label_0": sum(1 for row in bucket if row["label"] == 0),
                "label_1": sum(1 for row in bucket if row["label"] == 1),
            }
    for field in COUNT_FIELDS:
        values = [row[field] for row in materialized if row[field] is not None]
        if values:
            ordered = sorted(values)
            summary[field] = {
                "min": ordered[0],
                "median": ordered[len(ordered) // 2],
                "max": ordered[-1],
            }
    return summary
