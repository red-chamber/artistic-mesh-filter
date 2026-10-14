from __future__ import annotations

from pathlib import Path
from typing import Any

from mesh_filter.config import ProjectConfig
from mesh_filter.inspection import (
    evaluate_asset,
    has_warning,
    load_feature_options,
    safe_message,
)
from mesh_filter.manifest import read_csv, write_csv

STATUS_FIELDS = [
    "asset_id",
    "source_dataset",
    "label",
    "label_origin",
    "parse_status",
    "validity_status",
    "error_code",
    "error_message",
    "part_count",
    "vertex_count",
    "face_count",
    "edge_count",
    "component_count",
    "zero_area_face_count",
    "zero_area_face_ratio",
    "duplicate_face_count",
    "boundary_edge_count",
    "nonmanifold_edge_count",
    "equilateral_ratio",
    "sliver_ratio",
    "small_angle_ratio",
    "area_coefficient_of_variation",
]


def _reusable_rows(
    output: Path,
    resume: bool,
    current_asset_ids: set[str],
) -> dict[str, dict[str, str]]:
    """Rows from a previous run that can be kept instead of recomputed.

    Resuming assumes the read-only sources have not changed. A stale row is
    worse than a slow rerun, so this is opt-in through `runtime.resume`.
    """
    if not resume or not output.exists():
        return {}
    reusable: dict[str, dict[str, str]] = {}
    for row in read_csv(output):
        asset_id = row.get("asset_id", "")
        if asset_id not in current_asset_ids or row.get("parse_status") != "parsed":
            continue
        if all(field in row for field in STATUS_FIELDS):
            reusable[asset_id] = row
    return reusable


def validate_manifest(
    config: ProjectConfig,
    feature_config_path: str | Path | None = None,
) -> tuple[Path, dict[str, int]]:
    manifest_path = config.output_path("manifest")
    rows = read_csv(manifest_path)
    options = load_feature_options(feature_config_path)
    output = config.output_path("features")

    runtime = config.values.get("runtime", {})
    continue_on_error = bool(runtime.get("continue_on_error", True))
    reusable = _reusable_rows(
        output,
        bool(runtime.get("resume", False)),
        {row["asset_id"] for row in rows},
    )

    output_rows: list[dict[str, Any]] = []
    counts = {"total": len(rows), "parsed": 0, "failed": 0, "warnings": 0, "resumed": 0}

    for record in rows:
        base = {
            "asset_id": record["asset_id"],
            "source_dataset": record["source_dataset"],
            "label": record["label"],
            "label_origin": record["label_origin"],
        }

        previous = reusable.get(record["asset_id"])
        if previous is not None:
            output_rows.append(previous)
            counts["parsed"] += 1
            counts["resumed"] += 1
            counts["warnings"] += int(previous.get("validity_status") == "valid_with_warnings")
            continue

        mesh_path = config.source_root(record["source_dataset"]) / record["raw_relpath"]
        try:
            features = evaluate_asset(mesh_path, options)
            warned = has_warning(features)
            validity = "valid_with_warnings" if warned else "valid"
            output_rows.append(
                {
                    **base,
                    "parse_status": "parsed",
                    "validity_status": validity,
                    "error_code": "",
                    "error_message": "",
                    **features,
                }
            )
            counts["parsed"] += 1
            counts["warnings"] += int(warned)
        except Exception as error:  # per-asset containment is intentional
            output_rows.append(
                {
                    **base,
                    "parse_status": "failed",
                    "validity_status": "invalid",
                    "error_code": type(error).__name__,
                    "error_message": safe_message(error),
                    "part_count": "",
                }
            )
            counts["failed"] += 1
            if not continue_on_error:
                raise

    write_csv(output, output_rows, STATUS_FIELDS)
    return output, counts
