from __future__ import annotations

from pathlib import Path
from typing import Any

from mesh_filter.config import ProjectConfig, load_yaml
from mesh_filter.features.basic import compute_basic_features
from mesh_filter.manifest import read_csv, write_csv
from mesh_filter.mesh_io import concatenate_for_metrics, load_mesh_parts


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


def _safe_message(error: Exception, maximum: int = 500) -> str:
    return " ".join(str(error).split())[:maximum]


def validate_manifest(
    config: ProjectConfig,
    feature_config_path: str | Path | None = None,
) -> tuple[Path, dict[str, int]]:
    manifest_path = config.output_path("manifest")
    rows = read_csv(manifest_path)
    feature_options: dict[str, Any] = {}
    if feature_config_path is not None:
        raw = load_yaml(feature_config_path)
        for key in (
            "zero_area_epsilon",
            "equilateral_threshold",
            "sliver_threshold",
            "small_angle_threshold_deg",
        ):
            if key in raw:
                feature_options[key] = raw[key]

    continue_on_error = bool(config.values.get("runtime", {}).get("continue_on_error", True))
    output_rows: list[dict[str, Any]] = []
    counts = {"total": len(rows), "parsed": 0, "failed": 0, "warnings": 0}

    for record in rows:
        base = {
            "asset_id": record["asset_id"],
            "source_dataset": record["source_dataset"],
            "label": record["label"],
            "label_origin": record["label_origin"],
        }
        mesh_path = config.source_root(record["source_dataset"]) / record["raw_relpath"]
        try:
            parts = load_mesh_parts(mesh_path)
            mesh = concatenate_for_metrics(parts)
            features = compute_basic_features(mesh.vertices, mesh.faces, **feature_options)
            has_warning = any(
                features[name] > 0
                for name in (
                    "zero_area_face_count",
                    "duplicate_face_count",
                    "nonmanifold_edge_count",
                )
            )
            validity = "valid_with_warnings" if has_warning else "valid"
            output_rows.append(
                {
                    **base,
                    "parse_status": "parsed",
                    "validity_status": validity,
                    "error_code": "",
                    "error_message": "",
                    "part_count": len(parts),
                    **features,
                }
            )
            counts["parsed"] += 1
            counts["warnings"] += int(has_warning)
        except Exception as error:  # per-asset containment is intentional
            output_rows.append(
                {
                    **base,
                    "parse_status": "failed",
                    "validity_status": "invalid",
                    "error_code": type(error).__name__,
                    "error_message": _safe_message(error),
                    "part_count": "",
                }
            )
            counts["failed"] += 1
            if not continue_on_error:
                raise

    output = config.output_path("features")
    write_csv(output, output_rows, STATUS_FIELDS)
    return output, counts
