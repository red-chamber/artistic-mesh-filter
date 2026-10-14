"""Per-asset work shared by `validate` and the smoke test.

Both commands must agree on what "a parsed asset" means, on which feature
options are honoured, and on what counts as a warning. Keeping that in one place
stops the smoke verdict from drifting away from the real feature run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mesh_filter.config import load_yaml
from mesh_filter.features.basic import compute_basic_features
from mesh_filter.mesh_io import concatenate_for_metrics, load_mesh_parts

FEATURE_OPTION_KEYS = (
    "zero_area_epsilon",
    "equilateral_threshold",
    "sliver_threshold",
    "small_angle_threshold_deg",
)

WARNING_FEATURES = (
    "zero_area_face_count",
    "duplicate_face_count",
    "nonmanifold_edge_count",
)


def load_feature_options(path: str | Path | None) -> dict[str, Any]:
    """Read the subset of a feature config that the basic features accept."""
    if path is None:
        return {}
    raw = load_yaml(path)
    return {key: raw[key] for key in FEATURE_OPTION_KEYS if key in raw}


def evaluate_asset(mesh_path: str | Path, options: dict[str, Any]) -> dict[str, Any]:
    """Load one asset and compute basic features. Raises on unusable input."""
    parts = load_mesh_parts(mesh_path)
    mesh = concatenate_for_metrics(parts)
    features = compute_basic_features(mesh.vertices, mesh.faces, **options)
    return {"part_count": len(parts), **features}


def has_warning(features: dict[str, Any]) -> bool:
    return any(features.get(name, 0) > 0 for name in WARNING_FEATURES)


def safe_message(error: BaseException, maximum: int = 500) -> str:
    return " ".join(str(error).split())[:maximum]
