"""Per-asset work shared by `validate` and the smoke test.

Both commands must agree on what "a parsed asset" means, on which feature
options are honoured, and on what counts as a warning. Keeping that in one place
stops the smoke verdict from drifting away from the real feature run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mesh_filter.config import load_yaml
from mesh_filter.features.artistic import COLUMNS as ARTISTIC_COLUMNS
from mesh_filter.features.artistic import compute_artistic_features
from mesh_filter.features.basic import compute_basic_features
from mesh_filter.mesh_io import concatenate_for_metrics, load_mesh_parts

FEATURE_OPTION_KEYS = (
    "zero_area_epsilon_ratio",
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


def load_asset_geometry(
    mesh_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load one asset as a single concatenated triangle mesh.

    Returns the vertex array, the face array and how many parts were joined, so
    callers that need the geometry itself (fingerprints) do not have to parse a
    second time. Raises on unusable input.
    """
    parts = load_mesh_parts(mesh_path)
    mesh = concatenate_for_metrics(parts)
    return np.asarray(mesh.vertices), np.asarray(mesh.faces), len(parts)


def load_artistic_features(path: str | Path | None) -> frozenset[str]:
    """Read which artistic feature columns are switched on.

    The config switches carry the same names as the columns they produce.
    """
    if path is None:
        return frozenset()
    raw = load_yaml(path)
    enabled = raw.get("features", {})
    return frozenset(name for name, on in enabled.items() if on and name in ARTISTIC_COLUMNS)


def evaluate_asset(
    mesh_path: str | Path,
    options: dict[str, Any],
    artistic: frozenset[str] | set[str] = frozenset(),
) -> dict[str, Any]:
    """Load one asset and compute its features. Raises on unusable input.

    A mesh that parses but is too small or too degenerate for the artistic
    features is still a parsed asset: those columns stay empty and
    `artistic_status` records why, so the failure is visible without turning a
    usable mesh into a parse failure.
    """
    vertices, faces, part_count = load_asset_geometry(mesh_path)
    features = compute_basic_features(vertices, faces, **options)
    record: dict[str, Any] = {"part_count": part_count, **features}
    if artistic:
        try:
            record.update(compute_artistic_features(vertices, faces, artistic))
            record["artistic_status"] = "ok"
        except Exception as error:  # noqa: BLE001 - recorded in its own column
            record["artistic_status"] = f"failed:{type(error).__name__}"
    return record


def has_warning(features: dict[str, Any]) -> bool:
    return any(features.get(name, 0) > 0 for name in WARNING_FEATURES)


def safe_message(error: BaseException, maximum: int = 500) -> str:
    return " ".join(str(error).split())[:maximum]
