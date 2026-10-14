"""Rotation-, scale- and order-invariant geometry fingerprints.

Exact SHA-256 deduplication only catches byte-identical files. The same model
re-exported with reordered vertices, a different unit, or a different rotation
becomes a different hash, and if the two copies land in different splits the
evaluation leaks. The fingerprint closes that gap.

The signature is built only from quantities that survive rotation, uniform
scaling, translation, vertex reordering and mirroring:

- quantiles of the per-vertex distance to the centroid (normalised by its mean);
- quantiles of the per-triangle area (normalised by its mean);
- sorted covariance eigenvalues as ratios, which describe the shape's anisotropy.

Everything is computed on the concatenated mesh, exactly like the basic features,
so the counts used for bucketing agree with the features table.

Two assets can only be near-duplicates if their vertex count, face count and
component count are identical, which makes the candidate filter exact rather than
heuristic: an asset alone in its (vertex, face, component) bucket cannot have a
near-duplicate. That is why only multi-member buckets are parsed.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from mesh_filter.config import ConfigError, ProjectConfig
from mesh_filter.features.basic import GeometryError, compute_basic_features
from mesh_filter.inspection import load_asset_geometry, load_feature_options
from mesh_filter.manifest import read_csv, write_csv

RADIAL_QUANTILES = 64
AREA_QUANTILES = 16
DEFAULT_TOLERANCE = 1.0e-3
BUCKET_EIGEN_DIGITS = 1

FINGERPRINT_FIELDS = [
    "asset_id",
    "source_dataset",
    "label",
    "vertex_count",
    "face_count",
    "component_count",
    "radial_quantiles",
    "area_quantiles",
    "eigenvalue_ratios",
    "bucket_key",
]


def _quantiles(values: np.ndarray, count: int) -> list[float]:
    points = np.linspace(0.0, 1.0, count)
    return [float(value) for value in np.quantile(values, points)]


def geometric_signature(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    """Build the invariant signature of one triangle mesh."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(vertices) < 2 or len(faces) == 0:
        raise GeometryError("fingerprint needs at least two vertices and one face")

    centred = vertices - vertices.mean(axis=0)
    radial = np.linalg.norm(centred, axis=1)
    mean_radial = float(radial.mean())
    if mean_radial <= 0.0:
        raise GeometryError("all vertices coincide; no shape to fingerprint")

    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    mean_area = float(areas.mean())
    if mean_area <= 0.0:
        raise GeometryError("every face has zero area; no shape to fingerprint")

    eigenvalues = np.sort(np.linalg.eigvalsh(np.cov(centred.T)))[::-1]
    total = float(eigenvalues.sum())
    eigenvalue_ratios = (
        [float(value) for value in eigenvalues / total] if total > 0 else [0.0, 0.0, 0.0]
    )

    return {
        "radial_quantiles": _quantiles(radial / mean_radial, RADIAL_QUANTILES),
        "area_quantiles": _quantiles(areas / mean_area, AREA_QUANTILES),
        "eigenvalue_ratios": eigenvalue_ratios,
    }


def bucket_key(
    vertex_count: int,
    face_count: int,
    component_count: int,
    eigenvalue_ratios: Sequence[float],
) -> str:
    """Coarse key used to avoid comparing every asset with every other one.

    Eigenvalue ratios are rounded coarsely: identical geometry produces identical
    ratios, and coarse rounding keeps near-identical exports in the same bucket.
    """
    rounded = (round(float(value), BUCKET_EIGEN_DIGITS) for value in eigenvalue_ratios)
    return f"{vertex_count}:{face_count}:{component_count}:" + ":".join(f"{v:.1f}" for v in rounded)


def _serialise(values: Sequence[float]) -> str:
    return ";".join(f"{float(value):.6f}" for value in values)


def _deserialise(text: str) -> list[float]:
    return [float(value) for value in text.split(";")] if text else []


def load_fingerprints(config: ProjectConfig) -> list[dict[str, Any]]:
    path = config.output_path("fingerprints")
    if not path.exists():
        raise ConfigError(f"Fingerprint table not found: {path}. Run `deduplicate` first.")
    rows: list[dict[str, Any]] = []
    for row in read_csv(path):
        try:
            rows.append(
                {
                    "asset_id": row["asset_id"],
                    "vertex_count": int(row["vertex_count"]),
                    "face_count": int(row["face_count"]),
                    "component_count": int(row["component_count"]),
                    "radial_quantiles": _deserialise(row["radial_quantiles"]),
                    "area_quantiles": _deserialise(row["area_quantiles"]),
                    "eigenvalue_ratios": _deserialise(row["eigenvalue_ratios"]),
                    "bucket_key": row["bucket_key"],
                }
            )
        except (KeyError, ValueError):
            continue
    return rows


def signatures_match(left: dict[str, Any], right: dict[str, Any], tolerance: float) -> bool:
    """Compare two signatures.

    Radial and eigenvalue terms are compared with an absolute tolerance because
    both are normalised to order-one values. Area quantiles span orders of
    magnitude, so they use a relative tolerance instead: an absolute epsilon
    would be far stricter on a flat mesh than on a spiky one.
    """
    if (left["vertex_count"], left["face_count"], left["component_count"]) != (
        right["vertex_count"],
        right["face_count"],
        right["component_count"],
    ):
        return False
    for key, relative in (
        ("radial_quantiles", False),
        ("eigenvalue_ratios", False),
        ("area_quantiles", True),
    ):
        left_values, right_values = left[key], right[key]
        if len(left_values) != len(right_values):
            return False
        for left_value, right_value in zip(left_values, right_values, strict=True):
            if relative:
                limit = tolerance * max(1.0, abs(left_value), abs(right_value))
            else:
                limit = tolerance
            if abs(left_value - right_value) > limit:
                return False
    return True


def candidate_rows(config: ProjectConfig) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Manifest rows that can possibly have a near-duplicate.

    The bucket key comes from the feature table (counts are already computed
    there); the row itself comes from the manifest, because that is where the
    path and the label live.
    """
    features_path = config.output_path("features")
    if not features_path.exists():
        raise ConfigError(f"Features table not found: {features_path}. Run `validate` first.")
    manifest = {row["asset_id"]: row for row in read_csv(config.output_path("manifest"))}

    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    unparsed = 0
    missing = 0
    for row in read_csv(features_path):
        if row.get("parse_status") != "parsed":
            unparsed += 1
            continue
        manifest_row = manifest.get(row.get("asset_id", ""))
        if manifest_row is None:
            missing += 1
            continue
        key = (row.get("vertex_count"), row.get("face_count"), row.get("component_count"))
        buckets[str(key)].append(manifest_row)

    candidates: list[dict[str, Any]] = []
    singleton_buckets = 0
    for key in sorted(buckets):
        members = buckets[key]
        if len(members) == 1:
            singleton_buckets += 1
            continue
        candidates.extend(members)
    stats = {
        "unparsed": unparsed,
        "missing_from_manifest": missing,
        "candidate_assets": len(candidates),
        "singleton_buckets_skipped": singleton_buckets,
    }
    return candidates, stats


def write_fingerprints(
    config: ProjectConfig,
    feature_config_path: str | Path | None = None,
    tolerance: float | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Parse every candidate asset and write the fingerprint table."""
    options: dict[str, Any] = load_feature_options(feature_config_path)

    candidates, stats = candidate_rows(config)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for record in candidates:
        mesh_path = config.source_root(record["source_dataset"]) / record["raw_relpath"]
        try:
            vertices, faces, _ = load_asset_geometry(mesh_path)
            features = compute_basic_features(vertices, faces, **options)
            signature = geometric_signature(vertices, faces)
        except Exception as error:  # per-asset containment is intentional
            failures.append({"asset_id": record["asset_id"], "error": str(error)[:200]})
            continue
        rows.append(
            {
                "asset_id": record["asset_id"],
                "source_dataset": record["source_dataset"],
                "label": record["label"],
                "vertex_count": features["vertex_count"],
                "face_count": features["face_count"],
                "component_count": features["component_count"],
                "radial_quantiles": _serialise(signature["radial_quantiles"]),
                "area_quantiles": _serialise(signature["area_quantiles"]),
                "eigenvalue_ratios": _serialise(signature["eigenvalue_ratios"]),
                "bucket_key": bucket_key(
                    features["vertex_count"],
                    features["face_count"],
                    features["component_count"],
                    signature["eigenvalue_ratios"],
                ),
            }
        )

    output = config.output_path("fingerprints")
    write_csv(output, rows, FINGERPRINT_FIELDS)
    summary = {
        **stats,
        "fingerprinted": len(rows),
        "failed": len(failures),
        "failures": failures[:10],
        "tolerance": DEFAULT_TOLERANCE if tolerance is None else tolerance,
    }
    return output, summary
