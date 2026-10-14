"""Format probing for a real data source.

Synthetic fixtures cannot answer the questions that decide the remote
implementation: which formats are actually present, whether files parse, how
many geometries / instances / materials one asset contains, and what scale and
axis conventions the sources use.

The probe is read-only, samples a small deterministic subset, never raises for
a single bad file, and writes one JSON report that can be sent back for review.
"""

from __future__ import annotations

import json
import statistics
import time
import tracemalloc
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import trimesh

from mesh_filter.config import ProjectConfig
from mesh_filter.manifest import normalized_extensions
from mesh_filter.quarantine import INVALID_GEOMETRY, classify_failure
from mesh_filter.sampling import (
    count_extensions,
    deterministic_sample,
    extension_of,
    iter_all_files,
    iter_source_files,
    resolve_sample_size,
)

MAX_UNCONFIGURED_SAMPLE_EXTENSIONS = 5
UNCONFIGURED_SAMPLE_PER_EXTENSION = 3


def _safe(call: Callable[[], Any], default: Any = None) -> Any:
    """Probe helpers must never turn one unsupported trimesh call into a crash."""
    try:
        return call()
    except Exception:  # noqa: BLE001 - the whole point is to keep probing
        return default


def _identity_transform_count(loaded: trimesh.Scene) -> int | None:
    """Count scene nodes whose transform is not the identity.

    Instanced or pre-transformed nodes change what "one asset" means for the
    training pipeline, so the probe reports them instead of hiding them.
    """
    edges = _safe(lambda: loaded.graph.to_edgelist())
    if edges is None:
        return None
    identity = _safe(lambda: trimesh.transformations.identity_matrix())
    total = 0
    for entry in edges:
        transform = entry[1] if len(entry) > 1 else None
        if transform is None or identity is None:
            continue
        if not _safe(
            lambda transform=transform: bool((abs(transform - identity) < 1e-9).all()),
            False,
        ):
            total += 1
    return total


def _part_records(parts: list[Any]) -> dict[str, Any]:
    vertex_count = 0
    face_count = 0
    visual_kinds: Counter[str] = Counter()
    uv_parts = 0
    watertight = 0
    not_watertight = 0
    unknown_watertight = 0
    for part in parts:
        vertex_count += int(_safe(lambda part=part: len(part.vertices), 0))
        face_count += int(_safe(lambda part=part: len(part.faces), 0))
        kind = _safe(lambda part=part: str(part.visual.kind), "unknown")
        visual_kinds[str(kind)] += 1
        if _safe(lambda part=part: part.visual.uv is not None, False):
            uv_parts += 1
        flag = _safe(lambda part=part: bool(part.is_watertight), None)
        if flag is None:
            unknown_watertight += 1
        elif flag:
            watertight += 1
        else:
            not_watertight += 1
    return {
        "vertex_count": vertex_count,
        "face_count": face_count,
        "visual_kinds": dict(sorted(visual_kinds.items())),
        "parts_with_uv": uv_parts,
        "watertight_parts": watertight,
        "non_watertight_parts": not_watertight,
        "unknown_watertight_parts": unknown_watertight,
    }


def inspect_asset(path: Path, configured_extensions: set[str]) -> dict[str, Any]:
    """Inspect one file and always return a record, never raise."""
    extension = extension_of(path)
    record: dict[str, Any] = {
        "relpath": str(path),
        "extension": extension,
        "size_bytes": int(_safe(lambda: path.stat().st_size, 0)),
        "parse_status": "failed",
        "error_code": "",
        "error_message": "",
        "quarantine_bucket": "",
        "load_seconds": 0.0,
        "peak_python_mb": 0.0,
    }

    owns_tracing = not tracemalloc.is_tracing()
    if owns_tracing:
        tracemalloc.start()
    start = time.perf_counter()
    try:
        loaded = trimesh.load(path, process=False)
    except Exception as error:  # per-asset containment is intentional
        record["error_code"] = type(error).__name__
        record["error_message"] = " ".join(str(error).split())[:300]
        record["quarantine_bucket"] = classify_failure(error, extension, configured_extensions)
        return record
    finally:
        record["load_seconds"] = round(time.perf_counter() - start, 6)
        if owns_tracing:
            record["peak_python_mb"] = round(tracemalloc.get_traced_memory()[1] / 1024 / 1024, 3)
            tracemalloc.stop()

    record["parse_status"] = "parsed"
    record["container"] = type(loaded).__name__

    if isinstance(loaded, trimesh.Scene):
        geometry = loaded.geometry
        parts = [value for value in geometry.values() if isinstance(value, trimesh.Trimesh)]
        record["geometry_count"] = len(geometry)
        record["geometry_types"] = dict(
            sorted(Counter(type(value).__name__ for value in geometry.values()).items())
        )
        record["node_count"] = len(_safe(lambda: loaded.graph.nodes, []))
        record["non_identity_transform_count"] = _identity_transform_count(loaded)
        record["has_animation"] = bool(_safe(lambda: getattr(loaded, "animation", None), None))
    else:
        parts = [loaded] if isinstance(loaded, trimesh.Trimesh) else []
        record["geometry_count"] = len(parts)
        record["geometry_types"] = {type(loaded).__name__: 1}
        record["node_count"] = 1
        record["non_identity_transform_count"] = 0
        record["has_animation"] = False

    record["part_count"] = len(parts)
    record.update(_part_records(parts))

    bounds = _safe(lambda: loaded.bounds)
    if bounds is not None and len(bounds) == 2:
        extents = [float(value) for value in (bounds[1] - bounds[0])]
        longest = max(extents) if extents else 0.0
        centre = (bounds[0] + bounds[1]) / 2.0
        record["extents"] = [round(value, 6) for value in extents]
        record["longest_extent"] = round(longest, 6)
        record["centred_at_origin"] = bool(
            longest > 0 and float(abs(centre).max()) <= 1e-6 * longest
        )
    else:
        record["extents"] = None
        record["longest_extent"] = None
        record["centred_at_origin"] = None

    if not parts:
        # Loaded successfully but holds no triangle geometry: not usable as-is.
        record["parse_status"] = "failed"
        record["error_code"] = "NoTriangleGeometry"
        record["error_message"] = f"{record['container']} contains no triangle geometry"
        record["quarantine_bucket"] = INVALID_GEOMETRY
    return record


def _relative(path: Path, root: Path) -> str:
    return _safe(lambda: path.relative_to(root).as_posix(), str(path))


def _probe_unconfigured(
    root: Path,
    paths: list[Path],
    configured_extensions: set[str],
) -> dict[str, Any]:
    """Try to parse a couple of files per out-of-whitelist extension.

    If the sources use a format nobody put in `formats.extensions`, the pipeline
    silently ingests nothing. This is the loud answer to that failure mode.
    """
    by_extension: dict[str, list[Path]] = {}
    for path in paths:
        extension = extension_of(path)
        if extension not in configured_extensions:
            by_extension.setdefault(extension, []).append(path)

    results: dict[str, Any] = {}
    sampled_extensions = sorted(by_extension.items())[:MAX_UNCONFIGURED_SAMPLE_EXTENSIONS]
    for extension, extension_paths in sampled_extensions:
        attempts = []
        for path in extension_paths[:UNCONFIGURED_SAMPLE_PER_EXTENSION]:
            loaded = _safe(lambda path=path: trimesh.load(path, process=False))
            if loaded is None:
                attempts.append({"relpath": _relative(path, root), "loadable": False})
                continue
            parts = (
                [value for value in loaded.geometry.values() if isinstance(value, trimesh.Trimesh)]
                if isinstance(loaded, trimesh.Scene)
                else ([loaded] if isinstance(loaded, trimesh.Trimesh) else [])
            )
            attempts.append(
                {
                    "relpath": _relative(path, root),
                    "loadable": True,
                    "container": type(loaded).__name__,
                    "triangle_parts": len(parts),
                }
            )
        results[extension] = {"file_count": len(extension_paths), "attempts": attempts}
    return results


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = [record for record in records if record["parse_status"] == "parsed"]
    failures = [record for record in records if record["parse_status"] == "failed"]
    longest = [record["longest_extent"] for record in parsed if record["longest_extent"]]
    part_counts = [record["part_count"] for record in parsed]

    def spread(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        return {
            "min": round(min(values), 6),
            "median": round(statistics.median(values), 6),
            "max": round(max(values), 6),
        }

    uv_parts = sum(record["parts_with_uv"] for record in parsed)
    parts_total = sum(record["part_count"] for record in parsed)
    scene_records = [record for record in parsed if record["container"] == "Scene"]
    return {
        "sampled": len(records),
        "parsed": len(parsed),
        "failed": len(failures),
        "parse_rate": round(len(parsed) / len(records), 4) if records else None,
        "error_codes": dict(sorted(Counter(record["error_code"] for record in failures).items())),
        "quarantine_buckets": dict(
            sorted(Counter(record["quarantine_bucket"] for record in failures).items())
        ),
        "containers": dict(sorted(Counter(record["container"] for record in parsed).items())),
        "longest_extent": spread(longest),
        "part_count": spread([float(value) for value in part_counts]),
        "parts_total": parts_total,
        "parts_with_uv": uv_parts,
        "uv_part_share": round(uv_parts / parts_total, 4) if parts_total else None,
        "multi_geometry_share": (
            round(
                sum(1 for record in parsed if record["geometry_count"] > 1) / len(parsed),
                4,
            )
            if parsed
            else None
        ),
        "scene_share": round(len(scene_records) / len(parsed), 4) if parsed else None,
        "records_with_instancing": sum(
            1
            for record in scene_records
            if (record.get("non_identity_transform_count") or 0) > 0
        ),
        "mean_load_seconds": (
            round(
                sum(record["load_seconds"] for record in records) / len(records),
                6,
            )
            if records
            else None
        ),
        "mean_peak_python_mb": (
            round(
                sum(record["peak_python_mb"] for record in records) / len(records),
                3,
            )
            if records
            else None
        ),
    }


def probe_sources(
    config: ProjectConfig,
    sample_per_source: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    configured_extensions = normalized_extensions(config)
    sources = config.values.get("sources", {})
    size = resolve_sample_size(config, sample_per_source)

    report: dict[str, Any] = {
        "configured_extensions": sorted(configured_extensions),
        "sample_per_source": size,
        "sources": {},
    }

    for source_name in sorted(sources):
        root = config.source_root(source_name)
        try:
            all_files = iter_all_files(config, source_name)
            ingestable = iter_source_files(config, source_name)
        except Exception as error:  # a wrong path is a finding, not a crash
            report["sources"][source_name] = {
                "root": str(root),
                "file_count": 0,
                "ingestable_count": 0,
                "extension_counts": {},
                "unconfigured_extensions": {},
                "sampled_count": 0,
                "aggregate": {},
                "sampled": [],
                "unconfigured_probe": {},
                "notes": [f"Cannot list source: {error}"],
            }
            continue
        sampled = deterministic_sample(ingestable, size)
        records = [inspect_asset(path, configured_extensions) for path in sampled]
        for record, path in zip(records, sampled, strict=True):
            record["relpath"] = _relative(path, root)

        extension_counts = count_extensions(all_files)
        unconfigured = {
            extension: count
            for extension, count in extension_counts.items()
            if extension not in configured_extensions
        }
        notes: list[str] = []
        if not ingestable:
            notes.append(
                "No ingestable file found. Check formats.extensions against "
                f"the extensions actually present: {extension_counts}"
            )
        if unconfigured:
            notes.append(
                "Files outside formats.extensions are invisible to the pipeline: "
                f"{unconfigured}"
            )
        aggregate = _aggregate(records)
        if aggregate["failed"]:
            notes.append(
                f"{aggregate['failed']}/{aggregate['sampled']} sampled files failed: "
                f"{aggregate['error_codes']}"
            )

        report["sources"][source_name] = {
            "root": str(root),
            "file_count": len(all_files),
            "ingestable_count": len(ingestable),
            "extension_counts": extension_counts,
            "unconfigured_extensions": unconfigured,
            "sampled_count": len(sampled),
            "aggregate": aggregate,
            "sampled": records,
            "unconfigured_probe": _probe_unconfigured(root, all_files, configured_extensions),
            "notes": notes,
        }

    output = config.output_path("probe_report")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return output, report
