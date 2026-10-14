"""Smoke test for a real data source.

The local machine can prove the code is logically correct; it cannot prove the
code suits the real models. This command is the missing half: it runs the same
per-asset work as `validate` on a small deterministic sample of the real data and
returns one JSON verdict, so a remote run either passes the gate or says exactly
which check failed.
"""

from __future__ import annotations

import json
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import Any

from mesh_filter.config import ProjectConfig
from mesh_filter.inspection import evaluate_asset, has_warning, load_feature_options, safe_message
from mesh_filter.manifest import normalized_extensions
from mesh_filter.quarantine import classify_failure
from mesh_filter.sampling import (
    deterministic_sample,
    extension_of,
    iter_source_files,
    resolve_sample_size,
)

DEFAULT_THRESHOLDS: dict[str, float] = {
    "min_parse_rate": 0.90,
    "max_seconds_per_asset": 120.0,
    "max_peak_mb": 2048.0,
}

MAX_QUARANTINE_PATHS_REPORTED = 20

MAX_SELECTION_VIOLATIONS_REPORTED = 10


def _thresholds(config: ProjectConfig) -> dict[str, float]:
    configured = config.values.get("smoke", {})
    return {
        key: float(configured.get(key, default)) for key, default in DEFAULT_THRESHOLDS.items()
    }


def _check_asset(path: Path, options: dict[str, Any], configured: set[str]) -> dict[str, Any]:
    extension = extension_of(path)
    row: dict[str, Any] = {
        "relpath": str(path),
        "extension": extension,
        "size_bytes": int(path.stat().st_size) if path.exists() else 0,
        "parse_status": "failed",
        "validity_status": "invalid",
        "error_code": "",
        "error_message": "",
        "quarantine_bucket": "",
        "seconds": 0.0,
        "peak_python_mb": 0.0,
    }

    owns_tracing = not tracemalloc.is_tracing()
    if owns_tracing:
        tracemalloc.start()
    start = time.perf_counter()
    try:
        features = evaluate_asset(path, options)
    except Exception as error:  # per-asset containment is intentional
        row["error_code"] = type(error).__name__
        row["error_message"] = safe_message(error)
        row["quarantine_bucket"] = classify_failure(error, extension, configured)
        return row
    finally:
        row["seconds"] = round(time.perf_counter() - start, 6)
        if owns_tracing:
            row["peak_python_mb"] = round(tracemalloc.get_traced_memory()[1] / 1024 / 1024, 3)
            tracemalloc.stop()

    row["parse_status"] = "parsed"
    row["validity_status"] = "valid_with_warnings" if has_warning(features) else "valid"
    row["part_count"] = features["part_count"]
    row["vertex_count"] = features["vertex_count"]
    row["face_count"] = features["face_count"]
    return row


EMPTY_SUMMARY: dict[str, Any] = {
    "sampled": 0,
    "parsed": 0,
    "failed": 0,
    "warnings": 0,
    "sample_parse_rate": None,
    "ingestable_count": 0,
    "max_seconds": None,
    "mean_seconds": None,
    "max_peak_python_mb": None,
    "estimated_full_pass_seconds": None,
    "error_codes": {},
    "quarantine_buckets": {},
    "selection_limits": None,
    "selection_limit_violation_count": 0,
    "selection_limit_violations": [],
}


def _selection_violations(
    rows: list[dict[str, Any]],
    limits: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Sampled assets that fall outside the declared screening bounds.

    The negative set was filtered by hand before it reached this project. If the
    delivered directory does not match that screening rule, the proxy label no
    longer means what the data card says it means, so this must be loud.
    Counts are measured after triangulation, which is what the pipeline sees.
    """
    if not limits:
        return []
    max_faces = limits.get("max_face_count")
    max_vertices = limits.get("max_vertex_count")
    violations: list[dict[str, Any]] = []
    for row in rows:
        if row["parse_status"] != "parsed":
            continue
        faces = row.get("face_count")
        vertices = row.get("vertex_count")
        over_faces = max_faces is not None and faces is not None and faces > max_faces
        over_vertices = (
            max_vertices is not None and vertices is not None and vertices > max_vertices
        )
        if over_faces or over_vertices:
            violations.append(
                {
                    "relpath": row["relpath"],
                    "face_count": faces,
                    "vertex_count": vertices,
                }
            )
    return violations


def _summarize(
    rows: list[dict[str, Any]],
    ingestable_count: int,
    limits: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parsed = [row for row in rows if row["parse_status"] == "parsed"]
    failed = [row for row in rows if row["parse_status"] == "failed"]
    seconds = [row["seconds"] for row in rows]
    mean_seconds = statistics.fmean(seconds) if seconds else None
    violations = _selection_violations(rows, limits)
    return {
        "sampled": len(rows),
        "parsed": len(parsed),
        "failed": len(failed),
        "warnings": sum(1 for row in parsed if row["validity_status"] == "valid_with_warnings"),
        "sample_parse_rate": round(len(parsed) / len(rows), 4) if rows else None,
        "ingestable_count": ingestable_count,
        "max_seconds": round(max(seconds), 6) if seconds else None,
        "mean_seconds": round(mean_seconds, 6) if mean_seconds is not None else None,
        "max_peak_python_mb": max((row["peak_python_mb"] for row in rows), default=None),
        "estimated_full_pass_seconds": (
            round(mean_seconds * ingestable_count, 1)
            if mean_seconds is not None and ingestable_count
            else None
        ),
        "error_codes": _counts(row["error_code"] for row in failed),
        "quarantine_buckets": _counts(row["quarantine_bucket"] for row in failed),
        "selection_limits": limits,
        "selection_limit_violation_count": len(violations),
        "selection_limit_violations": violations[:MAX_SELECTION_VIOLATIONS_REPORTED],
    }


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _check(
    name: str,
    ok: bool,
    detail: str,
    *,
    hard: bool = True,
) -> dict[str, Any]:
    if ok:
        status = "pass"
    else:
        status = "fail" if hard else "warn"
    return {"name": name, "status": status, "detail": detail}


def _verdict_checks(
    per_source: dict[str, dict[str, Any]],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    totals = {"sampled": 0, "parsed": 0, "failed": 0}

    for source_name, summary in sorted(per_source.items()):
        for key in totals:
            totals[key] += summary[key]
        checks.append(
            _check(
                f"{source_name}.ingestable_not_empty",
                summary["ingestable_count"] > 0,
                summary.get("error")
                or f"ingestable files: {summary['ingestable_count']}",
            )
        )
        checks.append(
            _check(
                f"{source_name}.sample_not_empty",
                summary["sampled"] > 0,
                f"sampled files: {summary['sampled']}",
                hard=summary["ingestable_count"] > 0,
            )
        )
        checks.append(
            _check(
                f"{source_name}.no_silent_drops",
                summary["sampled"] == summary["parsed"] + summary["failed"],
                f"sampled {summary['sampled']} = parsed {summary['parsed']} "
                f"+ failed {summary['failed']}",
            )
        )
        if summary["ingestable_count"] > 0:
            rate = summary["sample_parse_rate"] or 0.0
            checks.append(
                _check(
                    f"{source_name}.parse_rate",
                    rate >= thresholds["min_parse_rate"],
                    f"sample parse rate {rate:.4f} vs minimum {thresholds['min_parse_rate']:.2f}",
                )
            )
            max_seconds = summary["max_seconds"] or 0.0
            checks.append(
                _check(
                    f"{source_name}.seconds_per_asset",
                    max_seconds <= thresholds["max_seconds_per_asset"],
                    f"slowest asset {max_seconds:.3f}s vs budget "
                    f"{thresholds['max_seconds_per_asset']:.1f}s",
                )
            )
            peak = summary["max_peak_python_mb"] or 0.0
            checks.append(
                _check(
                    f"{source_name}.peak_memory",
                    peak <= thresholds["max_peak_mb"],
                    f"peak python heap {peak:.1f}MB vs budget "
                    f"{thresholds['max_peak_mb']:.0f}MB "
                    "(python heap only; excludes C-level and GPU memory)",
                )
            )
            if summary.get("selection_limits"):
                checks.append(
                    _check(
                        f"{source_name}.selection_limits",
                        summary["selection_limit_violation_count"] == 0,
                        f"{summary['selection_limit_violation_count']}/{summary['parsed']} "
                        f"parsed assets exceed declared limits "
                        f"{summary['selection_limits']} "
                        "(counts measured after triangulation)",
                    )
                )

    checks.append(
        _check(
            "overall.parse_rate",
            (totals["parsed"] / totals["sampled"]) >= thresholds["min_parse_rate"]
            if totals["sampled"]
            else False,
            f"parsed {totals['parsed']}/{totals['sampled']} sampled assets",
        )
    )
    return checks


def run_smoke_test(
    config: ProjectConfig,
    sample_per_source: int | None = None,
    feature_config_path: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    configured = normalized_extensions(config)
    options = load_feature_options(feature_config_path)
    thresholds = _thresholds(config)
    size = resolve_sample_size(config, sample_per_source)
    sources = config.values.get("sources", {})

    per_source: dict[str, dict[str, Any]] = {}
    rows_by_source: dict[str, list[dict[str, Any]]] = {}
    quarantine: dict[str, list[str]] = {}

    for source_name in sorted(sources):
        root = config.source_root(source_name)
        try:
            ingestable = iter_source_files(config, source_name)
        except Exception as error:  # an unreachable source is a smoke finding, not a crash
            per_source[source_name] = {**EMPTY_SUMMARY, "error": safe_message(error)}
            rows_by_source[source_name] = []
            continue
        sampled = deterministic_sample(ingestable, size)
        rows = [_check_asset(path, options, configured) for path in sampled]
        for row, path in zip(rows, sampled, strict=True):
            row["relpath"] = (
                path.relative_to(root).as_posix() if root in path.parents else str(path)
            )
            if row["quarantine_bucket"]:
                quarantine.setdefault(row["quarantine_bucket"], []).append(row["relpath"])
        rows_by_source[source_name] = rows
        limits = sources[source_name].get("selection_limits")
        per_source[source_name] = _summarize(rows, len(ingestable), limits)

    checks = _verdict_checks(per_source, thresholds)
    verdict = "fail" if any(check["status"] == "fail" for check in checks) else "pass"

    report: dict[str, Any] = {
        "verdict": verdict,
        "sample_per_source": size,
        "thresholds": thresholds,
        "checks": checks,
        "sources": per_source,
        "quarantine_plan": {
            bucket: {
                "count": len(paths),
                "sample": paths[:MAX_QUARANTINE_PATHS_REPORTED],
            }
            for bucket, paths in sorted(quarantine.items())
        },
        "samples": rows_by_source,
    }

    output = config.output_path("smoke_report")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return output, report
