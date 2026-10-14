from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, Iterable

from mesh_filter.config import ConfigError, ProjectConfig

MANIFEST_FIELDS = [
    "asset_id",
    "source_dataset",
    "source_uid",
    "raw_relpath",
    "sha256",
    "file_format",
    "file_size_bytes",
    "label",
    "label_origin",
    "selection_rule",
    "parse_status",
    "validity_status",
    "error_code",
    "duplicate_group_id",
    "canonical_version",
    "split",
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_asset_id(source_name: str, relative_path: str) -> str:
    identity = f"{source_name}\0{relative_path}".encode("utf-8")
    return hashlib.sha256(identity).hexdigest()[:20]


def read_csv(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    if fieldnames is None:
        fieldnames = list(materialized[0]) if materialized else []
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)
    return output_path


def normalized_extensions(config: ProjectConfig) -> set[str]:
    extensions = config.values.get("formats", {}).get("extensions", [])
    normalized = {str(ext).lower() for ext in extensions}
    if not normalized:
        raise ConfigError("formats.extensions must contain at least one extension")
    return {ext if ext.startswith(".") else f".{ext}" for ext in normalized}


def build_manifest(config: ProjectConfig) -> tuple[Path, int]:
    sources = config.values.get("sources", {})
    if not sources:
        raise ConfigError("At least one source must be configured")

    extensions = normalized_extensions(config)
    runtime = config.values.get("runtime", {})
    hash_files = bool(runtime.get("hash_files", True))
    limit = runtime.get("limit_per_source")
    rows: list[dict[str, Any]] = []

    for source_name in sorted(sources):
        source = sources[source_name]
        root = config.source_root(source_name)
        if not root.is_dir():
            raise ConfigError(f"Source directory does not exist: {root}")

        paths = sorted(
            path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions
        )
        if limit is not None:
            paths = paths[: int(limit)]

        for path in paths:
            relative = path.relative_to(root).as_posix()
            rows.append(
                {
                    "asset_id": make_asset_id(source_name, relative),
                    "source_dataset": source_name,
                    "source_uid": relative,
                    "raw_relpath": relative,
                    "sha256": sha256_file(path) if hash_files else "",
                    "file_format": path.suffix.lower().lstrip("."),
                    "file_size_bytes": path.stat().st_size,
                    "label": int(source["label"]),
                    "label_origin": source.get("label_origin", "dataset_proxy"),
                    "selection_rule": source.get("selection_rule", "unspecified"),
                    "parse_status": "not_run",
                    "validity_status": "not_run",
                    "error_code": "",
                    "duplicate_group_id": "",
                    "canonical_version": "",
                    "split": "",
                }
            )

    output = config.output_path("manifest")
    write_csv(output, rows, MANIFEST_FIELDS)
    return output, len(rows)
