"""Deterministic sampling of read-only sources.

The probe and the smoke test both need the same thing: the list of files the
pipeline would actually ingest, and a reproducible small subset of it. A random
sample would make remote smoke runs hard to compare, so sampling is evenly
spaced and seeded only by sorted paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from mesh_filter.config import ConfigError, ProjectConfig
from mesh_filter.manifest import normalized_extensions

DEFAULT_SAMPLE_SIZE = 20


def iter_source_files(config: ProjectConfig, source_name: str) -> list[Path]:
    """Return the ingestable files of one source, sorted for reproducibility."""
    extensions = normalized_extensions(config)
    root = config.source_root(source_name)
    if not root.is_dir():
        raise ConfigError(f"Source directory does not exist: {root}")
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions
    )


def iter_all_files(config: ProjectConfig, source_name: str) -> list[Path]:
    """Return every file of one source, including formats not in the whitelist."""
    root = config.source_root(source_name)
    if not root.is_dir():
        raise ConfigError(f"Source directory does not exist: {root}")
    return sorted(path for path in root.rglob("*") if path.is_file())


def extension_of(path: Path) -> str:
    return path.suffix.lower() or "<none>"


def deterministic_sample(items: Sequence[Path], size: int | None) -> list[Path]:
    """Pick evenly spaced items so the sample covers the whole sorted range."""
    if size is None or size >= len(items):
        return list(items)
    if size <= 0:
        raise ValueError("Sample size must be positive")
    step = len(items) / size
    return [items[min(int(index * step), len(items) - 1)] for index in range(size)]


def resolve_sample_size(config: ProjectConfig, explicit: int | None = None) -> int | None:
    """Explicit CLI value wins, then runtime.sample_per_source, then the default."""
    if explicit is not None:
        return int(explicit)
    configured = config.values.get("runtime", {}).get("sample_per_source", DEFAULT_SAMPLE_SIZE)
    return None if configured is None else int(configured)


def count_extensions(paths: Iterable[Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in paths:
        key = extension_of(path)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
