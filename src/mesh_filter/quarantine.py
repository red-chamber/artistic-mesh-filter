"""Failure classification for assets that cannot enter the pipeline.

A file that fails here is not an Artistic negative sample. It is simply not
usable by the current pipeline, and the reason decides which quarantine bucket
it belongs to. Keeping the bucket machine-readable lets the remote run report
"12 files failed" together with "why", without silently dropping samples.
"""

from __future__ import annotations

from mesh_filter.features.basic import GeometryError
from mesh_filter.mesh_io import MeshLoadError

PARSE_ERROR = "parse_error"
INVALID_GEOMETRY = "invalid_geometry"
UNSUPPORTED_FORMAT = "unsupported_format"

BUCKETS = (PARSE_ERROR, INVALID_GEOMETRY, UNSUPPORTED_FORMAT)


def classify_failure(
    error: BaseException,
    extension: str,
    configured_extensions: set[str],
) -> str:
    normalized = extension if extension.startswith(".") else f".{extension}"
    if normalized.lower() not in configured_extensions:
        return UNSUPPORTED_FORMAT
    if isinstance(error, GeometryError):
        return INVALID_GEOMETRY
    if isinstance(error, MeshLoadError):
        return PARSE_ERROR
    return PARSE_ERROR
