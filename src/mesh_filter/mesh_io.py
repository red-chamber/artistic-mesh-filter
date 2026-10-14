from __future__ import annotations

from pathlib import Path

import trimesh


class MeshLoadError(RuntimeError):
    """Raised when a file does not contain usable triangle geometry."""


def load_mesh_parts(path: str | Path) -> list[trimesh.Trimesh]:
    mesh_path = Path(path)
    loaded = trimesh.load(mesh_path, process=False)

    if isinstance(loaded, trimesh.Trimesh):
        parts = [loaded]
    elif isinstance(loaded, trimesh.Scene):
        parts = [
            geometry
            for geometry in loaded.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
        ]
    else:
        parts = []

    parts = [part for part in parts if len(part.vertices) > 0 and len(part.faces) > 0]
    if not parts:
        raise MeshLoadError(f"No non-empty triangle mesh found in {mesh_path}")
    return parts


def concatenate_for_metrics(parts: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    if len(parts) == 1:
        return parts[0]
    return trimesh.util.concatenate(parts)
