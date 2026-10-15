"""Anonymised meshes for blind human review.

The audit sheet already decides which assets get reviewed and under which
anonymous name. This module turns that sheet into something a person can open:
one flat directory of meshes named `A0001.obj`, `A0002.obj`, ..., each rewritten
in a single minimal format, all with the same timestamp.

The rewriting is the blinding step, not cosmetics. The negatives were exported
by Blender and the positives come from the Toys4K release, so the raw files
differ in vertex ordering, normals, UVs and numeric formatting - a reviewer
could name the source off the file without judging any geometry. A minimal OBJ
has no comments, no materials, no UVs and no unused vertices, so both sources
look identical from the file's point of view.

Judging from a 3D viewer is also better for this task than judging from a fixed
set of renders: the criteria here are topological (edge flow, face density,
sliver triangles), and a viewer lets the reviewer rotate, cut wireframe in and
out, and look at the back of the model instead of guessing it. The viewer also
normalises framing, so the scale channel - Toys4K is always 2.0, Objaverse spans
three orders of magnitude - is not visible either.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from mesh_filter.config import ProjectConfig
from mesh_filter.inspection import load_asset_geometry
from mesh_filter.manifest import read_csv

PRECISION = 6
# One fixed timestamp for every file: creation dates would otherwise separate the
# Toys4K originals from the Blender re-exports. 2023-11-14T22:13:20Z, arbitrary.
REVIEW_MTIME = 1700000000
STALE_PATTERN = "A????.obj"

PROTOCOL = """# 盲审操作说明

1. 打开方式：用 VSCode 的 3D Viewer 插件直接打开本目录下的 `.obj`，或任何 DCC 工具。
   本目录里的模型已统一重写为同一种最简格式，看文件本身分辨不出来源。
2. 观察顺序：先开线框看边流与面密度，再切实体看整体形状。允许自由旋转和缩放。
3. 只判断拓扑质量，不判断外观、材质或贴图——这些信息在本目录里已被刻意移除。
4. 每个文件对应评审表里同名的一行，把结论填进 `human_label`：

   | 取值 | 含义 |
   | --- | --- |
   | `accept` | 拓扑整体符合 Artistic/Low-poly 使用目标 |
   | `reject` | 明显扫描化、无意义密集、随机三角化或严重退化 |
   | `uncertain` | 仅凭当前视图不能稳定判断 |
   | `invalid` | 无法可靠显示或几何不可用，不参与好坏判断 |

5. 不要打开审计对照表（`audit_gold_key.csv`）。它一旦被看过，这份评审就不再是独立证据。
6. 校准：先看前三个，把标准定下来，再开始正式评审；中途不要改判据。
"""


def _remap_to_used_vertices(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Drop vertices no face references.

    Exporters differ in how many orphan vertices they leave behind, and that
    count is a fingerprint. Renumbering keeps the geometry identical while
    making the file uniform.
    """
    used = np.unique(faces)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    return vertices[used], remap[faces]


def minimal_obj_text(vertices: np.ndarray, faces: np.ndarray) -> str:
    vertex_lines = [
        f"v {x:.{PRECISION}g} {y:.{PRECISION}g} {z:.{PRECISION}g}" for x, y, z in vertices
    ]
    face_lines = [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    return "\n".join(vertex_lines + face_lines) + "\n"


def _clear_stale_files(output_dir: Path) -> list[str]:
    """Remove leftovers from an earlier sample.

    A stale `A0137.obj` from a previous run would silently join the review and
    be judged under a name that no longer means the same asset.
    """
    stale = sorted(path.name for path in output_dir.glob(STALE_PATTERN))
    for name in stale:
        (output_dir / name).unlink()
    return stale


def prepare_review(
    config: ProjectConfig,
    limit: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    sheet_path = config.output_path("audit_gold")
    if not sheet_path.exists():
        raise FileNotFoundError(
            f"Audit sheet not found: {sheet_path}. Run `build-audit-gold` first."
        )
    sheet = read_csv(sheet_path)
    if limit is not None:
        sheet = sheet[: int(limit)]

    manifest = {row["asset_id"]: row for row in read_csv(config.output_path("manifest"))}
    output_dir = config.output_path("review_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    stale = _clear_stale_files(output_dir)

    written: list[str] = []
    missing: list[str] = []
    failures: list[dict[str, str]] = []
    total_vertices = 0
    for row in sheet:
        name = row.get("anonymous_name", "")
        asset_id = row.get("asset_id", "")
        manifest_row = manifest.get(asset_id)
        if not name or manifest_row is None:
            missing.append(asset_id or "<no asset_id>")
            continue
        source_path = (
            config.source_root(manifest_row["source_dataset"]) / manifest_row["raw_relpath"]
        )
        try:
            vertices, faces, _ = load_asset_geometry(source_path)
            vertices, faces = _remap_to_used_vertices(vertices, faces)
            target = output_dir / f"{name}.obj"
            target.write_text(minimal_obj_text(vertices, faces), encoding="ascii")
            os.utime(target, (REVIEW_MTIME, REVIEW_MTIME))
            written.append(name)
            total_vertices += len(vertices)
        except Exception as error:  # per-asset containment is intentional
            failures.append({"anonymous_name": name, "error": " ".join(str(error).split())[:200]})

    protocol_path = output_dir / "REVIEW_PROTOCOL.md"
    protocol_path.write_text(PROTOCOL, encoding="utf-8")

    summary: dict[str, Any] = {
        "output": str(output_dir),
        "protocol": str(protocol_path),
        "requested": len(sheet),
        "written": len(written),
        "total_vertices": total_vertices,
        "stale_files_removed": len(stale),
        "missing_from_manifest": missing[:10],
        "missing_count": len(missing),
        "failures": failures[:10],
        "failure_count": len(failures),
    }
    return output_dir, summary
