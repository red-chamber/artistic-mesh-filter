"""Correction sheet for calibrating the screening thresholds.

The blind audit asked a human to judge meshes without knowing where they came
from. This kit asks a different question — "is the machine's verdict wrong?" —
so the machine's label is shown on purpose and the reviewer only writes when
they disagree. That is a much cheaper pass than judging from scratch, and it is
how the `flag` thresholds move: they stay recorded-not-gating until a human says
where the boundary actually sits.

The sample is stratified by source and by the rule that decided the verdict, so
every rule gets looked at from both sides instead of the corpus average being
reviewed once more. Anonymisation is the audit kit's: minimal OBJ, one
timestamp, no source and no file names in the mesh directory. The key that maps
anonymous names back to assets stays one directory up, outside the review
folder, and must not be opened.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from mesh_filter.config import ProjectConfig
from mesh_filter.evaluation.review_kit import prepare_review
from mesh_filter.manifest import read_csv, write_csv
from mesh_filter.screening import screening_paths

SHEET_FIELDS = ["anonymous_name", "asset_id", "machine_label", "my_label", "note"]

KEY_FIELDS = [
    "anonymous_name",
    "asset_id",
    "source_dataset",
    "label",
    "machine_label",
    "failed_rules",
    "unknown_rules",
    "flags_passed",
    "face_count",
]

# The reviewer's vocabulary; the screening table says clean/flagged/rejected.
VERDICT_TO_LABEL = {"clean": "accept", "flagged": "flag", "rejected": "reject"}
LEVELS = ("accept", "flag", "reject")

# Which rule decided the verdict, used to stratify the sample. First match wins,
# so the order is the priority; an unlisted rule lands in "other" and is still
# sampled, just without a stratum of its own.
REJECT_PRIORITY = (
    "narrow_faces",
    "face_count_range",
    "nonmanifold_edges",
    "degenerate_faces",
    "parsed",
)
FLAG_PRIORITY = (
    "redundant_vertices",
    "homogeneous_faces",
    "uniform_remesh_signature",
    "density_follows_curvature",
    "degree_concentration",
)

NOTES = """# 筛选校准：怎么填这张表

这次不是盲审——**机器判定是给你看的**。你的任务是纠错，不是重新判断一遍。

1. `screening_review.csv` 每一行对应同目录的一个 `A####.obj`。`machine_label` 是脚本
   给的判定：`accept`（=clean，全过）、`flag`（有未校准的规则没过）、`reject`（硬门槛没过）。
2. 用 VSCode 的 3D Viewer 打开模型（先线框后实体，可以旋转），和 `machine_label` 比对：
   - 认为这个判定**明显不合适** → 在 `my_label` 填你的标签：`accept` = 1、`reject` = 2、
     `flag` = 3；
   - **同意，或拿不准** → 填 0。留空按 0 处理，但尽量显式填 0。
3. 只写不同意的行即可，不必逐行重判——这张表要的是"机器错在哪"。
4. 判据（按课题组口径）：
   - 面数与顶点数不能太大；
   - 面的形状：接近等边本身不是问题，**为了形状规则在平坦处铺冗余的面才是问题**；
   - 顶点分布整齐有规律；平坦处稀疏、转折处密集。
5. 不要打开上一层的 `screening_review_key.csv`（对照表，含来源与资产 id）。看过它，
   这份校准就不再是独立证据。
6. 本轮样本三个等级各 15 个，顺序已打乱；判据说明与观察顺序另见同目录的
   `REVIEW_PROTOCOL.md`。
"""


def _primary_reason(failed: list[str], priority: tuple[str, ...]) -> str:
    for name in priority:
        if name in failed:
            return name
    return "other"


def plan_screening_review(
    config: ProjectConfig, per_level: int = 15, seed: int = 11
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Pick the sample. Returns (sheet rows, key rows, report). No files touched."""
    table_path, _ = screening_paths(config)
    if not table_path.exists():
        raise FileNotFoundError(
            f"Screening table not found: {table_path}. Run `screen` first."
        )
    rows = read_csv(table_path)
    features = {row["asset_id"]: row for row in read_csv(config.output_path("features"))}

    strata: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        level = VERDICT_TO_LABEL.get(row.get("verdict", ""))
        if level is None:
            continue
        failed = [name for name in row.get("failed_rules", "").split(";") if name]
        if level == "reject":
            reason = _primary_reason(failed, REJECT_PRIORITY)
        elif level == "flag":
            reason = _primary_reason(failed, FLAG_PRIORITY)
        else:
            reason = "none"
        strata.setdefault((level, row.get("source_dataset", "unknown"), reason), []).append(row)

    picked: list[dict[str, str]] = []
    strata_report: list[dict[str, Any]] = []
    for level in LEVELS:
        keys = sorted(key for key in strata if key[0] == level)
        order = sorted(keys, key=lambda key: (-len(strata[key]), key))
        quota = dict.fromkeys(order, 0)
        remaining = per_level
        while remaining > 0:
            progressed = False
            for key in order:
                if remaining <= 0:
                    break
                if quota[key] < len(strata[key]):
                    quota[key] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                break
        for key in order:
            items = sorted(strata[key], key=lambda row: row["asset_id"])
            chosen = random.Random(f"{seed}:screen-kit:{key[0]}:{key[1]}:{key[2]}").sample(
                items, quota[key]
            )
            strata_report.append(
                {
                    "level": key[0],
                    "source": key[1],
                    "reason": key[2],
                    "available": len(items),
                    "sampled": quota[key],
                }
            )
            picked.extend(chosen)

    random.Random(f"{seed}:screen-kit:anonymise").shuffle(picked)
    sheet_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for position, row in enumerate(picked, start=1):
        anonymous_name = f"A{position:04d}"
        label = VERDICT_TO_LABEL[row["verdict"]]
        sheet_rows.append(
            {
                "anonymous_name": anonymous_name,
                "asset_id": row["asset_id"],
                "machine_label": label,
                "my_label": "",
                "note": "",
            }
        )
        key_rows.append(
            {
                "anonymous_name": anonymous_name,
                "asset_id": row["asset_id"],
                "source_dataset": row.get("source_dataset", ""),
                "label": row.get("label", ""),
                "machine_label": label,
                "failed_rules": row.get("failed_rules", ""),
                "unknown_rules": row.get("unknown_rules", ""),
                "flags_passed": row.get("flags_passed", ""),
                "face_count": features.get(row["asset_id"], {}).get("face_count", ""),
            }
        )

    report: dict[str, Any] = {
        "levels": {
            level: sum(1 for row in picked if VERDICT_TO_LABEL[row["verdict"]] == level)
            for level in LEVELS
        },
        "available": {
            level: sum(1 for row in rows if VERDICT_TO_LABEL.get(row.get("verdict", "")) == level)
            for level in LEVELS
        },
        "per_level": per_level,
        "seed": seed,
        "sampled": len(picked),
        "strata": strata_report,
    }
    return sheet_rows, key_rows, report


def build_screening_review(
    config: ProjectConfig, per_level: int = 15, seed: int = 11
) -> tuple[Path, dict[str, Any]]:
    """Write the sheet, the key, the notes and the anonymised meshes."""
    sheet_rows, key_rows, report = plan_screening_review(config, per_level, seed)
    if not sheet_rows:
        raise FileNotFoundError("Nothing to review; the screening table is empty.")

    root = screening_paths(config)[0].parent
    review_dir = root / "review"
    sheet_path = review_dir / "screening_review.csv"
    key_path = root / "screening_review_key.csv"
    review_dir.mkdir(parents=True, exist_ok=True)
    write_csv(sheet_path, sheet_rows, SHEET_FIELDS)
    write_csv(key_path, key_rows, KEY_FIELDS)
    notes_path = review_dir / "SCREENING_REVIEW_NOTES.md"
    notes_path.write_text(NOTES, encoding="utf-8")

    _, kit = prepare_review(config, None, sheet_path=sheet_path, output_dir=review_dir)
    report.update(
        {
            "output": str(sheet_path),
            "key_output": str(key_path),
            "review_dir": str(review_dir),
            "notes": str(notes_path),
            "kit": kit,
        }
    )
    return sheet_path, report
