"""Third calibration round: four arms aimed at what rounds 1-2 exposed.

    python scripts/12_build_round3_review.py --config configs/paths.remote.yaml

Design (2026-10-16), following the plan sketched in docs/OPEN_QUESTIONS.md:

- `clean_high_*` (12): 判定为 clean 但面数 ≥2000 的四段（2/4/4/2）。第一轮的教训是
  "面多且看得出冗余才坏"，而"看得出"正是现有列量不出来的东西——漏放风险最大的角落。
- `low_/high_<column>` (16): 四列已算好、从未进过规则的特征（度数熵、一环边长变异系数、
  边长方向各向异性、邻边方向一致性）在 clean/flagged 池（≥1000 面）里的两端极值
  （q05 以下 / q95 以上）各两个。能预测改判就升成规则，不能就删。
- `share_*` (12): 占比边界带五段 [0.05,0.08) 至 [0.18,0.25)。0.15 是分量最重的门槛，
  而 0.13–0.15 一带的人工边界自相矛盾（A0036 vs A0022），专门复核两侧。
- `regular_smooth_candidates` (8): share<0.05、度数众数 ≥0.6、等边度 ≥0.75、≥1500 面，
  按度数众数从大到小抽前 3×8 个里的 8 个——A0029 一族（规则铺面但 patch 度量隐形）
  的廉价代理。

已经被人看过的资产（一二轮样本 + 200 条盲审，共 283 个）全部排除，不会让人看第二遍。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from mesh_filter.config import load_project_config
from mesh_filter.evaluation.targeted_kit import TargetArm, build_targeted_review
from mesh_filter.manifest import read_csv
from mesh_filter.screening import screening_paths

CONFIG_DEFAULT = "configs/paths.remote.yaml"

UNUSED_COLUMNS = [
    "vertex_degree_entropy",
    "one_ring_edge_length_cv_median",
    "local_edge_direction_anisotropy_mean",
    "neighboring_direction_consistency_mean",
]

NOTES_ROUND3 = """# 筛选校准（第三轮）：怎么填这张表

和第一、二轮一样——**机器判定是给你看的**，你的任务是纠错，不是重新判断一遍。

1. `screening_review.csv` 每一行对应同目录的一个 `A####.obj`。`machine_label` 是脚本
   给出的判定：`accept`（=clean，全过）、`flag`（有未校准的规则没过）、`reject`（硬门槛
   没过）。
2. 用 VSCode 的 3D Viewer 打开模型（先线框后实体，可以旋转），和 `machine_label` 比对：
   - 认为这个判定**明显不合适** → 在 `my_label` 填你的标签：`accept` = 1、`reject` = 2、
     `flag` = 3；
   - **同意，或拿不准** → 填 0。留空按 0 处理，但尽量显式填 0。
3. 只写不同意的行即可，不必逐行重判。
4. 判据（按课题组口径）：
   - 面数与顶点数不能太大；
   - 面的形状：接近等边本身不是问题，**为了形状规则在平坦处铺冗余的面才是问题**；
   - 顶点分布整齐有规律；平坦处稀疏、转折处密集。
5. 本轮样本**不是按等级均匀抽取**，而是按若干针对性问题抽样：高面数的"干净"样本、
   冗余占比的边界带、以及几列"算了但从没用过"的特征的极值。各判定档在样本里的比例
   会与语料不同，这是设计使然——判断时只按模型本身和判据，不必考虑"它为什么被抽中"。
   抽取记录在对照表里，**不要打开**上一层的 `screening_review_key.csv`。
6. 两个必须避免的错误（同前）：不要按"这是什么物体"判断，也不要按"好不好看"判断；
   面少但边流干净应 `accept`，面多但随机铺满应 `reject`。
"""


def _number(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return math.nan


def _quantile(values: list[float], probability: float) -> float:
    finite = [value for value in values if not math.isnan(value)]
    return float(np.quantile(finite, probability))


def _member(ids: set[str]):
    return lambda screening, feature, ids=ids: feature.get("asset_id") in ids


def build_arms(
    rows: list[dict[str, str]],
    features: dict[str, dict[str, str]],
    excluded: set[str],
) -> list[TargetArm]:
    """This round's four questions, as pools. Pure: no sampling here."""

    def available(row: dict[str, str]) -> bool:
        return row["asset_id"] not in excluded

    def feature_of(row: dict[str, str]) -> dict[str, str]:
        return features.get(row["asset_id"], {})

    def number(row: dict[str, str], key: str) -> float:
        return _number(feature_of(row), key)

    arms: list[TargetArm] = []

    # A — clean verdicts at high face counts, four bins.
    for name, low, high, count in (
        ("clean_2000_4000", 2000, 4000, 2),
        ("clean_4000_8000", 4000, 8000, 4),
        ("clean_8000_16000", 8000, 16000, 4),
        ("clean_16000_plus", 16000, 20001, 2),
    ):
        ids = {
            row["asset_id"]
            for row in rows
            if available(row)
            and row["verdict"] == "clean"
            and low <= number(row, "face_count") < high
        }
        arms.append(TargetArm(name, count, _member(ids)))

    # B — both tails of each computed-but-unused column.
    pool_b = [
        row
        for row in rows
        if available(row)
        and row["verdict"] in ("clean", "flagged")
        and number(row, "face_count") >= 1000
    ]
    for column in UNUSED_COLUMNS:
        values = [_number(feature_of(row), column) for row in pool_b]
        low = _quantile(values, 0.05)
        high = _quantile(values, 0.95)
        low_ids = {row["asset_id"] for row in pool_b if number(row, column) <= low}
        high_ids = {row["asset_id"] for row in pool_b if number(row, column) >= high}
        arms.append(TargetArm(f"low_{column}", 2, _member(low_ids)))
        arms.append(TargetArm(f"high_{column}", 2, _member(high_ids)))

    # C — the redundancy boundary band, five segments around the 0.15 gate.
    for low, high, count in (
        (0.05, 0.08, 2),
        (0.08, 0.12, 2),
        (0.12, 0.15, 3),
        (0.15, 0.18, 3),
        (0.18, 0.25, 2),
    ):
        ids = {
            row["asset_id"]
            for row in rows
            if available(row) and low <= number(row, "redundant_face_share") < high
        }
        arms.append(TargetArm(f"share_{low:.2f}_{high:.2f}", count, _member(ids)))

    # D — regular-and-smooth candidates: the A0029 family's cheap proxy.
    ids = {
        row["asset_id"]
        for row in rows
        if available(row)
        and row["verdict"] in ("clean", "flagged")
        and number(row, "face_count") >= 1500
        and number(row, "redundant_face_share") < 0.05
        and number(row, "vertex_degree_mode_share") >= 0.6
        and number(row, "equilateral_median") >= 0.75
    }
    arms.append(
        TargetArm(
            "regular_smooth_candidates",
            8,
            _member(ids),
            rank=lambda screening, feature: -_number(feature, "vertex_degree_mode_share"),
        )
    )
    return arms


def _exclusion_ids(config) -> set[str]:
    """Assets a human has already judged: rounds 1-2 samples and the blind audit."""
    root = screening_paths(config)[0].parent
    ids: set[str] = set()
    for key_path in (
        root / "round1" / "screening_review_key.csv",
        root / "round2" / "screening_review_key.csv",
        config.output_path("audit_gold_key"),
    ):
        if Path(key_path).exists():
            ids |= {
                row["asset_id"] for row in read_csv(key_path) if row.get("asset_id")
            }
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG_DEFAULT)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    config = load_project_config(args.config)
    rows = read_csv(screening_paths(config)[0])
    features = {
        row["asset_id"]: row for row in read_csv(config.output_path("features"))
    }
    excluded = _exclusion_ids(config)
    arms = build_arms(rows, features, excluded)
    _, report = build_targeted_review(
        config, arms, exclude_ids=excluded, notes=NOTES_ROUND3, seed=args.seed
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
