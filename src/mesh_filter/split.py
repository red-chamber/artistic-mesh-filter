from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from mesh_filter.config import ProjectConfig, load_yaml
from mesh_filter.manifest import read_csv, write_csv


def _assign_groups(
    grouped_rows: dict[str, list[dict[str, str]]],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, str]:
    assignments: dict[str, str] = {}
    groups_by_label: dict[str, list[tuple[str, list[dict[str, str]]]]] = defaultdict(list)
    for group_id, rows in grouped_rows.items():
        labels = {row["label"] for row in rows}
        if len(labels) != 1:
            raise ValueError(f"Duplicate group {group_id} contains conflicting labels: {labels}")
        groups_by_label[next(iter(labels))].append((group_id, rows))

    for label, groups in sorted(groups_by_label.items()):
        generator = random.Random(f"{seed}:{label}")
        generator.shuffle(groups)
        total = sum(len(rows) for _, rows in groups)
        target = {name: ratio * total for name, ratio in ratios.items()}
        current = {name: 0 for name in ratios}

        for group_id, rows in groups:
            split_name = max(
                ratios,
                key=lambda name: (target[name] - current[name], ratios[name], name),
            )
            assignments[group_id] = split_name
            current[split_name] += len(rows)
    return assignments


def _ensure_split_coverage(
    assignments: dict[str, str],
    grouped_rows: dict[str, list[dict[str, str]]],
    ratios: dict[str, float],
) -> dict[str, str]:
    """Give every split with a positive ratio at least one group.

    With a handful of smoke-test models the greedy fill can starve a split
    (`test` ends up empty), which makes the smoke test unable to prove the split
    stage works. Moving the smallest group out of the largest split changes
    nothing once the sources are large enough to fill every split naturally.
    """
    counts: dict[str, int] = {name: 0 for name in ratios}
    for split_name in assignments.values():
        counts[split_name] = counts.get(split_name, 0) + 1

    for split_name in sorted(ratios):
        if ratios[split_name] <= 0 or counts.get(split_name, 0):
            continue
        donors = [name for name in ratios if counts.get(name, 0) > 1]
        if not donors:
            continue
        donor = max(donors, key=lambda name: (counts[name], name))
        candidates = [group for group, name in assignments.items() if name == donor]
        moved = min(candidates, key=lambda group: (len(grouped_rows[group]), group))
        assignments[moved] = split_name
        counts[donor] -= 1
        counts[split_name] = 1
    return assignments


def create_splits(
    config: ProjectConfig,
    split_config_path: str | Path | None = None,
) -> tuple[Path, dict[str, int]]:
    manifest_path = config.output_path("deduplicated_manifest")
    rows = read_csv(manifest_path)

    if split_config_path is not None:
        split_config: dict[str, Any] = load_yaml(split_config_path)
    else:
        split_config = config.values.get("split", {})

    seed = int(split_config.get("seed", 42))
    ratios = {
        "train": float(split_config.get("train_ratio", 0.70)),
        "val": float(split_config.get("val_ratio", 0.15)),
        "test": float(split_config.get("test_ratio", 0.15)),
    }
    if abs(sum(ratios.values()) - 1.0) > 1.0e-9 or any(value < 0 for value in ratios.values()):
        raise ValueError(f"Split ratios must be non-negative and sum to 1, got {ratios}")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        group_id = row.get("duplicate_group_id") or row["asset_id"]
        grouped[group_id].append(row)
    assignments = _assign_groups(grouped, ratios, seed)
    assignments = _ensure_split_coverage(assignments, grouped, ratios)

    output_root = config.output_path("split_root")
    output_root.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[dict[str, str]]] = {name: [] for name in ratios}
    all_rows: list[dict[str, str]] = []
    for row in rows:
        group_id = row.get("duplicate_group_id") or row["asset_id"]
        row["split"] = assignments[group_id]
        by_split[row["split"]].append(row)
        all_rows.append(row)

    fields = list(all_rows[0]) if all_rows else []
    for name, split_rows in by_split.items():
        write_csv(output_root / f"{name}.csv", split_rows, fields)
    write_csv(output_root / "all.csv", all_rows, fields)
    return output_root, {name: len(split_rows) for name, split_rows in by_split.items()}
