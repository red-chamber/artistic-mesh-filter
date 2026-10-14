from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from mesh_filter.config import ProjectConfig
from mesh_filter.manifest import MANIFEST_FIELDS, read_csv, write_csv


def mark_exact_duplicates(config: ProjectConfig) -> tuple[Path, int]:
    rows = read_csv(config.output_path("manifest"))
    by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        sha256 = row.get("sha256", "")
        key = sha256 if sha256 else f"asset:{row['asset_id']}"
        by_hash[key].append(row)

    duplicate_assets = 0
    for key, group in by_hash.items():
        if key.startswith("asset:"):
            group_id = group[0]["asset_id"]
        else:
            group_id = f"exact-{key[:16]}"
        for row in group:
            row["duplicate_group_id"] = group_id
        if len(group) > 1:
            duplicate_assets += len(group)

    output = config.output_path("deduplicated_manifest")
    write_csv(output, rows, MANIFEST_FIELDS)
    return output, duplicate_assets
