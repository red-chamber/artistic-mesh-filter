"""Group duplicate assets so no model can span two splits.

Two kinds of duplication are merged here:

- exact: identical file bytes, found by SHA-256, no parsing needed;
- near: the same geometry re-exported with reordered vertices, a different unit,
  a different rotation or a mirror image, found by the invariant fingerprints in
  `mesh_filter.fingerprint`.

Both end up in the same `duplicate_group_id`, which the split stage treats as
atomic. Merging is deliberately biased towards over-grouping: a false merge costs
a little training data, a missed merge leaks a model across splits and silently
inflates the evaluation.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from pathlib import Path
from typing import Any

from mesh_filter.config import ProjectConfig
from mesh_filter.fingerprint import (
    DEFAULT_TOLERANCE,
    load_fingerprints,
    signatures_match,
    write_fingerprints,
)
from mesh_filter.manifest import MANIFEST_FIELDS, read_csv, write_csv

DEFAULT_MAX_BUCKET_SIZE = 200


class _UnionFind:
    """Union-find whose root is always the smallest id, so results are stable."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self._parent.setdefault(item, item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        smaller, larger = sorted((left_root, right_root))
        self._parent[larger] = smaller


def _settings(config: ProjectConfig) -> dict[str, Any]:
    settings = config.values.get("deduplicate", {})
    return {
        "near_duplicate": bool(settings.get("near_duplicate", True)),
        "tolerance": float(settings.get("tolerance", DEFAULT_TOLERANCE)),
        "max_bucket_size": int(settings.get("max_bucket_size", DEFAULT_MAX_BUCKET_SIZE)),
    }


def _link_exact_groups(rows: list[dict[str, str]], groups: _UnionFind) -> int:
    by_hash: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row.get("sha256"):
            by_hash[row["sha256"]].append(row["asset_id"])
    for members in by_hash.values():
        for member in members[1:]:
            groups.union(members[0], member)
    return sum(len(members) for members in by_hash.values() if len(members) > 1)


def _link_near_groups(
    config: ProjectConfig,
    rows: list[dict[str, str]],
    groups: _UnionFind,
    tolerance: float,
    max_bucket_size: int,
) -> dict[str, Any]:
    fingerprints = load_fingerprints(config)
    known = {row["asset_id"] for row in rows}
    fingerprints = [entry for entry in fingerprints if entry["asset_id"] in known]

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in fingerprints:
        buckets[entry["bucket_key"]].append(entry)

    linked_pairs = 0
    oversized: list[dict[str, Any]] = []
    for key in sorted(buckets):
        members = sorted(buckets[key], key=lambda entry: entry["asset_id"])
        if len(members) > max_bucket_size:
            # Comparing every pair here would be quadratic on a bucket this large,
            # which usually means the bucket key is too coarse for this data.
            oversized.append({"bucket_key": key, "size": len(members)})
            continue
        for left, right in itertools.combinations(members, 2):
            if signatures_match(left, right, tolerance):
                groups.union(left["asset_id"], right["asset_id"])
                linked_pairs += 1
    return {
        "fingerprinted": len(fingerprints),
        "evaluated_buckets": len(buckets) - len(oversized),
        "linked_pairs": linked_pairs,
        "oversized_buckets": oversized[:20],
        "oversized_bucket_count": len(oversized),
    }


def mark_duplicates(config: ProjectConfig) -> tuple[Path, dict[str, Any]]:
    rows = read_csv(config.output_path("manifest"))
    settings = _settings(config)

    groups = _UnionFind()
    for row in rows:
        groups.find(row["asset_id"])

    exact_assets = _link_exact_groups(rows, groups)

    near_report: dict[str, Any] = {"fingerprinted": 0, "linked_pairs": 0}
    if settings["near_duplicate"]:
        fingerprint_output, fingerprint_summary = write_fingerprints(config)
        near_report = _link_near_groups(
            config,
            rows,
            groups,
            settings["tolerance"],
            settings["max_bucket_size"],
        )
        near_report["fingerprint_output"] = str(fingerprint_output)
        near_report["candidate_stats"] = {
            key: value for key, value in fingerprint_summary.items() if key != "failures"
        }
        near_report["fingerprint_failures"] = fingerprint_summary["failures"]

    by_root: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_root[groups.find(row["asset_id"])].append(row)

    duplicate_groups = 0
    assets_in_groups = 0
    near_groups = 0
    for root in sorted(by_root):
        members = by_root[root]
        hashes = {row["sha256"] for row in members if row.get("sha256")}
        if len(members) > 1 and len(hashes) > 1:
            group_id = f"dup-{min(row['asset_id'] for row in members)}"
            near_groups += 1
        elif hashes:
            group_id = f"exact-{sorted(hashes)[0][:16]}"
        else:
            group_id = members[0]["asset_id"]
        for row in members:
            row["duplicate_group_id"] = group_id
        if len(members) > 1:
            duplicate_groups += 1
            assets_in_groups += len(members)

    output = config.output_path("deduplicated_manifest")
    write_csv(output, rows, MANIFEST_FIELDS)

    summary: dict[str, Any] = {
        "output": str(output),
        "assets": len(rows),
        "exact_duplicate_assets": exact_assets,
        "near_duplicate": settings["near_duplicate"],
        "tolerance": settings["tolerance"],
        "duplicate_groups": duplicate_groups,
        "assets_in_duplicate_groups": assets_in_groups,
        "groups_merged_by_near_duplicate": near_groups,
        **near_report,
    }
    return output, summary
