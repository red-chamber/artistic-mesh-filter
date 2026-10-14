from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mesh_filter.config import load_project_config
from mesh_filter.deduplicate import mark_exact_duplicates
from mesh_filter.manifest import build_manifest
from mesh_filter.models.count_baseline import train_count_baseline
from mesh_filter.probe import probe_sources
from mesh_filter.smoke import run_smoke_test
from mesh_filter.split import create_splits
from mesh_filter.validate import validate_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mesh-filter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config(command: str, help_text: str) -> argparse.ArgumentParser:
        subparser = subparsers.add_parser(command, help=help_text)
        subparser.add_argument("--config", required=True, type=Path)
        return subparser

    add_config("build-manifest", "Scan configured read-only sources")

    validate = add_config("validate", "Parse assets and extract basic features")
    validate.add_argument(
        "--feature-config",
        type=Path,
        default=Path("configs/features/basic_v1.yaml"),
    )

    add_config("deduplicate", "Mark exact duplicates using SHA-256")

    split = add_config("split", "Create deterministic group-isolated splits")
    split.add_argument("--split-config", type=Path, default=None)

    probe = add_config("probe", "Inspect real formats and asset structure (read-only)")
    probe.add_argument("--sample-size", type=int, default=None)

    smoke = add_config("smoke", "Run the pre-full-pass smoke test on real data")
    smoke.add_argument("--sample-size", type=int, default=None)
    smoke.add_argument(
        "--feature-config",
        type=Path,
        default=Path("configs/features/basic_v1.yaml"),
    )

    train = add_config("train-count-baseline", "Train the count-only baseline")
    train.add_argument(
        "--experiment",
        type=Path,
        default=Path("configs/experiments/count_baseline.yaml"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_project_config(args.config)

    if args.command == "build-manifest":
        output, count = build_manifest(config)
        print(json.dumps({"output": str(output), "asset_count": count}, ensure_ascii=False))
    elif args.command == "validate":
        output, counts = validate_manifest(config, config.resolve(args.feature_config))
        print(json.dumps({"output": str(output), **counts}, ensure_ascii=False))
    elif args.command == "deduplicate":
        output, duplicate_assets = mark_exact_duplicates(config)
        print(
            json.dumps(
                {"output": str(output), "assets_in_duplicate_groups": duplicate_assets},
                ensure_ascii=False,
            )
        )
    elif args.command == "split":
        split_config = config.resolve(args.split_config) if args.split_config else None
        output, counts = create_splits(config, split_config)
        print(json.dumps({"output": str(output), **counts}, ensure_ascii=False))
    elif args.command == "probe":
        output, report = probe_sources(config, args.sample_size)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "sources": {
                        name: {
                            "file_count": entry["file_count"],
                            "ingestable_count": entry["ingestable_count"],
                            "sampled_count": entry["sampled_count"],
                            "parse_rate": entry["aggregate"].get("parse_rate"),
                            "unconfigured_extensions": entry["unconfigured_extensions"],
                            "notes": entry["notes"],
                        }
                        for name, entry in report["sources"].items()
                    },
                },
                ensure_ascii=False,
            )
        )
    elif args.command == "smoke":
        output, report = run_smoke_test(
            config,
            args.sample_size,
            config.resolve(args.feature_config),
        )
        print(
            json.dumps(
                {
                    "output": str(output),
                    "verdict": report["verdict"],
                    "sources": {
                        name: {
                            "ingestable_count": summary["ingestable_count"],
                            "sampled": summary["sampled"],
                            "parsed": summary["parsed"],
                            "failed": summary["failed"],
                            "max_seconds": summary["max_seconds"],
                            "selection_limit_violations": summary[
                                "selection_limit_violation_count"
                            ],
                            "estimated_full_pass_seconds": summary[
                                "estimated_full_pass_seconds"
                            ],
                        }
                        for name, summary in report["sources"].items()
                    },
                    "failed_checks": [
                        check["detail"]
                        for check in report["checks"]
                        if check["status"] == "fail"
                    ],
                    "quarantine_buckets": {
                        bucket: entry["count"]
                        for bucket, entry in report["quarantine_plan"].items()
                    },
                },
                ensure_ascii=False,
            )
        )
        return 0 if report["verdict"] == "pass" else 1
    elif args.command == "train-count-baseline":
        output, metrics = train_count_baseline(config, config.resolve(args.experiment))
        print(json.dumps({"output": str(output), "metrics": metrics}, ensure_ascii=False))
    else:  # pragma: no cover
        raise AssertionError(f"Unhandled command: {args.command}")
    return 0
