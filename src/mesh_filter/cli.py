from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mesh_filter.config import load_project_config
from mesh_filter.deduplicate import mark_duplicates
from mesh_filter.evaluation.blind_audit import build_audit_gold
from mesh_filter.evaluation.matched_test import build_matched_test
from mesh_filter.evaluation.review_kit import prepare_review
from mesh_filter.evaluation.screening_kit import build_screening_review
from mesh_filter.manifest import build_manifest
from mesh_filter.models.count_baseline import train_count_baseline
from mesh_filter.probe import probe_sources
from mesh_filter.screening import run_screening
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
    validate.add_argument(
        "--artistic-config",
        type=Path,
        default=Path("configs/features/artistic_v1.yaml"),
    )

    add_config(
        "deduplicate",
        "Group exact and near duplicates (needs `validate` to have run)",
    )

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

    matched = add_config(
        "build-matched-test",
        "Pair both labels inside the same face-count bins",
    )
    matched.add_argument(
        "--pool",
        nargs="+",
        default=None,
        help="Splits to draw from; defaults to matched_test.pool in the config.",
    )
    matched.add_argument("--max-per-bin-per-label", type=int, default=None)

    audit = add_config("build-audit-gold", "Draw the blind human-review sample")
    audit.add_argument("--size", type=int, default=None)

    review = add_config(
        "prepare-review",
        "Write anonymised meshes so the audit can be judged in a 3D viewer",
    )
    review.add_argument("--limit", type=int, default=None, help="Only the first N rows")

    screen = add_config("screen", "Apply the screening rules and rank the corpus")
    screen.add_argument(
        "--rules",
        type=Path,
        default=Path("configs/screening/artistic_v1.yaml"),
    )

    screen_review = add_config(
        "build-screening-review",
        "Sample the screening verdicts into a correction sheet with anonymised meshes",
    )
    screen_review.add_argument("--per-level", type=int, default=15)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_project_config(args.config)

    if args.command == "build-manifest":
        output, count = build_manifest(config)
        print(json.dumps({"output": str(output), "asset_count": count}, ensure_ascii=False))
    elif args.command == "validate":
        output, counts = validate_manifest(
            config,
            config.resolve(args.feature_config),
            config.resolve(args.artistic_config),
        )
        print(json.dumps({"output": str(output), **counts}, ensure_ascii=False))
    elif args.command == "deduplicate":
        output, report = mark_duplicates(config)
        print(json.dumps(report, ensure_ascii=False))
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
    elif args.command == "build-matched-test":
        output, report = build_matched_test(config, args.pool, args.max_per_bin_per_label)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "pool": report["pool"],
                    "pool_size": report["pool_size"],
                    "pairs": report["pairs"],
                    "assets": report["assets"],
                    "dropped_bins": report["dropped_bins"],
                    "face_count_selected": report["population"]["selected"]["face_count"],
                    "face_count_pool": report["population"]["pool"]["face_count"],
                },
                ensure_ascii=False,
            )
        )
    elif args.command == "build-audit-gold":
        output, report = build_audit_gold(config, args.size)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "key_output": report["key_output"],
                    "requested": report["requested"],
                    "sampled": report["sampled"],
                    "population": report["population"],
                    "strata": report["strata"],
                },
                ensure_ascii=False,
            )
        )
    elif args.command == "prepare-review":
        output, report = prepare_review(config, args.limit)
        print(json.dumps(report, ensure_ascii=False))
    elif args.command == "build-screening-review":
        output, report = build_screening_review(config, args.per_level)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "key_output": report["key_output"],
                    "review_dir": report["review_dir"],
                    "levels": report["levels"],
                    "available": report["available"],
                    "sampled": report["sampled"],
                    "written": report["kit"]["written"],
                    "mesh_failures": report["kit"]["failure_count"],
                },
                ensure_ascii=False,
            )
        )
    elif args.command == "screen":
        output, report = run_screening(config, config.resolve(args.rules))
        print(
            json.dumps(
                {
                    "output": str(output),
                    "report": report["report"],
                    "assets": report["assets"],
                    "verdicts": report["verdicts"],
                    "rule_failures": {
                        rule["name"]: {
                            source: counts["fail"]
                            for source, counts in rule["counts"].items()
                        }
                        for rule in report["rules"]
                    },
                    "flag_score": report["flag_score"],
                },
                ensure_ascii=False,
            )
        )
    elif args.command == "train-count-baseline":
        output, metrics = train_count_baseline(config, config.resolve(args.experiment))
        print(json.dumps({"output": str(output), "metrics": metrics}, ensure_ascii=False))
    else:  # pragma: no cover
        raise AssertionError(f"Unhandled command: {args.command}")
    return 0
