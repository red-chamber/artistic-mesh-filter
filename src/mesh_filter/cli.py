from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mesh_filter.config import load_project_config
from mesh_filter.deduplicate import mark_exact_duplicates
from mesh_filter.manifest import build_manifest
from mesh_filter.models.count_baseline import train_count_baseline
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
    elif args.command == "train-count-baseline":
        output, metrics = train_count_baseline(config, config.resolve(args.experiment))
        print(json.dumps({"output": str(output), "metrics": metrics}, ensure_ascii=False))
    else:  # pragma: no cover
        raise AssertionError(f"Unhandled command: {args.command}")
    return 0
