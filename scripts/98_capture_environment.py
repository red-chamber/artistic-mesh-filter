"""Capture the environment that produced a run.

Every remote run should ship this file alongside its metrics: two runs with the
same config are only comparable if the interpreter, key libraries, git commit
and accelerator match.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

PACKAGES = (
    "numpy",
    "PyYAML",
    "trimesh",
    "scikit-learn",
    "pandas",
    "pyarrow",
    "torch",
)


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def git_state() -> dict[str, object]:
    def run(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
    }


def accelerator() -> dict[str, object]:
    try:
        import torch
    except ImportError:
        return {"torch_available": False}
    return {
        "torch_available": True,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the JSON report here instead of stdout.",
    )
    args = parser.parse_args()

    report = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in PACKAGES},
        "git": git_state(),
        "accelerator": accelerator(),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
