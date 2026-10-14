from __future__ import annotations

import json
import platform
import sys
from importlib import metadata


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in ("numpy", "PyYAML", "trimesh", "scikit-learn", "torch")
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
