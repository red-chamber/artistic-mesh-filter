"""Screen a folder of meshes on its own — the deliverable form of the rules.

    python scripts/11_filter.py --input <mesh directory> --output results.csv

No project config or pipeline is needed: it scans the folder, computes the
features each rule reads, applies `configs/screening/artistic_v1.yaml` and
writes one row per mesh with its verdict and the rules it failed.
"""

import sys

from mesh_filter.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["filter", *sys.argv[1:]]))
