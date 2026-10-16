"""Apply the screening rules to the feature table and rank the corpus.

The rules and their provenance live in `configs/screening/artistic_v1.yaml`.
Nothing here is a final filter yet: the `reject` rules gate, the `flag` rules
only rank, and the extremes go to a human before any threshold moves.
"""

import sys

from mesh_filter.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["screen", *sys.argv[1:]]))
