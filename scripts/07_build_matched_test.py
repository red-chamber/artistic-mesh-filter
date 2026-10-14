"""Build the face-count-matched test set.

Run after `split`. The output is a diagnostic subset of the frozen pool in which
both labels share the same face-count bins, so a score on it cannot come from
"this class simply has fewer polygons".
"""

import sys

from mesh_filter.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["build-matched-test", *sys.argv[1:]]))
