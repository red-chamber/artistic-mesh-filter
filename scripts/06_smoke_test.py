"""Pre-full-pass smoke test on a small deterministic sample of real data.

Exits non-zero when the verdict is `fail`, so it can gate the full run in a
shell pipeline. The JSON report is the artifact to send back for review.
"""

import sys

from mesh_filter.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["smoke", *sys.argv[1:]]))
