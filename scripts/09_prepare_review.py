"""Write the anonymised meshes a reviewer opens, one per audit row.

The output directory is self-contained: files named by anonymous ID, one shared
timestamp, one minimal format, plus the review protocol. The audit key stays
behind and must not be opened by whoever does the review.
"""

import sys

from mesh_filter.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["prepare-review", *sys.argv[1:]]))
