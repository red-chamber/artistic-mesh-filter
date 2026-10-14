"""Group exact and near duplicates before the split is frozen.

Exact grouping uses SHA-256. Near grouping re-parses the assets that share a
(vertex, face, component) bucket and compares rotation-, scale- and
order-invariant fingerprints, so a re-exported copy cannot land in another split.
"""

import sys

from mesh_filter.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["deduplicate", *sys.argv[1:]]))
