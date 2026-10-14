"""Read-only format and structure probe for the real sources.

Run this first on the data device, before any pipeline stage, and send the JSON
report back. It answers the questions synthetic fixtures cannot: which formats
are actually present, whether they parse, and what one asset contains.
"""

import sys

from mesh_filter.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["probe", *sys.argv[1:]]))
