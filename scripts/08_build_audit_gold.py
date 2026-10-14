"""Draw the blind human-review sample.

Writes two files: the blind sheet the reviewer opens, and the key that maps back
to source and proxy label. The key must not be opened by the reviewer.
"""

import sys

from mesh_filter.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["build-audit-gold", *sys.argv[1:]]))
