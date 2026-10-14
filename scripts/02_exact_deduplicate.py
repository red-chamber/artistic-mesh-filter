import sys

from mesh_filter.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["deduplicate", *sys.argv[1:]]))
