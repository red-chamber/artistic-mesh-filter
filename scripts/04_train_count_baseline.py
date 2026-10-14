import sys

from mesh_filter.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train-count-baseline", *sys.argv[1:]]))
