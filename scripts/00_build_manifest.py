import sys

from mesh_filter.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["build-manifest", *sys.argv[1:]]))
