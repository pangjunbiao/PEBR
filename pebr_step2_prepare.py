#!/usr/bin/env python3
"""Prepare the prespecified PEBR revision without running corpus experiments."""
from pathlib import Path
import argparse
import sys
sys.dont_write_bytecode = True


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"): stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--closeout-dir", type=Path, help="Optional explicit passed step1-closeout directory")
    options = parser.parse_args(argv)
    from pebr_revision_step2.workspace import prepare
    try:
        return prepare(options.project_root, options.closeout_dir)
    except Exception as error:
        print("STEP 2 COULD NOT START | " + type(error).__name__ + ": " + str(error), file=sys.stderr)
        print("Existing experiments have not been retrained or replaced.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
