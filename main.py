"""Single entry point. Run this file in the selected PyCharm interpreter."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="backslashreplace")
            except (OSError, ValueError):
                pass
    if sys.version_info < (3, 10):
        print("ERROR: Python 3.10 or newer is required. Selected interpreter: " + sys.executable
              + " (" + sys.version.split()[0] + "). Select the Python 3.10+ interpreter that passed Execution Foundation in PyCharm Project Interpreter and Run Configuration.", file=sys.stderr)
        return 2
    root = Path(__file__).resolve().parent
    for folder in (root, root / "src"):
        sys.path.insert(0, str(folder))
    parser = argparse.ArgumentParser(description="Proposed transport-topic project")
    parser.add_argument("--config", help="Project-relative YAML overlay; default: configs/project.yaml")
    parser.add_argument("--mode", choices=("check", "run", "report"), default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    installation = parser.add_mutually_exclusive_group()
    installation.add_argument("--install-missing", dest="install_missing", action="store_true")
    installation.add_argument("--no-install", dest="install_missing", action="store_false")
    parser.set_defaults(install_missing=True)
    parser.add_argument("--resume", metavar="RUN_ID", help="Validate a compatible completed project run; ordinary interrupted receivers resume automatically")
    parser.add_argument("--run-id", help="Completed run to inspect with --mode report; defaults to latest")
    args = parser.parse_args()
    if args.run_id and args.mode != "report":
        parser.error("--run-id requires --mode report")
    if args.resume and args.mode == "report":
        parser.error("--resume cannot be combined with --mode report")
    from transport_topics.pipeline import execute
    return execute(root, args)


if __name__ == "__main__":
    raise SystemExit(main())
