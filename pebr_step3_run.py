"""Run the default-bank comparison from the accepted, isolated Step 2 inputs."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
    os.environ[name] = '1'
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path('.'))
    parser.add_argument('--step2-workspace', type=Path, default=None)
    parser.add_argument('--mode', choices=('reproduce', 'compare'), default='compare')
    args = parser.parse_args()
    from pebr_revision_step3.runner import execute
    return execute(args.project_root, args.step2_workspace, args.mode)

if __name__ == '__main__':
    raise SystemExit(main())
