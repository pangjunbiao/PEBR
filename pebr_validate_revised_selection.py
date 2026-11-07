"""Validate the frozen revised PEBR rule on four fresh selection-mask banks."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path('.'))
    parser.add_argument('--step3-workspace', type=Path)
    parser.add_argument('--candidate-workspace', type=Path)
    parser.add_argument('--mode', choices=('run', 'preflight'), default='run')
    args = parser.parse_args()
    from pebr_fresh_validation.engine import execute
    return execute(args.project_root, args.step3_workspace,
                   args.candidate_workspace, args.mode)

if __name__ == '__main__':
    raise SystemExit(main())
