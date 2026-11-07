"""Read-only baseline verification and traceable transportation evidence."""
import argparse
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path('.'))
    parser.add_argument('--protected-workspace', type=Path)
    args = parser.parse_args()
    from pebr_review7.engine import execute
    return execute(args.project_root, args.protected_workspace)

if __name__ == '__main__':
    raise SystemExit(main())
