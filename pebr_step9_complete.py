"""Complete the frozen revised-rule benchmarks, analyses and publication assets."""
import argparse
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=Path('.'))
    p.add_argument('--sensitivity-workspace',type=Path)
    p.add_argument('--phase',choices=('all','preflight','benchmarks','parameters','sizes','runtime','retrieval','assets'),default='all')
    a=p.parse_args()
    from pebr_complete9.engine import execute
    return execute(a.project_root,a.sensitivity_workspace,a.phase)
if __name__=='__main__':raise SystemExit(main())
