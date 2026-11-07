"""Replay the single fixed-cap candidate without the other development variants."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
from run_trial import Trial,read,write,digest,say


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dependencies',type=Path,default=None)
    parser.add_argument('--verify-existing',action='store_true')
    args=parser.parse_args();root=Path(__file__).resolve().parent
    if args.dependencies:sys.path.insert(0,str(args.dependencies.resolve()))
    sys.path[:0]=[str(root/'frozen_source'),str(root/'candidate')]
    import numpy as np
    from scipy import sparse
    from transport_topics.proposed.transfer import normalized_counts
    from revised_fixed_rule import select_and_transfer
    trial=Trial(root,root/'candidate_replay')
    reproduction=read(root/'run_v1/baseline_reproduction.json')
    assert reproduction['passed'] and len(reproduction['checks'])==28
    checks=[]
    for corpus in trial.protocol['corpora']:
        c=trial.collection(corpus);donors=normalized_counts(c['counts'])
        for k in trial.protocol['topic_counts']:
            refs=trial.references(corpus,k);projection=[r.projection(donors) for r in refs]
            p=trial.output/'selection'/corpus/('K%d'%k);p.mkdir(parents=True,exist_ok=True)
            expected_path=root/'forward_controls_v1/selection'/corpus/('K%d'%k)/'preserve_forward_fixed_cap/actions.json'
            expected=read(expected_path) if expected_path.exists() else None
            matrices=[];records=[];t=time.monotonic()
            for i in range(len(donors)):
                if c['eligible'][i] and np.any(c['candidates'][i]>=0):
                    f=int(c['folds'][i]);block=slice(i*20,(i+1)*20)
                    mixed,r=select_and_transfer(refs[f],c['visible'][block].toarray(),c['targets'][block].toarray(),
                        c['candidates'][i],donors,c['counts'][i],c['full'][i],projection[f])
                else:
                    mixed=donors[i];r={'donors':[],'offered_beta':0.,'actual_beta':0.,'available_donors':[],'selection_utility':0.}
                if expected is not None:
                    e=expected[i]
                    assert r['donors']==e['donors'] and r['available_donors']==e['available_donors']
                    assert abs(r['offered_beta']-e['offered_beta'])<1e-12 and abs(r['actual_beta']-e['actual_beta'])<1e-12
                matrices.append(sparse.csr_matrix(mixed[None]));records.append(r)
                if i==0 or i+1==len(donors) or time.monotonic()-t>15:
                    say('CANDIDATE',corpus,k,i+1,'/',len(donors));t=time.monotonic()
            matrix=sparse.vstack(matrices,format='csr');sparse.save_npz(p/'transferred.npz',matrix)
            write(p/'actions.json',records)
            if args.verify_existing:
                source=expected_path.parent/'transferred.npz'
                while not source.exists():
                    say('WAITING FOR COMPLETE EXISTING CANDIDATE BANK',corpus,k);time.sleep(5)
                assert (matrix-sparse.load_npz(source)).nnz==0
                checks.append({'corpus':corpus,'K':k,'actions':len(records),'transfer_matrix_identical':True})
            else:
                for seed in trial.protocol['final_model_seeds']:
                    trial.fit(corpus,k,'preserve_forward_fixed_cap',seed,matrix)
                trial.summarize()
    write(trial.output/'CANDIDATE_COMPLETE.json',{'mode':'kernel_verification' if args.verify_existing else 'fresh_candidate_replay',
        'kernel_sha256':digest(root/'revised_fixed_rule.py'),'selection_mask_base':2711,'cases':checks,
        'views':20,'fixed_positive_cap':1/3,'max_donors':3,'downside_weight':1.,
        'independent_mask_validation':False})
    say('CANDIDATE REPLAY COMPLETE')


if __name__=='__main__':main()
