"""Closest simpler-policy check under the SAME revised score and fixed cap.

Exploratory follow-up declared after the initial candidate outcomes. Evaluate
all six settings and all five final seeds, including a newly matched control.
"""
import argparse
from pathlib import Path
import sys
from run_trial import Trial,read,write,digest,say


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dependencies',type=Path,default=None)
    args=parser.parse_args();root=Path(__file__).resolve().parent
    if args.dependencies:sys.path.insert(0,str(args.dependencies.resolve()))
    sys.path[:0]=[str(root/'frozen_source'),str(root/'candidate')]
    import numpy as np
    from scipy import sparse
    from anchored_scoring import ReceiverAnchoredEvaluator
    from refined_search import rank_order
    from transport_topics.proposed.actions import BorrowingAction,NO_BORROWING
    from transport_topics.proposed.search import preferred
    from transport_topics.proposed.scoring import downside_utility
    from transport_topics.proposed.transfer import normalized_counts,transfer_action,check_transfer
    from transport_topics.experiments.sharing_controls import matched_action
    trial=Trial(root,root/'prefix_controls_v1')
    write(trial.output/'CONTROL_SPECIFICATION.json',{'source_sha256':digest(__file__),
        'scope':'development follow-up after the other candidate outcomes were examined; all settings retained',
        'ranking':'same mean reciprocal ranks over ordered masked-view lists as the old prefix control',
        'score':'same receiver-preserving score and downside utility as revised PEBR',
        'positive_cap':1/3,'final_seeds':[0,1,2,3,4],'selection_banks':1})
    policies=('preserve_fixed_prefix','matched_preserve_prefix')
    for corpus in trial.protocol['corpora']:
        c=trial.collection(corpus);donors=normalized_counts(c['counts'])
        for k in trial.protocol['topic_counts']:
            refs=trial.references(corpus,k);projection=[r.projection(donors) for r in refs]
            parent=trial.output/'selection'/corpus/('K%d'%k)
            if not (parent/'COMPLETE.json').exists():
                mats={p:[] for p in policies};records={p:[] for p in policies}
                for i in range(len(donors)):
                    chosen=NO_BORROWING;value=0.
                    if c['eligible'][i] and np.any(c['candidates'][i]>=0):
                        f=int(c['folds'][i]);b=slice(i*20,(i+1)*20);lists=c['candidates'][i]
                        ev=ReceiverAnchoredEvaluator(refs[f],c['visible'][b].toarray(),c['targets'][b].toarray(),
                            lists,donors,c['counts'][i].toarray(),projection[f],128)
                        order=rank_order(lists)
                        actions=[BorrowingAction(order[:s],1/3) for s in range(1,min(3,len(order))+1)]
                        gains=ev.gains(actions);utilities=downside_utility(gains,1.)
                        for action,score in zip(actions,utilities):
                            if preferred(action,float(score),chosen,value,0.,1e-8):chosen,value=action,float(score)
                    mixed,alpha,available=transfer_action(donors[i],chosen,c['full'][i],donors)
                    union=np.unique(c['candidates'][i][c['candidates'][i]>=0])
                    score=c['embeddings'][union]@c['queries'][i]
                    ranked=union[np.argsort(-score,kind='stable')].tolist()
                    matched=matched_action(chosen.donors,alpha,ranked,c['full'][i])
                    for p,a in zip(policies,(chosen,matched)):
                        q,beta,available=transfer_action(donors[i],a,c['full'][i],donors)
                        check_transfer(donors[i],q,beta,1/3)
                        mats[p].append(sparse.csr_matrix(q[None]));records[p].append({'donors':list(a.donors),
                            'offered_beta':a.beta,'actual_beta':beta,'available_donors':list(available)})
                    assert abs(records[policies[0]][-1]['actual_beta']-records[policies[1]][-1]['actual_beta'])<1e-12
                    assert len(records[policies[0]][-1]['available_donors'])==len(records[policies[1]][-1]['available_donors'])
                for p in policies:
                    target=parent/p;target.mkdir(parents=True,exist_ok=True)
                    sparse.save_npz(target/'transferred.npz',sparse.vstack(mats[p],format='csr'))
                    write(target/'actions.json',records[p])
                write(parent/'COMPLETE.json',{'policies':list(policies),'scope':'one existing selection bank'})
            for p in policies:
                learning=sparse.load_npz(parent/p/'transferred.npz').tocsr()
                for seed in trial.protocol['final_model_seeds']:
                    trial.fit(corpus,k,p,seed,learning);trial.summarize()
    assert len([r for r in trial.summarize() if r['policy'] in policies])==12
    write(trial.output/'PREFIX_COMPLETE.json',{'evaluated_seed_conditions':60,'all_declared_settings_retained':True,
        'independent_selection_validation':False})
    say('SAME-SCORE PREFIX COMPARISON COMPLETE','60 fresh fits')


if __name__=='__main__':main()
