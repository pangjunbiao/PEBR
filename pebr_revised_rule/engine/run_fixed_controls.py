"""Development follow-up: components of the fixed-cap receiver-preserving rule.

Declared after inspecting all three Indian development cases; before Chinese candidate outcomes. All six
corpus/topic settings and every final seed are retained. This is not blind
confirmation and no empirical superiority is assumed.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import hashlib
import json
import sys
import time
from run_trial import Trial, read, write, digest, say

POLICIES=('weighted_hidden_fixed','preserve_fixed_no_downside',
          'preserve_fixed_individual','preserve_fixed_single_view',
          'matched_preserve_fixed')


class Individual:
    def __init__(self,base):self.base,self.views,self.union=base,base.views,base.union
    def gains(self,actions):
        import numpy as np
        from transport_topics.proposed.actions import BorrowingAction
        unique=list({BorrowingAction((j,),a.beta):None for a in actions for j in a.donors})
        values=dict(zip(unique,self.base.gains(unique))) if unique else {}
        return np.asarray([np.mean([values[BorrowingAction((j,),a.beta)] for j in a.donors],axis=0)
                           if a.donors else np.zeros(self.views) for a in actions]).reshape(len(actions),self.views)


def select(trial,corpus,k,base_run,spec_hash):
    import numpy as np
    from scipy import sparse
    from anchored_scoring import ReceiverAnchoredEvaluator
    from transport_topics.proposed.actions import BorrowingAction,NO_BORROWING
    from transport_topics.proposed.search import select_action
    from transport_topics.proposed.scoring import downside_utility
    from transport_topics.proposed.transfer import normalized_counts,transfer_action,check_transfer
    from transport_topics.experiments.sharing_controls import matched_action
    c=trial.collection(corpus);refs=trial.references(corpus,k)
    base_path=base_run/'selection'/corpus/('K%d'%k)
    while not (base_path/'COMPLETE.json').exists():
        say('WAITING FOR FROZEN FORWARD BANK',corpus,k);time.sleep(10)
    anchor_path=base_path/'preserve_forward_fixed_cap/actions.json';anchor=read(anchor_path)
    parent=trial.output/'selection'/corpus/('K%d'%k)
    if (parent/'COMPLETE.json').exists():
        assert read(parent/'COMPLETE.json')['control_specification_sha256']==spec_hash
        return parent
    donors=normalized_counts(c['counts']);projections=[r.projection(donors) for r in refs]
    cap=trial.protocol['cap'];grid=[cap]
    margin=trial.protocol['admission_margin'];mats={p:[] for p in POLICIES};records={p:[] for p in POLICIES}
    start=time.perf_counter();tick=time.monotonic()
    for i in range(len(donors)):
        path=parent/'receivers'/('%06d.json'%i)
        if path.exists():
            saved=read(path)
            if saved['spec_hash']!=spec_hash:raise ValueError('Control checkpoint version differs')
        else:
            plans={p:NO_BORROWING for p in POLICIES};details={}
            if c['eligible'][i] and np.any(c['candidates'][i]>=0):
                fold=int(c['folds'][i]);block=slice(i*20,(i+1)*20)
                visible=c['visible'][block].toarray();targets=c['targets'][block].toarray();lists=c['candidates'][i]
                original=c['counts'][i].toarray()
                ev=ReceiverAnchoredEvaluator(refs[fold],visible,targets,lists,donors,original,projections[fold],128)
                class WeightedHidden(ReceiverAnchoredEvaluator):
                    def _diagnostics(self,probability,inactive):
                        hidden,shown,net,whole=super()._diagnostics(probability,inactive)
                        return hidden,shown,self.hidden_shares[None]*hidden,whole
                weighted=WeightedHidden(refs[fold],visible,targets,lists,donors,original,projections[fold],128)
                one=ReceiverAnchoredEvaluator(refs[fold],visible[:1],targets[:1],lists[:1],donors,original,projections[fold],128)
                decisions={
                    'weighted_hidden_fixed':select_action(weighted,[cap],3,1.,margin),
                    'preserve_fixed_no_downside':select_action(ev,grid,3,0.,margin),
                    'preserve_fixed_individual':select_action(Individual(ev),grid,3,1.,margin),
                    'preserve_fixed_single_view':select_action(one,grid,3,1.,margin)}
                for p,res in decisions.items():
                    plans[p]=res.action
                    validator=one if p=='preserve_fixed_single_view' else weighted if p=='weighted_hidden_fixed' else ev
                    actual=validator.direct_gains(res.action)
                    if p!='preserve_fixed_individual' and np.max(np.abs(actual-res.gains))>2e-8:
                        raise ArithmeticError('Control direct complete-action scoring differs')
                    if res.action.donors and res.utility<=margin:raise ArithmeticError('Nonpositive control admission')
                    details[p]={'selection_utility':float(res.utility),
                        'actual_selection_view_utility':float(downside_utility(actual,0. if p=='preserve_fixed_no_downside' else 1.)),
                        'selection_views':1 if p=='preserve_fixed_single_view' else 20,
                        'actual_full_view_utility':float(downside_utility(ev.gains([res.action])[0],0. if p=='preserve_fixed_no_downside' else 1.))}
                union=np.unique(lists[lists>=0]);cos=c['embeddings'][union]@c['queries'][i]
                ranked=union[np.argsort(-cos,kind='stable')].tolist()
                a=anchor[i]
                plans['matched_preserve_fixed']=matched_action(tuple(a['donors']),a['actual_beta'],ranked,c['full'][i])
            data={}
            for p,a in plans.items():
                q,alpha,available=transfer_action(donors[i],a,c['full'][i],donors)
                tv=check_transfer(donors[i],q,alpha,cap)
                data[p]={'donors':list(a.donors),'offered_beta':a.beta,'actual_beta':alpha,
                    'available_donors':list(available),'total_variation':tv,**details.get(p,{})}
            matched=data['matched_preserve_fixed']
            assert abs(anchor[i]['actual_beta']-matched['actual_beta'])<1e-12
            assert len(anchor[i]['available_donors'])==len(matched['available_donors'])
            saved={'receiver':i,'spec_hash':spec_hash,'plans':data};write(path,saved)
        for p,r in saved['plans'].items():
            a=BorrowingAction(tuple(r['donors']),r['offered_beta'])
            q,alpha,available=transfer_action(donors[i],a,c['full'][i],donors)
            assert abs(alpha-r['actual_beta'])<1e-12
            mats[p].append(sparse.csr_matrix(q[None]));records[p].append(r)
        if i==0 or i+1==len(donors) or time.monotonic()-tick>15:
            say('CONTROL SELECT',corpus,'K',k,i+1,'/',len(donors),'%.1fs'%(time.perf_counter()-start));tick=time.monotonic()
    for p in POLICIES:
        target=parent/p;target.mkdir(parents=True,exist_ok=True)
        sparse.save_npz(target/'transferred.npz',sparse.vstack(mats[p],format='csr'))
        write(target/'actions.json',records[p])
        write(target/'summary.json',{'posts':len(donors),'mean_actual_beta':float(np.mean([x['actual_beta'] for x in records[p]])),
            'borrowing_coverage':float(np.mean([x['actual_beta']>0 for x in records[p]])),
            'mean_contributing_donors':float(np.mean([len(x['available_donors']) for x in records[p]]))})
    write(parent/'COMPLETE.json',{'control_specification_sha256':spec_hash,'anchor_action_sha256':digest(anchor_path),
        'selection_seconds':time.perf_counter()-start,'variants':list(POLICIES)})
    return parent



def fit_or_reuse(trial,corpus,k,policy,seed,learning):
    import shutil
    from scipy import sparse
    target=trial.output/'fits'/corpus/('K%d'%k)/policy/('seed%d'%seed)
    if (target/'result.json').exists():return read(target/'result.json')
    for run in ('run_v1','forward_controls_v1'):
        parent=trial.root/run/'selection'/corpus/('K%d'%k)
        for bank in sorted(parent.glob('*/transferred.npz')):
            source=trial.root/run/'fits'/corpus/('K%d'%k)/bank.parent.name/('seed%d'%seed)
            if not (source/'result.json').exists():continue
            other=sparse.load_npz(bank).tocsr()
            if other.shape!=learning.shape or (learning-other).nnz:continue
            row=read(source/'result.json')
            assert row['trial_identity']==trial.identity
            row=dict(row,policy=policy,seconds=0.,source='reused_identical_transfer: '+str(source.relative_to(trial.root)))
            target.mkdir(parents=True,exist_ok=True)
            for name in ('topic_scores.npy','audit.json','quality.json'):shutil.copyfile(source/name,target/name)
            write(target/'reuse.json',{'identical_learning_matrix':True,'source':str(source.relative_to(trial.root)),
                'source_seed':seed,'source_policy':bank.parent.name,'seed_condition_not_independent':True})
            write(target/'result.json',row)
            say('REUSE IDENTICAL BANK',corpus,k,policy,seed,'from',bank.parent.name)
            return row
    return trial.fit(corpus,k,policy,seed,learning)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dependencies',type=Path,default=None)
    args=parser.parse_args();root=Path(__file__).resolve().parent
    if args.dependencies:sys.path.insert(0,str(args.dependencies.resolve()))
    sys.path[:0]=[str(root/'frozen_source'),str(root/'candidate')]
    import numpy as np
    from scipy import sparse
    base=root/'forward_controls_v1';output=root/'fixed_controls_v1'
    assert read(root/'run_v1/baseline_reproduction.json')['passed']
    spec={'scope':'exploratory fixed-cap controls after Indian development cases; all six settings retained',
        'script_sha256':digest(__file__),'policies':list(POLICIES),'selection_mask_base':2711,'final_seeds':[0,1,2,3,4],
        'corpora':['transport_grievance','chinese_pooled'],'topics':[8,10,12],
        'single_view_changes':'coverage and scoring views together; not an isolated view-count effect',
        'individual_control':'average singleton anchored gains; complete-mixture audit without veto'}
    spec_hash=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    trial=Trial(root,output);write(output/'CONTROL_SPECIFICATION.json',dict(spec,sha256=spec_hash))
    for corpus in trial.protocol['corpora']:
        for k in trial.protocol['topic_counts']:
            parent=select(trial,corpus,k,base,spec_hash)
            for p in POLICIES:
                learning=sparse.load_npz(parent/p/'transferred.npz').tocsr()
                for seed in trial.protocol['final_model_seeds']:
                    fit_or_reuse(trial,corpus,k,p,seed,learning);trial.summarize()
            del learning
    rows=trial.summarize();assert len([r for r in rows if r['policy'] in POLICIES])==30
    write(output/'CONTROLS_COMPLETE.json',{'scope':'focused exploratory forward-rule controls','evaluated_seed_conditions':150,
        'all_declared_conditions_retained':True,'selection_banks':1,'specification_sha256':spec_hash})
    say('FIXED-CAP CONTROLS COMPLETE','150 evaluated seed conditions; exact bank reuse recorded')


if __name__=='__main__':main()
