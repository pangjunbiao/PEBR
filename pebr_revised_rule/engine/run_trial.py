"""Focused development trial; frozen inputs, no encoder or source changes.

All variants/corpora/topic counts are fixed before observing new outcomes.
This is development evidence on known corpora, not independent confirmation.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from collections import Counter

sys.dont_write_bytecode = True
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def say(*args):
    print(time.strftime('%Y-%m-%dT%H:%M:%S'),*args,flush=True)


VARIANTS = ('preserve_forward','hidden_refined','preserve_refined',
            'preserve_refined_fixed_cap','matched_preserve_refined')


class Trial:
    def __init__(self, root, output):
        import importlib.metadata as meta
        self.root,self.output=Path(root),Path(output)
        self.input=self.root/'inputs';self.protocol=read(self.input/'protocol.json')
        env={name:meta.version(name) for name in ('numpy','scipy','scikit-learn','gensim')}
        expected=read(self.input/'provenance/environment.json')['packages']
        if any(env[name]!=expected[name] for name in env):
            raise ValueError('Numerical library versions must match: '+str(env))
        manifest=read(self.input/'provenance/snapshot_manifest.json')
        for rec in manifest['files']:
            if rec['destination'].startswith('src/'):
                p=self.root/'frozen_source'/Path(rec['destination']).relative_to('src')
                if digest(p)!=rec['sha256']:raise ValueError('Scientific source differs: '+str(p))
        self.output.mkdir(parents=True,exist_ok=True)
        identity={'inputs_sha256':digest(self.input/'export_manifest.json'),
            'runner_sha256':digest(__file__),
            'candidate_sha256':{p.name:digest(p) for p in sorted((self.root/'candidate').glob('*.py'))},
            'environment':env,'corpora':self.protocol['corpora'],'topics':self.protocol['topic_counts'],
            'variants':list(VARIANTS),'final_seeds':[0,1,2,3,4],
            'selection_mask_base':2711,'selection_repeats':1,
            'scope':'focused exploratory rule development on known corpora; no new confirmation'}
        self.identity=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        identity['identity']=self.identity
        old=self.output/'trial_identity.json'
        if old.exists() and read(old)!=identity:raise ValueError('Output belongs to another source/version')
        write(old,identity)
        self.oldrows=read(self.input/'step3/results/topic_words.json')
        self.old={(r['corpus'],int(r['K']),r['policy'],int(r['seed'])):r for r in self.oldrows}
        self.collections={};self.refs={}

    def collection(self, corpus):
        import numpy as np
        from scipy import sparse
        from transport_topics.evaluation.reference import TextReference
        if corpus in self.collections:return self.collections[corpus]
        p=self.input/'data'/corpus
        c=sparse.load_npz(p/'counts.npz').tocsr().astype(float)
        vocabulary=np.load(p/'vocabulary.npy',allow_pickle=False)
        folds=np.load(p/'folds.npy',allow_pickle=False);families=np.load(p/'families.npy',allow_pickle=False)
        tokens=read(p/'tokens.json');n,v=c.shape
        assert len(tokens)==n and len(vocabulary)==v
        lookup={str(w):j for j,w in enumerate(vocabulary)};rr=[];cc=[];vv=[]
        for i,words in enumerate(tokens):
            for word,count in Counter(words).items():
                if word in lookup:rr.append(i);cc.append(lookup[word]);vv.append(count)
        rebuilt=sparse.csr_matrix((vv,(rr,cc)),shape=c.shape)
        assert (rebuilt-c).nnz==0
        binary=c.astype(bool).astype('int64')
        reference=TextReference('focused-export-'+corpus,corpus,vocabulary,n,
            np.asarray(binary.sum(axis=0)).ravel(),(binary.T@binary).tocsr(),tokens,p)
        bank=p/'topic_learning'
        data={'counts':c,'vocabulary':vocabulary,'folds':folds,'families':families,
            'reference':reference,'eligible':np.load(bank/'eligible.npy',allow_pickle=False),
            'candidates':np.load(bank/'candidates.npy',allow_pickle=False),
            'full':np.load(bank/'full_candidates.npy',allow_pickle=False),
            'visible':sparse.load_npz(bank/'visible.npz').tocsr(),
            'targets':sparse.load_npz(bank/'targets.npz').tocsr(),
            'base':sparse.load_npz(bank/'base.npz').tocsr(),
            'embeddings':np.load(p/'embeddings.npy',allow_pickle=False),
            'queries':np.load(bank/'full_receiver_embeddings.npy',allow_pickle=False)}
        assert (data['base']-c).nnz==0
        assert data['visible'].shape==(n*20,v) and data['targets'].shape==(n*20,v)
        for i in range(n):
            ids=data['candidates'][i].ravel();ids=ids[ids>=0]
            assert np.all(folds[ids]!=folds[i]) and np.all(families[ids]!=families[i])
        self.collections[corpus]=data
        say('INPUT VERIFIED',corpus,n,'posts',v,'terms')
        return data

    def references(self, corpus, k):
        import numpy as np
        from transport_topics.proposed.reference import PredictiveReference
        if (corpus,k) in self.refs:return self.refs[corpus,k]
        c=self.collection(corpus);out=[]
        for f in range(5):
            p=self.input/'predictors'/corpus/('K%d'%k)/('fold%d'%f)
            train=np.load(p/'training_indices.npy',allow_pickle=False)
            assert np.array_equal(train,np.flatnonzero(c['folds']!=f))
            audit=read(p/'fit.json');assert audit['converged'] and audit['receiver_fold_excluded']
            out.append(PredictiveReference(np.load(p/'dictionary.npy',allow_pickle=False),
                np.load(p/'idf.npy',allow_pickle=False),np.load(p/'prior.npy',allow_pickle=False),.1))
        self.refs[corpus,k]=out;return out

    def old_bank(self,corpus,k,policy):
        import numpy as np
        from scipy import sparse
        from transport_topics.proposed.actions import BorrowingAction
        from transport_topics.proposed.transfer import normalized_counts,transfer_action,check_transfer
        c=self.collection(corpus)
        if policy=='nmf_original':return c['counts']
        rows=read(self.input/'actions'/corpus/('K%d'%k)/policy/'actions.json')
        donors=normalized_counts(c['counts']);matrix=[]
        for i,r in enumerate(rows):
            a=BorrowingAction(tuple(r['donors']),r['offered_beta'])
            q,alpha,available=transfer_action(donors[i],a,c['full'][i],donors)
            assert abs(alpha-r['actual_beta'])<1e-12 and list(available)==r['available_donors']
            check_transfer(donors[i],q,alpha,self.protocol['cap'])
            matrix.append(sparse.csr_matrix(q[None]))
        return sparse.vstack(matrix,format='csr')

    def fit(self,corpus,k,policy,seed,learning,reproduction=False):
        import numpy as np
        from threadpoolctl import threadpool_limits
        from transport_topics.topics.learning import fit_topics
        from transport_topics.evaluation.topic_quality import evaluate_topics
        c=self.collection(corpus)
        p=self.output/('reproduction' if reproduction else 'fits')/corpus/('K%d'%k)/policy/('seed%d'%seed)
        if (p/'result.json').exists():return read(p/'result.json')
        p.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
        with threadpool_limits(limits=1):
            model=fit_topics(c['counts'],learning,k,seed,self.protocol['final_nmf'])
            assert model.audit['fit']['converged']
            model.validate(c['counts'].shape[0],len(c['vocabulary']),k)
            quality=evaluate_topics(model.topic_scores,c['vocabulary'],c['reference'],10,True)
        row={'corpus':corpus,'K':k,'policy':policy,'seed':seed,'mask_base':2711,
            'selection_banks':1,'npmi':quality['npmi'],'cv':quality['cv'],'diversity':quality['diversity'],
            'top_words':quality['top_words'],'seconds':time.perf_counter()-started,
            'fit_converged':True,'source':'fresh_local_reproduction' if reproduction else 'fresh_focused_development',
            'trial_identity':self.identity}
        np.save(p/'topic_scores.npy',model.topic_scores,allow_pickle=False)
        write(p/'audit.json',model.audit);write(p/'quality.json',quality);write(p/'result.json',row)
        say('FIT',corpus,'K',k,policy,'seed',seed,'NPMI %.6f C_v %.6f TD %.6f'%(row['npmi'],row['cv'],row['diversity']))
        return row

    def reproduce(self):
        import numpy as np
        checks=[]
        for corpus in self.protocol['corpora']:
            for k in self.protocol['topic_counts']:
                self.references(corpus,k)
                for policy in ('nmf_original','proposed'):
                    bank=self.old_bank(corpus,k,policy)
                    seeds=[0,1,2,3,4] if k==12 else [0]
                    for seed in seeds:
                        row=self.fit(corpus,k,policy,seed,bank,True);old=self.old[corpus,k,policy,seed]
                        delta={m:abs(row[m]-old[m]) for m in ('npmi','cv','diversity')}
                        same=row['top_words']==old['top_words'];array_error=None
                        if seed==0:
                            oldh=np.load(self.input/'baseline_seed0'/corpus/('K%d'%k)/policy/'topic_scores.npy',allow_pickle=False)
                            newh=np.load(self.output/'reproduction'/corpus/('K%d'%k)/policy/'seed0/topic_scores.npy',allow_pickle=False)
                            array_error=float(np.max(np.abs(newh-oldh)))/max(1.,float(np.max(np.abs(oldh))))
                        passed=max(delta.values())<2e-6 and same and (array_error is None or array_error<2e-7)
                        checks.append({'corpus':corpus,'K':k,'policy':policy,'seed':seed,
                            'metric_absolute_errors':delta,'topic_words_identical':same,
                            'relative_dictionary_max_error':array_error,'passed':bool(passed)})
                        write(self.output/'baseline_reproduction.json',{'checks':checks,'passed':all(x['passed'] for x in checks)})
                        if not passed:raise ArithmeticError('Baseline reproduction failed; revision fitting blocked')
                    del bank
        say('BASELINE REPRODUCTION COMPLETE',len(checks),'fits; all passed')

    def select(self,corpus,k):
        import numpy as np
        from scipy import sparse
        from transport_topics.proposed.actions import BorrowingAction,NO_BORROWING
        from transport_topics.proposed.scoring import ActionEvaluator,downside_utility
        from transport_topics.proposed.search import select_action
        from transport_topics.proposed.transfer import normalized_counts,transfer_action,check_transfer
        from transport_topics.experiments.sharing_controls import matched_action
        from anchored_scoring import ReceiverAnchoredEvaluator
        from refined_search import select_refined_action
        c=self.collection(corpus);refs=self.references(corpus,k)
        parent=self.output/'selection'/corpus/('K%d'%k)
        if (parent/'COMPLETE.json').exists():return parent
        parent.mkdir(parents=True,exist_ok=True)
        donors=normalized_counts(c['counts']);projected=[r.projection(donors) for r in refs]
        old=read(self.input/'actions'/corpus/('K%d'%k)/'proposed/actions.json')
        prefixes=read(self.input/'actions'/corpus/('K%d'%k)/'predictively_scored_similarity_prefixes/actions.json')
        matrices={p:[] for p in VARIANTS};records={p:[] for p in VARIANTS};diagnostics=[]
        budgets=[self.protocol['cap']*f for f in self.protocol['positive_budget_cap_fractions']]
        cap=self.protocol['cap'];margin=self.protocol['admission_margin'];start=time.perf_counter();tick=time.monotonic()
        for i in range(len(donors)):
            checkpoint=parent/'receivers'/('%06d.json'%i)
            if checkpoint.exists():
                saved=read(checkpoint)
                if saved['trial_identity']!=self.identity:raise ValueError('Receiver checkpoint identity differs')
            else:
                plans={p:NO_BORROWING for p in VARIANTS};detail={};diag={'receiver':i,'eligible':bool(c['eligible'][i])}
                if c['eligible'][i] and np.any(c['candidates'][i]>=0):
                    fold=int(c['folds'][i]);block=slice(i*20,(i+1)*20)
                    visible=c['visible'][block].toarray();targets=c['targets'][block].toarray();lists=c['candidates'][i]
                    ev=ActionEvaluator(refs[fold],visible,targets,lists,donors,projected[fold],128)
                    ae=ReceiverAnchoredEvaluator(refs[fold],visible,targets,lists,donors,c['counts'][i].toarray(),projected[fold],128)
                    original=BorrowingAction(tuple(old[i]['donors']),old[i]['offered_beta'])
                    hg=ev.gains([original])[0]
                    error=float(np.max(np.abs(hg-np.asarray(old[i]['actual_view_gains']))))
                    if error>2e-8:raise ArithmeticError('Saved original selected-action score differs')
                    net=ae.gains([original])[0]
                    d=ae.diagnostics.get(original)
                    diag.update(old_score_max_error=error,old_hidden_utility=float(downside_utility(hg)),
                        old_prefix_utility=float(prefixes[i]['actual_mixture_utility']),
                        old_preservation_utility=float(downside_utility(net)),
                        old_visible_mean_gain=float(np.mean(d['visible_gain'])) if d else 0.,
                        old_whole_receiver_mean_gain=float(np.mean(d['whole_receiver_gain'])) if d else 0.,
                        old_visible_loss_view_fraction=float(np.mean(d['visible_gain'] < -1e-10)) if d else 0.)
                    decision={
                        'preserve_forward':select_action(ae,budgets,3,1.,margin),
                        'hidden_refined':select_refined_action(ev,lists,budgets,3,1.,margin),
                        'preserve_refined':select_refined_action(ae,lists,budgets,3,1.,margin),
                        'preserve_refined_fixed_cap':select_refined_action(ae,lists,[cap],3,1.,margin)}
                    for name,res in decision.items():
                        plans[name]=res.action
                        checker=ev if name=='hidden_refined' else ae
                        err=float(np.max(np.abs(checker.direct_gains(res.action)-res.gains)))
                        if err>2e-8:raise ArithmeticError('A revised complete-action score differs in word space')
                        detail[name]={'utility':float(res.utility),'proposals':res.proposals,'direct_score_max_error':err}
                    chosen=plans['preserve_refined']
                    _,alpha,available=transfer_action(donors[i],chosen,c['full'][i],donors)
                    union=np.unique(lists[lists>=0]);cos=c['embeddings'][union]@c['queries'][i]
                    ranked=union[np.argsort(-cos,kind='stable')].tolist()
                    plans['matched_preserve_refined']=matched_action(chosen.donors,alpha,ranked,c['full'][i])
                stored={}
                for name,a in plans.items():
                    q,alpha,available=transfer_action(donors[i],a,c['full'][i],donors)
                    tv=check_transfer(donors[i],q,alpha,cap)
                    stored[name]={'donors':list(a.donors),'offered_beta':a.beta,'actual_beta':alpha,
                        'available_donors':list(available),'total_variation':tv,**detail.get(name,{})}
                a,b=stored['preserve_refined'],stored['matched_preserve_refined']
                assert abs(a['actual_beta']-b['actual_beta'])<1e-12 and len(a['available_donors'])==len(b['available_donors'])
                saved={'trial_identity':self.identity,'receiver':i,'plans':stored,'diagnostic':diag}
                write(checkpoint,saved)
            for name,r in saved['plans'].items():
                a=BorrowingAction(tuple(r['donors']),r['offered_beta'])
                q,alpha,available=transfer_action(donors[i],a,c['full'][i],donors)
                assert abs(alpha-r['actual_beta'])<1e-12
                check_transfer(donors[i],q,alpha,cap)
                matrices[name].append(sparse.csr_matrix(q[None]))
                records[name].append(r)
            diagnostics.append(saved['diagnostic'])
            if i==0 or i+1==len(donors) or time.monotonic()-tick>15:
                say('SELECT',corpus,'K',k,i+1,'/',len(donors),'elapsed %.1fs'%(time.perf_counter()-start));tick=time.monotonic()
        for name in VARIANTS:
            p=parent/name;p.mkdir(parents=True,exist_ok=True)
            sparse.save_npz(p/'transferred.npz',sparse.vstack(matrices[name],format='csr'))
            write(p/'actions.json',records[name])
            write(p/'summary.json',{'posts':len(donors),'borrowing_coverage':float(np.mean([r['actual_beta']>0 for r in records[name]])),
                'mean_actual_beta':float(np.mean([r['actual_beta'] for r in records[name]])),
                'mean_offered_beta':float(np.mean([r['offered_beta'] for r in records[name]])),
                'positive_offers_at_cap':sum(r['offered_beta']>0 and abs(r['offered_beta']-cap)<1e-12 for r in records[name]),
                'positive_offers':sum(r['offered_beta']>0 for r in records[name]),
                'mean_contributing_donors':float(np.mean([len(r['available_donors']) for r in records[name]])),
                'trial_identity':self.identity})
        write(parent/'receiver_diagnostics.json',diagnostics)
        write(parent/'COMPLETE.json',{'trial_identity':self.identity,'selection_seconds':time.perf_counter()-start,'variants':list(VARIANTS)})
        return parent

    def summarize(self):
        import numpy as np
        from transport_topics.evaluation.stability import seed_stability
        fresh=[]
        for p in sorted((self.output/'fits').glob('*/*/*/*/result.json')):
            r=read(p)
            if r['trial_identity']!=self.identity:raise ValueError('A fitted output has another identity')
            fresh.append(r)
        old=[dict(r,source='saved_step3_policy_result') for r in self.oldrows]
        groups={}
        for row in old+fresh:groups.setdefault((row['corpus'],int(row['K']),row['policy']),[]).append(row)
        summary=[]
        for (corpus,k,policy),rows in sorted(groups.items()):
            if sorted(int(r['seed']) for r in rows)!=[0,1,2,3,4]:continue
            item={'corpus':corpus,'K':k,'policy':policy,'selection_banks':1,'final_seeds':5,'source':rows[0]['source']}
            for m in ('npmi','cv','diversity'):
                vals=np.asarray([r[m] for r in rows]);item[m+'_mean']=float(vals.mean());item[m+'_sd']=float(vals.std(ddof=1))
            item['stability']=seed_stability({int(r['seed']):r['top_words'] for r in rows})['stability'];summary.append(item)
        write(self.output/'results/fresh_seed_results.json',fresh)
        write(self.output/'results/all_summary.json',summary)
        for name,rows in [('all_summary.csv',summary),('fresh_seed_results.csv',[{k:v for k,v in r.items() if k!='top_words'} for r in fresh])]:
            if rows:
                p=self.output/'results'/name
                with p.open('w',encoding='utf-8',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        return summary

    def run(self,phase):
        if phase=='reproduce':return self.reproduce()
        report=read(self.output/'baseline_reproduction.json')
        if not report['passed'] or len(report['checks'])!=28:raise ValueError('All 28 baseline reproduction fits must pass first')
        from scipy import sparse
        for corpus in self.protocol['corpora']:
            for k in self.protocol['topic_counts']:
                paths=self.select(corpus,k)
                for policy in VARIANTS:
                    learning=sparse.load_npz(paths/policy/'transferred.npz').tocsr()
                    for seed in self.protocol['final_model_seeds']:
                        self.fit(corpus,k,policy,seed,learning)
                        self.summarize()
                    del learning
        summary=self.summarize()
        assert len([r for r in summary if r['policy'] in VARIANTS])==30
        write(self.output/'TRIAL_COMPLETE.json',{'status':'focused_development_trial_complete','new_final_fits':150,
            'selection_banks_per_setting':1,'independent_validation':False,'trial_identity':self.identity})
        say('FOCUSED TRIAL COMPLETE','150 new fits; both corpora and all topic counts retained')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['reproduce','trial','summarize'],required=True)
    parser.add_argument('--dependencies',type=Path,default=None)
    parser.add_argument('--output',type=Path,default=None)
    args=parser.parse_args();root=Path(__file__).resolve().parent
    if args.dependencies:sys.path.insert(0,str(args.dependencies.resolve()))
    sys.path[:0]=[str(root/'frozen_source'),str(root/'candidate')]
    trial=Trial(root,args.output or root/'run_v1')
    if args.phase=='summarize':trial.summarize()
    else:trial.run(args.phase)


if __name__=='__main__':main()
