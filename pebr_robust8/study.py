"""A bounded study: exact default replay, then separate reference/fold changes."""
import time
import numpy as np
from .common import read, write, table, digest, key
from .design import POLICIES, ANCHOR
from .results import record_key, export

def original_fit_path(step5,step3,corpus,k,policy,seed):
    if policy=='nmf_original':
        return step3/'fits'/corpus/('K%d'%k)/policy/('seed%d'%seed)
    return step5/'fits'/corpus/'mask3711'/('K%d'%k)/policy/('seed%d'%seed)

class Study:
    def __init__(self,step2,step3,step5,work,protocol,descriptions,logger):
        from pebr_revision_step3.runner import load_collections
        self.step2,self.step3,self.step5,self.work,self.protocol,self.logger=step2,step3,step5,work,protocol,logger
        self.corpora=load_collections(step2,descriptions,protocol)
        self.rows={};self.timings=[];self.reference_records=[];self.bank_records=[]
        self.reproductions=[];self.new_fit_count=0;self.reused_fit_count=0
        self.original_banks={};self.query_vectors={};self.positions={}

    def prepare(self):
        from pebr_fresh_validation.banks import load_bank,query_records
        from .banks import cached_queries,original_bank_check
        for corpus,c in self.corpora.items():
            col=c['collection']
            bank=load_bank(self.step5/'mask_banks'/corpus/'mask3711',col,self.protocol)
            records=query_records(corpus,col,self.work/'tokenizer_cache')
            vectors,positions,identity=cached_queries(self.step5,corpus,col,bank,records,self.protocol)
            check=original_bank_check(col,bank,vectors,positions,self.protocol)
            self.original_banks[corpus]=bank;self.query_vectors[corpus]=vectors;self.positions[corpus]=positions
            self.bank_records.append({'corpus':corpus,**identity,**check})
        write(self.work/'original_bank_reconstruction.json',self.bank_records)

    def reproduce(self):
        from scipy import sparse
        from pebr_fresh_validation.references import read_references
        from pebr_revision_step3.io import complete
        from .selection import select
        for corpus,c in self.corpora.items():
            col=c['collection'];refs=read_references(self.step3,col,12,self.protocol)
            target=self.work/'reproduction'/corpus/'K12'
            p=dict(self.protocol,current_case={'case':'default_replay','reference_seed':0,'fold_seed':2710})
            paths=select(col,self.original_banks[corpus],refs,p,target,self.logger)
            for policy,path in paths.items():
                old=self.step5/'selection'/corpus/'mask3711'/'K12'/policy
                if not complete(old):raise ValueError('Original selection bank receipt failed')
                left,right=read(path/'actions.json'),read(old/'actions.json')
                if len(left)!=len(right):raise ValueError('Original/replayed action row counts differ')
                for a,b in zip(left,right):
                    for field in ('donors','available_donors'):
                        if a[field]!=b[field]:raise ValueError('Frozen selector replay differs in '+field)
                    for field in ('offered_beta','actual_beta'):
                        if abs(a[field]-b[field])>1e-12:raise ValueError('Frozen selector replay differs in '+field)
                delta=(sparse.load_npz(path/'transferred.npz')-sparse.load_npz(old/'transferred.npz')).tocsr()
                error=float(np.max(np.abs(delta.data))) if delta.nnz else 0.
                if error>2e-12:raise ValueError('Frozen selector replay changed a transferred representation')
                self.reproductions.append({'corpus':corpus,'K':12,'policy':policy,'all_receivers_checked':len(left),
                    'selected_and_available_donors_match':True,'mass_matches':True,'max_representation_error':error})
        write(self.work/'default_selector_reproduction.json',{'passed':True,'checks':self.reproductions})
        self.logger.info('EXACT SELECTOR REPLAY PASSED | 4 policies, both corpora, every K12 receiver')

    def shared(self,corpus,k,case,policy,seed):
        from pebr_revision_step3.io import complete
        path=original_fit_path(self.step5,self.step3,corpus,k,policy,seed)
        if not complete(path):raise ValueError('Shared original fit receipt failed: '+str(path))
        old=read(path/'result.json')
        if (old['corpus'],int(old['K']),old['policy'],int(old['seed']))!=(corpus,k,policy,seed):
            raise ValueError('A shared original fit condition differs')
        return dict(corpus=corpus,K=k,case=case['case'],axis=case['axis'],reference_seed=case['reference_seed'],
            fold_seed=case['fold_seed'],policy=policy,seed=seed,npmi=old['npmi'],cv=old['cv'],
            diversity=old['diversity'],top_words=old['top_words'],fit_converged=old['fit_converged'],
            source='shared_unchanged_original_fit',path=str(path),
            unique_fit_identity=digest(path/'complete.json'))

    def fit(self,corpus,k,case,policy,seed,representation):
        from scipy import sparse
        from pebr_revision_step3.io import complete,save_receipt
        from transport_topics.topics.learning import fit_topics
        from transport_topics.evaluation.topic_quality import evaluate_topics
        directory=self.work/'fits'/case['case']/corpus/('K%d'%k)/policy/('s%d'%seed)
        if complete(directory):
            row=read(directory/'result.json')
            if record_key(row)!=(corpus,k,case['case'],policy,seed):raise ValueError('Resumed fit identity differs')
            self.reused_fit_count+=1
            return row
        c=self.corpora[corpus]
        learned=sparse.load_npz(representation/'transferred.npz').tocsr()
        self.logger.info('Fit | %s %s K%d %s seed%d',case['case'],corpus,k,policy,seed)
        started=time.perf_counter()
        model=fit_topics(c['collection'].counts,learned,k,seed,self.protocol['final_nmf'])
        if not model.audit['fit']['converged']:
            write(directory/'failed_fit.json',model.audit)
            raise RuntimeError('A sensitivity NMF fit did not converge; no automatic setting/seed change')
        model.validate(c['collection'].counts.shape[0],len(c['vocabulary']),k)
        elapsed=time.perf_counter()-started
        quality=evaluate_topics(model.topic_scores,c['vocabulary'],c['reference'],10,True)
        directory.mkdir(parents=True,exist_ok=True)
        for name,value in [('topic_scores',model.topic_scores),('doc_topics',model.doc_topics),
                           ('retrieval_features',model.retrieval_features),('vocabulary',c['vocabulary']),
                           ('source_rows',c['collection'].source_rows)]:
            np.save(directory/(name+'.npy'),value,allow_pickle=False)
        write(directory/'audit.json',model.audit);write(directory/'quality.json',quality)
        row=dict(corpus=corpus,K=k,case=case['case'],axis=case['axis'],reference_seed=case['reference_seed'],
            fold_seed=case['fold_seed'],policy=policy,seed=seed,npmi=quality['npmi'],cv=quality['cv'],
            diversity=quality['diversity'],top_words=quality['top_words'],fit_converged=True,
            fit_seconds=elapsed,source='new_sensitivity_fit',path=directory.relative_to(self.work).as_posix(),
            unique_fit_identity=key({'case':case,'corpus':corpus,'K':k,'policy':policy,'seed':seed,'protocol':key(self.protocol)}))
        write(directory/'result.json',row);save_receipt(directory)
        self.new_fit_count+=1
        return row

    def references(self,collection,k,case):
        from transport_topics.proposed.reference import reference_bank
        from pebr_revision_step3.io import complete
        setting=self.protocol['reference_predictor']
        settings={'folds':5,'seed':case['reference_seed'],'tolerance':setting['nmf_tolerance'],
            'max_iterations':setting['nmf_max_iterations'],'smoothing':setting['smoothing'],
            'unigram_smoothing':setting['unigram_smoothing']}
        refs,audits=reference_bank(self.work,collection.summary,collection.counts,collection.folds,k,settings,self.logger)
        for fold,(ref,audit) in enumerate(zip(refs,audits)):
            directory=self.work/audit['path']
            train=np.load(directory/'training_indices.npy',allow_pickle=False)
            if (not complete(directory) or not audit['converged'] or not audit['receiver_fold_excluded'] or
                not np.array_equal(train,np.flatnonzero(collection.folds!=fold))):
                raise ValueError('Reference convergence or receiving-fold exclusion failed')
            self.reference_records.append(dict(case=case['case'],collection=collection.summary['name'],K=k,
                excluded_fold=fold,reference_seed=case['reference_seed'],fold_seed=case['fold_seed'],
                path=audit['path'],identity_sha256=digest(directory/'identity.json'),
                training_rows_sha256=digest(directory/'training_indices.npy'),**{q:audit[q] for q in ('training_posts','iterations','converged','receiver_fold_excluded')}))
        return refs

    def run(self):
        from pebr_revision_step3.io import complete
        from .banks import alternate_collection,save_alternate_bank
        from .selection import select
        for case in self.protocol['cases']:
            for corpus,c in self.corpora.items():
                collection=c['collection'];bank=self.original_banks[corpus]
                if case['axis']=='family_folds':
                    collection,fold_identity=alternate_collection(collection,case['fold_seed'])
                    bank=save_alternate_bank(self.work/'fold_banks'/case['case']/corpus,collection,bank,
                        self.query_vectors[corpus],self.positions[corpus],self.protocol)
                    write(self.work/'fold_assignments'/case['case']/(corpus+'.json'),dict(fold_identity,
                        source_rows=collection.source_rows.tolist(),families=collection.families.tolist(),
                        changed_post_fraction=float(np.mean(collection.folds!=c['collection'].folds))))
                for k in self.protocol['topic_counts']:
                    paths={}
                    if case['case']!='default':
                        tick=time.perf_counter()
                        references=self.references(collection,k,case)
                        self.timings.append(dict(phase='reference_bank',case=case['case'],corpus=corpus,K=k,seconds=time.perf_counter()-tick))
                        tick=time.perf_counter()
                        paths=select(collection,bank,references,dict(self.protocol,current_case=case),
                                     self.work/'selection'/case['case']/corpus/('K%d'%k),self.logger)
                        self.timings.append(dict(phase='combined_four_policy_selection',case=case['case'],corpus=corpus,K=k,
                                                 seconds=time.perf_counter()-tick,standalone_pebr_runtime=False))
                        if case['axis']=='reference_initialization':
                            from scipy import sparse
                            original=self.step5/'selection'/corpus/'mask3711'/('K%d'%k)/'nmf_similarity'
                            if not complete(original):raise ValueError('Shared similarity bank receipt failed')
                            a=sparse.load_npz(paths['nmf_similarity']/'transferred.npz');b=sparse.load_npz(original/'transferred.npz')
                            difference=(a-b).tocsr()
                            if difference.nnz and np.max(np.abs(difference.data))>2e-12:
                                raise ValueError('Changing the predictive reference changed ordinary sharing')
                    for policy in self.protocol['policies']:
                        for seed in self.protocol['final_model_seeds']:
                            reuse=case['case']=='default' or policy=='nmf_original' or (case['axis']=='reference_initialization' and policy=='nmf_similarity')
                            row=self.shared(corpus,k,case,policy,seed) if reuse else self.fit(corpus,k,case,policy,seed,paths[policy])
                            if not row['fit_converged']:raise ValueError('Shared fit did not converge')
                            self.rows[record_key(row)]=row
                    export(self.work/'results',list(self.rows.values()),self.protocol)
                    table(self.work/'reference_fit_audits.csv',self.reference_records)
                    table(self.work/'phase_timings.csv',self.timings)
        if len(self.reference_records)!=120:raise ValueError('Expected exactly 120 fold-excluded reference fits')
        result=export(self.work/'results',list(self.rows.values()),self.protocol,True)
        unique_new={r['unique_fit_identity'] for r in self.rows.values() if r['source']=='new_sensitivity_fit'}
        unique_old={r['unique_fit_identity'] for r in self.rows.values() if r['source']=='shared_unchanged_original_fit'}
        if len(unique_new)!=420 or len(unique_old)!=150:raise ValueError('Unique model fitting/reuse accounting differs')
        write(self.work/'model_fit_accounting.json',{'new_unique_final_models':len(unique_new),'shared_unique_final_models':len(unique_old),
              'case_fit_records':len(self.rows),'shared_records_are_reused_not_new_replicates':True})
        return result
