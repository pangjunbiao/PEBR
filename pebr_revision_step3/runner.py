"""Reproduce the saved default, then execute a complete default-bank comparison."""
from __future__ import annotations
import csv
import datetime as dt
from importlib import metadata
import logging
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import zipfile
from . import VERSION
from .io import (PROTOCOL_SHA256, canonical_hash, complete, file_hash, inside, read_json,
                 resolve_step2, save_receipt, verify_step2, write_csv, write_json,
                 acquire_run_lock, release_run_lock)

def validate_environment(expected):
    names=('numpy','scipy','scikit-learn','gensim')
    actual={name:metadata.version(name) for name in names}
    wanted=expected.get('packages',{})
    mismatches={name:{'expected':wanted.get(name),'actual':actual[name]} for name in names if actual[name]!=wanted.get(name)}
    if mismatches:raise ValueError('Numerical environment differs from accepted Step 2; no automatic update: '+str(mismatches))
    return actual

def bind_frozen_source(step2):
    for p in (step2/'revision_code',step2/'src'):sys.path.insert(0,str(p))
    import transport_topics
    import pebr_revision_step2.policies
    for module in (transport_topics,pebr_revision_step2.policies):
        path=Path(module.__file__).resolve()
        try:path.relative_to(step2)
        except ValueError:raise RuntimeError('A scientific module was loaded outside the frozen snapshot: '+str(path))

def load_collections(step2, descriptions, protocol):
    import numpy as np
    from scipy import sparse
    from transport_topics.data.prepared import PreparedCollection, load_selection_bank
    from transport_topics.evaluation.reference import TextReference, tokenize_reference
    from collections import Counter
    result={}
    cfg={'method':{'views':{'count':protocol['views']},'candidates':{'per_view':protocol['candidates_per_view']}}}
    for row in descriptions:
        p=inside(step2,row['relative_directory']);summary=row['summary']
        if not complete(p):raise ValueError('Prepared full corpus lacks its receipt')
        counts=sparse.load_npz(p/'counts.npz').tocsr().astype(np.float64)
        words=np.load(p/'vocabulary.npy',allow_pickle=False)
        folds=np.load(p/'folds.npy',allow_pickle=False); families=np.load(p/'families.npy',allow_pickle=False)
        source_rows=np.load(p/'source_rows.npy',allow_pickle=False)
        n,v=counts.shape
        if (n,v)!=(summary['posts'],summary['vocabulary']) or words.shape!=(v,) or any(x.shape!=(n,) for x in (folds,families,source_rows)):
            raise ValueError('Prepared collection shape mismatch')
        if not np.isfinite(counts.data).all() or np.any(counts.data<0) or np.any(np.asarray(counts.sum(axis=1)).ravel()<=0):
            raise ValueError('Prepared counts contain invalid or zero rows')
        if not np.issubdtype(folds.dtype,np.integer) or np.any((folds<0)|(folds>=5)) or len(set(source_rows.tolist()))!=n:
            raise ValueError('Invalid prepared folds or source-row identities')
        if canonical_hash(read_json(p/'identity.json'))!=summary['identity_sha256']:
            raise ValueError('Prepared collection identity differs')
        for family in np.unique(families):
            if len(np.unique(folds[families==family]))!=1:raise ValueError('A family crosses folds')
        collection=PreparedCollection(summary,p,counts,folds,families,source_rows)
        bank=load_selection_bank(collection,'topic_learning',cfg)
        if (bank.base-counts).nnz:raise ValueError('Default topic-learning bank is not based on complete observed counts')
        posts=read_json(p/'posts.json')
        if len(posts)!=n:raise ValueError('Original post count differs')
        tokens=[r['tokens'] for r in posts] if row['corpus']=='chinese_pooled' else tokenize_reference([r['cleaned_text'] for r in posts])
        lookup={str(w):i for i,w in enumerate(words)};rr=[];cc=[];vv=[]
        for i,terms in enumerate(tokens):
            for word,count in Counter(terms).items():
                if word in lookup:rr.append(i);cc.append(lookup[word]);vv.append(count)
        rebuilt=sparse.csr_matrix((vv,(rr,cc)),shape=counts.shape,dtype=float)
        difference=(rebuilt-counts).tocsr()
        if difference.nnz and np.max(np.abs(difference.data))>1e-12:raise ValueError('Original tokens do not reproduce counts')
        binary=counts.astype(bool).astype(np.int64)
        identity=canonical_hash({'collection':summary['identity_sha256'],'tokens':tokens,'unaugmented':True})
        reference=TextReference(identity,row['corpus'],words,n,np.asarray(binary.sum(axis=0)).ravel(),
                    (binary.T@binary).tocsr(),tokens,p)
        result[row['corpus']]={'collection':collection,'bank':bank,'reference':reference,'vocabulary':words}
    return result

class Study:
    def __init__(self,project,step2,directory,protocol,descriptions,logger):
        self.project,self.step2,self.directory,self.protocol,self.logger=project,step2,directory,protocol,logger
        self.corpora=load_collections(step2,descriptions,protocol)
        self.canonical=read_json(step2/'provenance/step1_closeout/canonical_sources.json')
        self.models={};self.rows={};self.references={};self.timings=[];self.reproduction=[]
        self.fresh_fit_count=0;self.reused_fit_count=0
        for relative in self.canonical['model_directories']:
            p=inside(project,relative);info=read_json(p/'model.json')
            if info['topics']==12 and info['method'] in ('nmf_original','proposed'):
                key=(info['condition'],info['method'],int(info['seed']))
                if key in self.models:raise ValueError('Duplicate historical reproduction model')
                self.models[key]=(p,info)
        self.expected={}
        for p in sorted((step2/'provenance/historical_results').glob('*_seed_results.csv')):
            with p.open(encoding='utf-8-sig',newline='') as f:
                for row in csv.DictReader(f):
                    name=row.get('collection') or row.get('condition')
                    if name not in {'primary_full','chinese_pooled'} or row.get('study') not in {'main','chinese_full'}:continue
                    if int(row['K'])==12 and row['method'] in ('nmf_original','proposed'):
                        key=(name,row['method'],int(row['seed']))
                        if key in self.expected:raise ValueError('Duplicate historical default seed record')
                        self.expected[key]=row
        if len(self.models)!=20 or set(self.models)!=set(self.expected):
            raise ValueError('Expected 20 uniquely matched historical reproduction fits')

    def reference_bank(self,corpus,k):
        from threadpoolctl import threadpool_limits
        from transport_topics.proposed.reference import reference_bank
        key=(corpus,k)
        if key not in self.references:
            collection=self.corpora[corpus]['collection'];p=self.protocol['reference_predictor']
            settings={'folds':5,'seed':0,'tolerance':p['nmf_tolerance'],'max_iterations':p['nmf_max_iterations'],
                'smoothing':p['smoothing'],'unigram_smoothing':p['unigram_smoothing']}
            tick=time.perf_counter()
            with threadpool_limits(limits=1):
                refs,audits=reference_bank(self.directory,collection.summary,collection.counts,collection.folds,k,settings,self.logger)
            for ref,audit in zip(refs,audits):
                if not audit['converged'] or not audit['receiver_fold_excluded']:raise RuntimeError('Reference fit failed convergence or fold exclusion')
            self.references[key]=refs
            self.timings.append({'phase':'reference_bank','corpus':corpus,'K':k,'seconds':time.perf_counter()-tick,
                'scope':'five fold-excluded references; may include reuse'})
        return self.references[key]

    def select(self,corpus,k,only_proposed=False):
        from threadpoolctl import threadpool_limits
        from .selection import run_selection
        c=self.corpora[corpus];refs=self.reference_bank(corpus,k);tick=time.perf_counter()
        with threadpool_limits(limits=1):
            output=run_selection(c['collection'],c['bank'],refs,self.protocol,
                self.directory/'selection'/corpus/('K%d'%k),self.logger,
                requested=['proposed'] if only_proposed else None)
        self.timings.append({'phase':'policy_selection','corpus':corpus,'K':k,'seconds':time.perf_counter()-tick,
            'scope':'PEBR only' if only_proposed else 'combined policy loop; may reuse PEBR'})
        return output

    def fit(self,corpus,k,policy,seed,bank_path=None):
        import numpy as np
        from scipy import sparse
        from threadpoolctl import threadpool_limits
        from transport_topics.topics.learning import fit_topics
        from transport_topics.evaluation.topic_quality import evaluate_topics
        c=self.corpora[corpus];p=self.directory/'fits'/corpus/('K%d'%k)/policy/('seed%d'%seed)
        if complete(p):
            row=read_json(p/'result.json');self.rows[(corpus,k,policy,seed)]=row
            self.reused_fit_count+=1
            return row
        p.mkdir(parents=True,exist_ok=True)
        learned=c['collection'].counts if policy=='nmf_original' else sparse.load_npz(bank_path/'transferred.npz').tocsr()
        tick=time.perf_counter()
        self.logger.info('Fit | %s K=%d %s seed=%d',corpus,k,policy,seed)
        with threadpool_limits(limits=1):
            result=fit_topics(c['collection'].counts,learned,k,seed,self.protocol['final_nmf'])
        fit_seconds=time.perf_counter()-tick
        if not result.audit['fit']['converged']:
            write_json(p/'failed_fit.json',result.audit)
            raise RuntimeError('NMF did not converge; no settings changed automatically: '+str(p))
        result.validate(c['collection'].counts.shape[0],len(c['vocabulary']),k)
        tick=time.perf_counter()
        metrics=evaluate_topics(result.topic_scores,c['vocabulary'],c['reference'],10,True)
        evaluation_seconds=time.perf_counter()-tick
        for name,array in [('topic_scores',result.topic_scores),('doc_topics',result.doc_topics),
                          ('retrieval_features',result.retrieval_features),('source_rows',c['collection'].source_rows),
                          ('vocabulary',c['vocabulary'])]:np.save(p/(name+'.npy'),array,allow_pickle=False)
        write_json(p/'audit.json',result.audit);write_json(p/'quality.json',metrics)
        row={'corpus':corpus,'K':k,'policy':policy,'mask_base':'not_applicable' if policy=='nmf_original' else 2711,'seed':seed,
            'npmi':metrics['npmi'],'cv':metrics['cv'],'diversity':metrics['diversity'],
            'top_words':metrics['top_words'],'fit_seconds':fit_seconds,'evaluation_seconds':evaluation_seconds,
            'path':p.relative_to(self.directory).as_posix(),'fit_converged':True,
            'selection_banks':1,'scope':'default_bank_revision_not_selection_robustness'}
        write_json(p/'result.json',row);save_receipt(p)
        self.fresh_fit_count+=1
        self.rows[(corpus,k,policy,seed)]=row
        self.logger.info('Result | %s K=%d %s seed=%d | NPMI %.6f C_v %.6f TD %.6f',
            corpus,k,policy,seed,row['npmi'],row['cv'],row['diversity'])
        return row

    def reproduce(self):
        import numpy as np
        from scipy import sparse
        from .analysis import export
        initial_fits=self.fresh_fit_count
        for corpus in self.protocol['corpora']:
            c=self.corpora[corpus];paths=self.select(corpus,12,True)
            name=c['collection'].summary['name']
            old=[inside(self.project,r) for r in self.canonical['action_directories']
                 if '/'+name+'/K12/topic_learning/' in '/'+r]
            if len(old)!=1 or not complete(old[0]):raise ValueError('No unique intact historical PEBR action bank')
            records=read_json(paths['proposed']/'actions.json')
            selected=np.load(old[0]/'selected_donors.npy',allow_pickle=False)
            offered=np.load(old[0]/'offered_beta.npy',allow_pickle=False)
            actual=np.load(old[0]/'actual_beta.npy',allow_pickle=False)
            mismatch=[i for i,r in enumerate(records) if tuple(r['donors'])!=tuple(int(j) for j in selected[i] if j>=0)]
            beta_error=float(np.max(np.abs(offered-np.asarray([r['offered_beta'] for r in records]))))
            actual_error=float(np.max(np.abs(actual-np.asarray([r['actual_beta'] for r in records]))))
            diff=(sparse.load_npz(old[0]/'transferred.npz')-sparse.load_npz(paths['proposed']/'transferred.npz')).tocsr()
            transfer_error=float(np.max(np.abs(diff.data))) if diff.nnz else 0.
            check={'check':'fresh_action_bank','corpus':corpus,'K':12,'donor_mismatches':len(mismatch),
                'offered_beta_max_error':beta_error,'actual_beta_max_error':actual_error,
                'transferred_max_error':transfer_error,'tolerance':2e-12,
                'passed':not mismatch and max(beta_error,actual_error,transfer_error)<=2e-12}
            self.reproduction.append(check);write_json(self.directory/'reproduction.json',self.reproduction)
            if not check['passed']:raise ArithmeticError('Default PEBR actions do not reproduce; comparison is blocked')
            for policy in ('nmf_original','proposed'):
                for seed in self.protocol['final_model_seeds']:
                    row=self.fit(corpus,12,policy,seed,paths.get(policy))
                    saved_path,info=self.models[(name,policy,seed)]
                    if not complete(saved_path):raise ValueError('A historical fitted model changed')
                    expected=self.expected[(name,policy,seed)]
                    if expected['model_identity']!=info['identity']:raise ValueError('Historical table-to-model identity mismatch')
                    metric_error=max(abs(float(expected[m])-row[m]) for m in ('npmi','cv','diversity'))
                    fresh=self.directory/row['path'];array_errors={}
                    for array in ('topic_scores','doc_topics','retrieval_features'):
                        a=np.load(saved_path/(array+'.npy'),allow_pickle=False)
                        b=np.load(fresh/(array+'.npy'),allow_pickle=False)
                        if a.shape!=b.shape:raise ValueError('Reproduced output dimensions differ')
                        error=float(np.max(np.abs(a-b)));scale=max(1.,float(np.max(np.abs(a))))
                        array_errors[array]=error/scale
                    from transport_topics.evaluation.topic_quality import top_words
                    same_words=top_words(np.load(saved_path/'topic_scores.npy',allow_pickle=False),c['vocabulary']).tolist()==row['top_words']
                    passed=metric_error<=2e-6 and max(array_errors.values())<=2e-7 and same_words
                    self.reproduction.append({'check':'fresh_final_fit','corpus':corpus,'K':12,'policy':policy,'seed':seed,
                        'metric_max_error':metric_error,'normalized_array_errors':array_errors,
                        'top_words_identical':same_words,'metric_tolerance':2e-6,'array_tolerance':2e-7,'passed':passed})
                    write_json(self.directory/'reproduction.json',self.reproduction)
                    if not passed:raise ArithmeticError('Fresh final fit does not reproduce the saved default; comparison is blocked')
            export(self.directory/'results',list(self.rows.values()),self.protocol)
            self.logger.info('REPRODUCTION PASSED | %s | complete K=12 PEBR bank and ten fresh final fits',corpus)
        return {'passed':True,'checks':len(self.reproduction),'verified_final_fits':20,
            'fits_computed_this_invocation':self.fresh_fit_count-initial_fits,'verified_pebr_action_banks':2}

    def compare(self):
        from .analysis import export
        for corpus in self.protocol['corpora']:
            for k in self.protocol['topic_counts']:
                paths=self.select(corpus,k)
                for policy in self.protocol['policies']:
                    for seed in self.protocol['final_model_seeds']:
                        self.fit(corpus,k,policy,seed,paths.get(policy))
                        export(self.directory/'results',list(self.rows.values()),self.protocol)
        rows=list(self.rows.values())
        if len(rows)!=300:raise ValueError('Expected exactly 300 default-bank final fits')
        export(self.directory/'results',rows,self.protocol,True)
        return {'final_fits':300,'nontrivial_policy_banks':54,'selection_mask_bases':[2711],
                'five_mask_study_complete':False,'protected_word_audits_run':False}

def execute(project, explicit=None, mode='compare'):
    project=Path(project).resolve();directory=None;logger=None;lock=None;started=time.perf_counter()
    report={'version':VERSION,'mode':mode,'status':'starting','full_revision_complete':False}
    try:
        if not (project/'main.py').is_file():raise ValueError('Run beside your existing main.py')
        step2=resolve_step2(project,explicit)
        accepted,protocol,descriptions,manifest=verify_step2(step2)
        environment=validate_environment(read_json(step2/'environment.json'))
        own=Path(__file__).parent
        source={p.name:file_hash(p) for p in sorted(own.glob('*.py'))}
        source['pebr_step3_run.py']=file_hash(own.parent/'pebr_step3_run.py')
        identity={'version':VERSION,'protocol':PROTOCOL_SHA256,'snapshot':manifest['snapshot_hash'],
            'runner_source':source,'environment':environment,'executed_mask_base':2711}
        key=canonical_hash(identity)
        directory=project/'outputs/revision_study/step3'/(step2.name+'_'+key[:12])
        directory.mkdir(parents=True,exist_ok=True)
        lock=acquire_run_lock(directory/'RUNNING.lock')
        logger=logging.getLogger('pebr_step3_'+key);logger.setLevel(logging.INFO);logger.propagate=False
        formatter=logging.Formatter('%(asctime)s | %(message)s',datefmt='%Y-%m-%dT%H:%M:%S')
        for handler in (logging.StreamHandler(sys.stdout),logging.FileHandler(directory/'console_log.txt',encoding='utf-8')):
            handler.setFormatter(formatter);logger.addHandler(handler)
        logger.info('PEBR STEP 3 | v%s | %s | default mask 2711; no original files replaced',VERSION,mode)
        logger.info('Verified complete Step 2 | %d frozen files | both corpora',len(manifest['files']))
        write_json(directory/'identity.json',identity)
        for p in own.glob('*.py'):
            target=directory/'runner_source'/own.name/p.name;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copyfile(p,target)
        target=directory/'runner_source/pebr_step3_run.py'
        if not target.exists():shutil.copyfile(own.parent/'pebr_step3_run.py',target)
        bind_frozen_source(step2)
        report.update(step2_workspace=str(step2),step2_accepted=True,protocol_sha256=PROTOCOL_SHA256,
            scientific_source_modified=False,original_experiments_replaced=False,
            historical_K16_modified=False,selection_mask_bases_executed=[2711],
            selection_randomness_study_complete=False,protected_word_study_complete=False,
            baseline_tuning_review_complete=False,runner_identity=key)
        report['timing_scope']='This invocation includes new reference fitting, selection, final fitting and reporting. Frozen retrieval is reused; this is not a fresh end-to-end encoder/retrieval timing.'
        study=Study(project,step2,directory,protocol,descriptions,logger)
        reproduction=study.reproduce();report['reproduction']=reproduction
        write_json(directory/'step3_report.json',report)
        if mode=='compare':
            report['comparison']=study.compare();report['status']='default_bank_comparison_complete'
        else:report['status']='reproduction_complete'
        report['completed_final_fits_available']=len(study.rows)
        report['fits_computed_this_invocation']=study.fresh_fit_count
        report['completed_fit_artifact_reuses']=study.reused_fit_count
        write_csv(directory/'phase_timings.csv',study.timings)
        # Detect any accidental write to the isolated source/data snapshot.
        verify_step2(step2)
        report['step2_snapshot_reverified_after_run']=True
        logger.info('STEP 3 COMPLETE | %s | %d fitted/evaluated outputs available',report['status'],len(study.rows))
        logger.info('Selection-mask repeats, protected words and baseline tuning remain later stages; no superiority assumed.')
        return_code=0
    except BaseException as error:
        if isinstance(error,(KeyboardInterrupt,SystemExit)):message='Interrupted; completed receiver/model checkpoints are resumable.'
        else:message=type(error).__name__+': '+str(error)
        report.update(status='blocked',error=message)
        if directory is not None and logger is not None:
            (directory/'exception.txt').write_text(traceback.format_exc(),encoding='utf-8')
        if logger:logger.error('STEP 3 BLOCKED | %s',message)
        else:print('STEP 3 BLOCKED | '+message,flush=True)
        return_code=1
    finally:
        release_run_lock(lock)
        if directory is not None and logger is not None:
            report['invocation_elapsed_seconds']=time.perf_counter()-started
            write_json(directory/'step3_report.json',report)
            (directory/'READ_ME_FIRST.md').write_text(
                '# Step 3: default-bank controlled comparison\n\nStatus: '+report['status']+
                '\n\nThis stage uses the frozen default selection bank (mask base 2711), not five independent selection repeats. '
                'Final NMF seeds 0–4 assess fitting variation. Single-seed tables omit stability; five-seed tables use ten matched seed pairs. '
                'The default K=12 PEBR actions and 20 final fits are reproduced before other comparisons. '
                'All ten policies are retained at K=8,10,12 on both corpora in comparison mode.\n\n'
                'Inspect results/RESULTS.md, results/single_seed0_results.csv, results/multi_seed_summary.csv and '
                'results/paired_pebr_contrasts.csv. Negative and positive comparisons are both retained.\n\n'
                'Selection-repeat robustness, protected-word replication, reference/fold sensitivity and neural-baseline tuning '
                'are not completed by this stage. No favorable outcome or journal score is implied. '
                'Original source, prepared inputs and historical experiments remain unchanged. '
                'The frozen snapshot is verified again after successful execution.\n',encoding='utf-8')
            archive=directory/(directory.name+'_step3_inspection.zip')
            logger.info('RESULTS DIRECTORY: %s',directory)
            logger.info('SEND THIS INSPECTION ZIP: %s',archive)
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
                for name in ('step3_report.json','identity.json','reproduction.json','phase_timings.csv',
                             'READ_ME_FIRST.md','console_log.txt','exception.txt'):
                    p=directory/name
                    if p.is_file():z.write(p,'step3/'+name)
                for p in (directory/'results').glob('*'):
                    if p.is_file() and p.suffix in {'.csv','.md','.json'}:z.write(p,'results/'+p.name)
                for p in (directory/'runner_source').rglob('*.py'):z.write(p,p.relative_to(directory).as_posix())
                for p in (directory/'selection').glob('*/*/*/summary.json'):z.write(p,p.relative_to(directory).as_posix())
            logger.info('Inspection ZIP saved: %.2f MiB',archive.stat().st_size/1024**2)
            for handler in list(logger.handlers):handler.close();logger.removeHandler(handler)
    return return_code
