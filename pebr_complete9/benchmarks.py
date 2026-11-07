"""Complete K10 and a fixed 200-to-400 epoch budget check; no hyperparameter search."""
from contextlib import contextmanager
import copy
from pathlib import Path
import shutil
import numpy as np
from .common import read,write,table,key,digest,inside,csv_rows
from .core import save_model,shared,summary

class Benchmarks:
    def __init__(self,project,step7,step5,step3,work,corpora,protocol,logger):
        self.project,self.step7,self.step5,self.step3,self.work=project,step7,step5,step3,work
        self.corpora,self.protocol,self.logger=corpora,protocol,logger
        self.metadata=read(step7/'baseline_metadata.json');self.old={};self.contexts={}
        for r in self.metadata:
            c='chinese_pooled' if r['condition']=='chinese_pooled' else 'transport_grievance'
            self.old[(c,int(r['topics']),r['method'],int(r['seed']))]=r
        self.rows=[];self.histories=[]

    def preflight(self):
        from importlib import metadata
        from transport_topics.baselines.sources import FILES,COMMIT
        import torch
        if metadata.version('bitermplus')!='1.0.0':raise ValueError('Expected audited bitermplus 1.0.0')
        self.source_dir=self.project/('third_party/published/glocom_'+COMMIT[:12])
        for f,sha in FILES.items():
            if digest(self.source_dir/f)!=sha:raise ValueError('Pinned native source changed: '+f)
        for corpus,c in self.corpora.items():
            for method in ('knntm','glocom'):
                info=self.old[(corpus,12,method,0)];settings=info['audit']['settings']
                for k in (8,12):
                    for seed in range(5):
                        old=self.old[(corpus,k,method,seed)]
                        if old['audit']['settings']!=settings:raise ValueError('Historical settings vary across model seeds/K')
                        if digest(inside(self.project,old['audit']['checkpoint']))!=old['audit']['checkpoint_sha256']:
                            raise ValueError('An original neural checkpoint changed')
                if info['audit']['training_device'].startswith('cuda') and not torch.cuda.is_available():
                    raise RuntimeError('Audited baseline device is CUDA; no automatic CPU training substitution')
                if corpus=='chinese_pooled':from transport_topics.chinese.embeddings import load_word_vectors
                else:from transport_topics.representations.word_embeddings import load_word_vectors
                _,audit=load_word_vectors(self.project,c['vocabulary'],300 if method=='knntm' else 200)
                if audit['aligned_values_sha256']!=info['audit']['input_preparation']['word_vectors']['aligned_values_sha256']:
                    raise ValueError('Baseline word-vector alignment differs')
                if method=='knntm':self.ot_cache(info)
        return {'verified_historical_model_records':len(self.old),'bitermplus':'1.0.0','torch':torch.__version__,
                'main_neural_budget_precommitted':400,'old_checkpoints_preserved':True}

    def ot_cache(self,info):
        from transport_topics.data.preparation import check_receipt
        audit=info['audit']['input_preparation']['transport']
        path=self.project/'cache/baseline_inputs/knntm'/audit['identity'][:20]
        if not check_receipt(path):raise FileNotFoundError('Audited OT cache is missing; this runner will not rebuild original caches')
        return path,read(path/'audit.json')

    @contextmanager
    def readonly_inputs(self,module,corpus,info):
        from unittest.mock import patch
        from transport_topics.baselines.inputs import global_context
        def sources(root,logger,download=True):
            if Path(root).resolve()!=self.project:raise ValueError('Native input root changed')
            for f in info['audit']['implementation']['files']:
                if digest(inside(root,f['path']))!=f['sha256']:raise ValueError('Native source changed')
            return self.source_dir,copy.deepcopy(info['audit']['implementation'])
        def distances(*args,**kwargs):return self.ot_cache(info)
        def context(counts,embeddings,clusters,seed):
            ck=(corpus,int(clusters),int(seed))
            if ck not in self.contexts:self.contexts[ck]=global_context(counts,embeddings,clusters,seed)
            return self.contexts[ck]
        with patch.object(module,'prepare_sources',sources),patch.object(module,'transport_distances',distances),patch.object(module,'global_context',context):
            yield

    def old_record(self,corpus,k,method,seed):
        from pebr_revision_step3.io import complete
        from transport_topics.evaluation.topic_quality import evaluate_topics
        info=self.old[(corpus,k,method,seed)];directory=inside(self.project,info['path'])
        if not complete(directory):raise ValueError('Original baseline model receipt failed')
        q=evaluate_topics(np.load(directory/'topic_scores.npy',allow_pickle=False),self.corpora[corpus]['vocabulary'],self.corpora[corpus]['reference'],10,True)
        return dict(corpus=corpus,K=k,policy=method,seed=seed,epoch_budget=0 if method=='btm' else 200,
                    iteration_budget=1000 if method=='btm' else 0,npmi=q['npmi'],cv=q['cv'],diversity=q['diversity'],
                    top_words=q['top_words'],path=str(directory),source='verified_historical_baseline',fit_identity=info['identity'])

    def checkpoint_export(self,directory,checkpoint,stream_identity,model,corpus,info):
        import torch
        from pebr_review7.native_math import beta,theta
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if state['identity']!=stream_identity or int(state['epoch']) not in (200,400):raise ValueError('Neural stream identity/epoch differs')
        losses=state['losses'];epochs=int(state['epoch'])
        if [r['epoch'] for r in losses]!=list(range(1,epochs+1)) or not all(np.isfinite(r['loss']) for r in losses):raise ValueError('Neural loss coverage/nonfiniteness failed')
        cpu={name:value.detach().cpu() for name,value in state['model'].items() if name not in ('M_cos_dist','M_coo_dist','theta_bank')}
        method='knntm' if 'fcd1.weight' in cpu else 'glocom'
        arrays={n:v.numpy() for n,v in cpu.items()};settings=info['audit']['settings']
        actual=beta(arrays,method,settings.get('beta_temperature',.2))
        error=float(np.max(np.abs(actual-model.topic_scores)))
        if error>3e-5:raise ValueError('New native decoder does not reproduce its checkpoint')
        contexts=self.contexts[(corpus,settings['clusters'],settings['cluster_seed'])][0] if method=='glocom' else None
        reproduced=theta(self.corpora[corpus]['collection'].counts,arrays,method,contexts)
        theta_error=float(np.max(np.abs(reproduced-model.doc_topics)))
        if theta_error>3e-5:raise ValueError('New native memberships do not reproduce their checkpoint')
        compact=directory/'native_inference.pt';directory.mkdir(parents=True,exist_ok=True)
        torch.save({'native_stream_identity':stream_identity,'epoch':epochs,'model':cpu,
                    'omitted_training_only_tensors':['M_cos_dist','M_coo_dist','theta_bank'],
                    'static_input_source':'verified original count/OT/vector caches; unchanged'},compact)
        if directory.name=='e400_s0' and directory.parent.parent.name=='K12':
            names=('fcd1.weight',) if method=='knntm' else ('topic_embeddings','word_embeddings')
            np.savez_compressed(directory/'decoder_sample.npz',**{n:cpu[n].numpy() for n in names})
        table(directory/'training_losses.csv',losses)
        return {'path':str(compact),'sha256':digest(compact),'epoch':epochs,'native_decoder_max_error':error,'native_membership_max_error':theta_error,
                'temporary_training_checkpoint_sha256':digest(checkpoint),'optimizer_checkpoint_retained_after_completion':False}

    def prune(self,directory):
        directory.resolve().relative_to(self.work.resolve())
        for p in directory.glob('epoch_*.pt'):p.unlink()
        (directory/'latest.json').unlink(missing_ok=True)

    def neural(self,corpus,k,method,seed,budget,source200=None):
        from pebr_revision_step3.io import complete,save_receipt
        directory=self.work/'bench_models'/corpus/('K%d'%k)/method/('e%d_s%d'%(budget,seed))
        if complete(directory):return read(directory/'result.json')
        info=self.old[(corpus,12,method,0)];settings=copy.deepcopy(info['audit']['settings'])
        base_identity=(source200['stream_identity'] if source200 is not None else key({'corpus':corpus,'K':k,'method':method,'seed':seed,
            'settings':settings,'input_identity':self.corpora[corpus]['collection'].summary['identity_sha256'],'frozen_source':'step2'}))
        checkdir=self.work/'checkpoints'/corpus/('K%d'%k)/method/('e%d_s%d'%(budget,seed));checkdir.mkdir(parents=True,exist_ok=True)
        if budget==400 and not (checkdir/'latest.json').exists():
            if source200 is None:raise ValueError('Budget extension requires its 200-epoch source')
            src=Path(source200['checkpoint'])
            if digest(src)!=source200['sha256']:raise ValueError('Source-200 checkpoint changed')
            dst=checkdir/'epoch_0200.pt';shutil.copyfile(src,dst)
            write(checkdir/'latest.json',{'identity':base_identity,'epoch':200,'filename':dst.name,'sha256':digest(dst)})
        if corpus=='chinese_pooled':from transport_topics.chinese import native_training as native
        else:from transport_topics.baselines import neural as native
        self.logger.info('Baseline | %s %s K%d seed%d budget%d',corpus,method,k,seed,budget)
        integration={'download_sources':False,'checkpoint_epochs':20}
        with self.readonly_inputs(native,corpus,info):
            model=native.fit(self.project,self.corpora[corpus]['collection'],method,k,seed,settings,integration,
                info['audit']['training_device'],checkdir,base_identity,self.logger,epochs=budget)
        checkpoint=inside(self.project,model.audit['checkpoint'])
        receipt=self.checkpoint_export(directory,checkpoint,base_identity,model,corpus,info)
        raw=model.audit.pop('checkpoint');sha=model.audit.pop('checkpoint_sha256')
        model.audit.update(native_stream_identity=base_identity,inference_checkpoint=receipt,
            temporary_optimizer_checkpoint={'path_at_completion':raw,'sha256':sha,'pruned_after_verified_export':budget==400},
            unchanged_training_settings=settings,actual_epoch_budget=budget)
        record=save_model(directory,model,self.corpora[corpus],self.corpora[corpus]['reference'],k,seed,method,
                         {'corpus':corpus,'epoch_budget':budget,'iteration_budget':0,'native_stream_identity':base_identity})
        if budget==200:write(directory/'continuation_source.json',{'checkpoint':str(checkpoint),'sha256':sha,'stream_identity':base_identity})
        if budget==400:
            self.prune(checkdir)
            if k==10:
                previous=self.work/'checkpoints'/corpus/'K10'/method/('e200_s%d'%seed)
                self.prune(previous)
        # Rebuild the receipt after auxiliary continuation metadata was written.
        save_receipt(directory)
        return record

    def btm(self,corpus,k,seed):
        from pebr_revision_step3.io import complete
        directory=self.work/'bench_models'/corpus/('K%d'%k)/'btm'/('s%d'%seed)
        if complete(directory):return read(directory/'result.json')
        c=self.corpora[corpus];settings={'iterations':1000,'alpha_mass':50.,'beta':.01,'window':15}
        posts=read(c['collection'].directory/'posts.json')
        if corpus=='chinese_pooled':from transport_topics.chinese.models import fit_btm as fit
        else:from transport_topics.baselines.btm import fit
        model=fit(c['collection'].counts,posts,c['vocabulary'],k,seed,settings)
        return save_model(directory,model,c,c['reference'],k,seed,'btm',{'corpus':corpus,'epoch_budget':0,'iteration_budget':1000})

    def run(self):
        for corpus in self.corpora:
            for k in self.protocol['topic_counts']:
                for policy in ('nmf_original','nmf_similarity','preserve_forward_fixed_cap'):
                    for seed in range(5):self.rows.append(shared(self.step5,self.step3,corpus,k,policy,seed,{'epoch_budget':0,'iteration_budget':0}))
                for seed in range(5):
                    self.rows.append(self.btm(corpus,k,seed) if k==10 else self.old_record(corpus,k,'btm',seed))
                for method in ('knntm','glocom'):
                    for seed in range(5):
                        if k==10:
                            record=self.neural(corpus,k,method,seed,200)
                            source=read(Path(record['path'])/'continuation_source.json')
                        else:
                            record=self.old_record(corpus,k,method,seed);old=self.old[(corpus,k,method,seed)]
                            source={'checkpoint':str(inside(self.project,old['audit']['checkpoint'])),
                                    'sha256':old['audit']['checkpoint_sha256'],'stream_identity':old['identity']}
                        self.rows.append(record)
                        self.rows.append(self.neural(corpus,k,method,seed,400,source))
                self.export(False)
        return self.export(True)

    def export(self,require):
        directory=self.work/'benchmark_results';directory.mkdir(exist_ok=True)
        unique={(r['corpus'],r['K'],r['policy'],r['seed'],r['epoch_budget']) for r in self.rows}
        if len(unique)!=len(self.rows):raise ValueError('Duplicate benchmark record')
        expected={(c,k,p,s,e) for c in self.corpora for k in (8,10,12) for s in range(5)
                  for p,e in [('nmf_original',0),('nmf_similarity',0),('preserve_forward_fixed_cap',0),('btm',0),
                              ('knntm',200),('knntm',400),('glocom',200),('glocom',400)]}
        if not unique.issubset(expected):raise ValueError('Undeclared benchmark case')
        ok=unique==expected
        write(directory/'seed_records.json',self.rows)
        table(directory/'seed_metrics.csv',[{k:v for k,v in r.items() if k!='top_words'} for r in self.rows])
        # Only complete five-seed groups are summarized during a partial invocation.
        groups={tuple(r[k] for k in ('corpus','K','policy','epoch_budget')) for r in self.rows}
        complete_rows=[r for r in self.rows if sum(tuple(x[k] for k in ('corpus','K','policy','epoch_budget'))==tuple(r[k] for k in ('corpus','K','policy','epoch_budget')) for x in self.rows)==5]
        table(directory/'summaries.csv',summary(complete_rows,['corpus','K','policy','epoch_budget']))
        write(directory/'coverage.json',{'complete':ok,'records':len(unique),'expected':240,'new_fits_planned':90,
              'main_neural_budget':400,'optimizer_temporaries_pruned_after_inference_export':True})
        if require and not ok:raise ValueError('Incomplete benchmark grid')
        return ok
