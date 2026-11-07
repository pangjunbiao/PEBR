"""Verified model I/O and shared NMF/evaluation helpers."""
from pathlib import Path
import time
import numpy as np
from .common import read,write,key,digest
def save_model(directory,model,corpus,reference,k,seed,policy,context,source='new_fit'):
    from pebr_revision_step3.io import save_receipt
    from transport_topics.evaluation.topic_quality import evaluate_topics
    collection=corpus['collection'];words=corpus['vocabulary']
    model.validate(collection.counts.shape[0],len(words),k)
    q=evaluate_topics(model.topic_scores,words,reference,10,True)
    directory.mkdir(parents=True,exist_ok=True)
    for name,array in [('topic_scores',model.topic_scores),('doc_topics',model.doc_topics),('retrieval_features',model.retrieval_features),
                       ('source_rows',collection.source_rows),('vocabulary',words)]:np.save(directory/(name+'.npy'),array,allow_pickle=False)
    write(directory/'audit.json',model.audit);write(directory/'quality.json',q)
    record=dict(context,corpus=context.get('corpus',collection.summary['name']),K=k,seed=seed,policy=policy,
        npmi=q['npmi'],cv=q['cv'],diversity=q['diversity'],top_words=q['top_words'],path=str(directory),source=source,
        fit_identity=key({'collection':collection.summary['identity_sha256'],'K':k,'seed':seed,'policy':policy,
                          'context':{a:b for a,b in context.items() if not a.endswith('_seconds')}}))
    write(directory/'result.json',record);save_receipt(directory)
    return record
def fit_nmf(directory,corpus,learning,k,seed,policy,context,settings):
    from pebr_revision_step3.io import complete
    from transport_topics.topics.learning import fit_topics
    if complete(directory):return read(directory/'result.json')
    tick=time.perf_counter();model=fit_topics(corpus['collection'].counts,learning,k,seed,settings)
    if not model.audit['fit']['converged']:
        write(directory/'failed_fit.json',model.audit);raise RuntimeError('NMF did not converge; no automatic seed or tolerance change')
    return save_model(directory,model,corpus,corpus['reference'],k,seed,policy,dict(context,fit_seconds=time.perf_counter()-tick))
def original_path(step5,step3,corpus,k,policy,seed):
    return (step3/'fits'/corpus/('K%d'%k)/policy/('seed%d'%seed) if policy=='nmf_original'
            else step5/'fits'/corpus/'mask3711'/('K%d'%k)/policy/('seed%d'%seed))
def shared(step5,step3,corpus,k,policy,seed,context):
    from pebr_revision_step3.io import complete
    path=original_path(step5,step3,corpus,k,policy,seed)
    if not complete(path):raise ValueError('Shared model receipt failed: '+str(path))
    r=read(path/'result.json')
    if (r['corpus'],int(r['K']),r['policy'],int(r['seed']))!=(corpus,k,policy,seed):raise ValueError('Shared model identity differs')
    return dict(context,corpus=corpus,K=k,policy=policy,seed=seed,npmi=r['npmi'],cv=r['cv'],diversity=r['diversity'],
                top_words=r['top_words'],path=str(path),source='shared_verified_fit',fit_identity=digest(path/'complete.json'))
def summary(rows,fields):
    from collections import defaultdict
    from transport_topics.evaluation.stability import seed_stability
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[f] for f in fields)].append(r)
    result=[]
    for group,rr in sorted(groups.items(),key=lambda x:str(x[0])):
        if len(rr)!=5 or {int(r['seed']) for r in rr}!=set(range(5)):raise ValueError('A summary needs exactly five unique final seeds')
        item=dict(zip(fields,group));item['final_seeds']=5
        for f in ('npmi','cv','diversity'):
            item[f+'_mean']=float(np.mean([r[f] for r in rr]));item[f+'_sd']=float(np.std([r[f] for r in rr],ddof=1))
        item['stability']=seed_stability({int(r['seed']):r['top_words'] for r in rr})['stability'];result.append(item)
    return result
