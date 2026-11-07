"""Chinese input adapters with the original NMF and published model budgets."""
from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from ..baselines.base import ModelOutput
from ..baselines.registry import NMF_VARIANTS, fit_model
from ..config import canonical_hash
from ..data.preparation import check_receipt, complete, save_array
from ..runtime.artifacts import file_hash, write_json


def ordered_chinese_documents(posts,vocabulary):
    mapping={str(word):i for i,word in enumerate(vocabulary)}
    return [np.asarray([mapping[w] for w in post['tokens'] if w in mapping],dtype=np.int32) for post in posts]


def fit_btm(counts,posts,vocabulary,topics,seed,settings):
    import bitermplus as btm
    docs=ordered_chinese_documents(posts,vocabulary)
    for i,doc in enumerate(docs):
        if not np.array_equal(np.bincount(doc,minlength=len(vocabulary)),counts[i].toarray().ravel()):
            raise AssertionError('Ordered Chinese BTM tokens do not reproduce original counts')
    biterms=btm.get_biterms(docs,win=settings['window'],as_array=True)
    if not any(len(row) for row in biterms):raise ValueError('No Chinese biterms were constructed')
    model=btm.BTM(counts,np.asarray(vocabulary),T=topics,M=min(20,len(vocabulary)),
                  alpha=settings['alpha_mass']/topics,beta=settings['beta'],seed=int(seed),
                  win=settings['window'],has_background=False)
    model.fit(biterms,iterations=settings['iterations'],verbose=False)
    theta=np.asarray(model.transform(docs,infer_type='sum_b',verbose=False),dtype=np.float64)
    beta=np.asarray(model.matrix_topics_words_,dtype=np.float64)
    return ModelOutput(beta,theta,theta.copy(),{'implementation':'bitermplus','version':metadata.version('bitermplus'),
            'language_input':'ordered retained Jieba segments; never the English word regex',
            'iterations':settings['iterations'],'native_seed':seed,'window':settings['window'],
            'alpha':settings['alpha_mass']/topics,'beta':settings['beta'],
            'biterms':sum(len(r) for r in biterms),'training_device':'cpu',
            'termination':'fixed Gibbs iteration budget','annotations_used':False})


def fit_chinese_model(root,collection,method,topics,seed,protocol,device,logger,learning_path=None,control_identity=None):
    if method in NMF_VARIANTS:
        environment={'device_selection':{'torch_device_selected':device}}
        return fit_model(root,collection,method,topics,seed,protocol,environment,logger,
                         learning_path=learning_path,control_identity=control_identity)
    if method not in ('btm','knntm','glocom'):raise ValueError('Unknown Chinese model: '+method)
    settings=protocol['model_settings'][method]
    identity={'schema':1,'collection':collection.summary['identity_sha256'],'method':method,
              'topics':topics,'seed':seed,'settings':settings,'language':'zh',
              'device':device if method!='btm' else 'cpu','scope':'full',
              'adapter_hashes':{name:file_hash(Path(__file__).with_name(name))
                                for name in ('models.py','native_training.py','embeddings.py')},
              'native_input_code':file_hash(Path(__file__).parents[1]/'baselines/inputs.py'),
              'published_source_registry':file_hash(Path(__file__).parents[1]/'baselines/sources.py'),
              'versions':{name:metadata.version(name) for name in ('numpy','scipy','scikit-learn')},
              'word_vectors':'fixed multilingual encoder; deterministic PCA input adapter, native dimensions'}
    if method in ('knntm','glocom'):identity['versions']['torch']=metadata.version('torch')
    if method=='btm':identity['versions']['bitermplus']=metadata.version('bitermplus')
    if method=='knntm':identity['versions']['POT']=metadata.version('POT')
    key=canonical_hash(identity)
    directory=root/'outputs/models/chinese_topics'/collection.summary['name']/method/f'K{topics}'/f'seed{seed}'/key[:16]
    directory.mkdir(parents=True,exist_ok=True)
    if check_receipt(directory):
        info=json.loads((directory/'model.json').read_text(encoding='utf-8'))
        checkpoint=info['audit'].get('checkpoint')
        if checkpoint and file_hash(root/checkpoint)!=info['audit']['checkpoint_sha256']:
            raise ValueError('Chinese native model checkpoint changed')
        logger.info('Reusing complete Chinese model | %s K=%d seed=%d.',method,topics,seed);return info
    logger.info('CHINESE MODEL START | %s | %s K=%d seed=%d | full fixed budget.',collection.summary['name'],method,topics,seed)
    started=time.perf_counter()
    vocabulary=np.load(collection.directory/'vocabulary.npy',allow_pickle=False)
    if method=='btm':
        posts=json.loads((collection.directory/'posts.json').read_text(encoding='utf-8'))
        with threadpool_limits(limits=1):result=fit_btm(collection.counts,posts,vocabulary,topics,seed,settings)
    else:
        from .native_training import fit
        with threadpool_limits(limits=1):
            result=fit(root,collection,method,topics,seed,settings,protocol['integration'],device,
                       root/'outputs/checkpoints/chinese_models'/key,key,logger)
    result.validate(collection.counts.shape[0],collection.counts.shape[1],topics)
    for name,array in [('topic_scores',result.topic_scores),('doc_topics',result.doc_topics),
                       ('retrieval_features',result.retrieval_features),('source_rows',collection.source_rows),('vocabulary',vocabulary)]:
        save_array(directory/(name+'.npy'),array)
    info={'method':method,'condition':collection.summary['name'],'condition_identity':collection.summary['identity_sha256'],
          'topics':topics,'seed':seed,'scope':'full','development_only':False,'identity':key,
          'path':directory.relative_to(root).as_posix(),'posts':collection.counts.shape[0],
          'vocabulary':len(vocabulary),'retrieval_dimension':topics,'topic_model':True,
          'audit':result.audit,'seconds':time.perf_counter()-started,'performance_metrics_computed':False}
    write_json(directory/'identity.json',identity,overwrite=True);write_json(directory/'model.json',info,overwrite=True)
    complete(directory)
    logger.info('CHINESE MODEL COMPLETE | %s K=%d seed=%d | %.2fs.',method,topics,seed,info['seconds'])
    return info
