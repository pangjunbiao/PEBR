"""Fit registered models and save one aligned, checksummed output contract."""
from copy import deepcopy
from importlib import metadata
from pathlib import Path
import json
import time
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from ..config import canonical_hash
from ..schemas import number
from ..data.preparation import check_receipt, complete, save_array
from ..runtime.artifacts import file_hash, write_json

MAIN_METHODS = ('proposed','nmf_original','nmf_similarity','btm','knntm','glocom','minilm_retrieval')
NMF_VARIANTS = {'proposed':'proposed','nmf_original':'no_borrowing','nmf_similarity':'similarity_maximum',
                'similarity_matched':'similarity_matched','full_pool_loo':'full_pool_loo','no_downside':'no_downside'}


def enabled_methods(config):
    requested = config['method']['name']
    if requested != 'proposed':
        if requested not in MAIN_METHODS:
            raise ValueError('Unregistered main model: '+requested)
        return [requested]
    methods = ['proposed']+[name for name in MAIN_METHODS[1:] if config['baselines'][name]]
    if config['studies']['ablations']['enabled']:
        variants = config['studies']['ablations']['variants']
        methods += [name for name in ('similarity_matched','full_pool_loo','no_downside') if variants[name]]
    return methods


def model_settings(config, method):
    values = deepcopy(config['model_settings'].get(method,{}))
    for key in ('alpha_mass','beta','learning_rate','augmentation','prior_variance','ecr_weight','alpha','eta','rho','dropout','beta_temperature','sinkhorn_alpha'):
        if key in values: values[key] = number(values[key],method+'.'+key)
    return values


def fit_model(root, collection, method, topics, seed, config, environment, logger,
              learning_path=None, control_identity=None):
    smoke = config['integration']['scope']=='smoke'
    settings = model_settings(config,method)
    device = environment['device_selection']['torch_device_selected'] or 'cpu'
    epoch_budget = config['integration']['smoke_epochs'] if smoke else settings.get('epochs')
    iteration_budget = config['integration']['smoke_btm_iterations'] if smoke else settings.get('iterations')
    code = root/'src/transport_topics'
    implementation = {p.relative_to(code).as_posix():file_hash(p) for folder in ('baselines','topics')
                      for p in sorted((code/folder).glob('*.py'))}
    versions = {name:metadata.version(name) for name in ('numpy','scipy','scikit-learn')}
    if method in ('knntm','glocom'):versions['torch']=metadata.version('torch')
    if method=='knntm':versions['POT']=metadata.version('POT')
    if method=='btm':versions['bitermplus']=metadata.version('bitermplus')
    identity = {'schema':1,'condition':collection.summary['identity_sha256'],'method':method,
                'topics':None if method=='minilm_retrieval' else topics,
                'seed':None if method=='minilm_retrieval' else seed,
                'scope':'smoke' if smoke else 'full','settings':settings,
                'epoch_budget':epoch_budget if method in ('glocom','knntm') else None,
                'iteration_budget':iteration_budget if method=='btm' else None,
                'control_identity':control_identity,'versions':versions,'implementation':implementation,
                'inference_solver':file_hash(code/'numerics/nnls.py'),
                'nmf_code':file_hash(code/'numerics/nmf.py'),
                'tfidf_code':file_hash(code/'representations/tfidf.py'),
                'word_vector_code':file_hash(code/'representations/word_embeddings.py'),
                'device':device if method in ('glocom','knntm') else 'cpu',
                'nmf_settings':{k:config['method']['predictor'][k] for k in ('tolerance','max_iterations')}}
    key=canonical_hash(identity)
    part='model_smoke' if smoke else 'topic_models'
    directory=root/'outputs/models'/part/collection.summary['name']/method/('K%d'%topics if method!='minilm_retrieval' else 'direct')/('seed%d'%seed if method!='minilm_retrieval' else 'deterministic')/key[:16]
    directory.mkdir(parents=True,exist_ok=True)
    if check_receipt(directory):
        saved = json.loads((directory/'model.json').read_text(encoding='utf-8'))
        checkpoint_path = saved['audit'].get('checkpoint')
        if checkpoint_path and file_hash(root/checkpoint_path) != saved['audit']['checkpoint_sha256']:
            raise ValueError('A saved native model checkpoint changed')
        logger.info('Reusing complete model | %s %s K=%d seed=%d.',collection.summary['name'],method,topics,seed)
        return saved
    logger.info('Fitting model | %s %s K=%d seed=%d scope=%s.',collection.summary['name'],method,topics,seed,identity['scope'])
    started=time.perf_counter()
    if method in NMF_VARIANTS:
        from .nmf import fit
        rows=collection.counts if method=='nmf_original' else sparse.load_npz(learning_path).tocsr()
        with threadpool_limits(limits=1):
            result=fit(collection.counts,rows,topics,seed,config['method']['predictor'])
        if not result.audit['fit']['converged']:
            write_json(directory/'failed_fit.json',result.audit,overwrite=True)
            raise RuntimeError('Final NMF reported nonconvergence; inspect saved fit before proceeding')
    elif method=='minilm_retrieval':
        from .minilm import prepare
        result=prepare(np.load(collection.directory/'embeddings.npy',allow_pickle=False))
    elif method=='btm':
        from .btm import fit
        posts=json.loads((collection.directory/'posts.json').read_text(encoding='utf-8'))
        vocabulary=np.load(collection.directory/'vocabulary.npy',allow_pickle=False)
        result=fit(collection.counts,posts,vocabulary,topics,seed,settings,iteration_budget)
    else:
        from .neural import fit
        result=fit(root,collection,method,topics,seed,settings,config['integration'],device,
                   root/'outputs/checkpoints/models'/key,key,logger,epoch_budget)
    result.validate(collection.counts.shape[0],collection.counts.shape[1],topics,method=='minilm_retrieval')
    for name,value in (('retrieval_features',result.retrieval_features),('source_rows',collection.source_rows),
                        ('vocabulary',np.load(collection.directory/'vocabulary.npy',allow_pickle=False))):
        save_array(directory/(name+'.npy'),value)
    if result.topic_scores is not None:
        save_array(directory/'topic_scores.npy',result.topic_scores)
        save_array(directory/'doc_topics.npy',result.doc_topics)
        for name,value in result.extra_arrays.items():save_array(directory/(name+'.npy'),value)
    info={'method':method,'condition':collection.summary['name'],'condition_identity':collection.summary['identity_sha256'],
          'topics':None if method=='minilm_retrieval' else topics,'seed':None if method=='minilm_retrieval' else seed,
          'scope':identity['scope'],'development_only':smoke,'identity':key,
          'path':directory.relative_to(root).as_posix(),'posts':collection.counts.shape[0],
          'vocabulary':collection.counts.shape[1],'retrieval_dimension':result.retrieval_features.shape[1],
          'topic_model':method!='minilm_retrieval','audit':result.audit,'seconds':time.perf_counter()-started,
          'performance_metrics_computed':False}
    write_json(directory/'identity.json',identity,overwrite=True)
    write_json(directory/'model.json',info,overwrite=True)
    complete(directory)
    logger.info('MODEL COMPLETE | %s %s K=%s seed=%s | rows=%d features=%d | %s',
                collection.summary['name'],method,info['topics'],info['seed'],info['posts'],info['retrieval_dimension'],info['path'])
    return info
