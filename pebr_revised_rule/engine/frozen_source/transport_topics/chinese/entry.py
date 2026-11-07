"""Explicit main.py routes for Chinese-only runs and archived-result formatting."""
from __future__ import annotations

import json
from importlib import metadata
import os
from pathlib import Path
import platform
import sys
import time
import zipfile

from ..config import canonical_hash, read_yaml, within_project
from ..runtime.artifacts import RunStore, file_hash, source_identity, write_json
from ..runtime.bootstrap import ensure_foundation, _install
from ..runtime.checkpointing import ProjectLock
from ..runtime.logging import RunLogger, now_utc, run_log_snapshot


def ensure_dependencies(logger,install_missing,full_models):
    from packaging.requirements import Requirement
    required=['numpy>=1.26,<3','scipy>=1.13,<2','scikit-learn>=1.5,<2',
              'threadpoolctl>=3.5,<4','jieba==0.42.1']
    if full_models:required += ['tokenizers>=0.19,<1','gensim>=4.4,<5','bitermplus>=1.0,<1.1','POT>=0.9.5,<0.10']
    records=[]
    for text in required:
        req=Requirement(text);installed=False
        try:version=metadata.version(req.name)
        except metadata.PackageNotFoundError:
            if not install_missing:raise RuntimeError('Missing requirement in the current interpreter: '+text)
            _install(text,logger,timeout=900);version=metadata.version(req.name);installed=True
        if not req.specifier.contains(version,prereleases=True):
            raise RuntimeError(f'Existing {req.name}=={version} is outside {req.specifier}; it was preserved.')
        records.append({'distribution':req.name,'version':version,'installed_now':installed})
        logger.info('Chinese requirement OK: %s==%s (%s).',req.name,version,'installed missing' if installed else 'preserved')
    if full_models:
        try:metadata.version('torch')
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError('Use the Python 3.10 interpreter containing your verified PyTorch/CUDA installation. This update does not install or replace torch.') from exc
        import torch
        existing=[]
        for name in ('onnxruntime','onnxruntime-gpu'):
            try:existing.append({'distribution':name,'version':metadata.version(name),'installed_now':False})
            except metadata.PackageNotFoundError:pass
        if not existing:
            name='onnxruntime-gpu' if torch.cuda.is_available() else 'onnxruntime'
            if not install_missing:raise RuntimeError('Missing ONNX backend: '+name)
            _install(name+'==1.22.0',logger,timeout=900)
            existing=[{'distribution':name,'version':metadata.version(name),'installed_now':True}]
        records+=existing
    return records


def validate_cfg(cfg):
    if cfg['kind']!='chinese_full_topic_study':raise ValueError('Incorrect Chinese configuration kind')
    if cfg['model_seeds']!=[0,1,2,3,4]:raise ValueError('The full study uses exactly five fixed model seeds: 0--4')
    if not cfg['topic_counts'] or any(type(k)!=int or k<2 for k in cfg['topic_counts']):raise ValueError('Invalid topic counts')
    if len(set(cfg['topic_counts']))!=len(cfg['topic_counts']) or cfg['main_topic_count'] not in cfg['topic_counts']:
        raise ValueError('Topic counts must be unique and include the main topic count')
    if not cfg['collections'] or len(set(cfg['collections']))!=len(cfg['collections']):raise ValueError('Invalid Chinese collection scopes')
    if set(cfg['collections'])-{'pooled','beijing','shanghai','xiamen'}:raise ValueError('Unknown Chinese collection scope')
    if 'pooled' in cfg['collections'] and len(cfg['collections'])>1:raise ValueError('Choose pooled or separate city runs; do not count pooled and city data as independent replications')
    if set(cfg['methods'])-{'nmf_original','nmf_similarity','btm','knntm','glocom','proposed'}:raise ValueError('Unknown main model')
    if len(set(cfg['methods']))!=len(cfg['methods']) or not cfg['methods']:raise ValueError('Duplicate or empty main methods')
    if set(cfg['controls'])!={'similarity_matched','full_pool_loo','no_downside'}:raise ValueError('Missing or unknown control switches')
    if any(type(v)!=bool for v in cfg['controls'].values()):raise ValueError('Control switches must be true or false')
    if cfg['evaluation_top_words']!=10 or cfg['word_vectors']!='frozen_multilingual_pca':raise ValueError('Keep the declared metric/input protocol')


def device_for(cfg,args,logger):
    import torch
    requested=getattr(args,'device',None) or cfg['runtime']['device']
    if requested not in ('auto','cpu','cuda','gpu'):raise ValueError('Device must be auto, cpu or cuda')
    use_cuda=torch.cuda.is_available() and requested!='cpu'
    if requested in ('cuda','gpu') and not use_cuda:raise RuntimeError('CUDA was explicitly requested but torch cannot execute on a GPU')
    torch.set_num_threads(cfg['runtime']['cpu_threads'])
    if use_cuda:
        index=cfg['runtime']['gpu_index']
        if index>=torch.cuda.device_count():raise ValueError('GPU index is outside the available CUDA devices')
        device=f'cuda:{index}'
        a=torch.arange(16,dtype=torch.float32,device=device).reshape(4,4)
        value=a@a;torch.cuda.synchronize(index)
        if not torch.isfinite(value).all().item():raise RuntimeError('CUDA execution probe failed')
        logger.info('Chinese compute device: %s (%s); actual torch CUDA matrix multiplication verified.',device,torch.cuda.get_device_name(index))
    else:
        device='cpu';logger.info('Chinese compute device: CPU (explicit selection or no CUDA available).')
    return device,{'torch':torch.__version__,'selected':device,'cuda_execution_verified':use_cuda,
                   'gpu_name':torch.cuda.get_device_name(cfg['runtime']['gpu_index']) if use_cuda else None}


def inspection(root,store,maximum=6000000,extra=None):
    target=root/'outputs/inspection'/(store.run_id+'_inspection.zip')
    target.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(store.directory.rglob('*')):
            if path.is_file() and path.suffix in ('.json','.csv','.yaml','.txt','.md','.tex') and path.stat().st_size<=maximum:
                archive.write(path,'run/'+path.relative_to(store.directory).as_posix())
        if extra:
            for path in sorted(extra.rglob('*')):
                if path.is_file() and path.suffix in ('.json','.csv','.md','.tex') and path.stat().st_size<=maximum:
                    archive.write(path,'results/'+path.relative_to(extra).as_posix())
        archive.writestr('logs/run_log.txt',run_log_snapshot(root/'logs/run_log.txt',store.run_id))
    return target


def execute_extension(root,args,extension):
    root=Path(root).resolve();logger=None;store=None;lock=ProjectLock(root/'outputs/manifests/execution.lock')
    state='failed';error=None;result=None;output=None;cfg=None;started=time.monotonic()
    try:
        lock.acquire();store=RunStore(root,stage='chinese_topics' if extension=='chinese' else 'requested_tables')
        logger=RunLogger(root/'logs/run_log.txt',store.run_id)
        logger.info('PROPOSED TRANSPORT PROJECT | %s | interpreter: %s.',extension,sys.executable)
        ensure_foundation(logger,getattr(args,'install_missing',True))
        cfg=read_yaml(within_project(root,args.config))
        store.json('config.json',cfg)
        store.json('status.json',{'state':'running','stage':extension,'started_utc':store.started_utc})
        if extension=='saved_tables':
            if cfg['kind']!='saved_study_tables':raise ValueError('Incorrect saved-result table configuration')
            from .saved_results import export_requested
            output=root/cfg['outputs']/store.run_id
            result=export_requested(root,cfg,output,logger)
            result.update(output_directory=output.relative_to(root).as_posix(),complete=True)
            store.json('requested_tables.json',result)
        else:
            mode=getattr(args,'mode',None) or 'run'
            if mode not in ('run','check'):raise ValueError('Chinese config supports --mode check or --mode run; completed tables are already saved in the run output')
            validate_cfg(cfg)
            from ..runtime.reproducibility import configure_process
            protocol=read_yaml(within_project(root,cfg['protocol']))
            protocol['runtime'].update(cfg['runtime'])
            store.json('process_settings.json',configure_process(protocol['runtime'],cfg['model_seeds']))
            store.json('chinese_requirements.json',ensure_dependencies(logger,getattr(args,'install_missing',True),mode=='run'))
            from .data import read_records,prepare_collection
            expected=json.loads((root/'assets/chinese_study/locked_core_sha256.json').read_text(encoding='utf-8'))
            for relative,digest in expected.items():
                if file_hash(root/relative)!=digest:raise ValueError('A locked rule/NMF/numerical source file changed: '+relative)
            store.json('locked_core_check.json',{'files':len(expected),'all_unchanged':True})
            records,raw_audit=read_records(root,cfg,logger)
            store.json('chinese_raw_audit.json',raw_audit)
            protocol['method']['encoder'].update(cfg['encoder'])
            protocol['preparation']['min_document_frequency']=cfg['min_document_frequency']
            protocol['seeds']=cfg['model_seeds'];protocol['topics']['counts']=cfg['topic_counts']
            protocol['topics']['main_count']=cfg['main_topic_count']
            protocol['integration']['scope']='full';protocol['integration']['download_sources']=cfg['download_missing_assets']
            protocol['evaluation']['prediction']['enabled']=False
            protocol['borrowing_run']['receiver_limit']=0
            protocol['output']['resume']=True
            collections=[prepare_collection(root,cfg,protocol,records,name,logger) for name in cfg['collections']]
            store.json('chinese_collections.json',[c.summary for c,_ in collections])
            store.json('resolved_protocol.json',protocol)
            logger.info('CHINESE DATA VERIFIED | %d raw | %d normalized unique | full retained corpus; no 600-post pilot sample.',raw_audit['raw_posts'],len(records))
            if mode=='check':
                result={'complete':True,'operation':'data_and_configuration_check','new_model_fits':0,
                        'collections':[c.summary for c,_ in collections],'model_seeds':cfg['model_seeds'],
                        'next_command':'main.py --config configs/chinese/full_study.yaml --mode run'}
                store.json('chinese_check.json',result)
            else:
                device,device_audit=device_for(cfg,args,logger)
                store.json('environment.json',{'python':platform.python_version(),'device':device_audit})
                from .study import run_study
                output=root/cfg['outputs']/store.run_id
                result=run_study(root,cfg,protocol,records,collections,device,logger,store)
                output=root/result['output_directory'];store.json('chinese_study.json',result)
        state='completed'
    except KeyboardInterrupt:
        state='interrupted';error='Interrupted; rerun the same command to reuse verified caches and checkpoints.'
        if logger:logger.warning(error)
    except Exception as exc:
        error=str(exc)
        if logger:logger.exception('RUN FAILED: '+error)
        else:print('ERROR: '+error,file=sys.stderr)
    finally:
        if store and logger:
            try:
                store.json('status.json',{'state':state,'stage':extension,'finished_utc':now_utc(),
                           'error':error,'result':result},overwrite=True)
                store.json('timing.json',{'seconds':time.monotonic()-started})
                store.manifest({'configuration_sha256':canonical_hash(cfg) if cfg else None,
                                'source_sha256':source_identity(root)['sha256'],'extension':extension})
                bundle=inspection(root,store,cfg.get('inspection_max_bytes_per_file',6000000) if cfg else 6000000,output)
                write_json(root/'outputs/manifests'/('latest_'+extension+'.json'),
                           {'run_id':store.run_id,'state':state,'output_directory':str(output) if output else None,
                            'inspection_zip':bundle.relative_to(root).as_posix()},overwrite=True)
                logger.info('RUN %s | elapsed %.2fs | %s.',state.upper(),time.monotonic()-started,store.directory)
                if output:logger.info('RESULTS DIRECTORY: %s',output)
                logger.info('SEND THIS INSPECTION ZIP: %s',bundle)
                logger.info('Single append-only log: %s',root/'logs/run_log.txt')
            except Exception:
                state='failed';logger.exception('Could not finalize the extension run')
            logger.close()
        lock.release()
    return 0 if state=='completed' else (130 if state=='interrupted' else 1)
