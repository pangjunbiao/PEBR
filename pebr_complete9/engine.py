"""Final bounded implementation phase; completion requires every declared output."""
from pathlib import Path
from importlib import metadata
from types import SimpleNamespace
import datetime as dt
import logging
import shutil
import sys
import time
import traceback
import zipfile
from . import VERSION
from .common import read,write,key,digest

def verify_package(root):
    manifest=read(root/'PEBR_STEP9_MANIFEST.json')
    for name,h in manifest.items():
        p=(root/name).resolve();p.relative_to(root)
        if digest(p)!=h:raise ValueError('Completion runner changed: '+name)
    return manifest

def execute(project,explicit=None,phase='all'):
    project=Path(project).resolve();package=Path(__file__).resolve().parent.parent
    work=logger=lock=release=None;started=time.perf_counter()
    report={'version':VERSION,'status':'starting','step9_complete':False,'results_ready_for_manuscript_revision':False,
            'manuscript_text_updated':False,'rule_changed':False,'existing_experiments_replaced':False}
    try:
        if not (project/'main.py').is_file():raise ValueError('Run beside the existing main.py')
        own=verify_package(package)
        from .gate import bind
        s8,s7,s6,s5,s3,s2,fresh,descriptions,snapshot,gate=bind(project,explicit,package)
        from pebr_revision_step3.runner import validate_environment,load_collections
        from pebr_revision_step3.io import acquire_run_lock,release_run_lock,verify_step2,complete,save_receipt
        from threadpoolctl import threadpool_limits
        from .design import build
        from .benchmarks import Benchmarks
        environment=validate_environment(read(s2/'environment.json'));wanted=read(s2/'environment.json')['packages']
        for name in ['torch','jieba']:
            if wanted.get(name) and metadata.version(name)!=wanted[name]:raise ValueError('Audited package version differs: '+name)
        environment.update({n:metadata.version(n) for n in ['torch','POT','bitermplus','matplotlib']})
        import torch
        environment['cuda_device']=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        protocol=build(fresh)
        identity={'version':VERSION,'step8_identity':gate['step8_identity'],'protocol':protocol,
                  'runner_source':own,'environment':environment,'snapshot_hash':snapshot['snapshot_hash']}
        target=project/'outputs/revision_rule_lab/completion9'/('s9_'+key(identity)[:12]);target.mkdir(parents=True,exist_ok=True)
        lock=acquire_run_lock(target/'RUNNING.lock');release=release_run_lock;work=target
        logger=logging.getLogger('pebr_complete9_'+key(identity));logger.setLevel(logging.INFO);logger.propagate=False
        for h in [logging.StreamHandler(sys.stdout),logging.FileHandler(work/'console_log.txt',encoding='utf-8')]:
            h.setFormatter(logging.Formatter('%(asctime)s | %(message)s',datefmt='%Y-%m-%dT%H:%M:%S'));logger.addHandler(h)
        write(work/'identity.json',identity);write(work/'protocol.json',protocol);write(work/'accepted_stage_gate.json',gate)
        for name,h in own.items():
            p=work/'runner_source'/name;p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists() and digest(p)!=h:raise ValueError('Frozen completion runner copy changed')
            if not p.exists():shutil.copyfile(package/name,p)
        logger.info('PEBR STEP 9 | complete benchmark budgets, revised analyses and publication assets')
        logger.info('Rule/defaults frozen | both corpora | K8/10/12 | existing experiments preserved')
        logger.info('Workspace: %s',work)
        study=SimpleNamespace(project=project,step8=s8,step7=s7,step6=s6,step5=s5,step3=s3,step2=s2,work=work,
            protocol=protocol,logger=logger,lineage={'step8':gate['step8_identity'],'snapshot':snapshot['snapshot_hash']})
        study.corpora=load_collections(s2,descriptions,protocol)
        trainer=Benchmarks(project,s7,s5,s3,work,study.corpora,protocol,logger)
        with threadpool_limits(limits=1):
            from .retrieval import verify_source
            verify_source(study)
            checks=trainer.preflight();write(work/'preflight.json',checks)
            if phase!='preflight':
                from .analyses import parameters,sizes
                from .runtime import run as runtime
                from .retrieval import run as retrieval
                tasks=[('benchmarks','benchmark_results',trainer.run),('parameters','parameter_results',lambda:parameters(study)),
                       ('sizes','size_results',lambda:sizes(study)),('runtime','runtime_results',lambda:runtime(study)),
                       ('retrieval','retrieval_results',lambda:retrieval(study))]
                for name,result,operation in tasks:
                    if phase not in ('all',name):continue
                    directory=work/result
                    if complete(directory) and read(directory/'coverage.json').get('complete'):
                        logger.info('Verified complete phase reused: %s',name);continue
                    logger.info('PHASE START: %s',name)
                    if not operation():raise ValueError('Incomplete phase: '+name)
                    save_receipt(directory);logger.info('PHASE COMPLETE: %s',name)
                    if name=='benchmarks' and torch.cuda.is_available():torch.cuda.empty_cache()
                ready=all(complete(work/d) and read(work/d/'coverage.json').get('complete') for d in
                    ['benchmark_results','parameter_results','size_results','runtime_results','retrieval_results'])
                if phase in ('all','assets'):
                    if not ready:raise ValueError('All benchmark/parameter/size/runtime/retrieval phases are required before final assets')
                    from .assets import run as assets
                    assets(study)
                asset_ready=(work/'asset_coverage.json').is_file() and read(work/'asset_coverage.json').get('complete')
                report.update(step9_complete=bool(ready and asset_ready),results_ready_for_manuscript_revision=bool(ready and asset_ready),
                    phase_coverage={name:(complete(work/d) and read(work/d/'coverage.json').get('complete')) for name,d,_ in tasks},
                    assets_ready=bool(asset_ready))
        verify_step2(s2);bind(project,s8,package)
        report.update(runner_identity=key(identity),step8_workspace=str(s8),step7_workspace=str(s7),step6_workspace=str(s6),
            step5_workspace=str(s5),step3_workspace=str(s3),step2_workspace=str(s2),frozen_inputs_reverified_after_run=True,
            baseline_budget_review_scope='Same fixed architectures/input adapters and training settings; continuation 200 to 400 epochs, not optimal tuning.')
        report['status']=('preflight_complete_no_training' if phase=='preflight' else
                          'complete_results_export_ready_for_review' if report['step9_complete'] else 'requested_phase_complete_more_phases_pending')
        logger.info('STEP 9 FINISHED | %s',report['status']);code=0
    except BaseException as e:
        report.update(status='blocked',error=('Interrupted; rerun the same command to resume.' if isinstance(e,(KeyboardInterrupt,SystemExit)) else type(e).__name__+': '+str(e)))
        if work is None:
            work=project/'outputs/revision_rule_lab/completion9'/('blocked_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));work.mkdir(parents=True,exist_ok=True)
        (work/'exception.txt').write_text(traceback.format_exc(),encoding='utf-8');print('STEP 9 BLOCKED | '+report['error'],flush=True);code=1
    finally:
        if work is not None:
            report['invocation_seconds']=time.perf_counter()-started;write(work/'step9_report.json',report)
            (work/'READ_ME_FIRST.md').write_text('# Revised-rule results completion\n\nStatus: '+report['status']+
                '\n\nRead step9_report.json and each phase coverage.json. A phase success does not imply all results are complete. '
                'Only results_ready_for_manuscript_revision=true marks the complete declared results package. '
                'The manuscript itself has not been edited. Inspect publication_assets/, asset_manifest.json and the numerical source tables before using the outputs. '
                'The main neural budget is fixed at 400 epochs; 200-epoch outputs and all unfavorable comparisons remain. '
                'Parameter defaults remain cap1/3 and lambda1. Size curves refit minimum-DF2 vocabularies within each sample and record excluded zero rows. '
                'Runtime includes cold audited encoding, reference fitting and standalone PEBR selection. '
                'No result certifies universal superiority, semantic correctness or passenger-service improvement. '
                'Original files remain unchanged. Temporary optimizer checkpoints created here are pruned only after verified compact inference export.\n',encoding='utf-8')
            archive=work/'PEBR_step9_inspection.zip'
            patterns=['publication_assets/**/*','runner_source/**/*','*_results/*','retrieval_results/per_model/*.npz','sizes/*/*/data/identity.json',
                'parameter_selection/*/*/*/summary.json','runtime_completed/*/*.json','bench_models/*/*/*/*/audit.json']
            selected={p for pat in patterns for p in work.glob(pat) if p.is_file()}
            selected.update(p for p in work.iterdir() if p.is_file() and p!=archive and p.name!='RUNNING.lock')
            # Models and optimizer states stay local; compact selected dictionaries aid independent inspection.
            selected.update(p for p in work.glob('bench_models/*/*/*/*/topic_scores.npy'))
            selected.update(p for p in work.glob('bench_models/*/K12/*/e400_s0/decoder_sample.npz'))
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
                for p in sorted(selected):z.write(p,p.relative_to(work).as_posix())
            print('RESULTS DIRECTORY: '+str(work),flush=True);print('SEND THIS INSPECTION ZIP: '+str(archive),flush=True)
        if logger:
            for h in list(logger.handlers):h.close();logger.removeHandler(h)
        if release is not None:release(lock)
    return code
