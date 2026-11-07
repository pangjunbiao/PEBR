"""Frozen sensitivity study with resumable, isolated outputs and explicit scope."""
from pathlib import Path
import datetime as dt
import logging
import shutil
import sys
import time
import traceback
import zipfile
from . import VERSION
from .common import read,write,digest,key

def verify_package(package):
    manifest=read(package/'PEBR_STEP8_MANIFEST.json')
    for relative,sha in manifest.items():
        path=(package/relative).resolve();path.relative_to(package)
        if digest(path)!=sha:raise ValueError('Step 8 package changed: '+relative)
    return manifest

def execute(project,explicit=None,mode='run'):
    project=Path(project).resolve();package=Path(__file__).resolve().parent.parent
    work=logger=lock=release=study=None;started=time.perf_counter()
    report={'version':VERSION,'status':'starting','step8_complete':False,'full_revision_complete':False,
            'revised_rule_changed':False,'protected_targets_used_for_tuning':False,'encoder_calls':0,
            'existing_experiments_replaced':False,'new_final_topic_fits_planned':420,'new_reference_fits_planned':120}
    try:
        if not (project/'main.py').is_file():raise ValueError('Run beside the existing main.py')
        source=verify_package(package)
        from .gate import bind
        step7,step6,step5,step3,step2,fresh,descriptions,snapshot,checks=bind(project,explicit,package)
        from pebr_revision_step3.runner import validate_environment
        from pebr_revision_step3.io import acquire_run_lock,release_run_lock,verify_step2
        from threadpoolctl import threadpool_limits
        from .design import build
        from .study import Study
        environment=validate_environment(read(step2/'environment.json'))
        protocol=build(fresh)
        identity={'version':VERSION,'step7_identity':checks['step7_identity'],'snapshot_hash':snapshot['snapshot_hash'],
                  'protocol':protocol,'runner_source':source,'environment':environment}
        target_work=project/'outputs/revision_rule_lab/robustness8'/('s8_'+key(identity)[:12])
        target_work.mkdir(parents=True,exist_ok=True)
        lock=acquire_run_lock(target_work/'RUNNING.lock');release=release_run_lock
        work=target_work
        logger=logging.getLogger('pebr_robust8_'+key(identity));logger.setLevel(logging.INFO);logger.propagate=False
        for handler in (logging.StreamHandler(sys.stdout),logging.FileHandler(work/'console_log.txt',encoding='utf-8')):
            handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s',datefmt='%Y-%m-%dT%H:%M:%S'));logger.addHandler(handler)
        logger.info('PEBR STEP 8 | reference initialization and family-fold sensitivity | revised rule unchanged')
        logger.info('Plan: 120 new fold-excluded predictive references; 420 new final NMF fits; zero encoder calls')
        logger.info('Same two corpora | K8/10/12 | mask3711 | final seeds0-4 | separate sensitivity axes')
        logger.info('Workspace: %s',work)
        write(work/'identity.json',identity);write(work/'protocol.json',protocol);write(work/'accepted_stage_gate.json',checks)
        for relative,sha in source.items():
            target=work/'runner_source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and digest(target)!=sha:raise ValueError('Frozen Step 8 runner changed')
            if not target.exists():shutil.copyfile(package/relative,target)
        report.update(runner_identity=key(identity),step7_workspace=str(step7),step6_workspace=str(step6),
            step5_workspace=str(step5),step3_workspace=str(step3),step2_workspace=str(step2),
            sensitivity_scope=protocol['inferential_scope'],planned_case_fit_records=750)
        study=Study(step2,step3,step5,work,protocol,descriptions,logger)
        from .transport_notes import export as export_transport_notes
        report['transportation_clarification']=export_transport_notes(step6,work,study.corpora['transport_grievance']['collection'])
        with threadpool_limits(limits=1):
            study.prepare()
            if mode=='run':
                study.reproduce()
                result=study.run()
        verify_step2(step2)
        bind(project,step7,package)
        if mode=='preflight':
            report.update(status='preflight_complete_no_new_selection_or_fitting',step8_complete=False)
            logger.info('STEP 8 PREFLIGHT PASSED | cached text/query/candidate identities verified | no new fits')
        else:
            report.update(status='reference_fold_sensitivity_complete',step8_complete=True,
                available_case_fit_records=len(study.rows),available_reference_fits=len(study.reference_records),
                case_policy_summaries=len(result['summaries']),paired_effect_rows=len(result['contrasts']),
                newly_fitted_this_invocation=study.new_fit_count,resumed_new_fits_this_invocation=study.reused_fit_count,
                frozen_inputs_reverified_after_run=True,default_selector_reproduction_passed=True)
            logger.info('STEP 8 COMPLETE | 750 case-fit records | 150 case-policy summaries | rule frozen')
        code=0
    except BaseException as error:
        report.update(status='blocked',error=('Interrupted; rerun the same command to resume saved checkpoints.'
            if isinstance(error,(KeyboardInterrupt,SystemExit)) else type(error).__name__+': '+str(error)))
        if work is None:
            work=project/'outputs/revision_rule_lab/robustness8'/('blocked_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
            work.mkdir(parents=True,exist_ok=True)
        (work/'exception.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print('STEP 8 BLOCKED | '+report['error'],flush=True);code=1
    finally:
        if work is not None:
            if study is not None:
                from .results import export
                from .common import table
                export(work/'results',list(study.rows.values()),study.protocol)
                table(work/'reference_fit_audits.csv',study.reference_records)
                table(work/'phase_timings.csv',study.timings)
            report['invocation_seconds']=time.perf_counter()-started
            write(work/'step8_report.json',report)
            (work/'READ_ME_FIRST.md').write_text(
                '# Step 8 reference/fold sensitivity\n\nStatus: '+report['status']+
                '\n\nRead results/RESULTS.md, results/case_summaries.csv, results/paired_effects.csv and results/coverage.json. '
                'Reference seeds1/2 change the frozen predictor initialization only. Fold seeds3710/4710 change both the predictor fitting partitions and eligible donor lists. '
                'Both axes retain mask3711, the same counts/vocabulary and final NMF seeds0-4. PEBR, ordinary sharing, matched-amount sharing and a same-score prefix remain visible alongside no borrowing. '
                'All adverse results are retained. The default first-fresh bank is reused; it is not a new validation dataset. '
                'Shared NMF and similarity fits are not additional independent experiments. The five-fit SD is conditional on each particular reference/fold setting. '
                'The run preserves the revised score, cap and downside weight. No encoder is loaded and no protected-word outcomes enter selection. '
                'Reference convergence failures block the run; the script does not silently change tolerances, seeds or budgets.\n\n'
                'Still pending after this study: genuine K10 BTM/neural benchmark fits, a final review of baseline training budgets, regenerated corpus-size/parameter assets for the revised rule, and the final table/figure/claim ledger. '
                'The full manuscript revision remains incomplete until retained analyses match the final rule.\n',encoding='utf-8')
            archive=work/'PEBR_step8_inspection.zip'
            patterns=('results/*','runner_source/**/*','transportation_clarification/*','reproduction/*/*/*/actions.json','reproduction/*/*/*/summary.json',
                'selection/*/*/*/*/actions.json','selection/*/*/*/*/summary.json','fold_assignments/*/*.json',
                'fold_banks/*/*/*.npy','fold_banks/*/*/identity.json',
                'cache/predictors/*/dictionary.npy','cache/predictors/*/idf.npy','cache/predictors/*/prior.npy',
                'cache/predictors/*/training_indices.npy','cache/predictors/*/identity.json','cache/predictors/*/fit.json',
                'fits/*/*/K12/preserve_forward_fixed_cap/s*/*.npy','fits/*/*/K12/preserve_forward_fixed_cap/s*/*.json')
            selected={p for pattern in patterns for p in work.glob(pattern) if p.is_file()}
            selected.update(p for p in work.iterdir() if p.is_file() and p!=archive and p.name!='RUNNING.lock')
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
                for p in sorted(selected):z.write(p,p.relative_to(work).as_posix())
            print('RESULTS DIRECTORY: '+str(work),flush=True)
            print('SEND THIS INSPECTION ZIP: '+str(archive),flush=True)
        if logger:
            for handler in list(logger.handlers):handler.close();logger.removeHandler(handler)
        if release is not None:release(lock)
    return code
