"""Step 7: preserve the rule and inspect saved evidence before manuscript revision."""
from pathlib import Path
import datetime as dt
import logging
import shutil
import sys
import time
import traceback
import zipfile
from . import VERSION
from .common import read, digest, key, write, Findings

PENDING = [
    'Inspect baseline loss histories, decoder replay and adaptation settings; correct demonstrated defects before interpreting superiority.',
    'Reference-initialization and family-fold sensitivity remain separate from repeated masking and final NMF seeds.',
    'Regenerate corpus-size and parameter analyses for the exact revised fixed-cap rule, or remove outdated assets and corresponding claims.',
    'K10 all-method benchmarks require genuine K10 baseline fits if a complete benchmark at that count is claimed; no historical K16 relabelling.',
    'Review the new original-source cases, including the unsuccessful example, before selecting concise manuscript illustrations.',
    'Update introduction, algorithm, proofs and claims to the exact revised score and fixed-cap rule; retain conditional and adverse results.',
]

def verify_package(package):
    manifest = read(package/'PEBR_STEP7_MANIFEST.json')
    for relative, sha in manifest.items():
        path = (package/relative).resolve()
        path.relative_to(package)
        if digest(path) != sha:
            raise ValueError('Step 7 package changed: '+relative)
    return manifest

def execute(project, explicit=None):
    project = Path(project).resolve()
    package = Path(__file__).resolve().parent.parent
    started = time.perf_counter()
    work = logger = lock = release = findings = None
    report = dict(version=VERSION, status='starting', step7_execution_complete=False,
                  full_revision_complete=False, new_topic_model_fits=0, rule_changed=False,
                  existing_global_context_clustering_reconstructed=True,
                  existing_experiments_replaced=False, protected_gains_used_to_tune=False,
                  pending_reference_fold_sensitivity=True, pending_updated_size_parameter_results=True)
    try:
        if not (project/'main.py').is_file():
            raise ValueError('Run this script beside your existing main.py')
        own_source = verify_package(package)
        from .gate import bind
        step6, step5, step3, step2, protocol, descriptions, snapshot, checks = bind(project, explicit, package)
        from pebr_revision_step3.runner import validate_environment, load_collections
        from pebr_revision_step3.io import acquire_run_lock, release_run_lock, verify_step2
        from threadpoolctl import threadpool_limits
        environment = validate_environment(read(step2/'environment.json'))
        identity = dict(version=VERSION, step6_identity=checks['step6_identity'],
                        runner_source=own_source, snapshot_hash=snapshot['snapshot_hash'], environment=environment,
                        baseline_saved_counts=[8, 12], case_topic_count=12, case_selection_bank=3711,
                        case_outer_bank=3712, case_final_seeds=[0,1,2,3,4],
                        original_project_root=str(project))
        work = project/'outputs/revision_rule_lab/review7'/('s7_'+key(identity)[:12])
        work.mkdir(parents=True, exist_ok=True)
        lock = acquire_run_lock(work/'RUNNING.lock')
        release = release_run_lock
        logger = logging.getLogger('pebr_review7_'+key(identity))
        logger.setLevel(logging.INFO)
        logger.propagate = False
        for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(work/'console_log.txt', encoding='utf-8')):
            handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%dT%H:%M:%S'))
            logger.addHandler(handler)
        findings = Findings(work, logger)
        write(work/'identity.json', identity)
        write(work/'completed_stage_gate.json', checks)
        for relative in own_source:
            target = work/'runner_source'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and digest(target) != own_source[relative]:
                raise ValueError('Frozen Step 7 runner copy changed')
            if not target.exists():
                shutil.copyfile(package/relative, target)
        logger.info('PEBR STEP 7 | baseline verification and traceable transportation cases')
        logger.info('Read-only completed inputs | no neural/NMF/BTM training | zero downloads | revised rule unchanged')
        logger.info('Existing GloCOM clustering inputs are reconstructed with their fixed original settings.')
        logger.info('Work directory: %s', work)
        corpora = load_collections(step2, descriptions, protocol)
        from .baselines import Audit
        with threadpool_limits(limits=1):
            audit = Audit(project, step2, work, corpora, findings, logger)
            fidelity = audit.run()
        from .transport import run as run_transport
        with threadpool_limits(limits=1):
            transport = run_transport(step6, step5, step3, work, corpora['transport_grievance'], protocol, findings, logger)
        verify_step2(step2)
        bind(project, step6, package)
        report.update(step6_workspace=str(step6), step5_workspace=str(step5), step3_workspace=str(step3),
                      step2_workspace=str(step2), baseline_fidelity_checks_complete=fidelity,
                      baseline_tuning_adequacy_established=False, transportation_export_complete=transport,
                      frozen_inputs_reverified_after_run=True, findings= findings.counts())
        complete = fidelity and transport and findings.counts()['FAIL'] == 0 and findings.counts()['INCOMPLETE'] == 0
        report.update(status='review_export_complete' if complete else 'review_finished_with_issues',
                      step7_execution_complete=complete)
        logger.info('STEP 7 FINISHED | %s | %s', report['status'], findings.counts())
        code = 0 if complete else 1
    except BaseException as error:
        if work is None:
            work = project/'outputs/revision_rule_lab/review7'/('blocked_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
            work.mkdir(parents=True, exist_ok=True)
        report.update(status='blocked', error=type(error).__name__+': '+str(error))
        (work/'exception.txt').write_text(traceback.format_exc(), encoding='utf-8')
        print('STEP 7 BLOCKED | '+report['error'], flush=True)
        code = 1
    finally:
        if work is not None:
            report['invocation_seconds'] = time.perf_counter()-started
            if findings is not None:
                report['findings'] = findings.counts()
            write(work/'step7_report.json', report)
            write(work/'remaining_validation.json', dict(full_revision_complete=False, pending=PENDING))
            notes = ['# Step 7 review', '', 'Status: '+report['status'], '',
                'This run does not train any neural, NMF or BTM topic model or change the revised PEBR rule. It reconstructs GloCOM\'s fixed clustering input. The old source, checkpoints and experiment outputs remain unchanged.', '',
                'Read baseline_coverage.json, findings.json, baseline_summaries.csv and baseline_diagnostics.json first. '
                'A faithful implementation replay does not prove adequate tuning, universal superiority or significance.', '',
                'Transportation files are in transportation/. PROTECTED_CASES.md explains saved-action prediction cases; '
                'transport_cases_revised.tex and the recurrence plots describe final topic dictionaries. '
                'They use different saved action banks and must not be presented as one before/after pipeline trace.', '',
                'The first four anchors are fixed. Added contrast/failure cases are disclosed outcome-selected illustrations, not an independent evaluation cohort. '
                'The method does not establish an incident cause or improve passenger services merely by improving a word-prediction score.', '',
                'Old corpus-size/parameter figures and old illustrative topic lists do not automatically apply to the revised rule.', '',
                '## Remaining decisions', '']+['- '+p for p in PENDING]
            (work/'READ_ME_FIRST.md').write_text('\n'.join(notes)+'\n', encoding='utf-8')
            archive = work/'PEBR_step7_inspection.zip'
            with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
                for path in sorted(work.rglob('*')):
                    if not path.is_file() or path == archive or path.name == 'RUNNING.lock' or '__pycache__' in path.parts:
                        continue
                    z.write(path, path.relative_to(work).as_posix())
            print('RESULTS DIRECTORY: '+str(work), flush=True)
            print('SEND THIS INSPECTION ZIP: '+str(archive), flush=True)
        if logger:
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)
        if release is not None:
            release(lock)
    return code
