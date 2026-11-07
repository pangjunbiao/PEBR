"""Five protected-target banks; no rule changes, final fitting or automatic tuning."""
from __future__ import annotations
from pathlib import Path
from importlib import metadata
import datetime as dt
import json
import logging
import shutil
import sys
import time
import traceback
import zipfile
from . import VERSION
from .helpers import read_json, digest, canonical_hash

def verify_package(root):
    manifest = read_json(root/'PEBR_STEP6_MANIFEST.json')
    for relative, wanted in manifest.items():
        p = (root/relative).resolve()
        p.relative_to(root.resolve())
        if digest(p) != wanted:
            raise ValueError('The protected-word runner changed: '+relative)
    return manifest

class Study:
    def __init__(self, step2, step3, work, protocol, descriptions, logger):
        from pebr_revision_step3.runner import load_collections
        self.step2, self.step3, self.work, self.protocol, self.logger = step2, step3, work, protocol, logger
        self.corpora = load_collections(step2, descriptions, protocol)
        self.rows = {}
        self.timings = []
        self.completed_configurations = 0

    def run(self, preflight=False):
        from threadpoolctl import threadpool_limits
        from pebr_revision_step3.io import complete, write_csv, write_json
        from pebr_fresh_validation.banks import query_records, make_encoder
        from pebr_fresh_validation.references import read_references
        from .banks import masked_inputs, load_bank, prepare_bank
        from .selection import select_actions
        from .outcomes import audit_configuration
        from .aggregate import export
        checks = []
        for corpus in self.protocol['corpora']:
            c = self.corpora[corpus]
            collection = c['collection']
            records = query_records(corpus, collection, self.work/'tokenizer_cache')
            references = {k: read_references(self.step3, collection, k, self.protocol) for k in self.protocol['topic_counts']}
            encoder = None
            for outer_seed in self.protocol['outer_target_bases']:
                bank_directory = self.work/'protected_banks'/corpus/('outer%d'%outer_seed)
                if preflight:
                    view = masked_inputs(corpus, collection, records, self.protocol, outer_seed)
                    checks.append({'corpus': corpus, 'outer_seed': outer_seed,
                                   'eligible_posts': int(view['eligible'].sum()),
                                   'inner_queries': len(view['queries']),
                                   'protected_numeric_and_query_checks_passed': True})
                    continue
                started = time.perf_counter()
                if complete(bank_directory):
                    bank = load_bank(bank_directory, collection, self.protocol)
                else:
                    if encoder is None:
                        encoder = make_encoder(corpus, self.step2, self.work, collection, self.protocol, self.logger)
                    bank = prepare_bank(corpus, collection, records, self.protocol, outer_seed, encoder, bank_directory, self.logger)
                self.timings.append({'phase': 'protected_retrieval', 'corpus': corpus, 'outer_seed': outer_seed,
                                     'K': 'shared', 'seconds_this_invocation': time.perf_counter()-started,
                                     'may_include_checkpoint_reuse': True})
                for k in self.protocol['topic_counts']:
                    selection_directory = self.work/'selection'/corpus/('outer%d'%outer_seed)/('K%d'%k)
                    outcome_directory = self.work/'audits'/corpus/('outer%d'%outer_seed)/('K%d'%k)
                    started = time.perf_counter()
                    with threadpool_limits(limits=1):
                        actions = select_actions(collection, bank, references[k], self.protocol,
                                                 outer_seed, selection_directory, self.logger)
                    self.timings.append({'phase': 'ten_policy_selection', 'corpus': corpus, 'outer_seed': outer_seed,
                                         'K': k, 'seconds_this_invocation': time.perf_counter()-started,
                                         'may_include_checkpoint_reuse': True})
                    started = time.perf_counter()
                    with threadpool_limits(limits=1):
                        summaries = audit_configuration(corpus, k, outer_seed, collection, bank, references[k], actions,
                                                        self.protocol, selection_directory, outcome_directory, self.logger)
                    self.timings.append({'phase': 'protected_scoring', 'corpus': corpus, 'outer_seed': outer_seed,
                                         'K': k, 'seconds_this_invocation': time.perf_counter()-started,
                                         'may_include_checkpoint_reuse': True})
                    for row in summaries:
                        self.rows[(corpus, k, outer_seed, row['policy'])] = row
                    self.completed_configurations += 1
                    export(self.work/'results', list(self.rows.values()), self.protocol)
                    write_csv(self.work/'phase_timings.csv', self.timings)
            del encoder
        if preflight:
            write_json(self.work/'preflight_mask_checks.json', checks)
            return None
        return export(self.work/'results', list(self.rows.values()), self.protocol, True)

def execute(project, explicit=None, mode='run'):
    project = Path(project).resolve()
    work = logger = lock = study = release = None
    started = time.perf_counter()
    report = {'version': VERSION, 'status': 'starting', 'protected_word_study_complete': False,
              'full_revision_complete': False, 'new_final_models_fitted': 0}
    try:
        if not (project/'main.py').is_file():
            raise ValueError('Run beside the existing project main.py')
        package = Path(__file__).resolve().parent.parent
        own_source = verify_package(package)
        from .gate import bind, audit_protocol
        step5, step3, step2, fresh, descriptions, snapshot, checks = bind(project, explicit, package)
        from pebr_revision_step3.io import write_json, write_csv, acquire_run_lock, release_run_lock, verify_step2
        from pebr_revision_step3.runner import validate_environment
        release = release_run_lock
        expected_environment = read_json(step2/'environment.json')
        environment = validate_environment(expected_environment)
        wanted = expected_environment['packages']
        encoders = {}
        for name in ('jieba', 'tokenizers', 'onnxruntime-gpu', 'torch'):
            if wanted.get(name) is not None:
                found = metadata.version(name)
                if found != wanted[name]:
                    raise ValueError('The audited encoder/tokenizer version differs: '+name)
                encoders[name] = found
        protocol = audit_protocol(fresh)
        identity = {'version': VERSION, 'completed_step5_identity': checks['step5_identity'],
                    'snapshot': snapshot['snapshot_hash'], 'protocol': protocol,
                    'runner_source': own_source, 'environment': environment, 'encoder_environment': encoders}
        key = canonical_hash(identity)
        from .storage import workspace_path, verify_cache_write
        target_work = workspace_path(project, key)
        target_work.mkdir(parents=True, exist_ok=True)
        lock = acquire_run_lock(target_work/'RUNNING.lock')
        work = target_work
        logger = logging.getLogger('pebr_protected_'+key)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')
        for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(work/'console_log.txt', encoding='utf-8')):
            handler.setFormatter(fmt)
            logger.addHandler(handler)
        logger.info('PEBR STEP 6 | v%s | protected words | 5 outer banks, both corpora, K=8/10/12', VERSION)
        logger.info('Frozen revised rule and controls; zero final-model fits; protected gains do not tune selection.')
        write_json(work/'identity.json', identity)
        write_json(work/'protocol.json', protocol)
        write_json(work/'step5_gate.json', checks)
        for name, h in own_source.items():
            out = work/'runner_source'/name
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists():
                if digest(out) != h:
                    raise ValueError('A frozen protected-audit runner file changed')
            else:
                shutil.copyfile(package/name, out)
        report.update(step5_workspace=str(step5), step3_workspace=str(step3), step2_workspace=str(step2),
                      runner_identity=key, planned_audit_configurations=30,
                      outer_target_bases=protocol['outer_target_bases'], inner_mask_base=2711,
                      original_source_or_experiments_replaced=False,
                      revised_rule_changed=False, reference_predictors_reused=True,
                      protected_scores_used_for_selection=False,
                      reference_fold_sensitivity_complete=False, neural_baseline_audit_complete=False)
        cache_check = verify_cache_write(work)
        write_json(work/'cache_write_check.json', cache_check)
        report['cache_write_check_passed'] = True
        logger.info('Embedding cache write/read check passed | %d-character temporary path',
                    cache_check['temporary_cache_path_characters'])
        study = Study(step2, step3, work, protocol, descriptions, logger)
        result = study.run(mode == 'preflight')
        verify_step2(step2)
        # Recheck the original selection source and completed result coverage after the audit.
        bind(project, step5, package)
        if mode == 'preflight':
            report['status'] = 'preflight_complete_no_new_selection_encoding_or_scoring'
            logger.info('STEP 6 PREFLIGHT COMPLETE | 10 protected mask/query constructions checked; no encoder loaded')
        else:
            report.update(status='protected_word_audit_complete', protected_word_study_complete=True,
                          completed_audit_configurations=study.completed_configurations,
                          policy_bank_summaries=len(study.rows),
                          paired_protected_contrasts=len(result['paired_protected_contrasts']),
                          primary_fresh_protected_contrasts=len(result['paired_fresh_protected_contrasts']),
                          frozen_inputs_reverified_after_run=True)
            logger.info('STEP 6 COMPLETE | 30 protected configurations | 330 policy-bank summaries | zero final fits')
        code = 0
    except BaseException as error:
        report.update(status='blocked', error=('Interrupted; completed checkpoints are resumable.'
            if isinstance(error, (KeyboardInterrupt, SystemExit)) else type(error).__name__+': '+str(error)))
        if work is None:
            work = project/'outputs/revision_rule_lab/protected_words'/('blocked_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
            work.mkdir(parents=True, exist_ok=True)
        (work/'exception.txt').write_text(traceback.format_exc(), encoding='utf-8')
        if logger:
            logger.error('STEP 6 BLOCKED | %s', report['error'])
        else:
            print('STEP 6 BLOCKED | '+report['error'], flush=True)
        code = 1
    finally:
        if work is not None:
            if study is not None:
                from pebr_revision_step3.io import write_csv
                from .aggregate import export
                export(work/'results', list(study.rows.values()), study.protocol)
                write_csv(work/'phase_timings.csv', study.timings)
            report['invocation_elapsed_seconds'] = time.perf_counter()-started
            (work/'step6_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            (work/'READ_ME_FIRST.md').write_text(
                '# Repeated protected-word audit\n\nStatus: '+report['status']+
                '\n\nRead results/RESULTS.md, results/coverage.json and results/paired_fresh_protected_contrasts.csv. '
                'For prediction gain, larger is better; for loss frequency and negative-part magnitude, smaller is better. '
                'All policies use common query cohorts and the same protected targets in each bank. '
                'Four fresh protected banks form the primary summary; known bank 2712 is excluded and remains in secondary all-five-bank summaries. '
                'These are overlapping word holdouts, not final NMF seeds or independent corpora. '
                'The model, mask fraction, cap, lambda and policy definitions remain fixed. '
                'Do not infer a positive protected gain from positive selection-view utility. '
                'Both favorable and adverse outcomes are retained. '
                'This stage does not finish neural baseline review, reference/fold sensitivity or regenerated transportation examples.\n', encoding='utf-8')
            archive = work/'PEBR_step6_protected_word_inspection.zip'
            with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
                for name in ('step6_report.json', 'identity.json', 'protocol.json', 'step5_gate.json',
                             'preflight_mask_checks.json', 'cache_write_check.json', 'phase_timings.csv', 'READ_ME_FIRST.md', 'console_log.txt', 'exception.txt'):
                    path = work/name
                    if path.is_file():
                        z.write(path, name)
                for pattern in ('results/*', 'runner_source/**/*', 'protected_banks/*/*/summary.json',
                                'encoder_audits/*/*.json', 'selection/*/*/*/selection_summary.json',
                                'selection/*/*/*/*/summary.json',
                                'selection/*/*/*/preserve_forward_fixed_cap/actions.json',
                                'audits/*/*/*/post_outcomes.csv', 'audits/*/*/*/audit_identity.json'):
                    for path in work.glob(pattern):
                        if path.is_file():
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
