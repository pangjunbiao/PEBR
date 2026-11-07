"""Run the frozen revised rule on new masks; old workspaces are read-only inputs."""
from __future__ import annotations
from importlib import metadata
from pathlib import Path
import datetime as dt
import json
import logging
import shutil
import sys
import time
import traceback
import zipfile
from . import VERSION
from .bootstrap import resolve_and_bind, digest, canonical_hash, read_json

def verify_package(root):
    manifest = read_json(root/'PACKAGE_MANIFEST.json')
    for name, expected in manifest.items():
        target = (root/name).resolve()
        target.relative_to(root.resolve())
        if digest(target) != expected:
            raise ValueError('The validation package changed: '+name)
    return manifest

def verify_default_fits(step3, protocol):
    from pebr_revision_step3.io import complete, inside
    from .aggregate import row_key
    rows = read_json(step3/'results/topic_words.json')
    expected = {(c, k, p, 'not_applicable' if p == 'nmf_original' else '2711', s)
                for c in protocol['corpora'] for k in protocol['topic_counts'] for p in protocol['policies']
                for s in protocol['final_model_seeds']}
    if len(rows) != 300 or {row_key(r) for r in rows} != expected:
        raise ValueError('The accepted original Step 3 comparison is incomplete')
    for row in rows:
        p = inside(step3, row['path'])
        if not complete(p) or read_json(p/'result.json') != row:
            raise ValueError('A completed Step 3 fit record changed')
    # Only the 30 no-borrowing fits are reused in the fresh-mask summaries.
    return [dict(row, path=str(inside(step3, row['path'])), origin='accepted_step3_shared_no_borrowing')
            for row in rows if row['policy'] == 'nmf_original']

class Study:
    def __init__(self, step2, step3, work, protocol, descriptions, baseline_rows, logger):
        from pebr_revision_step3.runner import load_collections
        from .aggregate import row_key
        self.step2, self.step3, self.work, self.protocol, self.logger = step2, step3, work, protocol, logger
        self.corpora = load_collections(step2, descriptions, protocol)
        self.rows = {row_key(r): r for r in baseline_rows}
        self.fresh_fits, self.reused_fits = 0, 0
        self.timings, self.bank_checks = [], []
        self.chinese_query_identity = None
        audit_path = step2/'provenance/step1_closeout/audit_report.json'
        if audit_path.is_file():
            findings = [r for r in read_json(audit_path)['all_findings'] if
                        r.get('check') == 'chinese_masked_query_cache' and r.get('status') == 'PASS']
            if len(findings) != 1 or not findings[0].get('cache_identity'):
                raise ValueError('Accepted Chinese query-cache reconstruction identity is missing')
            self.chinese_query_identity = findings[0]['cache_identity']

    def fit(self, corpus, k, mask, policy, seed, representation):
        import numpy as np
        from scipy import sparse
        from threadpoolctl import threadpool_limits
        from transport_topics.topics.learning import fit_topics
        from transport_topics.evaluation.topic_quality import evaluate_topics
        from pebr_revision_step3.io import complete, save_receipt, write_json
        from .aggregate import row_key
        p = self.work/'fits'/corpus/('mask%d'%mask)/('K%d'%k)/policy/('seed%d'%seed)
        if complete(p):
            row = read_json(p/'result.json')
            if row_key(row) != (corpus, k, policy, str(mask), seed):
                raise ValueError('Completed fit condition changed')
            self.rows[row_key(row)] = row
            self.reused_fits += 1
            return row
        p.mkdir(parents=True, exist_ok=True)
        c = self.corpora[corpus]
        learned = sparse.load_npz(representation/'transferred.npz').tocsr()
        self.logger.info('Fit | %s mask=%d K=%d %s seed=%d', corpus, mask, k, policy, seed)
        started = time.perf_counter()
        with threadpool_limits(limits=1):
            result = fit_topics(c['collection'].counts, learned, k, seed, self.protocol['final_nmf'])
        fit_seconds = time.perf_counter()-started
        if not result.audit['fit']['converged']:
            write_json(p/'failed_fit.json', result.audit)
            raise RuntimeError('NMF did not converge. No automatic seed/parameter change: '+str(p))
        result.validate(c['collection'].counts.shape[0], len(c['vocabulary']), k)
        started = time.perf_counter()
        quality = evaluate_topics(result.topic_scores, c['vocabulary'], c['reference'], 10, True)
        evaluation_seconds = time.perf_counter()-started
        for name, array in [('topic_scores', result.topic_scores), ('doc_topics', result.doc_topics),
                            ('retrieval_features', result.retrieval_features),
                            ('source_rows', c['collection'].source_rows), ('vocabulary', c['vocabulary'])]:
            np.save(p/(name+'.npy'), array, allow_pickle=False)
        write_json(p/'quality.json', quality)
        write_json(p/'audit.json', result.audit)
        row = {'corpus': corpus, 'K': k, 'policy': policy, 'mask_base': int(mask), 'seed': int(seed),
               'npmi': quality['npmi'], 'cv': quality['cv'], 'diversity': quality['diversity'],
               'top_words': quality['top_words'], 'fit_seconds': fit_seconds,
               'evaluation_seconds': evaluation_seconds, 'path': str(p),
               'fit_converged': True, 'origin': 'fresh_mask_revised_validation'}
        write_json(p/'result.json', row)
        save_receipt(p)
        self.rows[row_key(row)] = row
        self.fresh_fits += 1
        self.logger.info('Result | NPMI %.6f C_v %.6f TD %.6f', row['npmi'], row['cv'], row['diversity'])
        return row

    def run(self):
        from threadpoolctl import threadpool_limits
        from pebr_revision_step3.io import complete, write_csv, write_json
        from .banks import query_records, validate_default_reconstruction, make_encoder, prepare_bank, load_bank
        from .selection import select_bank
        from .references import read_references
        from .aggregate import export
        records = {}
        for corpus in self.protocol['corpora']:
            c = self.corpora[corpus]
            records[corpus] = query_records(corpus, c['collection'], self.work/'tokenizer_cache')
            self.bank_checks.append(validate_default_reconstruction(
                corpus, c['collection'], c['bank'], records[corpus], self.protocol,
                self.chinese_query_identity if corpus == 'chinese_pooled' else None))
        write_json(self.work/'default_mask_reconstruction.json', self.bank_checks)
        for corpus in self.protocol['corpora']:
            c = self.corpora[corpus]
            collection = c['collection']
            encoder = None
            references = {k: read_references(self.step3, collection, k, self.protocol) for k in self.protocol['topic_counts']}
            for mask in self.protocol['selection_mask_bases']:
                directory = self.work/'mask_banks'/corpus/('mask%d'%mask)
                started = time.perf_counter()
                if complete(directory):
                    bank = load_bank(directory, collection, self.protocol)
                else:
                    if encoder is None:
                        encoder = make_encoder(corpus, self.step2, self.work, collection, self.protocol, self.logger)
                    bank = prepare_bank(corpus, collection, c['bank'], records[corpus], self.protocol, mask, encoder, directory, self.logger)
                self.timings.append({'phase': 'masked_retrieval', 'corpus': corpus, 'mask_base': mask, 'K': 'shared',
                                     'seconds_this_invocation': time.perf_counter()-started, 'may_include_reuse': True})
                for k in self.protocol['topic_counts']:
                    started = time.perf_counter()
                    with threadpool_limits(limits=1):
                        paths = select_bank(collection, bank, references[k], self.protocol, mask,
                            self.work/'selection'/corpus/('mask%d'%mask)/('K%d'%k), self.logger)
                    self.timings.append({'phase': 'ten_policy_selection', 'corpus': corpus, 'mask_base': mask, 'K': k,
                                         'seconds_this_invocation': time.perf_counter()-started, 'may_include_reuse': True})
                    for policy in self.protocol['policies']:
                        if policy == 'nmf_original':
                            continue
                        for seed in self.protocol['final_model_seeds']:
                            self.fit(corpus, k, mask, policy, seed, paths[policy])
                    export(self.work/'results', list(self.rows.values()), self.protocol)
                    write_csv(self.work/'phase_timings.csv', self.timings)
            del encoder
        return export(self.work/'results', list(self.rows.values()), self.protocol, True)

def execute(project, explicit=None, candidate_workspace=None, mode='run'):
    project = Path(project).resolve()
    work = logger = lock = study = release = None
    started = time.perf_counter()
    report = {'version': VERSION, 'status': 'starting', 'full_revision_complete': False,
              'selection_validation_complete': False}
    try:
        if not (project/'main.py').is_file():
            raise ValueError('Run beside the existing main.py')
        package_root = Path(__file__).resolve().parent.parent
        source_manifest = verify_package(package_root)
        step3, step2, accepted, identity, original_protocol, descriptions, manifest = resolve_and_bind(project, explicit)
        from pebr_revision_step3.io import write_json, write_csv, acquire_run_lock, release_run_lock, verify_step2
        from pebr_revision_step3.runner import validate_environment
        from .protocol import build_protocol
        from .gate import verify_candidate
        release = release_run_lock
        protocol = build_protocol(original_protocol)
        environment = validate_environment(read_json(step2/'environment.json'))
        wanted = read_json(step2/'environment.json')['packages']
        encoder_environment = {}
        for name in ('jieba', 'tokenizers', 'onnxruntime-gpu', 'torch'):
            if wanted.get(name) is not None:
                found = metadata.version(name)
                if found != wanted[name]:
                    raise ValueError('Encoder/tokenizer environment changed: '+name)
                encoder_environment[name] = found
        candidate, candidate_check = verify_candidate(project, candidate_workspace,
                                       package_root/'accepted_development', descriptions, step2)
        baseline_rows = verify_default_fits(step3, original_protocol)
        run_identity = {'version': VERSION, 'step3_runner_identity': accepted['runner_identity'],
                        'step2_snapshot': manifest['snapshot_hash'], 'protocol': protocol,
                        'source_manifest': source_manifest, 'environment': environment,
                        'encoder_environment': encoder_environment}
        key = canonical_hash(run_identity)
        proposed_work = project/'outputs/revision_rule_lab/fresh_selection'/(step3.name+'_'+key[:12])
        proposed_work.mkdir(parents=True, exist_ok=True)
        lock = acquire_run_lock(proposed_work/'RUNNING.lock')
        work = proposed_work
        logger = logging.getLogger('pebr_fresh_'+key)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')
        for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(work/'console_log.txt', encoding='utf-8')):
            handler.setFormatter(fmt)
            logger.addHandler(handler)
        logger.info('REVISED PEBR FRESH-SELECTION VALIDATION | 4 new masks | 1200 new final fits')
        logger.info('Known development mask 2711 is excluded from validation summaries. No parameter tuning.')
        for name in source_manifest:
            target = work/'runner_source'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file():
                shutil.copyfile(package_root/name, target)
            elif digest(target) != source_manifest[name]:
                raise ValueError('Frozen validation runner source changed')
        write_json(work/'identity.json', run_identity)
        write_json(work/'protocol.json', protocol)
        write_json(work/'candidate_gate.json', dict(candidate_check, candidate_workspace=str(candidate)))
        report.update(step2_workspace=str(step2), step3_workspace=str(step3), candidate_workspace=str(candidate),
                      runner_identity=key, selection_mask_bases=protocol['selection_mask_bases'],
                      planned_new_final_fits=1200, shared_no_borrowing_fits=30,
                      development_bank_in_validation=False, original_experiments_replaced=False,
                      revised_scientific_rule_changed_in_this_stage=False,
                      preprocessing_changed=False, reference_predictors_reused=True,
                      protected_word_study_complete=False, reference_fold_sensitivity_complete=False,
                      neural_baseline_audit_complete=False)
        study = Study(step2, step3, work, protocol, descriptions, baseline_rows, logger)
        if mode == 'preflight':
            from .references import read_references
            from .banks import query_records, validate_default_reconstruction
            for c, value in study.corpora.items():
                for k in protocol['topic_counts']:
                    read_references(step3, value['collection'], k, protocol)
                records = query_records(c, value['collection'], work/'tokenizer_cache')
                study.bank_checks.append(validate_default_reconstruction(c, value['collection'], value['bank'],
                    records, protocol, study.chinese_query_identity if c == 'chinese_pooled' else None))
            report['status'] = 'preflight_complete_no_new_selection_or_fitting'
            logger.info('PREFLIGHT COMPLETE | candidate/input/reference/mask checks passed; no new fits')
        else:
            result = study.run()
            verify_step2(step2)
            verify_default_fits(step3, original_protocol)
            report.update(status='fresh_selection_validation_complete', selection_validation_complete=True,
                          completed_new_fit_artifacts=1200, total_unique_fit_records=len(study.rows),
                          fitted_this_invocation=study.fresh_fits, fit_reuses_this_invocation=study.reused_fits,
                          paired_fresh_contrasts=len(result['paired_fresh_bank_contrasts']),
                          frozen_inputs_reverified_after_run=True)
            logger.info('VALIDATION COMPLETE | 4 fresh selection banks | 1200 new fits + 30 shared NMF fits')
        code = 0
    except BaseException as error:
        report.update(status='blocked', error=('Interrupted; saved checkpoints are resumable.'
            if isinstance(error, (KeyboardInterrupt, SystemExit)) else type(error).__name__+': '+str(error)))
        if work is None:
            work = project/'outputs/revision_rule_lab/fresh_selection'/('blocked_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
            work.mkdir(parents=True, exist_ok=True)
        (work/'exception.txt').write_text(traceback.format_exc(), encoding='utf-8')
        if logger:
            logger.error('VALIDATION BLOCKED | %s', report['error'])
        else:
            print('VALIDATION BLOCKED | '+report['error'], flush=True)
        code = 1
    finally:
        if work is not None:
            if study is not None:
                from pebr_revision_step3.io import write_csv, write_json
                from .aggregate import export
                export(work/'results', list(study.rows.values()), study.protocol)
                write_csv(work/'phase_timings.csv', study.timings)
                write_json(work/'default_mask_reconstruction.json', study.bank_checks)
            report['invocation_elapsed_seconds'] = time.perf_counter()-started
            (work/'validation_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            (work/'READ_ME_FIRST.md').write_text(
                '# Revised-rule fresh-selection validation\n\nStatus: '+report['status']+
                '\n\nRead results/RESULTS.md, results/coverage.json and results/paired_fresh_bank_contrasts.csv. '
                'The four fresh-mask results are separate from development mask 2711. '
                'Five final NMF seeds are averaged within each bank; between-bank SDs describe mask variation, not confidence intervals or new-corpus uncertainty. '
                'Original no-borrowing fits are reused. Policy proposed means ORIGINAL PEBR; preserve_forward_fixed_cap means REVISED PEBR. '
                'Matched controls target the revised action mass and contributing donor count. '
                'Inspect all positive and negative effects before writing conclusions. '
                'This stage does not complete protected-word evaluation, reference/fold sensitivity, neural baseline checks or transportation case regeneration.\n', encoding='utf-8')
            archive = work/'PEBR_fresh_selection_inspection.zip'
            with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
                for name in ('validation_report.json', 'identity.json', 'protocol.json', 'candidate_gate.json',
                             'phase_timings.csv', 'default_mask_reconstruction.json', 'READ_ME_FIRST.md',
                             'console_log.txt', 'exception.txt'):
                    path = work/name
                    if path.is_file():
                        z.write(path, name)
                for path in (work/'results').glob('*'):
                    if path.is_file():
                        z.write(path, path.relative_to(work).as_posix())
                for path in (work/'runner_source').rglob('*'):
                    if path.is_file():
                        z.write(path, path.relative_to(work).as_posix())
                for pattern in ('mask_banks/*/*/summary.json', 'encoder_audits/*/*.json', 'selection/*/*/*/*/summary.json',
                                'selection/*/*/*/preserve_forward_fixed_cap/actions.json'):
                    for path in work.glob(pattern):
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
