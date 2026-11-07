"""Checkpointed execution of the proposed rule on prepared evidence banks."""
from __future__ import annotations

from pathlib import Path
import json
import time
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from ..config import canonical_hash
from ..schemas import number
from ..runtime.artifacts import file_hash, safe_path, write_json
from ..data.preparation import check_receipt, complete, save_array, save_matrix
from ..data.prepared import preparation_dependency, load_collection, load_selection_bank
from ..proposed.actions import BorrowingAction, NO_BORROWING
from ..proposed.reference import reference_bank
from ..proposed.scoring import ActionEvaluator, downside_utility
from ..proposed.search import select_action
from ..proposed.transfer import normalized_counts, transfer_action, check_transfer


def rule_settings(config):
    method = config['method']
    if method['name'] not in ('proposed', 'nmf_original'):
        raise NotImplementedError('The requested baseline is connected in Topic Models and Baseline Integration; this phase does not substitute the proposed rule for it')
    enabled = method['borrowing']['enabled']
    if enabled and method['selection']['strategy'] != 'predictive':
        raise NotImplementedError('This phase executes the proposed predictive selector or no borrowing; similarity and LOO controls are connected in Topic Models and Baseline Integration')
    mode = method['transfer']['budget_mode']
    if enabled and mode not in ('grid', 'maximum'):
        raise NotImplementedError('Matched controls require the proposed anchor actions and belong to the baseline-integration phase')
    cap = number(method['transfer']['max_weight'], 'max_weight')
    grid = [cap] if mode == 'maximum' else [cap * number(f, 'grid_fractions') for f in method['transfer']['grid_fractions']]
    downside = method['selection']['downside']
    return {'enabled': enabled, 'budgets': grid, 'cap': cap,
            'max_donors': method['selection']['max_donors'],
            'downside_weight': number(downside['weight'], 'downside.weight') if downside['enabled'] else 0.0,
            'margin': number(method['selection']['margin'], 'margin')}


def _row_receipt(path, identity, compute):
    if path.is_file():
        envelope = json.loads(path.read_text(encoding='utf-8'))
        if envelope['identity'] != identity or envelope['sha256'] != canonical_hash(envelope['record']):
            raise ValueError('A receiver checkpoint changed; preserve and inspect: ' + str(path))
        return envelope['record'], True
    record = compute()
    write_json(path, {'identity': identity, 'sha256': canonical_hash(record), 'record': record})
    return record, False


def _select_receiver(i, collection, bank, references, donors, projections, config, settings):
    started = time.perf_counter()
    views = config['method']['views']['count']
    reference = references[int(collection.folds[i])]
    calls, fallbacks = reference.solver.rows_solved, reference.solver.direct_fallback_rows
    action, gains, value, trace, proposals = NO_BORROWING, np.zeros(views), 0.0, [], 0
    reason = 'borrowing_disabled' if not settings['enabled'] else 'insufficient_masked_types'
    score_error = 0.0
    availability = []
    union = np.unique(bank.candidates[i][bank.candidates[i] >= 0])
    if settings['enabled'] and bank.eligible[i] and len(union):
        block = slice(i * views, (i + 1) * views)
        evaluator = ActionEvaluator(reference, bank.visible[block].toarray(), bank.targets[block].toarray(),
                                    bank.candidates[i], donors, projections[int(collection.folds[i])],
                                    batch_size=config['borrowing_run']['proposal_batch_size'])
        result = select_action(evaluator, settings['budgets'], settings['max_donors'],
                               settings['downside_weight'], settings['margin'])
        action, gains, value, trace, proposals = result.action, result.gains, result.utility, result.trace, result.proposals
        direct = evaluator.direct_gains(action)
        score_error = float(np.max(np.abs(gains - direct)))
        if score_error > 2e-8:
            raise ArithmeticError('Selected action differs between projected scoring and exact word-space transfer')
        availability = [float(np.mean(evaluator.availability[:, evaluator.lookup[j]])) for j in action.donors]
        reason = 'admitted' if action.donors else 'no_improving_extension'
    elif settings['enabled'] and bank.eligible[i]:
        reason = 'empty_candidate_union'
    if action.donors and value <= settings['margin']:
        raise ArithmeticError('A nonempty action failed the positive-utility admission rule')
    mean_gain = float(gains.mean())
    downside = float(np.maximum(-gains, 0).mean())
    if action.donors and mean_gain <= settings['downside_weight'] * downside + settings['margin']:
        raise ArithmeticError('Admitted action failed the empirical predictive-surplus property')
    return {'document_index': int(i), 'source_row': int(collection.source_rows[i]),
            'fold': int(collection.folds[i]), 'eligible': bool(bank.eligible[i]),
            'selected_donors': list(action.donors),
            'selected_donor_source_rows': collection.source_rows[list(action.donors)].tolist(),
            'offered_beta': action.beta, 'utility': value, 'view_gains': gains.tolist(),
            'mean_view_gain': mean_gain, 'mean_downside': downside,
            'negative_view_fraction': float(np.mean(gains < -1e-10)),
            'selected_availability': availability, 'candidate_union_count': len(union),
            'accepted_extensions': trace, 'proposals_scored': proposals, 'reason': reason,
            'score_transfer_max_error': score_error,
            'nnls_rows': reference.solver.rows_solved - calls,
            'direct_nnls_fallback_rows': reference.solver.direct_fallback_rows - fallbacks,
            'max_normalized_kkt_residual': reference.solver.max_kkt_residual,
            'seconds': time.perf_counter() - started}


def run_bank(root, collection, bank, references, predictor_audits, topics, config, logger):
    settings = rule_settings(config)
    run_settings = config['borrowing_run']
    code = root / 'src/transport_topics'
    implementation = {name: file_hash(code / name) for name in (
        'numerics/nnls.py', 'proposed/reference.py', 'proposed/actions.py',
        'proposed/scoring.py', 'proposed/search.py', 'proposed/transfer.py',
        'experiments/borrowing.py', 'evaluation/predictive_audit.py')}
    identity = {'schema': 1, 'condition': collection.summary['identity_sha256'],
                'scope': bank.scope, 'topics': topics, 'settings': settings,
                'predictor_keys': [item['cache_key'] for item in predictor_audits],
                'implementation': implementation, 'receiver_limit': run_settings['receiver_limit']}
    key = canonical_hash(identity)
    directory = root / 'outputs/models/proposed' / collection.summary['name'] / ('K%d' % topics) / bank.scope / key[:16]
    checkpoints = root / 'outputs/checkpoints/borrowing' / key / 'receivers'
    directory.mkdir(parents=True, exist_ok=True)
    if check_receipt(directory):
        if not config['output']['resume']:
            raise ValueError('Completed action artifacts exist and output.resume is disabled')
        logger.info('Reusing complete borrowing bank | %s K=%d %s.', collection.summary['name'], topics, bank.scope)
        return json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    n, vocabulary = collection.counts.shape
    donor_rows = normalized_counts(collection.counts)
    receiver_rows = normalized_counts(bank.base)
    projections = [reference.projection(donor_rows) for reference in references]
    transferred = receiver_rows.copy()
    selected = np.full((n, settings['max_donors']), -1, dtype=np.int64)
    offered, actual, utilities = np.zeros(n), np.zeros(n), np.zeros(n)
    processed = np.zeros(n, dtype=bool)
    view_gains = np.zeros((n, config['method']['views']['count']))
    eligible_indices = np.flatnonzero(bank.eligible)
    limit = run_settings['receiver_limit']
    if limit:
        eligible_indices = eligible_indices[:limit]
    work = np.sort(np.concatenate((eligible_indices, np.flatnonzero(~bank.eligible))))
    records, reused, max_tv, max_error = [], 0, 0.0, 0.0
    started, last_progress = time.perf_counter(), 0.0
    logger.info('Selecting actions | %s K=%d %s | eligible=%d requested=%d.',
                collection.summary['name'], topics, bank.scope, int(bank.eligible.sum()), len(eligible_indices))
    for position, i in enumerate(work):
        row_identity = {'bank_key': key, 'document_index': int(i)}
        checkpoint = checkpoints / ('%06d.json' % i)
        if checkpoint.exists() and not config['output']['resume']:
            raise ValueError('Receiver checkpoints exist and output.resume is disabled')
        record, was_reused = _row_receipt(
            checkpoint, row_identity,
            lambda i=int(i): _select_receiver(i, collection, bank, references, donor_rows, projections, config, settings))
        reused += int(was_reused)
        action = BorrowingAction(tuple(record['selected_donors']), record['offered_beta'])
        union = set(int(j) for j in bank.candidates[i].ravel() if j >= 0)
        if (not set(action.donors).issubset(union) or len(action.donors) > settings['max_donors']
                or action.beta > settings['cap'] + 1e-12):
            raise ValueError('A saved action is outside the feasible candidate/budget space')
        mixed, alpha, available = transfer_action(receiver_rows[i], action, bank.full_candidates[i], donor_rows)
        tv = check_transfer(receiver_rows[i], mixed, alpha, settings['cap'])
        transferred[i] = mixed
        selected[i, :len(action.donors)] = action.donors
        offered[i], actual[i], utilities[i] = action.beta, alpha, record['utility']
        view_gains[i] = record['view_gains']; processed[i] = True
        records.append(dict(record, actual_beta=alpha, available_selected_donors=list(available),
                            available_donor_source_rows=collection.source_rows[list(available)].tolist(),
                            total_variation=tv))
        max_tv = max(max_tv, tv); max_error = max(max_error, record['score_transfer_max_error'])
        elapsed = time.perf_counter() - started
        if position == 0 or position + 1 == len(work) or elapsed-last_progress >= run_settings['progress_seconds']:
            logger.info('Action progress | %s K=%d %s | %d/%d receivers | elapsed %.1fs | reused %d.',
                        collection.summary['name'], topics, bank.scope, position+1, len(work), elapsed, reused)
            last_progress = elapsed
    for label, value in (('selected_donors', selected), ('offered_beta', offered), ('actual_beta', actual),
                         ('utility', utilities), ('view_gains', view_gains), ('processed', processed),
                         ('source_rows', collection.source_rows)):
        save_array(directory / (label + '.npy'), value)
    matrix = sparse.csr_matrix(transferred)
    save_matrix(directory / 'transferred.npz', matrix)
    write_json(directory / 'actions.json', records, overwrite=True)
    write_json(directory / 'identity.json', identity, overwrite=True)
    prediction = None
    if bank.scope == 'outer_audit':
        # Outer targets are loaded only after selection and transfer artifacts exist.
        from ..evaluation.predictive_audit import audit_actions
        outer = sparse.load_npz(bank.directory / 'outer.npz').tocsr()
        expanded = outer[np.repeat(np.arange(n), config['method']['views']['count'])]
        if expanded.multiply(bank.visible + bank.targets).nnz:
            raise ValueError('Protected outer words entered an inner selection view')
        prediction = audit_actions(references, collection.folds, collection.source_rows,
                                   bank.base, outer, matrix, bank.eligible, processed, topics,
                                   directory / 'outer_prediction.csv')
        write_json(directory / 'outer_prediction.json', prediction, overwrite=True)
        logger.info('Protected prediction audit | %s K=%d | posts=%d mean gain=%.6f nats/token.',
                    collection.summary['name'], topics, prediction['eligible_evaluated_posts'],
                    prediction['mean_post_gain_nats_per_token'] or 0.0)
    admitted = sum(bool(record['selected_donors']) for record in records)
    summary = {'condition': collection.summary['name'], 'condition_identity': collection.summary['identity_sha256'],
               'topics': topics, 'scope': bank.scope, 'bank_key': key,
               'path': directory.relative_to(root).as_posix(), 'total_posts': n,
               'eligible_receivers': int(bank.eligible.sum()), 'processed_receivers': int(processed.sum()),
               'processed_eligible_receivers': len(eligible_indices), 'admitted_receivers': admitted,
               'complete_bank': bool(processed.all()), 'development_only': bool(limit),
               'settings': settings, 'max_total_variation': max_tv,
               'max_score_transfer_error': max_error, 'donor_boundary_violations': 0,
               'admission_property_violations': 0, 'borrowing_bound_violations': 0,
               'no_borrowing_receivers': int(processed.sum())-admitted,
               'mean_offered_beta': float(offered[processed].mean()) if processed.any() else 0.0,
               'mean_actual_beta': float(actual[processed].mean()) if processed.any() else 0.0,
               'offered_at_cap_receivers': int(np.sum(np.isclose(offered[processed], settings['cap'], atol=1e-12))),
               'selected_donor_count_histogram': {str(k): sum(len(r['selected_donors']) == k for r in records)
                                                 for k in range(settings['max_donors']+1)},
               'scored_proposals': sum(r['proposals_scored'] for r in records),
               'nnls_scored_rows': sum(r['nnls_rows'] for r in records),
               'direct_nnls_fallback_rows': sum(r['direct_nnls_fallback_rows'] for r in records),
               'max_normalized_kkt_residual': max((r['max_normalized_kkt_residual'] for r in records), default=0.0),
               'nnls_kkt_tolerance': references[0].solver.kkt_tolerance,
               'selection_compute_seconds': float(sum(r['seconds'] for r in records)),
               'outer_prediction': prediction, 'topic_models_fitted': False,
               'performance_based_setting_changes': False}
    write_json(directory / 'summary.json', summary, overwrite=True)
    complete(directory)
    logger.info('BORROWING BANK COMPLETE | %s K=%d %s | admitted=%d actual mean=%.4f | %s',
                collection.summary['name'], topics, bank.scope, admitted, summary['mean_actual_beta'], summary['path'])
    return summary


def run(root, config, environment, logger, store):
    preparation, dependency = preparation_dependency(root, config)
    store.json('preparation_dependency.json', dependency)
    conditions = preparation['conditions']
    if config['borrowing_run']['collections'] == 'primary':
        conditions = conditions[:1]
    scopes = config['borrowing_run']['banks']
    if not config['evaluation']['prediction']['enabled']:
        scopes = [scope for scope in scopes if scope != 'outer_audit']
        logger.info('Protected prediction audit disabled by configuration.')
    if not scopes:
        raise ValueError('No borrowing banks remain enabled')
    logger.info('Reusing verified preparation run %s; no data downloads or re-encoding.', dependency['preparation_run_id'])
    logger.info('Rule execution: float64 CPU reference/NNLS systems; prepared encoder vectors retain their verified provider provenance.')
    banks, fits = [], []
    # Small NNLS systems avoid multithread BLAS overhead. This numerical policy
    # applies equally to all reference fits and later common sharing controls.
    with threadpool_limits(limits=1):
        for condition in conditions:
            collection = load_collection(root, condition, config)
            for topics in config['topics']['counts']:
                references, audits = reference_bank(root, condition, collection.counts, collection.folds,
                                                    topics, config['method']['predictor'], logger)
                fits.extend(dict(item, condition=condition['name']) for item in audits)
                store.json('predictor_audit.json', {'fits': fits, 'all_folds_excluded': all(a['receiver_fold_excluded'] for a in fits)}, overwrite=True)
                for scope in scopes:
                    bank = load_selection_bank(collection, scope, config)
                    result = run_bank(root, collection, bank, references, audits, topics, config, logger)
                    banks.append(result)
                    store.json('selected_actions_index.json', {'banks': banks}, overwrite=True)
    prediction = [dict(item['outer_prediction'], condition=item['condition'], topics=item['topics'])
                  for item in banks if item['outer_prediction'] is not None]
    active = rule_settings(config)
    registered_scope = (len(conditions) == 6 and preparation['scope'] == 'full'
                        and config['topics']['counts'] == [8, 12, 16]
                        and set(scopes) == {'topic_learning', 'outer_audit'}
                        and config['borrowing_run']['receiver_limit'] == 0
                        and active['enabled'] and active['downside_weight'] == 1.0
                        and abs(active['cap'] - 1/3) < 1e-15
                        and active['max_donors'] == 3
                        and len(active['budgets']) == 4
                        and np.allclose(active['budgets'], [1/12, 1/6, 1/4, 1/3], atol=1e-15))
    summary = {'method_label': 'proposed', 'complete_requested_run': all(b['complete_bank'] for b in banks),
               'active_method': config['method']['name'],
               'complete_registered_proposed_scope': registered_scope and all(b['complete_bank'] for b in banks),
               'development_only': bool(config['borrowing_run']['receiver_limit']) or len(conditions) != 6 or preparation['scope'] != 'full',
               'condition_count': len(conditions), 'topic_counts': config['topics']['counts'],
               'bank_count': len(banks), 'reference_fit_count': len(fits),
               'all_reference_fits_converged': all(f['converged'] for f in fits),
               'prepared_evidence_reused': True, 'source_preparation_run': dependency['preparation_run_id'],
               'sample_manifest_sha256': preparation['sample_manifest_sha256'],
               'total_processed_receivers': sum(b['processed_receivers'] for b in banks),
               'max_score_transfer_error': max(b['max_score_transfer_error'] for b in banks),
               'donor_boundary_violations': 0, 'admission_property_violations': 0,
               'borrowing_bound_violations': 0, 'final_topics_fitted': False,
               'baseline_comparisons_run': False, 'outer_prediction_audited': bool(prediction),
               'next_phase': 'Topic Models and Baseline Integration'}
    store.json('borrowing_rule.json', summary)
    store.json('outer_prediction_audit.json', {'results': prediction, 'selection_or_tuning_use': False})
    if not summary['all_reference_fits_converged']:
        raise RuntimeError('At least one frozen NMF reference reported nonconvergence. Artifacts were saved; inspect predictor_audit.json before proceeding.')
    logger.info('PROPOSED RULE COMPLETE | collections=%d banks=%d reference fits=%d; final topic/baseline results are pending.',
                len(conditions), len(banks), len(fits))
    return summary, dependency
