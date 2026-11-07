"""Common receiver loop for the frozen default bank and all ten policies."""
from __future__ import annotations
import time
import numpy as np
from scipy import sparse
from transport_topics.proposed.actions import BorrowingAction, NO_BORROWING
from transport_topics.proposed.scoring import ActionEvaluator, downside_utility
from transport_topics.proposed.transfer import normalized_counts, transfer_action, check_transfer
from transport_topics.experiments.sharing_controls import full_pool_marginals, matched_action
from pebr_revision_step2.policies import select_predictive_variant
from .io import canonical_hash, checkpoint, complete, read_json, save_receipt, write_json

def choose_policies(evaluator, single_evaluator, candidate_lists, ranked, full_available,
                    base, donors, protocol, cached=None, requested=None):
    """Ordinary and LOO controls receive no new predictive rejection threshold."""
    requested = list(requested or [p for p in protocol['policies'] if p != 'nmf_original'])
    result = dict(cached or {})
    cap, limit = protocol['cap'], protocol['max_donors']
    budgets = [cap*f for f in protocol['positive_budget_cap_fractions']]
    margin, weight = protocol['admission_margin'], protocol['downside_weight']
    names = ['proposed'] + [p for p in requested if p != 'proposed']
    for name in names:
        if name in result: continue
        selection_value, trace, proposals = None, [], 0
        ev = single_evaluator if name == 'single_view' else evaluator
        if name == 'nmf_similarity':
            action = BorrowingAction(tuple(ranked[:limit]), cap) if ranked else NO_BORROWING
        elif name == 'similarity_matched':
            anchor = result['proposed']
            action = matched_action(tuple(anchor['donors']), anchor['actual_beta'], ranked, full_available)
        elif name == 'full_pool_loo':
            values = full_pool_marginals(evaluator, cap, weight)
            order = np.argsort(-values, kind='stable')[:limit]
            action = BorrowingAction(tuple(evaluator.union[order].tolist()), cap) if len(order) else NO_BORROWING
            proposals = len(evaluator.union)+1
        else:
            lists = candidate_lists[:1] if name == 'single_view' else candidate_lists
            decision = select_predictive_variant(name, ev, lists, budgets, cap, limit, weight, margin)
            action, selection_value = decision.action, decision.selection_utility
            trace, proposals = decision.trace, decision.proposals
        gains = ev.gains([action])[0]
        direct = ev.direct_gains(action)
        score_error = float(np.max(np.abs(gains-direct)))
        if score_error > 2e-8:
            raise ArithmeticError('Projected and direct word-space action scores disagree')
        policy_weight = 0. if name == 'no_downside' else weight
        actual_utility = float(downside_utility(gains, policy_weight))
        if name not in {'nmf_similarity', 'similarity_matched', 'full_pool_loo'}:
            if action.donors and selection_value <= margin:
                raise ArithmeticError('Predictive selector admitted nonpositive selection utility')
            if name != 'individual_donor' and abs(actual_utility-selection_value) > 2e-8:
                raise ArithmeticError('Complete-mixture selection and actual utility differ')
        mixed, alpha, available = transfer_action(base, action, full_available, donors)
        tv = check_transfer(base, mixed, alpha, cap)
        result[name] = {'donors': list(action.donors), 'offered_beta': float(action.beta),
            'actual_beta': float(alpha), 'available_donors': list(available),
            'selection_utility': selection_value, 'actual_mixture_utility': actual_utility,
            'actual_view_gains': gains.tolist(), 'trace': trace, 'proposals_scored': proposals,
            'score_transfer_max_error': score_error, 'total_variation': float(tv)}
    return {name: result[name] for name in names}

def empty_record():
    return {'donors': [], 'offered_beta': 0., 'actual_beta': 0., 'available_donors': [],
            'selection_utility': 0., 'actual_mixture_utility': 0., 'actual_view_gains': [],
            'trace': [], 'proposals_scored': 0, 'score_transfer_max_error': 0., 'total_variation': 0.}

def run_selection(collection, bank, references, protocol, directory, logger, requested=None):
    directory.mkdir(parents=True, exist_ok=True)
    requested = list(requested or [p for p in protocol['policies'] if p != 'nmf_original'])
    if 'proposed' not in requested: requested.insert(0, 'proposed')
    paths = {p: directory/p for p in requested}
    pending = [p for p in requested if not complete(paths[p])]
    if not pending:
        logger.info('Reusing complete policy banks | %s | %s', collection.summary['name'], directory.parent.name)
        return paths
    start = time.perf_counter(); n, v = collection.counts.shape
    donors, base = normalized_counts(collection.counts), normalized_counts(bank.base)
    projections = [r.projection(donors) for r in references]
    embeddings = np.load(collection.directory/'embeddings.npy', allow_pickle=False)
    queries = np.load(bank.directory/'full_receiver_embeddings.npy', allow_pickle=False)
    matrices = {p: np.empty((n, v), dtype=np.float64) for p in pending}
    records = {p: [] for p in pending}
    views = protocol['views']; tick = time.monotonic()
    bank_identity = canonical_hash({'collection': collection.summary['identity_sha256'],
        'topics': references[0].k, 'scope': bank.scope, 'mask_base': 2711,
        'protocol': protocol, 'runner': directory.parents[2].name})
    reused = 0
    for i in range(n):
        row_identity = {'bank': bank_identity, 'receiver': i}
        cached = {}
        for p in requested:
            path = directory/'receivers'/p/('%06d.json' % i)
            if path.is_file():
                value, _ = checkpoint(path, row_identity, lambda: None)
                cached[p] = value; reused += 1
        needed = [p for p in pending if p not in cached]
        if needed:
            if bank.eligible[i] and np.any(bank.candidates[i] >= 0):
                union = np.unique(bank.candidates[i][bank.candidates[i] >= 0])
                scores = embeddings[union] @ queries[i]
                ranked = union[np.argsort(-scores, kind='stable')].tolist()
                fold = int(collection.folds[i]); block = slice(i*views, (i+1)*views)
                visible, targets = bank.visible[block].toarray(), bank.targets[block].toarray()
                ev = ActionEvaluator(references[fold], visible, targets, bank.candidates[i], donors, projections[fold], 128)
                one = ActionEvaluator(references[fold], visible[:1], targets[:1], bank.candidates[i,:1], donors, projections[fold], 128) if 'single_view' in needed else None
                chosen = choose_policies(ev, one, bank.candidates[i], ranked, bank.full_candidates[i],
                    base[i], donors, protocol, cached, needed)
            else: chosen = {p: empty_record() for p in needed + ['proposed']}
            for p in needed:
                path = directory/'receivers'/p/('%06d.json' % i)
                cached[p], _ = checkpoint(path, row_identity, lambda p=p: chosen[p])
        for p in pending:
            rec = cached[p]
            action = BorrowingAction(tuple(rec['donors']), rec['offered_beta'])
            union = set(int(x) for x in bank.candidates[i].ravel() if x >= 0)
            if not set(action.donors).issubset(union) or len(action.donors)>protocol['max_donors']:
                raise ValueError('Checkpoint donor action violates the candidate universe')
            mixed, alpha, available = transfer_action(base[i], action, bank.full_candidates[i], donors)
            check_transfer(base[i], mixed, alpha, protocol['cap'])
            if abs(alpha-rec['actual_beta'])>1e-12 or list(available)!=rec['available_donors']:
                raise ArithmeticError('Checkpoint final availability or mass changed')
            matrices[p][i] = mixed
            records[p].append(dict(rec, document_index=i, source_row=int(collection.source_rows[i]),
                                   eligible=bool(bank.eligible[i]), fold=int(collection.folds[i])))
        if 'similarity_matched' in pending:
            a, b = cached['proposed'], cached['similarity_matched']
            if abs(a['actual_beta']-b['actual_beta'])>1e-12 or len(a['available_donors'])!=len(b['available_donors']):
                raise ArithmeticError('Matched mass or contributing donor count differs')
        if i==0 or i+1==n or time.monotonic()-tick>=10:
            logger.info('Selection | %s %s | %d/%d receivers | %.1fs', collection.summary['name'], directory.name, i+1, n, time.perf_counter()-start)
            tick = time.monotonic()
    elapsed = time.perf_counter()-start
    for p in pending:
        target = paths[p]; target.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(target/'transferred.npz', sparse.csr_matrix(matrices[p]))
        write_json(target/'actions.json', records[p])
        write_json(target/'summary.json', {'policy': p, 'posts': n, 'complete_bank': True,
            'mask_base': 2711, 'mean_actual_beta': float(np.mean([r['actual_beta'] for r in records[p]])),
            'borrowing_coverage': float(np.mean([r['actual_beta']>0 for r in records[p]])),
            'mean_contributing_donors': float(np.mean([len(r['available_donors']) for r in records[p]])),
            'shared_receiver_loop_seconds': elapsed, 'timing_scope': 'combined requested policies, not per-policy standalone runtime',
            'matched_mass_count_verified': p=='similarity_matched', 'reused_receiver_records': reused})
        save_receipt(target)
    return paths
