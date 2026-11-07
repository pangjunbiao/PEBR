"""Common-predictor NMF sharing controls, separate from the proposed method."""
from copy import deepcopy
from pathlib import Path
import json
import time
import numpy as np
from scipy import sparse

from ..config import canonical_hash
from ..data.preparation import check_receipt, complete, save_matrix, save_array
from ..data.prepared import load_selection_bank
from ..runtime.artifacts import file_hash, write_json
from ..proposed.actions import BorrowingAction, NO_BORROWING
from ..proposed.reference import reference_bank
from ..proposed.scoring import ActionEvaluator, downside_utility
from ..proposed.search import select_action
from ..proposed.transfer import normalized_counts, transfer_action, check_transfer
from .borrowing import run_bank, rule_settings

VARIANTS = ('no_borrowing', 'similarity_maximum', 'similarity_matched',
            'full_pool_loo', 'no_downside', 'proposed')


def full_pool_marginals(evaluator, cap, downside_weight):
    union=evaluator.union
    if not len(union):return np.empty(0)
    full=BorrowingAction(tuple(union.tolist()),cap)
    leave=[BorrowingAction(tuple(int(j) for j in union if j!=removed),cap)
           if len(union)>1 else NO_BORROWING for removed in union]
    gain=evaluator.gains([full]+leave)
    return downside_utility(gain[0][None]-gain[1:],downside_weight)


def matched_action(selected, actual_weight, ranked, full_available):
    available=set(int(j) for j in full_available if j>=0)
    count=sum(j in available for j in selected)
    matched=[int(j) for j in ranked if j in available][:count]
    if len(matched)!=count:
        raise AssertionError('The shared available universe cannot match the proposed donor count')
    return BorrowingAction(tuple(matched),float(actual_weight)) if count else NO_BORROWING


def make_controls(root, collection, topics, scope, config, logger, requested_variants=VARIANTS):
    variants=tuple(name for name in VARIANTS if name in set(requested_variants)|{'proposed','no_borrowing'})
    anchor_config = deepcopy(config)
    if config['method']['name'] != 'proposed':
        anchor_config['method']['name'] = 'proposed'
        anchor_config['method']['borrowing']['enabled'] = True
        anchor_config['method']['selection']['strategy'] = 'predictive'
        anchor_config['method']['selection']['downside']['enabled'] = True
        anchor_config['method']['transfer']['budget_mode'] = 'grid'
    anchor_config['borrowing_run']['receiver_limit'] = 0
    settings = rule_settings(anchor_config)
    references, fit_audits = reference_bank(root, collection.summary, collection.counts, collection.folds,
                                            topics, config['method']['predictor'], logger)
    bank = load_selection_bank(collection, scope, config)
    anchor = run_bank(root, collection, bank, references, fit_audits, topics, anchor_config, logger)
    if not anchor['complete_bank']:
        raise ValueError('Controlled comparisons require a complete proposed anchor bank')
    spec = {'schema': 1, 'anchor_bank_key': anchor['bank_key'], 'settings': settings,'variants':variants,
            'implementation': file_hash(Path(__file__)),
            'loo': 'full-pool minus each donor, fixed total offered cap; downside utility of viewwise marginal gain; top-k ranking',
            'similarity': 'candidate union ranked by full-receiver MiniLM cosine; same final availability convention',
            'matched': 'same realized donor count and normalized count mass as proposed'}
    key = canonical_hash(spec)
    directory = root/'outputs/models/sharing_controls'/collection.summary['name']/('K%d'%topics)/scope/key[:16]
    directory.mkdir(parents=True, exist_ok=True)
    if check_receipt(directory):
        logger.info('Reusing aligned sharing controls | %s K=%d %s.', collection.summary['name'], topics, scope)
        return directory, json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    source = root/anchor['path']
    anchor_selected = np.load(source/'selected_donors.npy', allow_pickle=False)
    anchor_offered = np.load(source/'offered_beta.npy', allow_pickle=False)
    anchor_actual = np.load(source/'actual_beta.npy', allow_pickle=False)
    proposed = sparse.load_npz(source/'transferred.npz').tocsr()
    donors = normalized_counts(collection.counts)
    base = normalized_counts(bank.base)
    n, vocabulary = base.shape
    representations = {name: base.copy() for name in variants}
    representations['proposed'] = proposed.toarray()
    offered = {name: np.zeros(n) for name in variants}
    realized = {name: np.zeros(n) for name in variants}
    donor_count = {name: np.zeros(n, dtype=int) for name in variants}
    embeddings = np.load(collection.directory/'embeddings.npy', allow_pickle=False)
    queries = np.load(bank.directory/'full_receiver_embeddings.npy', allow_pickle=False)
    projections = [reference.projection(donors) for reference in references]
    views = config['method']['views']['count']
    records = []
    checkpoints = root/'outputs/checkpoints/sharing_controls'/key
    from .borrowing import _row_receipt
    tick = time.monotonic()
    for i in range(n):
        def choose(i=i):
            plans = {name: NO_BORROWING for name in variants}
            if bank.eligible[i]:
                union = np.unique(bank.candidates[i][bank.candidates[i] >= 0])
                if len(union):
                    score = embeddings[union] @ queries[i]
                    ranked = union[np.argsort(-score, kind='stable')].tolist()
                    if 'similarity_maximum' in variants:
                        plans['similarity_maximum'] = BorrowingAction(tuple(ranked[:settings['max_donors']]), settings['cap'])
                    block = slice(i*views, (i+1)*views)
                    fold = int(collection.folds[i])
                    evaluator = ActionEvaluator(references[fold], bank.visible[block].toarray(),
                                                bank.targets[block].toarray(), bank.candidates[i], donors,
                                                projections[fold], config['borrowing_run']['proposal_batch_size']) if any(v in variants for v in ('no_downside','full_pool_loo')) else None
                    if 'no_downside' in variants:
                        mean = select_action(evaluator, settings['budgets'], settings['max_donors'], 0.0, settings['margin'])
                        plans['no_downside'] = mean.action
                    if 'full_pool_loo' in variants:
                        marginal = full_pool_marginals(evaluator,settings['cap'],settings['downside_weight'])
                        order = np.argsort(-marginal, kind='stable')
                        plans['full_pool_loo'] = BorrowingAction(tuple(union[order[:settings['max_donors']]].tolist()), settings['cap'])
                    chosen = tuple(int(j) for j in anchor_selected[i] if j >= 0)
                    plans['proposed'] = BorrowingAction(chosen, float(anchor_offered[i])) if chosen else NO_BORROWING
                    if 'similarity_matched' in variants:
                        plans['similarity_matched'] = matched_action(chosen,anchor_actual[i],ranked,bank.full_candidates[i])
            return {'plans': {name: {'donors': list(a.donors), 'beta': a.beta} for name,a in plans.items()}}
        payload, _ = _row_receipt(checkpoints/('%06d.json'%i), {'key': key, 'row': i}, choose)
        row = {'document_index': i, 'source_row': int(collection.source_rows[i]), 'variants': {}}
        for name, stored in payload['plans'].items():
            action = BorrowingAction(tuple(stored['donors']), stored['beta'])
            mixed, alpha, available = transfer_action(base[i], action, bank.full_candidates[i], donors)
            check_transfer(base[i], mixed, alpha, settings['cap'])
            representations[name][i] = mixed
            offered[name][i], realized[name][i], donor_count[name][i] = action.beta, alpha, len(available)
            row['variants'][name] = {'selected_source_rows': collection.source_rows[list(action.donors)].tolist(),
                                     'available_source_rows': collection.source_rows[list(available)].tolist(),
                                     'offered_beta': action.beta, 'actual_beta': alpha}
        if 'similarity_matched' in variants and abs(realized['similarity_matched'][i]-anchor_actual[i]) > 1e-12:
            raise ArithmeticError('Matched sharing does not match the proposed realized weight')
        if 'similarity_matched' in variants and donor_count['similarity_matched'][i] != donor_count['proposed'][i]:
            raise ArithmeticError('Matched sharing does not match the proposed realized donor count')
        if not np.allclose(representations['proposed'][i], proposed[i].toarray()[0], atol=2e-12, rtol=0):
            raise ArithmeticError('The control pipeline changed the proposed representation')
        records.append(row)
        if i==0 or i+1==n or time.monotonic()-tick >= 10:
            logger.info('Sharing-control progress | %s K=%d %s | %d/%d.', collection.summary['name'], topics, scope, i+1, n)
            tick = time.monotonic()
    for name, rows in representations.items():
        save_matrix(directory/(name+'.npz'), sparse.csr_matrix(rows))
    for name, values in (('actual_beta',realized),('offered_beta',offered),('actual_donor_count',donor_count)):
        np.savez_compressed(directory/(name+'.npz'), **values)
    save_array(directory/'source_rows.npy', collection.source_rows)
    write_json(directory/'actions.json', records, overwrite=True)
    prediction = {}
    if scope == 'outer_audit':
        from ..evaluation.predictive_audit import audit_actions
        outer = sparse.load_npz(bank.directory/'outer.npz').tocsr()
        for name, rows in representations.items():
            prediction[name] = audit_actions(references, collection.folds, collection.source_rows,
                bank.base, outer, sparse.csr_matrix(rows), bank.eligible, np.ones(n,dtype=bool), topics,
                directory/(name+'_outer_prediction.csv'))
            csv_path = directory/(name+'_outer_prediction.csv')
            csv_path.write_text(csv_path.read_text(encoding='utf-8').replace('proposed_log_score','action_log_score'), encoding='utf-8')
    summary = {'condition': collection.summary['name'], 'topics': topics, 'scope': scope,
               'identity': key, 'variants': list(variants), 'anchor_path': anchor['path'],
               'matched_amount_count_violations': 0, 'proposed_representation_changes': 0,
               'bounds_violations': 0, 'outer_prediction': prediction,
               'mean_actual_beta': {k:float(v.mean()) for k,v in realized.items()},
               'loo_is_atlas_implementation': False,
               'setting_changes_from_results': False}
    write_json(directory/'identity.json', spec, overwrite=True)
    write_json(directory/'summary.json', summary, overwrite=True)
    complete(directory)
    return directory, summary
