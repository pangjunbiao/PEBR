"""The accepted revised kernel and prespecified controls, with row checkpoints."""
from __future__ import annotations
import time
import numpy as np
from scipy import sparse
from transport_topics.proposed.actions import BorrowingAction, NO_BORROWING
from transport_topics.proposed.scoring import ActionEvaluator, downside_utility
from transport_topics.proposed.search import select_action, preferred
from transport_topics.proposed.transfer import normalized_counts, transfer_action, check_transfer
from transport_topics.experiments.sharing_controls import full_pool_marginals, matched_action
from pebr_revision_step3.io import canonical_hash, checkpoint, complete, save_receipt, write_json
from .anchored_scoring import ReceiverAnchoredEvaluator
from .protocol import ANCHOR

class Individual:
    def __init__(self, base):
        self.base, self.views, self.union = base, base.views, base.union

    def gains(self, actions):
        unique = list({BorrowingAction((j,), a.beta): None for a in actions for j in a.donors})
        values = dict(zip(unique, self.base.gains(unique))) if unique else {}
        return np.asarray([np.mean([values[BorrowingAction((j,), a.beta)] for j in a.donors], axis=0)
                           if a.donors else np.zeros(self.views) for a in actions]).reshape(len(actions), self.views)

class WeightedHidden(ReceiverAnchoredEvaluator):
    def _diagnostics(self, probability, inactive):
        hidden, shown, net, whole = super()._diagnostics(probability, inactive)
        return hidden, shown, self.hidden_shares[None]*hidden, whole

def rank_order(lists):
    scores = {}
    for row in lists:
        ids = [int(j) for j in row if j >= 0]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate candidates in a masked view')
        for rank, donor in enumerate(ids, 1):
            scores[donor] = scores.get(donor, 0.) + 1./rank
    return tuple(sorted(scores, key=lambda j: (-scores[j], j)))

def choose(reference, visible, targets, lists, donors, original, projection,
           ranked, full_available, protocol):
    """Return actions only from permitted donors; labels/quality metrics are unavailable."""
    cap, limit = protocol['cap'], protocol['max_donors']
    weight, margin = protocol['downside_weight'], protocol['admission_margin']
    grid = [cap*f for f in protocol['positive_budget_cap_fractions']]
    base = normalized_counts(original).reshape(-1)
    ev = ReceiverAnchoredEvaluator(reference, visible, targets, lists, donors, original, projection, 128)
    old = ActionEvaluator(reference, visible, targets, lists, donors, projection, 128)
    weighted = WeightedHidden(reference, visible, targets, lists, donors, original, projection, 128)
    one = ReceiverAnchoredEvaluator(reference, visible[:1], targets[:1], lists[:1], donors, original, projection, 128)
    decisions = {
        ANCHOR: select_action(ev, [cap], limit, weight, margin),
        'proposed': select_action(old, grid, limit, weight, margin),
        'weighted_hidden_fixed': select_action(weighted, [cap], limit, weight, margin),
        'preserve_fixed_no_downside': select_action(ev, [cap], limit, 0., margin),
        'preserve_fixed_individual': select_action(Individual(ev), [cap], limit, weight, margin),
        'preserve_fixed_single_view': select_action(one, [cap], limit, weight, margin)}
    plans = {p: r.action for p, r in decisions.items()}
    details = {}
    for p, result in decisions.items():
        validator = old if p == 'proposed' else weighted if p == 'weighted_hidden_fixed' else one if p == 'preserve_fixed_single_view' else ev
        actual = validator.direct_gains(result.action)
        if p != 'preserve_fixed_individual' and np.max(np.abs(actual-result.gains)) > 2e-8:
            raise ArithmeticError('Projected and direct selection scores disagree')
        if result.action.donors and result.utility <= margin:
            raise ArithmeticError('A predictive selector admitted nonpositive selection utility')
        lam = 0. if p == 'preserve_fixed_no_downside' else weight
        details[p] = {'selection_utility': float(result.utility),
                      'actual_selection_view_utility': float(downside_utility(actual, lam)),
                      'score_direct_error': float(np.max(np.abs(actual-result.gains))),
                      'selection_views': 1 if p == 'preserve_fixed_single_view' else len(visible)}
        # The individual control deliberately audits, but never vetoes or reranks, its actual mixture.
        if p == 'preserve_fixed_individual':
            details[p]['score_direct_error'] = None
        if p == ANCHOR and result.action.donors:
            d = ev.diagnostics[result.action]
            if float(np.mean(d['whole_receiver_gain'])) <= margin:
                raise ArithmeticError('Selection-view average receiver-word surplus failed')
            details[p].update(mean_hidden_gain=float(np.mean(d['hidden_gain'])),
                              mean_visible_gain=float(np.mean(d['visible_gain'])),
                              mean_whole_receiver_gain=float(np.mean(d['whole_receiver_gain'])),
                              property_scope='selection views under fixed predictor only')
    order = rank_order(lists)
    actions = [BorrowingAction(order[:size], cap) for size in range(1, min(limit, len(order))+1)]
    chosen, value = NO_BORROWING, 0.
    for action, score in zip(actions, downside_utility(ev.gains(actions), weight)):
        if preferred(action, float(score), chosen, value, 0., margin):
            chosen, value = action, float(score)
    plans['preserve_fixed_prefix'] = chosen
    if chosen.donors and value <= margin:
        raise ArithmeticError('The predictive prefix admitted a nonpositive utility')
    if np.max(np.abs(ev.gains([chosen])[0]-ev.direct_gains(chosen))) > 2e-8:
        raise ArithmeticError('Prefix projected/direct score mismatch')
    details['preserve_fixed_prefix'] = {'selection_utility': value, 'selection_views': len(visible)}
    plans['nmf_similarity'] = BorrowingAction(tuple(ranked[:limit]), cap) if ranked else NO_BORROWING
    marginals = full_pool_marginals(old, cap, weight)
    ids = np.argsort(-marginals, kind='stable')[:limit]
    plans['full_pool_loo'] = BorrowingAction(tuple(old.union[ids].tolist()), cap) if len(ids) else NO_BORROWING
    _, alpha, _ = transfer_action(base, plans[ANCHOR], full_available, donors)
    plans['matched_preserve_fixed'] = matched_action(plans[ANCHOR].donors, alpha, ranked, full_available)
    records = {}
    for p, action in plans.items():
        mixed, mass, available = transfer_action(base, action, full_available, donors)
        tv = check_transfer(base, mixed, mass, cap)
        records[p] = {'donors': list(action.donors), 'offered_beta': float(action.beta),
                      'actual_beta': float(mass), 'available_donors': list(available),
                      'total_variation': float(tv), **details.get(p, {})}
    a, b = records[ANCHOR], records['matched_preserve_fixed']
    if abs(a['actual_beta']-b['actual_beta']) > 1e-12 or len(a['available_donors']) != len(b['available_donors']):
        raise ArithmeticError('Matched control differs in realized mass or contributing donor count')
    return records

def select_bank(collection, bank, references, protocol, mask, directory, logger):
    directory.mkdir(parents=True, exist_ok=True)
    policies = [p for p in protocol['policies'] if p != 'nmf_original']
    paths = {p: directory/p for p in policies}
    pending = [p for p in policies if not complete(paths[p])]
    if not pending:
        return paths
    started = time.perf_counter()
    n, v = collection.counts.shape
    donors, base = normalized_counts(collection.counts), normalized_counts(bank.base)
    projections = [r.projection(donors) for r in references]
    embeddings = np.load(collection.directory/'embeddings.npy', allow_pickle=False)
    queries = np.load(bank.directory/'full_receiver_embeddings.npy', allow_pickle=False)
    matrices = {p: np.empty((n, v), dtype=np.float64) for p in pending}
    records = {p: [] for p in pending}
    views, tick = protocol['views'], time.monotonic()
    identity = canonical_hash({'collection': collection.summary['identity_sha256'],
                              'K': references[0].k, 'mask': int(mask), 'protocol': protocol})
    for i in range(n):
        def compute():
            if not bank.eligible[i] or not np.any(bank.candidates[i] >= 0):
                return {p: {'donors': [], 'offered_beta': 0., 'actual_beta': 0.,
                            'available_donors': [], 'total_variation': 0.} for p in policies}
            union = np.unique(bank.candidates[i][bank.candidates[i] >= 0])
            score = embeddings[union]@queries[i]
            ranked = union[np.argsort(-score, kind='stable')].tolist()
            fold = int(collection.folds[i])
            block = slice(i*views, (i+1)*views)
            return choose(references[fold], bank.visible[block].toarray(), bank.targets[block].toarray(),
                          bank.candidates[i], donors, collection.counts[i].toarray(), projections[fold],
                          ranked, bank.full_candidates[i], protocol)
        plans, reused = checkpoint(directory/'receivers'/('%06d.json' % i), {'bank': identity, 'row': i}, compute)
        if set(plans) != set(policies):
            raise ValueError('Receiver checkpoint omits a declared control')
        for p in pending:
            rec = plans[p]
            action = BorrowingAction(tuple(rec['donors']), rec['offered_beta'])
            lists = bank.candidates[i, :1] if p == 'preserve_fixed_single_view' else bank.candidates[i]
            if not set(action.donors).issubset(set(int(j) for j in lists.ravel() if j >= 0)) or len(action.donors) > protocol['max_donors']:
                raise ValueError('Receiver checkpoint violates donor eligibility')
            mixed, mass, available = transfer_action(base[i], action, bank.full_candidates[i], donors)
            check_transfer(base[i], mixed, mass, protocol['cap'])
            if abs(mass-rec['actual_beta']) > 1e-12 or list(available) != rec['available_donors']:
                raise ValueError('Checkpoint transfer cannot be reconstructed')
            matrices[p][i] = mixed
            records[p].append(dict(rec, document_index=i, source_row=int(collection.source_rows[i]),
                                   eligible=bool(bank.eligible[i]), fold=int(collection.folds[i])))
        a, b = plans[ANCHOR], plans['matched_preserve_fixed']
        if abs(a['actual_beta']-b['actual_beta']) > 1e-12 or len(a['available_donors']) != len(b['available_donors']):
            raise ArithmeticError('Checkpoint matched mass/count mismatch')
        if i == 0 or i+1 == n or time.monotonic()-tick > 15:
            logger.info('Selection | %s mask=%d K=%d | %d/%d', collection.summary['name'], mask, references[0].k, i+1, n)
            tick = time.monotonic()
    for p in pending:
        target = paths[p]
        target.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(target/'transferred.npz', sparse.csr_matrix(matrices[p]))
        write_json(target/'actions.json', records[p])
        write_json(target/'summary.json', {'policy': p, 'posts': n, 'mask_base': int(mask),
                   'mean_actual_beta': float(np.mean([r['actual_beta'] for r in records[p]])),
                   'positive_borrowing_fraction': float(np.mean([r['actual_beta'] > 0 for r in records[p]])),
                   'mean_contributing_donors': float(np.mean([len(r['available_donors']) for r in records[p]])),
                   'shared_selection_seconds': time.perf_counter()-started,
                   'timing_scope': 'combined ten-policy receiver loop, not per-policy runtime',
                   'matched_mass_count_verified': p == 'matched_preserve_fixed'})
        save_receipt(target)
    return paths
