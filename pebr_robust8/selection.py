"""Exact four-policy subset of the frozen Step 5 selector; validate by replay."""
import time
import numpy as np
from .design import ANCHOR
from .common import write

def choose(reference, visible, targets, lists, donors, original, projection, ranked, full_available, protocol):
    from pebr_fresh_validation.anchored_scoring import ReceiverAnchoredEvaluator
    from pebr_fresh_validation.selection import rank_order
    from transport_topics.proposed.actions import BorrowingAction, NO_BORROWING
    from transport_topics.proposed.search import select_action, preferred
    from transport_topics.proposed.scoring import downside_utility
    from transport_topics.proposed.transfer import normalized_counts, transfer_action, check_transfer
    from transport_topics.experiments.sharing_controls import matched_action
    cap, limit, weight, margin = (protocol[k] for k in ('cap', 'max_donors', 'downside_weight', 'admission_margin'))
    ev = ReceiverAnchoredEvaluator(reference, visible, targets, lists, donors, original, projection, 128)
    selected = select_action(ev, [cap], limit, weight, margin)
    actual = ev.direct_gains(selected.action)
    if np.max(np.abs(actual-selected.gains)) > 2e-8:
        raise ArithmeticError('Revised selection projected/direct gain differs')
    details = {ANCHOR: {'selection_utility': float(selected.utility),
                        'actual_selection_view_utility': float(downside_utility(actual, weight))}}
    if selected.action.donors:
        d = ev.diagnostics[selected.action]
        if selected.utility <= margin or float(np.mean(d['whole_receiver_gain'])) <= margin:
            raise ArithmeticError('Revised selection-view admission property failed')
        details[ANCHOR].update(mean_hidden_gain=float(np.mean(d['hidden_gain'])),
                              mean_visible_gain=float(np.mean(d['visible_gain'])),
                              mean_whole_receiver_gain=float(np.mean(d['whole_receiver_gain'])))
    order = rank_order(lists)
    prefix, value = NO_BORROWING, 0.
    prefixes = [BorrowingAction(order[:size], cap) for size in range(1, min(limit,len(order))+1)]
    for action, score in zip(prefixes, downside_utility(ev.gains(prefixes), weight)):
        if preferred(action, float(score), prefix, value, 0., margin):
            prefix, value = action, float(score)
    if np.max(np.abs(ev.gains([prefix])[0]-ev.direct_gains(prefix))) > 2e-8:
        raise ArithmeticError('Prefix projected/direct gain differs')
    details['preserve_fixed_prefix'] = {'selection_utility': float(value)}
    base = normalized_counts(original).reshape(-1)
    _, mass, _ = transfer_action(base, selected.action, full_available, donors)
    plans = {ANCHOR: selected.action, 'preserve_fixed_prefix': prefix,
             'nmf_similarity': BorrowingAction(tuple(ranked[:limit]), cap) if ranked else NO_BORROWING,
             'matched_preserve_fixed': matched_action(selected.action.donors, mass, ranked, full_available)}
    result = {}
    for policy, action in plans.items():
        q, alpha, available = transfer_action(base, action, full_available, donors)
        tv = check_transfer(base, q, alpha, cap)
        result[policy] = {'donors': list(action.donors), 'offered_beta': float(action.beta), 'actual_beta': float(alpha),
                          'available_donors': list(available), 'total_variation': float(tv), **details.get(policy,{})}
    a, b = result[ANCHOR], result['matched_preserve_fixed']
    if abs(a['actual_beta']-b['actual_beta']) > 1e-12 or len(a['available_donors']) != len(b['available_donors']):
        raise ArithmeticError('Matched per-receiver realized mass/count differs')
    return result

def select(collection, bank, references, protocol, directory, logger):
    from scipy import sparse
    from pebr_revision_step3.io import complete, checkpoint, save_receipt, canonical_hash, read_json
    from transport_topics.proposed.actions import BorrowingAction
    from transport_topics.proposed.transfer import normalized_counts, transfer_action, check_transfer
    policies = [p for p in protocol['policies'] if p != 'nmf_original']
    if complete(directory):
        return {p: directory/p for p in policies}
    directory.mkdir(parents=True, exist_ok=True)
    n, v = collection.counts.shape
    donors, base = normalized_counts(collection.counts), normalized_counts(bank.base)
    projections = [r.projection(donors) for r in references]
    embeddings = np.load(collection.directory/'embeddings.npy', allow_pickle=False)
    queries = np.load(bank.directory/'full_receiver_embeddings.npy', allow_pickle=False)
    matrices, records = {p: np.empty((n,v), dtype=float) for p in policies}, {p: [] for p in policies}
    identity = canonical_hash({'collection': collection.summary['identity_sha256'], 'K': references[0].k,
                               'protocol': protocol, 'reference_files_bound_by_parent_case': True})
    tick = time.monotonic()
    for i in range(n):
        def compute():
            if not bank.eligible[i] or not np.any(bank.candidates[i] >= 0):
                return {p: {'donors': [], 'offered_beta': 0., 'actual_beta': 0., 'available_donors': [], 'total_variation': 0.} for p in policies}
            union = np.unique(bank.candidates[i][bank.candidates[i] >= 0])
            ranked = union[np.argsort(-(embeddings[union]@queries[i]), kind='stable')].tolist()
            fold, block = int(collection.folds[i]), slice(i*protocol['views'], (i+1)*protocol['views'])
            return choose(references[fold], bank.visible[block].toarray(), bank.targets[block].toarray(),
                bank.candidates[i], donors, collection.counts[i].toarray(), projections[fold], ranked, bank.full_candidates[i], protocol)
        plans, _ = checkpoint(directory/'receivers'/('%06d.json'%i), {'case': identity,'row':i}, compute)
        if set(plans) != set(policies):
            raise ValueError('A resumed receiver omits a declared policy')
        for policy in policies:
            rec = plans[policy]
            action = BorrowingAction(tuple(rec['donors']), rec['offered_beta'])
            union = set(int(j) for j in bank.candidates[i].ravel() if j >= 0)
            if len(action.donors) > protocol['max_donors'] or not set(action.donors).issubset(union):
                raise ValueError('Selected donor violates the permitted candidate set')
            q, mass, available = transfer_action(base[i], action, bank.full_candidates[i], donors)
            check_transfer(base[i], q, mass, protocol['cap'])
            if abs(mass-rec['actual_beta']) > 1e-12 or list(available) != rec['available_donors']:
                raise ValueError('Saved action does not reproduce transfer')
            matrices[policy][i] = q
            records[policy].append(dict(rec,document_index=i,source_row=int(collection.source_rows[i]),fold=int(collection.folds[i])))
        a,b=plans[ANCHOR],plans['matched_preserve_fixed']
        if abs(a['actual_beta']-b['actual_beta']) > 1e-12 or len(a['available_donors']) != len(b['available_donors']):
            raise ArithmeticError('Resumed action bank no longer matches quantity')
        if i == 0 or i+1 == n or time.monotonic()-tick > 15:
            logger.info('Selection | %s K%d | %d/%d',collection.summary['name'],references[0].k,i+1,n)
            tick = time.monotonic()
    for policy in policies:
        p = directory/policy
        p.mkdir(parents=True,exist_ok=True)
        sparse.save_npz(p/'transferred.npz',sparse.csr_matrix(matrices[policy]))
        write(p/'actions.json',records[policy])
        write(p/'summary.json',{'posts':n,'policy':policy,'positive_transfer_fraction':float(np.mean([r['actual_beta']>0 for r in records[policy]])),
                               'mean_mass':float(np.mean([r['actual_beta'] for r in records[policy]]))})
    save_receipt(directory)
    return {p:directory/p for p in policies}
