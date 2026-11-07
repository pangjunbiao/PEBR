"""Save every action before opening protected targets; use the frozen Step 5 rule."""
import time
import numpy as np
from pebr_revision_step3.io import canonical_hash, checkpoint, complete, read_json, save_receipt, write_json
from pebr_fresh_validation.selection import choose
from pebr_fresh_validation.protocol import ANCHOR
from transport_topics.proposed.actions import BorrowingAction
from transport_topics.proposed.transfer import normalized_counts, transfer_action, check_transfer

def select_actions(collection, bank, references, protocol, outer_seed, directory, logger):
    policies = [p for p in protocol['policies'] if p != 'nmf_original']
    if complete(directory):
        return {p: read_json(directory/p/'actions.json') for p in policies}
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    donors = normalized_counts(collection.counts)
    base = normalized_counts(bank.base)
    projections = [r.projection(donors) for r in references]
    embeddings = np.load(collection.directory/'embeddings.npy', allow_pickle=False)
    queries = np.load(bank.directory/'full_receiver_embeddings.npy', allow_pickle=False)
    n, m = collection.counts.shape[0], protocol['views']
    values = {p: [] for p in policies}
    identity = canonical_hash({'collection': collection.summary['identity_sha256'],
                              'topics': references[0].k, 'outer_seed': int(outer_seed), 'protocol': protocol})
    tick = time.monotonic()
    for i in range(n):
        def compute():
            if not bank.eligible[i] or not np.any(bank.candidates[i] >= 0):
                return {p: {'donors': [], 'offered_beta': 0., 'actual_beta': 0.,
                            'available_donors': [], 'total_variation': 0.} for p in policies}
            ids = np.unique(bank.candidates[i][bank.candidates[i] >= 0])
            scores = embeddings[ids]@queries[i]
            ranked = ids[np.argsort(-scores, kind='stable')].tolist()
            f = int(collection.folds[i])
            block = slice(i*m, (i+1)*m)
            # Receiver counts are OUTER-VISIBLE; complete original receiver counts are never supplied.
            return choose(references[f], bank.visible[block].toarray(), bank.targets[block].toarray(),
                          bank.candidates[i], donors, bank.base[i].toarray(), projections[f],
                          ranked, bank.full_candidates[i], protocol)
        plans, reused = checkpoint(directory/'receivers'/('%06d.json'%i), {'configuration': identity, 'row': i}, compute)
        if set(plans) != set(policies):
            raise ValueError('Protected-selection checkpoint omits a declared policy')
        for p in policies:
            rec = plans[p]
            action = BorrowingAction(tuple(rec['donors']), rec['offered_beta'])
            lists = bank.candidates[i, :1] if p == 'preserve_fixed_single_view' else bank.candidates[i]
            if not set(action.donors).issubset(set(int(j) for j in lists.ravel() if j >= 0)) or len(action.donors) > protocol['max_donors']:
                raise ArithmeticError('Protected action violates the permitted donor set')
            mixed, mass, available = transfer_action(base[i], action, bank.full_candidates[i], donors)
            check_transfer(base[i], mixed, mass, protocol['cap'])
            if abs(mass-rec['actual_beta']) > 1e-12 or list(available) != rec['available_donors']:
                raise ArithmeticError('Protected transfer mass/availability cannot be reconstructed')
            values[p].append(dict(rec, document_index=i, source_row=int(collection.source_rows[i]),
                                  fold=int(collection.folds[i]), eligible=bool(bank.eligible[i])))
        a, b = plans[ANCHOR], plans['matched_preserve_fixed']
        if abs(a['actual_beta']-b['actual_beta']) > 1e-12 or len(a['available_donors']) != len(b['available_donors']):
            raise ArithmeticError('Protected matched control differs in realized mass/donor count')
        if i == 0 or i+1 == n or time.monotonic()-tick > 15:
            logger.info('Protected selection | %s outer=%d K=%d | %d/%d',
                        collection.summary['name'], outer_seed, references[0].k, i+1, n)
            tick = time.monotonic()
    for p in policies:
        write_json(directory/p/'actions.json', values[p])
        write_json(directory/p/'summary.json', {'policy': p, 'posts': n,
                   'mean_mass_all_posts': float(np.mean([r['actual_beta'] for r in values[p]])),
                   'positive_transfer_fraction_all_posts': float(np.mean([r['actual_beta'] > 0 for r in values[p]])),
                   'protected_targets_supplied_to_selector': False,
                   'matched_mass_count_verified': p == 'matched_preserve_fixed'})
    write_json(directory/'selection_summary.json', {'all_actions_saved_before_protected_scoring': True,
               'seconds_all_ten_policies': time.perf_counter()-started,
               'timing_scope': 'Combined ten-policy loop, not standalone runtime per policy'})
    save_receipt(directory)
    return values
