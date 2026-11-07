"""Read protected targets only after action selection; never select using these scores."""
import math
import statistics
import numpy as np
from scipy import sparse
from scipy.stats import rankdata
from pebr_revision_step3.io import complete, read_json, save_receipt, write_csv, write_json
from transport_topics.proposed.actions import BorrowingAction, NO_BORROWING
from transport_topics.proposed.transfer import normalized_counts, transfer_action, check_transfer

def gain_from_probabilities(probabilities, target_weights):
    p = np.asarray(probabilities, dtype=float)
    t = np.asarray(target_weights, dtype=float)
    if p.ndim != 2 or p.shape[1] != len(t) or np.any(p <= 0) or not np.isfinite(p).all() or np.any(t < 0) or t.sum() <= 0:
        raise ValueError('Invalid protected probabilities or token counts')
    scores = np.log(p)@t/t.sum()
    return scores, scores-scores[0]

def correlation(left, right, ranks=False):
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    if ranks:
        a, b = rankdata(a), rankdata(b)
    return float(np.corrcoef(a, b)[0, 1])

def summarize_posts(rows, threshold):
    if not rows:
        raise ValueError('The common protected cohort is empty')
    gains = np.asarray([r['gain_nats_per_token'] for r in rows], dtype=float)
    tokens = np.asarray([r['target_tokens'] for r in rows], dtype=float)
    admitted = [r for r in rows if r['offered_beta'] > 0]
    paired = [r for r in rows if r.get('selection_utility') is not None]
    def corr(subset, ranks=False):
        return correlation([r['selection_utility'] for r in subset],
                           [r['gain_nats_per_token'] for r in subset], ranks)
    admitted_paired = [r for r in paired if r['offered_beta'] > 0]
    return {'eligible_posts': len(rows), 'protected_target_tokens': int(tokens.sum()),
            'mean_post_gain': float(gains.mean()),
            'token_weighted_gain': float(np.average(gains, weights=tokens)),
            'median_post_gain': float(np.median(gains)),
            'post_loss_fraction': float(np.mean(gains < -threshold)),
            'post_positive_fraction': float(np.mean(gains > threshold)),
            'mean_negative_part_all_posts': float(np.mean(np.maximum(-gains, 0))),
            'p05_post_gain': float(np.quantile(gains, .05)),
            'p95_post_gain': float(np.quantile(gains, .95)),
            'minimum_post_gain': float(gains.min()),
            'mean_realized_mass_audited_posts': float(np.mean([r['actual_beta'] for r in rows])),
            'positive_transfer_fraction_audited_posts': float(np.mean([r['actual_beta'] > 0 for r in rows])),
            'selection_utility_available_posts': len(paired),
            'inner_outer_pearson_all_utility_posts': corr(paired),
            'inner_outer_spearman_all_utility_posts': corr(paired, True),
            'inner_outer_pearson_admitted_posts': corr(admitted_paired),
            'inner_outer_spearman_admitted_posts': corr(admitted_paired, True),
            'admitted_posts': len(admitted),
            'correlation_scope': 'Descriptive within a fixed bank; no p-value, independent-post or causal claim'}

def audit_configuration(corpus, k, outer_seed, collection, bank, references, actions,
                        protocol, selection_directory, directory, logger):
    if complete(directory):
        return read_json(directory/'bank_summaries.json')
    if not complete(selection_directory):
        raise ValueError('All actions must be saved before protected targets are opened')
    # This is deliberately the first read of protected counts in the outcome stage.
    outer = sparse.load_npz(bank.directory/'outer.npz').tocsr()
    if outer.multiply(bank.base).nnz:
        raise ArithmeticError('Protected targets overlap receiver inputs')
    cohort = np.flatnonzero(np.asarray(bank.eligible) & (np.asarray(outer.sum(axis=1)).ravel() > 0))
    if not len(cohort):
        raise ValueError('There are no eligible protected queries')
    policies = protocol['policies']
    if policies[0] != 'nmf_original':
        raise ValueError('No-borrowing must be the common predictive reference')
    donors, base = normalized_counts(collection.counts), normalized_counts(bank.base)
    rows = {p: [] for p in policies}
    for i in cohort:
        target = outer[int(i)].toarray().ravel()
        ids = np.flatnonzero(target)
        mixed_rows = [base[i]]
        records = [{'donors': [], 'offered_beta': 0., 'actual_beta': 0.,
                    'available_donors': [], 'selection_utility': 0.}]
        for p in policies[1:]:
            rec = actions[p][int(i)]
            if rec['document_index'] != int(i):
                raise ValueError('Action rows and protected queries are misaligned')
            action = BorrowingAction(tuple(rec['donors']), rec['offered_beta'])
            q, alpha, available = transfer_action(base[i], action, bank.full_candidates[i], donors)
            check_transfer(base[i], q, alpha, protocol['cap'])
            if abs(alpha-rec['actual_beta']) > 1e-12 or list(available) != rec['available_donors']:
                raise ArithmeticError('The audited action differs from the saved selected action')
            mixed_rows.append(q)
            records.append(rec)
        probability = references[int(collection.folds[i])].predict(np.asarray(mixed_rows), ids)
        scores, gains = gain_from_probabilities(probability, target[ids])
        if gains[0] != 0. or not np.isfinite(gains).all():
            raise ArithmeticError('Invalid no-borrowing reference or protected gain')
        for p, rec, score, gain in zip(policies, records, scores, gains):
            if rec['actual_beta'] == 0 and abs(gain) > 2e-8:
                raise ArithmeticError('Zero transfer unexpectedly changed protected prediction')
            rows[p].append({'corpus': corpus, 'K': int(k), 'outer_seed': int(outer_seed), 'policy': p,
                'document_index': int(i), 'source_row': int(collection.source_rows[i]),
                'fold': int(collection.folds[i]), 'target_tokens': int(target.sum()),
                'no_borrowing_log_score': float(scores[0]), 'policy_log_score': float(score),
                'gain_nats_per_token': float(gain), 'offered_beta': rec['offered_beta'],
                'actual_beta': rec['actual_beta'], 'contributing_donors': len(rec['available_donors']),
                'selection_utility': rec.get('selection_utility')})
    summaries = []
    for p in policies:
        if p != 'nmf_original':
            mass_all = float(np.mean([r['actual_beta'] for r in actions[p]]))
        else:
            mass_all = 0.
        summary = dict(corpus=corpus, K=int(k), outer_seed=int(outer_seed), policy=p,
                       mean_realized_mass_all_posts=mass_all,
                       **summarize_posts(rows[p], protocol['loss_threshold']))
        summaries.append(summary)
    directory.mkdir(parents=True, exist_ok=True)
    write_csv(directory/'post_outcomes.csv', [row for p in policies for row in rows[p]])
    write_json(directory/'bank_summaries.json', summaries)
    write_json(directory/'audit_identity.json', {'corpus': corpus, 'K': int(k), 'outer_seed': int(outer_seed),
        'fresh_protected_bank': int(outer_seed) in protocol['fresh_outer_target_bases'],
        'common_cohort_source_rows': collection.source_rows[cohort].tolist(),
        'protected_type_fraction': protocol['outer_type_fraction'], 'policies': policies,
        'targets_enter_retrieval_or_selection': False, 'final_models_fitted': 0,
        'scope': 'Same fixed fold-excluded predictor; protected words opened after all actions saved'})
    save_receipt(directory)
    logger.info('Protected outcomes | %s outer=%d K=%d | %d queries, %d policies', corpus, outer_seed, k, len(cohort), len(policies))
    return summaries
