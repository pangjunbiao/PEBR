"""Protected word prediction, called only after actions have been saved."""
from __future__ import annotations

import csv
import io
import numpy as np

from ..runtime.artifacts import atomic_write
from ..proposed.transfer import normalized_counts


def audit_actions(references, folds, original_source_rows, base_counts, outer_targets,
                  transferred, eligible, processed, topics, output_path):
    if base_counts.shape != outer_targets.shape or transferred.shape != base_counts.shape:
        raise ValueError('Outer audit dimensions differ')
    if base_counts.multiply(outer_targets).nnz:
        raise ValueError('Protected outer targets overlap the receiver input')
    records = []
    for i in np.flatnonzero(np.asarray(eligible) & np.asarray(processed)):
        target = outer_targets[i].toarray().ravel()
        ids = np.flatnonzero(target)
        if not len(ids):
            raise ValueError('An outer-audit receiver has no protected targets')
        reference = references[int(folds[i])]
        base = normalized_counts(base_counts[i])[0]
        borrowed = transferred[i].toarray()[0]
        base_log = float(np.average(np.log(reference.predict(base, ids)[0]), weights=target[ids]))
        action_log = float(np.average(np.log(reference.predict(borrowed, ids)[0]), weights=target[ids]))
        records.append({'source_row': int(original_source_rows[i]), 'document_index': int(i),
                        'fold': int(folds[i]), 'topics': int(topics),
                        'target_tokens': int(target.sum()), 'no_borrowing_log_score': base_log,
                        'proposed_log_score': action_log, 'gain_nats_per_token': action_log-base_log})
    stream = io.StringIO(newline='')
    fields = ['source_row', 'document_index', 'fold', 'topics', 'target_tokens',
              'no_borrowing_log_score', 'proposed_log_score', 'gain_nats_per_token']
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader(); writer.writerows(records)
    atomic_write(output_path, stream.getvalue().encode('utf-8'), overwrite=True)
    gains = np.array([row['gain_nats_per_token'] for row in records])
    tokens = np.array([row['target_tokens'] for row in records])
    return {'eligible_evaluated_posts': len(records), 'target_tokens': int(tokens.sum()),
            'mean_post_gain_nats_per_token': float(gains.mean()) if len(gains) else None,
            'token_weighted_gain_nats_per_token': float(np.average(gains, weights=tokens)) if len(gains) else None,
            'post_loss_frequency': float(np.mean(gains < -1e-10)) if len(gains) else None,
            'loss_numerical_threshold': 1e-10,
            'reference': 'same frozen predictor, same receiver without borrowing',
            'used_for_selection_or_tuning': False,
            'interpretation': 'Prediction audit only; final topic quality and baseline comparisons are pending'}
