"""Receiver-preserving, fixed-cap PEBR development candidate.

This is the same kernel as preserve_forward_fixed_cap in the focused study.
The caller supplies a frozen fold-excluded predictor and permitted donors.
"""
from __future__ import annotations
import numpy as np
from anchored_scoring import ReceiverAnchoredEvaluator
from transport_topics.proposed.search import select_action
from transport_topics.proposed.transfer import normalized_counts,transfer_action,check_transfer


def select_and_transfer(reference,visible,targets,candidate_lists,donors,
                        receiver_counts,full_available,donor_projection=None,
                        cap=1/3,max_donors=3,downside_weight=1.,margin=1e-8):
    counts=receiver_counts.toarray() if hasattr(receiver_counts,'toarray') else np.asarray(receiver_counts)
    ev=ReceiverAnchoredEvaluator(reference,visible,targets,candidate_lists,donors,
        counts,donor_projection,128)
    decision=select_action(ev,[cap],max_donors,downside_weight,margin)
    direct=ev.direct_gains(decision.action)
    if np.max(np.abs(direct-decision.gains))>2e-8:
        raise ArithmeticError('Direct and projected complete-action scores differ')
    base=normalized_counts(counts).reshape(-1)
    mixed,alpha,available=transfer_action(base,decision.action,full_available,donors)
    variation=check_transfer(base,mixed,alpha,cap)
    if decision.action.donors and decision.utility<=margin:
        raise ArithmeticError('A nonpositive borrowing action was admitted')
    d=ev.diagnostics.get(decision.action)
    if d is not None and np.mean(d['whole_receiver_gain'])<=margin:
        raise ArithmeticError('The selection-view receiver-word surplus failed')
    return mixed,{'donors':list(decision.action.donors),'offered_beta':decision.action.beta,
        'actual_beta':alpha,'available_donors':list(available),
        'selection_utility':float(decision.utility),'total_variation':variation,
        'score_direct_max_error':float(np.max(np.abs(direct-decision.gains))),
        'selection_views':ev.views,'offered_weight_search':False,
        'property_scope':'tested masked selection views; no final-coherence or unseen-word guarantee'}
