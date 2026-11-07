"""Predictive policy controls using the audited PEBR evaluator and search.

These functions select actions; they do not fit models, retrieve text, alter
post content, or write project files. Existing NMF/sharing/LOO controls are
retained in the frozen project source for the subsequent experiment runner.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from transport_topics.proposed.actions import BorrowingAction, NO_BORROWING
from transport_topics.proposed.scoring import downside_utility
from transport_topics.proposed.search import SearchResult, preferred, select_action


def reciprocal_rank_order(candidate_lists):
    """Mean reciprocal rank across the ordered visible-view lists; absent=0."""
    lists = [list(map(int, row)) for row in candidate_lists]
    if not lists:
        return ()
    scores = {}
    for row in lists:
        if any(j < -1 for j in row):
            raise ValueError("Only -1 is an allowed padding index")
        ids = [j for j in row if j >= 0]
        if len(set(ids)) != len(ids):
            raise ValueError("A candidate list contains duplicate donors")
        seen_padding = False
        for rank, donor in enumerate(row, start=1):
            if donor < 0:
                seen_padding = True
                continue
            if seen_padding:
                raise ValueError("Candidate padding must follow the ranked donors")
            scores[donor] = scores.get(donor, 0.) + 1. / rank / len(lists)
    return tuple(sorted(scores, key=lambda j: (-scores[j], j)))


def select_similarity_prefix(evaluator, candidate_lists, budgets, max_donors=3,
                             downside_weight=1., margin=1e-8):
    grid = tuple(map(float, budgets))
    if not grid or grid != tuple(sorted(set(grid))) or not 0 < grid[0] <= grid[-1] < 1:
        raise ValueError("Require increasing positive budgets below one")
    if max_donors < 1 or downside_weight < 0 or margin <= 0:
        raise ValueError("Invalid policy settings")
    if len(candidate_lists) != evaluator.views:
        raise ValueError("Rank lists and scoring views differ")
    order = reciprocal_rank_order(candidate_lists)
    if set(order) != set(map(int, evaluator.union)):
        raise ValueError("Prefix control and evaluator do not share the candidate universe")
    proposals = [BorrowingAction(tuple(order[:size]), beta)
                 for size in range(1, min(max_donors, len(order)) + 1) for beta in grid]
    best, best_value, best_gains = NO_BORROWING, 0., np.zeros(evaluator.views)
    if proposals:
        gains = evaluator.gains(proposals)
        values = downside_utility(gains, downside_weight)
        for action, value, row in zip(proposals, values, gains):
            if preferred(action, float(value), best, best_value, 0., margin):
                best, best_value, best_gains = action, float(value), row.copy()
    trace = [] if not best.donors else [{"donors": list(best.donors), "beta": best.beta,
              "previous_utility": 0., "utility": best_value, "improvement": best_value,
              "search": "all prespecified similarity prefixes; not a forward-admission path"}]
    return SearchResult(best, best_value, best_gains, trace, len(proposals))


class SingletonAverageEvaluator:
    """Score groups by the mean of singleton gains evaluated at the same beta."""
    def __init__(self, evaluator):
        self.base = evaluator
        self.views, self.union = evaluator.views, evaluator.union

    def gains(self, actions):
        singletons = {}
        for action in actions:
            for donor in action.donors:
                item = BorrowingAction((donor,), action.beta)
                singletons[item] = None
        unique = list(singletons)
        if unique:
            for action, values in zip(unique, self.base.gains(unique)):
                singletons[action] = values
        return np.asarray([np.mean([singletons[BorrowingAction((j,), action.beta)] for j in action.donors], axis=0)
                           if action.donors else np.zeros(self.views) for action in actions]).reshape(len(actions), self.views)


@dataclass
class PredictiveDecision:
    action: BorrowingAction
    selection_utility: float
    selection_gains: np.ndarray
    actual_mixture_utility: float
    actual_mixture_gains: np.ndarray
    proposals: int
    trace: list


def select_predictive_variant(name, evaluator, candidate_lists, budgets, cap=1/3,
                              max_donors=3, downside_weight=1., margin=1e-8):
    """Dispatch only predictive variants; audit actual mixtures without vetoing controls."""
    budgets = tuple(map(float, budgets))
    if len(candidate_lists) != evaluator.views:
        raise ValueError("Rank lists and scoring views differ")
    if not 0 < cap < 1 or any(float(beta) > cap + 1e-12 for beta in budgets):
        raise ValueError("Budgets must respect the offered cap")
    if name in {"proposed", "no_downside", "single_view", "fixed_cap_pebr"}:
        if name == "single_view" and evaluator.views != 1:
            raise ValueError("Construct the evaluator from fixed view 0 and its candidate list only")
        grid = (float(cap),) if name == "fixed_cap_pebr" else budgets
        weight = 0. if name == "no_downside" else downside_weight
        result = select_action(evaluator, grid, max_donors, weight, margin)
    elif name == "predictively_scored_similarity_prefixes":
        result = select_similarity_prefix(evaluator, candidate_lists, budgets, max_donors,
                                          downside_weight, margin)
        weight = downside_weight
    elif name == "individual_donor":
        result = select_action(SingletonAverageEvaluator(evaluator), budgets, max_donors,
                               downside_weight, margin)
        weight = downside_weight
    else:
        raise ValueError("Unknown predictive policy: " + str(name))
    actual = evaluator.gains([result.action])[0]
    return PredictiveDecision(result.action, result.utility, result.gains,
                              float(downside_utility(actual, weight)), actual.copy(),
                              result.proposals, result.trace)
