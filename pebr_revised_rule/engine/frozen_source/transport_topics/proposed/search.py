"""The specified forward admission rule with deterministic numerical ties."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .actions import BorrowingAction, NO_BORROWING
from .scoring import downside_utility


@dataclass
class SearchResult:
    action: BorrowingAction
    utility: float
    gains: np.ndarray
    trace: list[dict]
    proposals: int


def preferred(candidate, candidate_value, best, best_value, current_value, margin):
    if candidate_value > best_value + margin:
        return True
    return (abs(candidate_value - best_value) <= margin
            and candidate_value > current_value + margin
            and candidate.tie_key < best.tie_key)


def select_action(evaluator, budgets, max_donors=3, downside_weight=1.0, margin=1e-8):
    grid = tuple(float(b) for b in budgets)
    if not grid or grid != tuple(sorted(set(grid))) or not 0 < grid[0] <= grid[-1] < 1:
        raise ValueError('Selection requires an increasing positive borrowing grid below one')
    if max_donors < 1 or downside_weight < 0 or margin <= 0:
        raise ValueError('Invalid admission settings')
    current = NO_BORROWING
    current_value = 0.0
    current_gains = np.zeros(evaluator.views)
    trace, proposals = [], 0
    for _ in range(max_donors):
        candidates = [BorrowingAction(current.donors + (int(j),), beta)
                      for j in evaluator.union if j not in current.donors for beta in grid]
        if not candidates:
            break
        gains = evaluator.gains(candidates)
        values = downside_utility(gains, downside_weight)
        proposals += len(candidates)
        best, best_value, best_gains = current, current_value, current_gains
        for action, value, view_gains in zip(candidates, values, gains):
            if preferred(action, float(value), best, best_value, current_value, margin):
                best, best_value, best_gains = action, float(value), view_gains.copy()
        if best_value <= current_value + margin:
            break
        trace.append({'donors': list(best.donors), 'beta': best.beta,
                      'previous_utility': current_value, 'utility': best_value,
                      'improvement': best_value - current_value})
        current, current_value, current_gains = best, best_value, best_gains
    return SearchResult(current, current_value, current_gains, trace, proposals)
