"""Exploratory search repair; no claim about final topic coherence.

Includes forward PEBR and all fixed reciprocal-rank prefixes as incumbents,
then checks complete add/delete/swap actions and every allowed budget.
Every change strictly improves the same utility. No benchmark metric or label
is accessible to this selector. This file is not installed in the project.
"""
from __future__ import annotations

import numpy as np
from transport_topics.proposed.actions import BorrowingAction, NO_BORROWING
from transport_topics.proposed.scoring import downside_utility
from transport_topics.proposed.search import SearchResult, select_action


def rank_order(candidate_lists):
    scores = {}
    for row in candidate_lists:
        ids = [int(j) for j in row if j >= 0]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate candidates in one view')
        for rank, donor in enumerate(ids, 1):
            scores[donor] = scores.get(donor, 0.) + 1. / rank
    return tuple(sorted(scores, key=lambda j: (-scores[j], j)))


def select_refined_action(evaluator, candidate_lists, budgets, max_donors=3,
                          downside_weight=1., margin=1e-8):
    grid = tuple(float(b) for b in budgets)
    # The audited forward selector validates the grid and admission settings.
    forward = select_action(evaluator, grid, max_donors, downside_weight, margin)
    order = rank_order(candidate_lists)
    if len(candidate_lists) != evaluator.views or set(order) != set(map(int, evaluator.union)):
        raise ValueError('All candidates and views must be shared')
    count = forward.proposals
    prefixes = [BorrowingAction(order[:size], beta)
                for size in range(1, min(max_donors, len(order)) + 1)
                for beta in grid]
    seeds = sorted(set([NO_BORROWING, forward.action] + prefixes), key=lambda a: a.tie_key)

    def best_of(actions, incumbent, value, gains):
        nonlocal count
        if not actions:
            return incumbent, value, gains
        values_by_view = evaluator.gains(actions)
        scores = downside_utility(values_by_view, downside_weight)
        count += len(actions)
        for action, score, row in zip(actions, scores, values_by_view):
            score = float(score)
            # Exact maxima retain a deterministically ordered tied incumbent.
            if score > value:
                incumbent, value, gains = action, score, row.copy()
        return incumbent, value, gains

    current, value, gains = best_of(seeds, NO_BORROWING, 0., np.zeros(evaluator.views))
    if value <= margin:
        current, value, gains = NO_BORROWING, 0., np.zeros(evaluator.views)
    trace = []
    if current.donors:
        trace.append({'operation': 'forward-or-prefix initialization',
                      'donors': list(current.donors), 'beta': current.beta,
                      'previous_utility': 0., 'utility': value})
    universe = set(map(int, evaluator.union))
    while current.donors:
        selected = set(current.donors)
        donor_sets = {tuple(current.donors)}
        for j in selected:
            donor_sets.add(tuple(sorted(selected - {j})))
            for other in universe - selected:
                donor_sets.add(tuple(sorted((selected - {j}) | {other})))
        if len(selected) < max_donors:
            for other in universe - selected:
                donor_sets.add(tuple(sorted(selected | {other})))
        neighbours = {BorrowingAction(group, beta) for group in donor_sets if group for beta in grid}
        neighbours.discard(current)
        proposals = sorted(neighbours, key=lambda a: a.tie_key)
        candidate, candidate_value, candidate_gains = best_of(proposals, current, value, gains)
        if candidate_value <= value + margin:
            break
        trace.append({'operation': 'complete-action add/delete/swap/budget check',
                      'donors': list(candidate.donors), 'beta': candidate.beta,
                      'previous_utility': value, 'utility': candidate_value})
        current, value, gains = candidate, candidate_value, candidate_gains
    if value + margin < forward.utility:
        raise ArithmeticError('Refinement degraded its forward incumbent')
    return SearchResult(current, value, gains, trace, count)
