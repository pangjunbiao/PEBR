"""Predictive gain of complete actions, including unavailable-donor views."""
from __future__ import annotations

import numpy as np
from .actions import BorrowingAction
from .transfer import mix_views, mixture_weights


def downside_utility(gains, weight=1.0):
    values = np.asarray(gains, dtype=np.float64)
    if not np.isfinite(values).all() or values.shape[-1] == 0 or weight < 0:
        raise ValueError('Utility requires finite gains, nonempty views and a nonnegative penalty')
    return values.mean(axis=-1) - weight * np.maximum(-values, 0).mean(axis=-1)


class ActionEvaluator:
    """This API receives inner targets only; protected outer targets are absent."""
    def __init__(self, reference, visible, targets, candidate_lists, donors,
                 donor_projection=None, batch_size=128):
        self.reference = reference
        self.visible = np.asarray(visible, dtype=np.float64)
        self.targets = np.asarray(targets, dtype=np.float64)
        if self.visible.ndim != 2 or self.targets.shape != self.visible.shape:
            raise ValueError('Visible rows and inner targets must have matching view-by-word shapes')
        if not np.isfinite(self.visible).all() or not np.isfinite(self.targets).all():
            raise ValueError('Masked evidence must be finite')
        if (self.visible < 0).any() or (self.targets < 0).any():
            raise ValueError('Masked evidence must be nonnegative')
        if not np.allclose(self.visible.sum(axis=1), 1.0, atol=1e-12):
            raise ValueError('Every selection view needs a normalized nonzero receiver')
        self.views = len(self.visible)
        if len(candidate_lists) != self.views or batch_size < 1:
            raise ValueError('Invalid candidate lists or proposal batch size')
        self.donors = np.asarray(donors, dtype=np.float64)
        self.lists = [set(int(j) for j in row if j >= 0) for row in candidate_lists]
        self.union = np.array(sorted(set().union(*self.lists)), dtype=np.int64)
        if len(self.union) and self.union[-1] >= len(self.donors):
            raise ValueError('A candidate donor index is outside the collection')
        self.lookup = {int(j): position for position, j in enumerate(self.union)}
        self.availability = np.array([[j in row for j in self.union] for row in self.lists], dtype=bool)
        self.base_projection = reference.projection(self.visible)
        self.donor_projection = (reference.projection(self.donors) if donor_projection is None
                                 else np.asarray(donor_projection, dtype=np.float64))
        self.word_ids = np.flatnonzero(self.targets.sum(axis=0))
        self.target_weights = self.targets[:, self.word_ids]
        self.target_mass = self.target_weights.sum(axis=1)
        if not len(self.word_ids) or (self.target_mass <= 0).any():
            raise ValueError('Every selection view needs nonempty hidden evidence')
        baseline = reference.predict(self.visible, self.word_ids)
        self.baseline_score = (self.target_weights * np.log(baseline)).sum(axis=1) / self.target_mass
        self.batch_size = batch_size
        self.cache = {}
        self.action_evaluations = 0

    def _selected_availability(self, action):
        try:
            positions = [self.lookup[j] for j in action.donors]
        except KeyError as error:
            raise ValueError('The action contains a donor outside the candidate union') from error
        return self.availability[:, positions]

    def _fallback_inputs(self, chunk, flat_rows):
        output = np.empty((len(flat_rows), self.visible.shape[1]), dtype=np.float64)
        actions, masks = np.divmod(flat_rows, self.views)
        for index in np.unique(actions):
            positions = np.flatnonzero(actions == index)
            action = chunk[int(index)]
            view_ids = masks[positions]
            availability = self._selected_availability(action)[view_ids]
            mixed, _ = mix_views(self.visible[view_ids], action, availability,
                                  self.donors[list(action.donors)])
            output[positions] = mixed * self.reference.idf
        return output

    def gains(self, actions):
        output = np.empty((len(actions), self.views), dtype=np.float64)
        pending = []
        for position, action in enumerate(actions):
            if not action.donors:
                output[position] = 0.0
            elif action in self.cache:
                output[position] = self.cache[action]
            else:
                pending.append((position, action))
        for start in range(0, len(pending), self.batch_size):
            entries = pending[start:start + self.batch_size]
            chunk = [action for _, action in entries]
            projection = np.empty((len(chunk), self.views, self.reference.k), dtype=np.float64)
            inactive = np.empty((len(chunk), self.views), dtype=bool)
            for position, action in enumerate(chunk):
                availability = self._selected_availability(action)
                alpha, weights = mixture_weights(action, availability)
                projection[position] = ((1.0 - alpha[:, None]) * self.base_projection
                                        + weights @ self.donor_projection[list(action.donors)])
                inactive[position] = alpha == 0
            probability = self.reference.from_projection(
                projection.reshape(-1, self.reference.k), self.word_ids,
                input_factory=lambda ids: self._fallback_inputs(chunk, ids))
            probability = probability.reshape(len(chunk), self.views, -1)
            score = (np.log(probability) * self.target_weights[None]).sum(axis=2) / self.target_mass
            gain = score - self.baseline_score[None]
            gain[inactive] = 0.0
            if not np.isfinite(gain).all():
                raise ArithmeticError('A predictive action gain is not finite')
            for (destination, action), row in zip(entries, gain):
                self.cache[action] = row.copy()
                output[destination] = row
            self.action_evaluations += len(chunk)
        return output

    def direct_gains(self, action):
        if not action.donors:
            return np.zeros(self.views)
        availability = self._selected_availability(action)
        mixed, alpha = mix_views(self.visible, action, availability,
                                  self.donors[list(action.donors)])
        probability = self.reference.predict(mixed, self.word_ids)
        score = (self.target_weights * np.log(probability)).sum(axis=1) / self.target_mass
        gain = score - self.baseline_score
        gain[alpha == 0] = 0.0
        return gain
