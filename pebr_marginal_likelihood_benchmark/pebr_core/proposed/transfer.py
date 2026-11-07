"""One mixing operation for masked scoring and final evidence transfer."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from .actions import BorrowingAction


def normalized_counts(counts):
    x = counts.toarray() if sparse.issparse(counts) else np.asarray(counts, dtype=np.float64)
    mass = x.sum(axis=-1, keepdims=True)
    return np.divide(x, mass, out=np.zeros_like(x, dtype=np.float64), where=mass > 0)


def mixture_weights(action, availability):
    av = np.atleast_2d(np.asarray(availability, dtype=bool))
    if av.shape[1] != len(action.donors):
        raise ValueError('Availability columns must match the complete selected donor set')
    if not action.donors:
        return np.zeros(len(av)), np.zeros_like(av, dtype=np.float64)
    weights = (action.beta / len(action.donors)) * av
    return weights.sum(axis=1), weights


def mix_views(receiver, action, availability, donor_rows):
    u = np.atleast_2d(np.asarray(receiver, dtype=np.float64))
    donors = np.asarray(donor_rows, dtype=np.float64)
    alpha, weights = mixture_weights(action, availability)
    if len(alpha) != len(u) or donors.shape != (len(action.donors), u.shape[1]):
        raise ValueError('Mixture input shapes differ')
    return (1.0 - alpha[:, None]) * u + weights @ donors, alpha


def transfer_action(receiver, action, available, all_donors):
    u = np.asarray(receiver, dtype=np.float64)
    available = set(int(j) for j in available if j >= 0)
    flags = np.array([[j in available for j in action.donors]], dtype=bool)
    result, alpha = mix_views(u, action, flags,
                             np.asarray(all_donors)[list(action.donors)])
    return result[0], float(alpha[0]), tuple(j for j in action.donors if j in available)


def check_transfer(receiver, transferred, actual_weight, cap, tolerance=2e-10):
    p, r = np.asarray(receiver), np.asarray(transferred)
    if not np.isfinite(r).all() or np.min(r) < -tolerance:
        raise ArithmeticError('Transferred evidence is not finite and nonnegative')
    expected_mass = 1.0 if p.sum() > 0 else 0.0
    if abs(r.sum() - expected_mass) > tolerance:
        raise ArithmeticError('Transferred evidence does not preserve normalized count mass')
    tv = float(0.5 * np.abs(r - p).sum())
    if not -tolerance <= actual_weight <= cap + tolerance or tv > actual_weight + tolerance:
        raise ArithmeticError('The normalized count-space borrowing bound failed')
    return tv
