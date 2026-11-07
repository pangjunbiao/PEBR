"""Infer original-post coefficients and scale-invariant topic contributions."""
import numpy as np
from ..numerics.nnls import NonnegativeProjector


def contribution_features(coefficients, dictionary):
    w, h = np.asarray(coefficients, dtype=np.float64), np.asarray(dictionary, dtype=np.float64)
    if w.ndim != 2 or h.ndim != 2 or w.shape[1] != h.shape[0]:
        raise ValueError('Dictionary and original-post memberships have different topic dimensions')
    mass = w * h.sum(axis=1)[None, :]
    denominator = mass.sum(axis=1, keepdims=True)
    return np.divide(mass, denominator, out=np.zeros_like(mass), where=denominator > 0)


def infer_original_memberships(original_tfidf, dictionary):
    solver = NonnegativeProjector(dictionary)
    memberships = solver.solve(original_tfidf)
    return memberships, contribution_features(memberships, dictionary), solver.diagnostics()
