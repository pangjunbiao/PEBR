"""Checked batch NNLS with a direct-dictionary, rank-tolerant fallback.

The fast path solves the unregularized Gram-system quadratic. A fallback
always receives the actual word-space target and solves the same objective
with nonnegative bounds; no Cholesky factorization or ridge is required.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear


class NonnegativeProjector:
    def __init__(self, dictionary, active_iterations=24, kkt_tolerance=5e-9):
        self.H = np.asarray(dictionary, dtype=np.float64)
        if self.H.ndim != 2 or not all(self.H.shape):
            raise ValueError('The dictionary must be a nonempty topic-by-word matrix')
        if not np.isfinite(self.H).all() or (self.H < 0).any():
            raise ValueError('The dictionary must be finite and nonnegative')
        if active_iterations < 0 or kkt_tolerance <= 0:
            raise ValueError('Invalid NNLS solver settings')
        self.k, self.v = self.H.shape
        self.gram = self.H @ self.H.T
        self.inverse = np.linalg.pinv(self.gram)
        self.rank = int(np.linalg.matrix_rank(self.H))
        self.active_iterations = active_iterations
        self.kkt_tolerance = float(kkt_tolerance)
        self._supports = {}
        self.rows_solved = 0
        self.direct_fallback_rows = 0
        self.max_kkt_residual = 0.0

    def _residual(self, weights, projection):
        gradient = weights @ self.gram - projection
        scale = np.maximum(1.0, np.max(np.abs(projection), axis=1))
        weight_scale = np.maximum(1.0, np.max(np.abs(weights), axis=1))
        primal = np.maximum(-np.min(weights, axis=1), 0.0)
        dual = np.maximum(-np.min(gradient, axis=1), 0.0) / scale
        complementarity = np.max(np.abs(weights * gradient), axis=1) / (scale * weight_scale)
        return np.maximum.reduce((primal, dual, complementarity))

    def solve_projection(self, projection, input_factory=None):
        """Infer coefficients from X H^T; materialize X only for fallback rows."""
        b = np.atleast_2d(np.asarray(projection, dtype=np.float64))
        if b.shape[1] != self.k or not np.isfinite(b).all():
            raise ValueError('Invalid projected NNLS target')
        if not len(b):
            return np.empty((0, self.k), dtype=np.float64)
        self.rows_solved += len(b)
        w = np.maximum(b @ self.inverse, 0.0)
        support = w > 0
        pending = np.arange(len(b))
        tolerance = 2e-11 * np.maximum(1.0, np.max(np.abs(b), axis=1))
        bits = (1 << np.arange(self.k, dtype=np.int64)) if self.k < 63 else None
        for _ in range(self.active_iterations):
            if not len(pending):
                break
            if bits is not None:
                codes = (support[pending] * bits).sum(axis=1)
                _, groups = np.unique(codes, return_inverse=True)
            else:
                _, groups = np.unique(support[pending], axis=0, return_inverse=True)
            for group in np.unique(groups):
                rows = pending[groups == group]
                ids = np.flatnonzero(support[rows[0]])
                key = tuple(ids.tolist())
                if key not in self._supports:
                    self._supports[key] = (np.linalg.pinv(self.gram[np.ix_(ids, ids)])
                                           if len(ids) else np.empty((0, 0)))
                w[rows] = 0.0
                if len(ids):
                    w[np.ix_(rows, ids)] = b[np.ix_(rows, ids)] @ self._supports[key]
            gradient = w[pending] @ self.gram - b[pending]
            primal = w[pending] < -tolerance[pending, None]
            dual = (~support[pending]) & (gradient < -tolerance[pending, None])
            bad = primal.any(axis=1) | dual.any(axis=1)
            changed = support[pending[bad]].copy()
            changed[primal[bad]] = False
            changed[dual[bad]] = True
            support[pending[bad]] = changed
            pending = pending[bad]
        w = np.maximum(w, 0.0)
        bad_rows = np.flatnonzero(self._residual(w, b) > self.kkt_tolerance)
        if len(bad_rows):
            if input_factory is None:
                raise ArithmeticError('Direct NNLS fallback requires the actual word-space inputs')
            targets = input_factory(bad_rows)
            targets = targets.toarray() if sparse.issparse(targets) else np.asarray(targets)
            if targets.shape != (len(bad_rows), self.v) or not np.isfinite(targets).all():
                raise ValueError('Invalid word-space fallback inputs')
            if not np.allclose(targets @ self.H.T, b[bad_rows], rtol=2e-9, atol=2e-11):
                raise ArithmeticError('Fallback inputs do not match the scored projection')
            for row, target in zip(bad_rows, targets):
                result = lsq_linear(self.H.T, target, bounds=(0.0, np.inf),
                                    method='bvls', lsq_solver='exact', tol=1e-12,
                                    max_iter=max(100, 100 * self.k))
                if not result.success or not np.isfinite(result.x).all():
                    raise ArithmeticError('Direct bounded least-squares fallback did not converge')
                w[row] = np.maximum(result.x, 0.0)
            self.direct_fallback_rows += len(bad_rows)
        residual = self._residual(w, b)
        self.max_kkt_residual = max(self.max_kkt_residual, float(residual.max()))
        if (residual > self.kkt_tolerance).any():
            raise ArithmeticError('NNLS optimality check failed after direct fallback')
        return w

    def solve(self, targets):
        x = targets.tocsr().astype(np.float64) if sparse.issparse(targets) else np.atleast_2d(np.asarray(targets, dtype=np.float64))
        if x.shape[1] != self.v or not np.isfinite(x.data if sparse.issparse(x) else x).all():
            raise ValueError('Invalid NNLS word-space targets')
        return self.solve_projection(np.asarray(x @ self.H.T), lambda rows: x[rows])

    def diagnostics(self):
        return {'topic_rows': self.k, 'dictionary_rank': self.rank,
                'rows_solved': self.rows_solved,
                'direct_dictionary_fallback_rows': self.direct_fallback_rows,
                'max_normalized_kkt_residual': self.max_kkt_residual,
                'kkt_tolerance': self.kkt_tolerance, 'active_iterations': self.active_iterations,
                'fallback': 'scipy.optimize.lsq_linear, bvls/exact, tol=1e-12, nonnegative bounds',
                'regularization_added': False, 'precision': 'float64'}
