"""Explicit conventional NMF settings shared by predictors and later models."""
from __future__ import annotations

import time
import warnings
import numpy as np
from scipy import sparse
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning


def fit_nmf(matrix, topics, seed=0, tolerance=1e-4, max_iterations=2000):
    # Small corpora permit dense float64 fits, matching the original pilot.
    x = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not all(x.shape) or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('NMF requires a finite, nonnegative, nonempty matrix')
    if topics < 1 or not np.any(x):
        raise ValueError('NMF requires a positive topic count and nonzero fitting data')
    estimator = NMF(n_components=topics, init='random', solver='cd', beta_loss='frobenius',
                    random_state=seed, tol=tolerance, max_iter=max_iterations,
                    alpha_W=0.0, alpha_H=0.0, l1_ratio=0.0, shuffle=False)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        coefficients = estimator.fit_transform(x)
    dictionary = np.asarray(estimator.components_, dtype=np.float64)
    audit = {'topics': int(topics), 'seed': int(seed), 'iterations': int(estimator.n_iter_),
             'converged': not any(issubclass(w.category, ConvergenceWarning) for w in caught),
             'warnings': [{'type': w.category.__name__, 'message': str(w.message)} for w in caught],
             'seconds': time.perf_counter() - started, 'solver': 'cd',
             'loss': 'frobenius', 'tolerance': float(tolerance),
             'max_iterations': int(max_iterations), 'precision': 'float64',
             'reconstruction_error': float(estimator.reconstruction_err_)}
    return coefficients, dictionary, audit
