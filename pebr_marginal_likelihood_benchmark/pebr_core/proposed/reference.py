"""Fixed word predictor from the PEBR reference implementation."""
import numpy as np
from scipy import sparse
from ..numerics.nnls import NonnegativeProjector


class PredictiveReference:
    def __init__(self, dictionary, idf, prior, smoothing=0.1, **solver_settings):
        self.solver = NonnegativeProjector(dictionary, **solver_settings)
        self.H = self.solver.H
        self.idf = np.asarray(idf, dtype=np.float64)
        self.prior = np.asarray(prior, dtype=np.float64)
        self.smoothing = float(smoothing)
        if self.idf.shape != (self.solver.v,) or self.prior.shape != self.idf.shape:
            raise ValueError('Predictor vocabulary dimensions differ')
        if not np.isfinite(self.idf).all() or (self.idf <= 0).any():
            raise ValueError('Predictor IDF weights must be positive')
        if not np.isfinite(self.prior).all() or (self.prior <= 0).any():
            raise ValueError('The predictive prior must be strictly positive')
        if not 0 < self.smoothing < 1:
            raise ValueError('Predictive smoothing must be in (0,1)')
        self.prior = self.prior / self.prior.sum()
        self.word_map = self.H / self.idf[None, :]
        self.word_mass = self.word_map.sum(axis=1)
        self.k = self.solver.k

    def projection(self, rows):
        weighted = rows.multiply(self.idf) if sparse.issparse(rows) else np.atleast_2d(np.asarray(rows)) * self.idf
        return np.asarray(weighted @ self.H.T)

    def from_projection(self, projection, word_ids=None, input_factory=None):
        coefficients = self.solver.solve_projection(projection, input_factory)
        word_map = self.word_map if word_ids is None else self.word_map[:, word_ids]
        prior = self.prior if word_ids is None else self.prior[word_ids]
        reconstructed = coefficients @ word_map
        mass = coefficients @ self.word_mass
        probability = np.divide(reconstructed, mass[:, None],
                                out=np.zeros_like(reconstructed), where=mass[:, None] > 0)
        probability = (1.0 - self.smoothing) * probability + self.smoothing * prior
        probability[mass <= 0] = prior
        return probability

    def predict(self, rows, word_ids=None):
        x = rows.tocsr() if sparse.issparse(rows) else np.atleast_2d(np.asarray(rows, dtype=np.float64))
        weighted = x.multiply(self.idf).tocsr() if sparse.issparse(x) else x * self.idf
        return self.from_projection(np.asarray(weighted @ self.H.T), word_ids,
                                    input_factory=lambda ids: weighted[ids])
