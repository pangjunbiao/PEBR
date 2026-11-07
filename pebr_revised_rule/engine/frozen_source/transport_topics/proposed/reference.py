"""Fold-excluded frozen predictors; fitting accepts counts and folds only."""
from __future__ import annotations

from importlib import metadata
from pathlib import Path
import json
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

from ..config import canonical_hash
from ..schemas import number
from ..data.preparation import check_receipt, complete, save_array, frozen_json
from ..runtime.artifacts import file_hash, write_json
from ..representations.tfidf import idf_weights
from ..numerics.nmf import fit_nmf
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


def fit_reference(counts, folds, excluded_fold, topics, settings):
    train = np.flatnonzero(np.asarray(folds) != excluded_fold)
    if not len(train):
        raise ValueError('A predictive reference has no fitting posts outside its receiving fold')
    x = counts[train].toarray() if sparse.issparse(counts) else np.asarray(counts)[train]
    x = np.asarray(x, dtype=np.float64)
    weights = idf_weights(sparse.csr_matrix(x))
    fitting = normalize(x * weights, norm='l2')
    _, dictionary, fit = fit_nmf(fitting, topics, settings['seed'],
                                number(settings['tolerance'], 'tolerance'), settings['max_iterations'])
    prior = x.sum(axis=0) + number(settings['unigram_smoothing'], 'unigram_smoothing')
    prior /= prior.sum()
    reference = PredictiveReference(dictionary, weights, prior,
                                    number(settings['smoothing'], 'smoothing'))
    audit = {**fit, 'excluded_fold': int(excluded_fold), 'training_posts': len(train),
             'receiver_fold_excluded': bool(np.all(np.asarray(folds)[train] != excluded_fold)),
             'idf_and_prior_use_fitting_posts_only': True,
             'dictionary_rank': reference.solver.rank}
    return reference, train, audit


def reference_bank(root, condition, counts, folds, topics, settings, logger):
    spec = {'schema': 1, 'condition': condition['identity_sha256'], 'topics': topics,
            'settings': settings,
            'versions': {name: metadata.version(name) for name in ('numpy', 'scipy', 'scikit-learn')},
            'fitting_code': file_hash(Path(__file__).with_name('reference.py')),
            'nmf_code': file_hash(Path(__file__).parents[1] / 'numerics/nmf.py')}
    references, audits = [], []
    for fold in range(settings['folds']):
        key = canonical_hash(dict(spec, excluded_fold=fold))
        directory = root / 'cache/predictors' / (condition['name'] + '_' + key[:16])
        directory.mkdir(parents=True, exist_ok=True)
        if check_receipt(directory):
            reference = PredictiveReference(
                np.load(directory / 'dictionary.npy', allow_pickle=False),
                np.load(directory / 'idf.npy', allow_pickle=False),
                np.load(directory / 'prior.npy', allow_pickle=False),
                number(settings['smoothing'], 'smoothing'))
            train = np.load(directory / 'training_indices.npy', allow_pickle=False)
            audit = json.loads((directory / 'fit.json').read_text(encoding='utf-8'))
            if not np.array_equal(train, np.flatnonzero(folds != fold)):
                raise ValueError('Cached reference uses different fitting folds')
            logger.info('Reusing reference | %s K=%d fold=%d.', condition['name'], topics, fold)
        else:
            reference, train, audit = fit_reference(counts, folds, fold, topics, settings)
            for name, value in (('dictionary', reference.H), ('idf', reference.idf),
                                ('prior', reference.prior), ('training_indices', train)):
                save_array(directory / (name + '.npy'), value)
            frozen_json(directory / 'identity.json', dict(spec, excluded_fold=fold))
            write_json(directory / 'fit.json', audit, overwrite=True)
            complete(directory)
            logger.info('Reference fitted | %s K=%d fold=%d posts=%d iterations=%d converged=%s.',
                        condition['name'], topics, fold, len(train), audit['iterations'], audit['converged'])
        references.append(reference)
        audits.append(dict(audit, cache_key=key, path=directory.relative_to(root).as_posix()))
    return references, audits
