import unittest
import numpy as np
from scipy import sparse
from scipy.optimize import nnls

from transport_topics.proposed.reference import PredictiveReference, fit_reference


class ReferenceTests(unittest.TestCase):
    def test_predictions_match_explicit_normalized_word_space_nnls(self):
        rng = np.random.default_rng(9)
        for topics in (2, 8, 12, 16):
            h = rng.gamma(.5, 1, (topics, 51)); idf = 1 + 4*rng.random(51)
            prior = 1 + rng.random(51); prior /= prior.sum()
            x = rng.gamma(.3, 1, (30, 51))
            predictor = PredictiveReference(h, idf, prior)
            expected = []
            for row in x:
                weighted = row*idf; weighted /= np.linalg.norm(weighted)
                coefficients = nnls(h.T, weighted, maxiter=100*topics)[0]
                p = (coefficients@h)/idf; p /= p.sum()
                expected.append(.9*p + .1*prior)
            actual = predictor.predict(x)
            np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-9)
            np.testing.assert_allclose(actual.sum(axis=1), 1)
            np.testing.assert_allclose(predictor.predict(7*x), actual, atol=2e-9)

    def test_zero_reconstruction_uses_positive_prior(self):
        predictor = PredictiveReference(np.zeros((2, 4)), np.ones(4), np.ones(4))
        np.testing.assert_allclose(predictor.predict(np.ones((3, 4))), .25)

    def test_fitting_fold_exclusion_applies_to_dictionary_idf_and_prior(self):
        rng = np.random.default_rng(8)
        x = rng.poisson(2, (18, 9)).astype(float); folds = np.arange(18) % 3
        settings = {'seed': 0, 'tolerance': 1e-4, 'max_iterations': 2000,
                    'unigram_smoothing': .1, 'smoothing': .1}
        first, train, audit = fit_reference(sparse.csr_matrix(x), folds, 0, 2, settings)
        x[folds == 0] += 100
        second, other_train, _ = fit_reference(sparse.csr_matrix(x), folds, 0, 2, settings)
        np.testing.assert_array_equal(first.H, second.H)
        np.testing.assert_array_equal(first.idf, second.idf)
        np.testing.assert_array_equal(first.prior, second.prior)
        np.testing.assert_array_equal(train, other_train)
        self.assertTrue(audit['receiver_fold_excluded'])

    def test_absent_fitting_words_receive_prior_probability(self):
        x = sparse.csr_matrix([[2., 0., 1.], [1., 0., 0.], [0., 4., 1.], [1., 5., 0.]])
        settings = {'seed': 0, 'tolerance': 1e-4, 'max_iterations': 2000,
                    'unigram_smoothing': .1, 'smoothing': .1}
        reference, _, _ = fit_reference(x, np.array([1, 1, 0, 0]), 0, 1, settings)
        self.assertGreater(reference.predict([[1., 0., 0.]])[0, 1], 0)
