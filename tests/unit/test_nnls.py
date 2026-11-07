import unittest
import numpy as np
from scipy.optimize import lsq_linear, nnls

from transport_topics.numerics.nnls import NonnegativeProjector


class NNLSTests(unittest.TestCase):
    def test_batch_reconstructions_match_independent_scipy_nnls(self):
        rng = np.random.default_rng(2710)
        for topics in (2, 8, 12, 16):
            h = rng.gamma(0.5, 1, (topics, 43))
            x = rng.gamma(0.3, 1, (40, 43))
            projector = NonnegativeProjector(h)
            actual = projector.solve(x)
            expected = np.array([nnls(h.T, row, maxiter=100 * topics)[0] for row in x])
            np.testing.assert_allclose(actual @ h, expected @ h, atol=2e-9, rtol=2e-9)
            self.assertTrue((actual >= 0).all())

    def test_duplicate_topic_rows_are_supported(self):
        h = np.array([[1., 2., 0.], [1., 2., 0.], [0., 0., 1.]])
        x = np.array([[2., 3., 1.], [0., 1., 2.]])
        projector = NonnegativeProjector(h)
        self.assertEqual(projector.rank, 2)
        actual = projector.solve(x)
        expected = np.array([lsq_linear(h.T, row, bounds=(0, np.inf), method='bvls', tol=1e-12).x for row in x])
        np.testing.assert_allclose(actual @ h, expected @ h, atol=2e-10)

    def test_direct_fallback_solves_original_dictionary_objective(self):
        h = np.array([[1., 0., 1.], [0., 1., 1.]])
        x = np.array([[1., 0., 0.]])
        projector = NonnegativeProjector(h, active_iterations=0)
        actual = projector.solve(x)
        np.testing.assert_allclose(actual, [[0.5, 0]], atol=2e-10)
        self.assertEqual(projector.direct_fallback_rows, 1)

    def test_fallback_requires_matching_word_space_input(self):
        h = np.array([[1., 0., 1.], [0., 1., 1.]])
        projector = NonnegativeProjector(h, active_iterations=0)
        with self.assertRaisesRegex(ArithmeticError, 'word-space'):
            projector.solve_projection([[1., 0.]])
        with self.assertRaisesRegex(ArithmeticError, 'do not match'):
            projector.solve_projection([[1., 0.]], lambda rows: np.zeros((len(rows), 3)))

    def test_zero_and_dependent_dictionary_rows_are_valid(self):
        projector = NonnegativeProjector(np.zeros((3, 4)))
        np.testing.assert_array_equal(projector.solve(np.ones((2, 4))), np.zeros((2, 3)))


if __name__ == '__main__':
    unittest.main()
