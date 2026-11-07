import unittest
import numpy as np
from sklearn.preprocessing import normalize

from transport_topics.proposed.actions import BorrowingAction, NO_BORROWING
from transport_topics.proposed.transfer import (check_transfer, mix_views,
                                                normalized_counts, transfer_action)


class TransferTests(unittest.TestCase):
    def test_missing_donor_share_returns_to_receiver(self):
        p = np.array([.2, .8])
        donors = np.array([[.8, .2], [.6, .4]])
        action = BorrowingAction((0, 1), 1/3)
        actual, alpha, used = transfer_action(p, action, [0], donors)
        np.testing.assert_allclose(actual, 5/6 * p + 1/6 * donors[0])
        self.assertEqual(alpha, 1/6)
        self.assertEqual(used, (0,))

    def test_no_borrowing_and_all_unavailable_are_exact_fallbacks(self):
        p = np.array([.2, .8]); donors = np.array([[.8, .2]])
        for action in (NO_BORROWING, BorrowingAction((0,), 1/3)):
            actual, alpha, _ = transfer_action(p, action, [], donors)
            np.testing.assert_array_equal(actual, p)
            self.assertEqual(alpha, 0)

    def test_thousand_randomized_mass_and_total_variation_checks(self):
        rng = np.random.default_rng(2710)
        for _ in range(1000):
            p = normalized_counts(rng.random(15))
            donors = normalized_counts(rng.random((3, 15)))
            action = BorrowingAction((0, 1, 2), float(rng.uniform(0.001, 1/3)))
            available = np.flatnonzero(rng.random(3) > .5)
            actual, alpha, _ = transfer_action(p, action, available, donors)
            self.assertLessEqual(check_transfer(p, actual, alpha, 1/3), alpha + 2e-10)

    def test_additive_counts_give_same_normalized_tfidf(self):
        p = np.array([.2, .5, .3]); donor = np.array([.7, .1, .2]); beta = 1/3
        mixed = (1-beta) * p + beta * donor
        additive = p + beta/(1-beta) * donor
        idf = np.array([1., 3., 2.])
        np.testing.assert_allclose(normalize([mixed*idf]), normalize([additive*idf]))

    def test_zero_receiver_no_borrowing_remains_zero(self):
        actual, alpha, _ = transfer_action(np.zeros(3), NO_BORROWING, [], np.zeros((1, 3)))
        self.assertEqual(check_transfer(np.zeros(3), actual, alpha, 1/3), 0)

    def test_actions_reject_repeated_donors_and_inconsistent_zero_weight(self):
        for args in (((1, 1), .2), ((1,), 0), ((), .2)):
            with self.assertRaises(ValueError): BorrowingAction(*args)
