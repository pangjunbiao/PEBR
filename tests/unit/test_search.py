import unittest
import numpy as np

from transport_topics.proposed.actions import BorrowingAction, NO_BORROWING
from transport_topics.proposed.reference import PredictiveReference
from transport_topics.proposed.scoring import ActionEvaluator
from transport_topics.proposed.search import preferred, select_action


class SearchTests(unittest.TestCase):
    def evaluator(self, donor_rows):
        return ActionEvaluator(PredictiveReference(np.eye(2), np.ones(2), np.ones(2)),
                               [[.2, .8]], [[1., 0.]], [list(range(len(donor_rows)))], donor_rows)

    def test_helpful_redundant_donors_need_not_be_unique_to_be_accepted(self):
        evaluator = self.evaluator([[.9, .1], [.9, .1]])
        result = select_action(evaluator, (1/12, 1/3))
        self.assertEqual(result.action, BorrowingAction((0,), 1/3))
        self.assertGreater(result.utility, 0)

    def test_harmful_donors_return_no_borrowing(self):
        result = select_action(self.evaluator([[.1, .9], [.01, .99]]), (1/12, 1/3))
        self.assertEqual(result.action, NO_BORROWING)
        self.assertEqual(result.utility, 0)

    def test_ties_prefer_smaller_budget_then_source_order(self):
        current = NO_BORROWING
        best = BorrowingAction((3,), 1/3)
        self.assertTrue(preferred(BorrowingAction((5,), 1/12), 1., best, 1., 0., 1e-8))
        self.assertTrue(preferred(BorrowingAction((2,), 1/3), 1., best, 1., 0., 1e-8))
        self.assertFalse(preferred(BorrowingAction((0,), 1/12), 0., current, 0., 0., 1e-8))

    def test_near_tie_cannot_admit_a_nonimproving_extension(self):
        best = BorrowingAction((3,), .3)
        self.assertFalse(preferred(BorrowingAction((0,), .1), 1e-8,
                                   best, 1.5e-8, 0., 1e-8))

    def test_every_accepted_extension_satisfies_admission_margin(self):
        result = select_action(self.evaluator([[.8, .2], [.9, .1], [.95, .05]]),
                               (1/12, 1/6, 1/4, 1/3), max_donors=3)
        for step in result.trace: self.assertGreater(step['improvement'], 1e-8)
        self.assertLessEqual(len(result.action.donors), 3)

    def test_complete_sets_are_scored_even_with_greedy_construction(self):
        evaluator = self.evaluator([[.8, .2], [.95, .05]])
        select_action(evaluator, (1/12, 1/3))
        self.assertIn(BorrowingAction((0, 1), 1/12), evaluator.cache)
        self.assertIn(BorrowingAction((0, 1), 1/3), evaluator.cache)
