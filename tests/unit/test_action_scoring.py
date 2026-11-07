import unittest
import numpy as np

from transport_topics.proposed.actions import BorrowingAction, NO_BORROWING
from transport_topics.proposed.reference import PredictiveReference
from transport_topics.proposed.scoring import ActionEvaluator, downside_utility
from transport_topics.proposed.transfer import normalized_counts


class ActionScoringTests(unittest.TestCase):
    def fixture(self, batch_size=128):
        rng = np.random.default_rng(18)
        reference = PredictiveReference(rng.gamma(.4, 1, (8, 43)),
                                        1+rng.random(43), 1+rng.random(43))
        visible = normalized_counts(rng.random((7, 43)))
        targets = rng.integers(0, 3, (7, 43)).astype(float)
        donors = normalized_counts(rng.random((5, 43)))
        lists = [[j for j in range(5) if rng.random() > .4] for _ in range(7)]
        return ActionEvaluator(reference, visible, targets, lists, donors, batch_size=batch_size)

    def test_projected_action_score_matches_exact_transferred_words(self):
        evaluator = self.fixture()
        actions = [BorrowingAction((0,), 1/12), BorrowingAction((1, 2), 1/4),
                   BorrowingAction((0, 2, 4), 1/3), NO_BORROWING]
        actual = evaluator.gains(actions)
        for action, row in zip(actions, actual):
            np.testing.assert_allclose(row, evaluator.direct_gains(action), atol=2e-9)

    def test_proposal_batch_size_and_cache_do_not_change_scores(self):
        a, b = self.fixture(1), self.fixture(128)
        actions = [BorrowingAction((j,), beta) for j in range(5) for beta in (1/12, 1/3)]
        first = a.gains(actions)
        np.testing.assert_allclose(first, b.gains(actions), atol=2e-9)
        np.testing.assert_array_equal(a.gains(actions), first)
        self.assertEqual(a.action_evaluations, len(actions))

    def test_missing_donor_view_remains_in_objective_with_zero_gain(self):
        reference = PredictiveReference(np.eye(2), np.ones(2), np.ones(2))
        evaluator = ActionEvaluator(reference, np.array([[.2, .8], [.2, .8]]),
                                    np.array([[1., 0.], [1., 0.]]), [[0], []],
                                    np.array([[.8, .2]]))
        gain = evaluator.gains([BorrowingAction((0,), 1/3)])[0]
        self.assertGreater(gain[0], 0)
        self.assertEqual(gain[1], 0)
        self.assertEqual(downside_utility(gain), gain[0]/2)

    def test_downside_penalty_and_switch_have_declared_arithmetic(self):
        gains = np.array([.9, -.2, 0., 0.])
        self.assertAlmostEqual(float(downside_utility(gains, 1)), .125)
        self.assertAlmostEqual(float(downside_utility(gains, 0)), .175)

    def test_direct_word_space_fallback_remains_available_in_scoring(self):
        reference = PredictiveReference([[1., 0., 1.], [0., 1., 1.]],
                                        np.ones(3), np.ones(3), active_iterations=0)
        evaluator = ActionEvaluator(reference, [[1., 0., 0.]], [[0., 1., 0.]],
                                    [[0]], [[0., 1., 0.]])
        action = BorrowingAction((0,), 1/3)
        np.testing.assert_allclose(evaluator.gains([action])[0], evaluator.direct_gains(action), atol=2e-9)
        self.assertGreater(reference.solver.direct_fallback_rows, 0)
