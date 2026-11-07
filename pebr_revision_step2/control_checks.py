"""Run synthetic checks of new controls against the frozen audited source."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import sys
import unittest


def run(source):
    source = Path(source).resolve()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(source))
    import numpy as np
    from transport_topics.proposed.actions import BorrowingAction, NO_BORROWING
    from transport_topics.proposed.scoring import ActionEvaluator
    from transport_topics.proposed.reference import PredictiveReference
    from transport_topics.proposed.search import select_action
    from pebr_revision_step2.policies import reciprocal_rank_order, select_predictive_variant, SingletonAverageEvaluator

    class TableEvaluator:
        def __init__(self, views=1, union=(0, 1)):
            self.views, self.union = views, np.asarray(union, dtype=int)
            self.table, self.calls = {}, []

        def set(self, donors, beta, values):
            self.table[(tuple(sorted(donors)), round(beta, 12))] = np.asarray(values, dtype=float)
            return self

        def gains(self, actions):
            out = []
            for action in actions:
                self.calls.append(action)
                value = np.zeros(self.views) if not action.donors else self.table.get(
                    (action.donors, round(action.beta, 12)), -np.ones(self.views))
                out.append(value)
            return np.asarray(out).reshape(len(actions), self.views)

    class ControlChecks(unittest.TestCase):
        def test_fixed_cap_never_offers_lower_positive_budget(self):
            e = TableEvaluator().set((0,), 1/3, [.1])
            r = select_predictive_variant("fixed_cap_pebr", e, [[0, 1]], (1/12, 1/6, 1/3))
            self.assertEqual(r.action.beta, 1/3)
            self.assertTrue(all(a.beta == 1/3 for a in e.calls if a.donors))

        def test_fixed_cap_can_choose_no_borrowing(self):
            e = TableEvaluator()
            r = select_predictive_variant("fixed_cap_pebr", e, [[0, 1]], (1/6, 1/3))
            self.assertEqual(r.action, NO_BORROWING)

        def test_weight_search_and_fixed_cap_can_differ(self):
            e = TableEvaluator().set((0,), 1/6, [.5]).set((0,), 1/3, [-.1]).set((1,), 1/3, [.2])
            a = select_predictive_variant("proposed", e, [[0, 1]], (1/6, 1/3))
            b = select_predictive_variant("fixed_cap_pebr", e, [[0, 1]], (1/6, 1/3))
            self.assertEqual(a.action, BorrowingAction((0,), 1/6))
            self.assertEqual(b.action, BorrowingAction((1,), 1/3))

        def test_prefix_ranking_uses_all_view_ranks(self):
            # Scores are 0:.75, 2:.5 and 1:(1/3+1/2)/2.
            self.assertEqual(reciprocal_rank_order([[2, 0, 1], [0, 1, -1]]), (0, 2, 1))

        def test_prefix_ranking_has_source_order_ties(self):
            self.assertEqual(reciprocal_rank_order([[1, 0], [0, 1]]), (0, 1))

        def test_missing_view_contributes_zero_rank_score(self):
            self.assertEqual(reciprocal_rank_order([[2, -1], [-1, -1], [0, 2]]), (2, 0))

        def test_duplicate_donors_rejected(self):
            with self.assertRaises(ValueError): reciprocal_rank_order([[0, 0]])

        def test_nonterminal_padding_rejected(self):
            with self.assertRaises(ValueError): reciprocal_rank_order([[-1, 0]])

        def test_invalid_negative_donor_rejected(self):
            with self.assertRaises(ValueError): reciprocal_rank_order([[-2]])

        def test_prefix_scores_complete_groups(self):
            e = TableEvaluator().set((0,), 1/3, [-.2]).set((0, 1), 1/3, [.6])
            r = select_predictive_variant("predictively_scored_similarity_prefixes", e, [[0, 1]], (1/3,))
            self.assertEqual(r.action.donors, (0, 1))
            self.assertAlmostEqual(r.selection_utility, .6)

        def test_prefix_does_not_search_arbitrary_nonprefix_sets(self):
            e = TableEvaluator(union=(0, 1, 2)).set((2,), 1/3, [10.]).set((0,), 1/3, [.1])
            r = select_predictive_variant("predictively_scored_similarity_prefixes", e, [[0, 1, 2]], (1/3,))
            self.assertEqual(r.action.donors, (0,))
            self.assertNotIn(BorrowingAction((2,), 1/3), e.calls)

        def test_prefix_all_negative_returns_zero_action(self):
            e = TableEvaluator()
            r = select_predictive_variant("predictively_scored_similarity_prefixes", e, [[0, 1]], (1/3,))
            self.assertEqual(r.action, NO_BORROWING)
            self.assertEqual(r.selection_utility, 0.)

        def test_prefix_empty_union_returns_zero_action(self):
            e = TableEvaluator(union=())
            r = select_predictive_variant("predictively_scored_similarity_prefixes", e, [[-1, -1]], (1/3,))
            self.assertEqual(r.action, NO_BORROWING)
            self.assertEqual(r.proposals, 0)

        def test_prefix_rejects_different_candidate_universe(self):
            with self.assertRaises(ValueError):
                select_predictive_variant("predictively_scored_similarity_prefixes", TableEvaluator(), [[0]], (1/3,))

        def test_prefix_penalizes_losses(self):
            e = TableEvaluator(views=2).set((0,), 1/3, [.3, -.25])
            r = select_predictive_variant("predictively_scored_similarity_prefixes", e, [[0, 1], [0, 1]], (1/3,))
            self.assertEqual(r.action, NO_BORROWING)

        def test_no_penalty_is_only_zero_lambda_change(self):
            e = TableEvaluator(views=2).set((0,), 1/3, [.3, -.25])
            r = select_predictive_variant("no_downside", e, [[0, 1], [0, 1]], (1/3,))
            self.assertEqual(r.action.donors, (0,))
            self.assertGreater(r.selection_utility, 0.)

        def test_individual_proxy_averages_same_budget_singletons(self):
            e = TableEvaluator(views=2).set((0,), 1/3, [1., -.1]).set((1,), 1/3, [-.1, 1.])
            proxy = SingletonAverageEvaluator(e)
            np.testing.assert_allclose(proxy.gains([BorrowingAction((0, 1), 1/3)]), [[.45, .45]])

        def test_individual_control_does_not_veto_negative_actual_mixture(self):
            e = TableEvaluator(views=2).set((0,), 1/3, [1., -.1]).set((1,), 1/3, [-.1, 1.]).set((0, 1), 1/3, [-.5, -.5])
            r = select_predictive_variant("individual_donor", e, [[0, 1], [0, 1]], (1/3,))
            self.assertEqual(r.action.donors, (0, 1))
            self.assertGreater(r.selection_utility, 0.)
            self.assertLess(r.actual_mixture_utility, 0.)

        def test_single_view_rejects_full_view_evaluator(self):
            with self.assertRaises(ValueError):
                select_predictive_variant("single_view", TableEvaluator(views=2), [[0, 1], [0, 1]], (1/3,))

        def test_single_view_accepts_one_view(self):
            e = TableEvaluator().set((0,), 1/3, [.1])
            r = select_predictive_variant("single_view", e, [[0, 1]], (1/3,))
            self.assertEqual(r.action.donors, (0,))

        def test_cap_violation_rejected(self):
            with self.assertRaises(ValueError):
                select_predictive_variant("proposed", TableEvaluator(), [[0, 1]], (.5,), cap=1/3)

        def test_unknown_variant_rejected(self):
            with self.assertRaises(ValueError):
                select_predictive_variant("unknown", TableEvaluator(), [[0, 1]], (1/3,))

        def test_standard_pebr_matches_audited_search(self):
            e = TableEvaluator().set((0,), 1/6, [.5]).set((1,), 1/3, [.2])
            expected = select_action(e, (1/6, 1/3))
            actual = select_predictive_variant("proposed", e, [[0, 1]], (1/6, 1/3))
            self.assertEqual(expected.action, actual.action)
            self.assertEqual(expected.utility, actual.selection_utility)

        def test_real_word_predictor_works_with_new_controls(self):
            reference = PredictiveReference(np.eye(2), np.ones(2), np.ones(2))
            visible = np.array([[.2, .8], [.3, .7]])
            targets = np.array([[1., 0.], [1., 0.]])
            donors = np.array([[.8, .2], [.9, .1]])
            before = visible.copy(), targets.copy(), donors.copy()
            lists = [[0, 1], [1, 0]]
            e = ActionEvaluator(reference, visible, targets, lists, donors)
            for name in ("proposed", "fixed_cap_pebr", "predictively_scored_similarity_prefixes", "no_downside", "individual_donor"):
                r = select_predictive_variant(name, e, lists, (1/6, 1/3))
                self.assertLessEqual(r.action.beta, 1/3)
                self.assertTrue(np.isfinite(r.actual_mixture_gains).all())
            for value, old in zip((visible, targets, donors), before): np.testing.assert_array_equal(value, old)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ControlChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    payload = {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
               "skipped": len(result.skipped), "passed": result.wasSuccessful() and result.testsRun > 0 and not result.skipped,
               "synthetic_only": True, "corpus_results_computed": False,
               "frozen_core_source_root": str(source),
               "policy_module_sha256": hashlib.sha256(Path(__file__).with_name("policies.py").read_bytes()).hexdigest()}
    Path("control_tests.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(run(sys.argv[1]))
    except Exception as error:
        Path("control_tests.json").write_text(json.dumps({"tests_run": 0, "passed": False,
                                "error": type(error).__name__ + ": " + str(error)}, indent=2))
        raise
