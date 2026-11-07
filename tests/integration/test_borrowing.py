from copy import deepcopy
from pathlib import Path
import json
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy import sparse

from transport_topics.config import canonical_hash, load_configuration
from transport_topics.data.prepared import PreparedCollection, SelectionBank
from transport_topics.data.preparation import save_matrix
from transport_topics.evaluation.predictive_audit import audit_actions
from transport_topics.experiments.borrowing import _row_receipt, rule_settings, run_bank
from transport_topics.proposed.reference import fit_reference

ROOT = Path(__file__).resolve().parents[2]


class BorrowingIntegrationTests(unittest.TestCase):
    def fixture(self, root, outer=False):
        shutil.copytree(ROOT / 'src', root / 'src', ignore=shutil.ignore_patterns('__pycache__'))
        config = deepcopy(load_configuration(ROOT).values)
        config['method']['views']['count'] = 3
        config['method']['candidates']['per_view'] = 3
        config['borrowing_run']['progress_seconds'] = 100
        rng = np.random.default_rng(2710)
        counts = sparse.csr_matrix(rng.integers(1, 5, (20, 12)).astype(float))
        folds = np.arange(20) % 5
        directory = root / 'data/processed/synthetic'
        scope = 'outer_audit' if outer else 'topic_learning'
        bank_dir = directory / scope; bank_dir.mkdir(parents=True)
        summary = {'identity_sha256': canonical_hash('integration-fixture'), 'name': 'synthetic',
                   'posts': 20, 'vocabulary': 12}
        collection = PreparedCollection(summary, directory, counts, folds, np.arange(20), np.arange(20))
        base = counts.toarray(); hidden_outer = np.zeros_like(base)
        if outer:
            hidden_outer[:, 0] = base[:, 0]; base[:, 0] = 0
        visible, targets = np.zeros((60, 12)), np.zeros((60, 12))
        candidates = np.full((20, 3, 3), -1, dtype=int)
        full = np.full((20, 3), -1, dtype=int)
        for i in range(20):
            ids = np.flatnonzero(folds != folds[i])[:3]
            candidates[i] = ids; full[i] = ids
            for m in range(3):
                row = i*3+m
                hidden = np.array([1+m, 6+m])
                targets[row, hidden] = base[i, hidden]
                visible[row] = base[i]; visible[row, hidden] = 0
                visible[row] /= visible[row].sum()
        bank = SelectionBank(scope, bank_dir, sparse.csr_matrix(base), sparse.csr_matrix(visible),
                             sparse.csr_matrix(targets), np.ones(20, dtype=bool), candidates, full)
        save_matrix(bank_dir / 'outer.npz', sparse.csr_matrix(hidden_outer))
        references, audits = [], []
        for fold in range(5):
            reference, _, audit = fit_reference(counts, folds, fold, 2, config['method']['predictor'])
            references.append(reference); audits.append(dict(audit, cache_key='synthetic-fold-%d' % fold))
        return config, collection, bank, references, audits

    def test_completed_action_bank_is_checked_and_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, collection, bank, references, audits = self.fixture(root)
            result = run_bank(root, collection, bank, references, audits, 2, config, Mock())
            self.assertTrue(result['complete_bank'])
            self.assertEqual(result['processed_receivers'], 20)
            with patch('transport_topics.experiments.borrowing._select_receiver', side_effect=AssertionError('must reuse')):
                other = run_bank(root, collection, bank, references, audits, 2, config, Mock())
            self.assertEqual(result, other)

    def test_interrupted_selection_resumes_at_unfinished_receivers(self):
        import transport_topics.experiments.borrowing as module
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, collection, bank, references, audits = self.fixture(root)
            original = module._select_receiver
            state = {'calls': 0}
            def interrupt(*args):
                state['calls'] += 1
                if state['calls'] == 4: raise KeyboardInterrupt()
                return original(*args)
            with patch.object(module, '_select_receiver', side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    run_bank(root, collection, bank, references, audits, 2, config, Mock())
            self.assertEqual(len(list((root/'outputs/checkpoints').rglob('*.json'))), 3)
            with patch.object(module, '_select_receiver', wraps=original) as selector:
                result = run_bank(root, collection, bank, references, audits, 2, config, Mock())
            self.assertEqual(selector.call_count, 17)
            self.assertTrue(result['complete_bank'])

    def test_disabled_borrowing_is_exact_original_representation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, collection, bank, references, audits = self.fixture(root)
            config['method']['borrowing']['enabled'] = False
            result = run_bank(root, collection, bank, references, audits, 2, config, Mock())
            self.assertEqual(result['admitted_receivers'], 0)
            self.assertEqual(result['scored_proposals'], 0)
            actual = sparse.load_npz(root/result['path']/'transferred.npz').toarray()
            expected = collection.counts.toarray(); expected /= expected.sum(axis=1, keepdims=True)
            np.testing.assert_array_equal(actual, expected)

    def test_downside_and_budget_configuration_switches_are_effective(self):
        config = deepcopy(load_configuration(ROOT).values)
        config['method']['selection']['downside']['enabled'] = False
        config['method']['transfer']['max_weight'] = '1/2'
        settings = rule_settings(config)
        self.assertEqual(settings['downside_weight'], 0)
        self.assertEqual(settings['budgets'], [1/8, 1/4, 3/8, 1/2])

    def test_outer_evaluation_runs_after_action_artifacts_are_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, collection, bank, references, audits = self.fixture(root, outer=True)
            def inspect(*args):
                output = args[-1].parent
                self.assertTrue((output/'selected_donors.npy').is_file())
                self.assertTrue((output/'transferred.npz').is_file())
                return audit_actions(*args)
            with patch('transport_topics.evaluation.predictive_audit.audit_actions', side_effect=inspect):
                result = run_bank(root, collection, bank, references, audits, 2, config, Mock())
            self.assertEqual(result['outer_prediction']['eligible_evaluated_posts'], 20)
            self.assertFalse(result['outer_prediction']['used_for_selection_or_tuning'])

    def test_changing_outer_targets_cannot_change_saved_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, collection, bank, references, audits = self.fixture(root, outer=True)
            result = run_bank(root, collection, bank, references, audits, 2, config, Mock())
            output = root/result['path']
            selected_before = (output/'selected_donors.npy').read_bytes()
            outer = sparse.load_npz(bank.directory/'outer.npz').tocsr()*7
            transferred = sparse.load_npz(output/'transferred.npz').tocsr()
            audit_actions(references, collection.folds, collection.source_rows, bank.base, outer,
                          transferred, bank.eligible, np.ones(20, dtype=bool), 2, root/'changed_outer.csv')
            self.assertEqual((output/'selected_donors.npy').read_bytes(), selected_before)

    def test_corrupt_receiver_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'row.json'; identity = {'row': 0}
            _row_receipt(path, identity, lambda: {'donors': [2]})
            envelope = json.loads(path.read_text()); envelope['record']['donors'] = [7]
            path.write_text(json.dumps(envelope))
            with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                _row_receipt(path, identity, lambda: self.fail('corrupt checkpoint must not be used'))

    def test_development_limit_is_not_reported_as_full_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, collection, bank, references, audits = self.fixture(root)
            config['borrowing_run']['receiver_limit'] = 3
            result = run_bank(root, collection, bank, references, audits, 2, config, Mock())
            self.assertTrue(result['development_only'])
            self.assertFalse(result['complete_bank'])
            self.assertEqual(result['processed_eligible_receivers'], 3)
