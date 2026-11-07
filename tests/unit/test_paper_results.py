import html
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from tests.report_fixture import completed_exports, refresh_manifest, revise_json
from transport_topics.reporting.common import escape, numeric, one, read_csv, write_csv
from transport_topics.reporting.provenance import method_source_check, result_readout
from transport_topics.reporting.tables import dataset_rows
from transport_topics.reporting.topic_reports import make_topic_report, _html_report, verify_receipt
from transport_topics.reporting.validation import resolve_study, validate_study
from transport_topics.runtime.artifacts import file_hash


class PublicationValidationTests(unittest.TestCase):
    def test_actual_completed_study_reconciles_all_registered_exports(self):
        with completed_exports() as root:
            bundle = validate_study(root)
        self.assertTrue(bundle['audit']['passed'])
        self.assertEqual(bundle['audit']['unique_models'], 1536)
        self.assertEqual(bundle['audit']['logical_evaluations'], 1651)
        self.assertEqual(bundle['audit']['converged_final_nmf_fits'], 1260)
        self.assertEqual(bundle['audit']['verified_implementation_tests'], 170)

    def test_failed_run_is_rejected_even_with_valid_checksums(self):
        with completed_exports() as root:
            revise_json(root, 'status.json', lambda d: d.update(state='failed'))
            with self.assertRaisesRegex(ValueError, 'completed full-study'):
                validate_study(root)

    def test_smoke_results_cannot_be_published_as_full_results(self):
        with completed_exports() as root:
            revise_json(root, 'evaluation_summary.json', lambda d: d.update(development_only=True))
            with self.assertRaisesRegex(ValueError, 'smoke or partial'):
                validate_study(root)

    def test_csv_byte_tampering_is_rejected(self):
        with completed_exports() as root:
            path = root / 'results/topic_quality_summary.csv'
            path.write_bytes(path.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'missing or changed'):
                validate_study(root)

    def test_incorrect_aggregate_is_detected_after_checksum_refresh(self):
        with completed_exports() as root:
            path = root / 'results/topic_quality_summary.csv'
            rows = read_csv(path)
            rows[0]['npmi_mean'] = '0.9'
            write_csv(path, rows)
            refresh_manifest(root, 'results/topic_quality_summary.csv')
            with self.assertRaisesRegex(ValueError, 'aggregation differs'):
                validate_study(root)

    def test_category_cohort_change_is_detected(self):
        with completed_exports() as root:
            path = root / 'results/retrieval_category_results.csv'
            rows = read_csv(path)
            rows[0]['category_queries'] = '30'
            write_csv(path, rows)
            refresh_manifest(root, 'results/retrieval_category_results.csv')
            with self.assertRaisesRegex(ValueError, 'cohort differ'):
                validate_study(root)

    def test_wrong_corpus_size_pairing_is_detected(self):
        with completed_exports() as root:
            path = root / 'results/corpus_size_deltas.csv'
            rows = read_csv(path)
            rows[0]['npmi_change_from_no_borrowing'] = '0.8'
            write_csv(path, rows)
            refresh_manifest(root, 'results/corpus_size_deltas.csv')
            with self.assertRaisesRegex(ValueError, 'paired corpus-size'):
                validate_study(root)

    def test_sampling_sizes_are_reported_as_saved_not_rounded_to_600(self):
        with completed_exports() as root:
            bundle = validate_study(root)
            rows = dataset_rows(bundle['data']['dataset_evaluation_profile'])
        self.assertEqual([r['retained_posts'] for r in rows], [619, 618, 602, 600, 602, 600])

    def test_external_sample_and_seed_uncertainty_are_kept_separate(self):
        with completed_exports() as root:
            bundle = validate_study(root)
            r = one(bundle['tables']['topic_quality_summary.csv'], dataset='external', method='proposed', K=12)
        self.assertTrue(r['npmi_sample_sd'])
        self.assertTrue(r['npmi_mean_within_sample_seed_sd'])
        self.assertEqual(r['npmi_seed_sd'], '')

    def test_result_readout_preserves_unfavorable_retrieval_comparisons(self):
        with completed_exports() as root:
            text = result_readout(validate_study(root))
        self.assertIn('0.4132', text)
        self.assertIn('lies below zero', text)
        self.assertIn('includes zero', text)

    def test_resolver_skips_failed_and_report_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, status in [('20260101T000000Z_good', {'stage': 'full_study', 'state': 'completed', 'requested_evaluation_complete': True}),
                                 ('20260102T000000Z_failed', {'stage': 'full_study', 'state': 'failed'}),
                                 ('20260103T000000Z_report', {'stage': 'reports', 'state': 'completed'})]:
                p = root / 'outputs/runs' / name
                p.mkdir(parents=True)
                (p / 'status.json').write_text(json.dumps(status), encoding='utf-8')
            self.assertEqual(resolve_study(root).name, '20260101T000000Z_good')

    def test_run_id_cannot_escape_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                resolve_study(Path(tmp), '../../outside')

    def test_scientific_source_change_is_not_hidden_by_report_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / 'src/transport_topics/proposed/search.py'
            p.parent.mkdir(parents=True)
            p.write_text('original', encoding='utf-8')
            record = {'files': {'src/transport_topics/proposed/search.py': file_hash(p)}}
            self.assertTrue(method_source_check(root, record)['all_scientific_sources_unchanged'])
            p.write_text('changed', encoding='utf-8')
            self.assertFalse(method_source_check(root, record)['all_scientific_sources_unchanged'])

    def test_fraction_configuration_and_tex_symbols_are_supported(self):
        self.assertAlmostEqual(numeric('1/3'), 1 / 3)
        self.assertEqual(escape('a_b&c%'), r'a\_b\&c\%')
        with self.assertRaises(ValueError):
            numeric('nan')

    def test_publication_inspection_contains_previews_but_no_original_topic_text(self):
        from transport_topics.runtime.artifacts import RunStore, make_inspection
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RunStore(root, stage='reports')
            store.json('status.json', {'state': 'completed'})
            for area in ('paper', 'supplement'):
                for folder in ('figures', 'tables', 'topic_reports'):
                    (root / 'outputs' / area / folder).mkdir(parents=True)
            (root / 'outputs/paper/tables/table.tex').write_text('TABLE', encoding='utf-8')
            (root / 'outputs/paper/figures/figure.png').write_bytes(b'TEST PREVIEW')
            (root / 'outputs/supplement/topic_reports/original.html').write_text('ORIGINAL TEXT', encoding='utf-8')
            store.json('publication_summary.json', {'complete': True, 'paper_directory': 'outputs/paper', 'supplement_directory': 'outputs/supplement'})
            archive = make_inspection(root, store.run_id)
            with zipfile.ZipFile(archive) as z:
                self.assertIn('publication/paper/figures/figure.png', z.namelist())
                self.assertIn('publication/paper/tables/table.tex', z.namelist())
                self.assertFalse(any('original.html' in n for n in z.namelist()))


class TopicEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.model = {'identity': 'test-only', 'condition': 'primary_full', 'method': 'proposed', 'topics': 2, 'seed': 0}
        self.posts = [{'source_row': 1, 'source_id': 'p:1', 'original_text': '<script>alert(1)</script> taxi fare', 'cleaned_text': 'taxi fare'},
                      {'source_row': 3, 'source_id': 'p:3', 'original_text': 'bus delay', 'cleaned_text': 'bus delay'}]
        self.scores = np.array([[1., 1., 0.], [0., 0., 1.]])
        self.features = np.array([[.8, .2], [.3, .7]])
        self.vocabulary = np.array(['taxi', 'fare', 'delay'])

    def report(self, **kwargs):
        return make_topic_report(self.model, self.scores, self.features, self.vocabulary, self.posts, **kwargs)

    def test_donor_words_are_not_inserted_into_original_text(self):
        actions = {1: {'available_source_rows': [3], 'actual_beta': .2}, 3: {'available_source_rows': [], 'actual_beta': 0}}
        report = self.report(actions=actions, families=np.array([0, 1]))
        first = report['topics'][0]['ranked_original_posts'][0]
        self.assertEqual(first['original_text'], self.posts[0]['original_text'])
        self.assertNotIn('delay', first['top_words_present_in_original_cleaned_text'])
        self.assertEqual(report['topics'][0]['original_posts_with_positive_association'], 2)
        self.assertFalse(report['semantic_validation_performed'])

    def test_html_original_text_is_escaped(self):
        text = _html_report(self.report())
        self.assertNotIn('<script>', text)
        self.assertIn('&lt;script&gt;', text)

    def test_zero_associations_preserve_topics_without_inventing_support(self):
        self.features[:] = 0
        report = self.report()
        self.assertEqual(len(report['topics']), 2)
        self.assertTrue(all(not t['ranked_original_posts'] for t in report['topics']))

    def test_ties_use_saved_vocabulary_and_source_order(self):
        self.features[:] = .5
        report = self.report()
        self.assertEqual(report['topics'][0]['top_words'][:2], ['taxi', 'fare'])
        self.assertEqual(report['topics'][0]['ranked_original_posts'][0]['source_row'], 1)

    def test_misaligned_or_repeated_original_posts_are_rejected(self):
        self.posts[1]['source_row'] = 1
        with self.assertRaisesRegex(ValueError, 'distinct'):
            self.report()

    def test_topic_evidence_rejects_modified_saved_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            p = directory / 'original.txt'
            p.write_text('original', encoding='utf-8')
            (directory / 'complete.json').write_text(json.dumps({'files': {'original.txt': file_hash(p)}}), encoding='utf-8')
            verify_receipt(directory)
            p.write_text('changed', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'receipt failed'):
                verify_receipt(directory)
