from pathlib import Path
import tempfile
import unittest
import zipfile

from tests.report_fixture import completed_exports
from transport_topics.reporting.common import read_csv
from transport_topics.reporting.provenance import portable_bundle
from transport_topics.reporting.run import render_numeric_results
from transport_topics.reporting.validation import validate_study


class PaperExportTests(unittest.TestCase):
    def test_topic_export_reads_saved_originals_and_full_borrowing_provenance(self):
        import json
        import numpy as np
        from transport_topics.reporting.topic_reports import export_topic_reports
        from transport_topics.runtime.artifacts import file_hash
        class Log:
            def info(self, *args):
                pass
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, model, control = root / 'data/test', root / 'models/test', root / 'controls/test'
            for p in (data, model, control):
                p.mkdir(parents=True)
            posts = [{'source_row': 1, 'source_id': 'p:1', 'original_text': 'Taxi fare', 'cleaned_text': 'taxi fare'},
                     {'source_row': 3, 'source_id': 'p:3', 'original_text': 'Bus delay', 'cleaned_text': 'bus delay'}]
            (data / 'posts.json').write_text(json.dumps(posts), encoding='utf-8')
            np.save(data / 'source_rows.npy', [1, 3])
            np.save(data / 'families.npy', [0, 1])
            np.save(model / 'source_rows.npy', [1, 3])
            np.save(model / 'topic_scores.npy', [[2., 1., 0.], [0., 0., 2.]])
            np.save(model / 'retrieval_features.npy', [[.8, .2], [.3, .7]])
            np.save(model / 'vocabulary.npy', ['taxi', 'fare', 'delay'])
            actions = [{'source_row': 1, 'variants': {'proposed': {'selected_source_rows': [3], 'available_source_rows': [3], 'offered_beta': .2, 'actual_beta': .2}}},
                       {'source_row': 3, 'variants': {'proposed': {'selected_source_rows': [], 'available_source_rows': [], 'offered_beta': 0., 'actual_beta': 0.}}}]
            (control / 'actions.json').write_text(json.dumps(actions), encoding='utf-8')
            for directory in (data, model, control):
                receipt = {'files': {p.name: file_hash(p) for p in directory.iterdir()}}
                (directory / 'complete.json').write_text(json.dumps(receipt), encoding='utf-8')
            bundle = {'config': {'topics': {'main_count': 2, 'top_words': 2}}, 'data': {
                'dataset_evaluation_profile': {'preparation': {'conditions': [{'name': 'primary_full', 'path': 'data/test'}]}},
                'models_index': {'models': [{'topic_model': True, 'topics': 2, 'path': 'models/test', 'condition': 'primary_full', 'method': 'proposed', 'seed': 0, 'identity': 'test-only'}]},
                'study_controls_index': {'banks': [{'job': {'study': 'main'}, 'condition': 'primary_full', 'topics': 2, 'scope': 'topic_learning', 'path': 'controls/test'}]}}}
            result = export_topic_reports(root, bundle, root / 'export', Log())
            self.assertEqual(result['model_reports'], 1)
            self.assertEqual(result['topic_reports'], 2)
            report = json.loads((root / 'export/primary_full_proposed_K2_seed0.json').read_text(encoding='utf-8'))
            support = report['topics'][0]['ranked_original_posts'][0]
            self.assertEqual(support['original_text'], 'Taxi fare')
            self.assertEqual(support['borrowing_provenance']['available_source_ids'], ['p:3'])
            self.assertTrue((root / 'export/index.html').is_file())

    def test_actual_results_export_tables_figures_sources_and_exclude_raw_text_bundle(self):
        with completed_exports() as source, tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper, supplement = root / 'paper', root / 'supplement'
            render_numeric_results(validate_study(source), paper, supplement)
            self.assertEqual(len(list((paper / 'tables').glob('*.tex'))), 4)
            self.assertEqual(len(list((paper / 'figures').glob('*.pdf'))), 2)
            self.assertEqual(len(list((paper / 'figures').glob('*.png'))), 2)
            self.assertEqual(len(list((supplement / 'tables').glob('*.tex'))), 7)
            # All category effects are retained; the order does not select favorable effects.
            categories = read_csv(paper / 'source_data/figure_2_transportation_categories.csv')
            self.assertEqual(len(categories), 20)
            self.assertEqual(len({r['category'] for r in categories}), 10)
            self.assertTrue(any(float(r['difference']) < 0 for r in categories))
            # Exact CSV bytes are preserved, avoiding double rounding of scientific outputs.
            for path in (source / 'results').glob('*.csv'):
                self.assertEqual(path.read_bytes(), (supplement / 'source_data' / path.name).read_bytes())
            direct = next(r for r in read_csv(paper / 'source_data/table_3_transportation_retrieval.csv') if r['method'] == 'minilm_retrieval')
            self.assertEqual(direct['K'], '')
            self.assertEqual(direct['macro_ndcg_seed_sd'], '')
            (supplement / 'topic_reports').mkdir()
            (supplement / 'topic_reports/private.html').write_text('ORIGINAL COMPLAINT TEXT', encoding='utf-8')
            archive = portable_bundle(paper, supplement, root / 'numerical_results.zip')
            with zipfile.ZipFile(archive) as z:
                self.assertIsNone(z.testzip())
                self.assertFalse(any('topic_reports' in n for n in z.namelist()))
                self.assertTrue(any(n.endswith('figure_1_corpus_size.pdf') for n in z.namelist()))

    def test_generation_does_not_fit_models_or_mutate_source_results(self):
        from transport_topics.runtime.artifacts import file_hash
        import sys
        imported_before = set(sys.modules)
        with completed_exports() as source, tempfile.TemporaryDirectory() as tmp:
            before = {p.relative_to(source).as_posix(): file_hash(p) for p in source.rglob('*') if p.is_file()}
            bundle = validate_study(source)
            render_numeric_results(bundle, Path(tmp) / 'paper', Path(tmp) / 'supplement')
            after = {p.relative_to(source).as_posix(): file_hash(p) for p in source.rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            # Display/export imports do not load fitting adapters or a GPU backend.
            self.assertNotIn('transport_topics.baselines.neural', set(sys.modules) - imported_before)
