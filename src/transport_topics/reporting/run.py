"""Read-only paper-result generation from one complete phase-five study."""
from __future__ import annotations

import json
from pathlib import Path

from ..runtime.artifacts import atomic_write, file_hash, write_json
from .common import RESULT_NAMES, write_text
from .figures import main_figures, sensitivity_figure
from .provenance import method_source_check, portable_bundle, result_readout, write_reproduction_sources
from .tables import main_tables, supplementary_tables
from .topic_reports import export_topic_reports
from .validation import resolve_study, validate_study


def render_numeric_results(bundle, paper, supplement):
    """Reusable core: also exercised on the actual uploaded inspection CSVs."""
    for base in (paper, supplement):
        for folder in ('tables', 'figures', 'source_data'):
            (base / folder).mkdir(parents=True, exist_ok=True)
    for name in RESULT_NAMES:
        atomic_write(supplement / 'source_data' / name, (bundle['directory'] / 'results' / name).read_bytes(), overwrite=True)
    main_tables(bundle, paper)
    supplementary_tables(bundle, supplement)
    main_figures(bundle, paper)
    sensitivity_figure(bundle, supplement)
    write_text(paper / 'RESULTS_READOUT.md', result_readout(bundle))
    write_text(paper / 'FIGURE_CAPTIONS.tex',
        '% Include these captions alongside the corresponding figure PDF.\n'
        '\\caption{Corpus-size effects at $K=12$. Changes are paired with original-only NMF within each '
        'ladder and model seed. Points average five ladder means after averaging five model seeds; '
        'error bars are SD across ladder means. Family preservation can raise actual counts above the target; '
        'exact sizes are given in the source CSV. Overlapping ladders are not independent datasets.}\n\n'
        '\\caption{Service-category retrieval differences on the Indian corpus at $K=12$, $B=10$. '
        'All ten categories are shown in alphabetical order with query counts. Intervals use 2,000 paired '
        'query-family bootstrap resamples within categories after averaging model seeds. They are conditional '
        'on the fitted models and are not multiplicity-adjusted.}\n')
    write_text(paper / 'README.md',
        '# Paper results\n\nFour LaTeX tables, two PDF/PNG figures and their exact CSV sources. '
        'The source run fixes every topic count, model seed and document budget. Tables require booktabs; '
        'table* is used for wide tables. Include figures with graphicx. '
        'Copy table snippets into the manuscript; no figure data are created from table screenshots.\n\n'
        'Read RESULTS_READOUT.md before writing claims. Supplemental source CSVs preserve every unfavorable and inconclusive result.\n')
    write_text(supplement / 'README.md',
        '# Supplementary experimental results\n\n'
        'Compile supplement_results.tex from this directory using a normal LaTeX installation (booktabs, longtable, graphicx, geometry). '
        'Compilation is optional; generation does not install TeX. All 16 source CSV exports are copied without numerical edits.\n\n'
        'Primary summaries use seed SD; external summaries use SD of sample means and separately retain within-sample seed SD. '
        'Topic reports contain all main K=12 models and seeds with original complaints and borrowing provenance. '
        'They are illustrative associations, not a human relevance assessment. No donor text is attributed to a receiving author.\n')


def run(root, run_id, logger, store):
    root = Path(root)
    source = resolve_study(root, run_id)
    logger.info('PAPER RESULTS | source run: %s. No fitting, encoder calls or parameter search will run.', source.name)
    bundle = validate_study(source)
    store.json('phase5_acceptance.json', bundle['audit'])
    changes = method_source_check(root, bundle['data']['source_fingerprints'])
    store.json('method_source_check.json', changes)
    if not changes['all_scientific_sources_unchanged']:
        raise ValueError('Scientific source files differ from the completed study; preserve and reconcile the change before publishing.')
    logger.info('SOURCE ACCEPTED | %d models; %d evaluations; all saved aggregates reconcile.',
                bundle['audit']['unique_models'], bundle['audit']['logical_evaluations'])
    # A report run gets its own output trees. Previous papers and failed/completed runs stay intact.
    paper = root / 'outputs/paper' / store.run_id
    supplement = root / 'outputs/supplement' / store.run_id
    logger.info('EXPORT | four main tables, two main figures and full supplementary numerical results.')
    render_numeric_results(bundle, paper, supplement)
    evidence = export_topic_reports(root, bundle, supplement / 'topic_reports', logger)
    store.json('topic_report_summary.json', evidence)
    write_reproduction_sources(root, bundle, supplement / 'reproducibility')
    write_json(supplement / 'reproducibility/topic_report_summary.json', evidence, overwrite=True)
    # Hash every generated artifact, including local topic evidence; do not copy raw text into inspection.
    outputs = {}
    for base in (paper, supplement):
        for path in sorted(base.rglob('*')):
            if path.is_file():
                outputs[path.relative_to(root).as_posix()] = {'sha256': file_hash(path), 'bytes': path.stat().st_size}
    store.json('publication_outputs.json', outputs)
    write_json(supplement / 'reproducibility/publication_outputs.json', outputs, overwrite=True)
    archive = portable_bundle(paper, supplement, root / 'outputs/reproducibility' / (store.run_id + '_paper_results.zip'))
    summary = {'complete': True, 'source_run_id': source.name, 'models_retrained': False,
        'scientific_settings_changed': False, 'result_values_recomputed_or_rounded_at_source': False,
        'main_tables': 4, 'main_figures': 2, 'supplementary_tables': 7, 'supplementary_figures': 1,
        'paper_directory': paper.relative_to(root).as_posix(), 'supplement_directory': supplement.relative_to(root).as_posix(),
        'portable_numerical_bundle': archive.relative_to(root).as_posix(), 'portable_bundle_sha256': file_hash(archive),
        'topic_evidence': evidence, 'source_study_identity': bundle['data']['evaluation_summary']['identity'],
        'source_acceptance': bundle['audit'], 'runtime_device': 'cpu; saved model outputs only',
        'tex_compilation_required_for_generation': False}
    store.json('publication_summary.json', summary)
    store.json('report_target.json', {'target_run_id': source.name, 'operation': 'paper_results_from_saved_outputs'})
    store.text('results_readout.md', result_readout(bundle))
    # Small tables/figure sources accompany the inspection; no original complaint text.
    for path in (paper / 'source_data').glob('*.csv'):
        atomic_write(store.directory / 'results' / path.name, path.read_bytes())
    write_json(root / 'outputs/manifests/latest_publication.json', {'run_id': store.run_id, **summary}, overwrite=True)
    logger.info('PAPER RESULTS COMPLETE | %d model reports; %d topic records.', evidence['model_reports'], evidence['topic_reports'])
    logger.info('Paper tables and figures: %s', paper)
    logger.info('Full supplementary results: %s', supplement)
    logger.info('Portable numerical result bundle: %s', archive)
    return summary
