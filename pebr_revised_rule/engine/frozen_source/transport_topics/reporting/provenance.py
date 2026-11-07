"""Source provenance, portable result bundles and carefully bounded result descriptions."""
from __future__ import annotations

from pathlib import Path
import zipfile

from ..runtime.artifacts import file_hash, safe_path, source_identity, write_json
from .common import LABELS, TOPIC_METHODS, numeric, one, write_text


def method_source_check(root, original):
    """Report formatting must not silently revise fitting, selection or evaluation code."""
    prefixes = tuple('src/transport_topics/' + name + '/' for name in
                     ('proposed', 'numerics', 'representations', 'topics', 'baselines', 'application', 'evaluation', 'data', 'experiments'))
    config_prefixes = ('configs/data/', 'configs/methods/', 'configs/models/', 'configs/studies/', 'configs/evaluation/')
    fixed = ('configs/project.yaml', 'src/transport_topics/reporting/aggregate.py')
    records = []
    for name, expected in original['files'].items():
        if name.startswith(prefixes + config_prefixes) or name in fixed:
            path = safe_path(root, name)
            current = file_hash(path) if path.is_file() else None
            records.append({'path': name, 'study_sha256': expected, 'current_sha256': current, 'unchanged': current == expected})
    return {'all_scientific_sources_unchanged': all(r['unchanged'] for r in records), 'files': records}


def result_readout(bundle):
    """Use fixed configurations and actual signs; no claim of superiority is scripted."""
    tables, config, audit = bundle['tables'], bundle['config'], bundle['audit']
    k, b = config['topics']['main_count'], config['retrieval']['main_budget']
    lines = ['# Phase-five results reviewed for publication', '',
        'Source run: ' + audit['source_run_id'], '',
        'The registered study is complete. Completion establishes execution and coverage; it does not establish overall methodological superiority.', '',
        '%d distinct saved model outputs; %d logical evaluations; %d converged final NMF fits; %d recorded CUDA neural fits.' %
        (audit['unique_models'], audit['logical_evaluations'], audit['converged_final_nmf_fits'], audit['cuda_neural_fits']), '',
        '## Topic coherence', '', '| Corpus | Method | K=8 | K=12 | K=16 |', '|---|---|---:|---:|---:|']
    for dataset in ('primary', 'external'):
        for method in TOPIC_METHODS:
            scores = [one(tables['topic_quality_summary.csv'], dataset=dataset, method=method, K=count) for count in config['topics']['counts']]
            lines.append('| %s | %s | %s |' % (dataset, LABELS[method], ' | '.join('%.4f' % numeric(r['npmi_mean']) for r in scores)))
    lines += ['', 'Primary values average five seeds. External values average five within-sample seed means. '
        'External samples come from one corpus and can overlap; they are not independent datasets.', '',
        '## Primary category retrieval', '', '| Method | Macro nDCG@%d | Macro Recall@%d |' % (b, b), '|---|---:|---:|']
    for method in TOPIC_METHODS + ('minilm_retrieval',):
        r = one(tables['retrieval_summary.csv'], method=method, K='' if method == 'minilm_retrieval' else k, B=b)
        lines.append('| %s | %.4f | %.4f |' % (LABELS[method], numeric(r['macro_ndcg_mean']), numeric(r['macro_recall_mean'])))
    lines += ['', '## Paired retrieval evidence', '']
    for method in ('nmf_original', 'nmf_similarity', 'similarity_matched', 'minilm_retrieval'):
        r = one(tables['retrieval_contrasts.csv'], K=k, B=b, metric='ndcg', comparator=method)
        low, high = numeric(r['ci_low']), numeric(r['ci_high'])
        direction = 'includes zero' if low <= 0 <= high else ('lies above zero' if low > 0 else 'lies below zero')
        lines += ['- Proposed minus %s: %.5f; paired 95%% interval [%.5f, %.5f], which %s.' %
                  (LABELS[method], numeric(r['difference']), low, high, direction)]
    lines += ['', 'Intervals are conditional on the fitted, seed-averaged models. They are not adjusted for multiple contrasts. '
        'No superiority claim should be based on interval signs alone across an unplanned selection of comparisons.', '',
        '## Interpretation boundaries', '',
        '- Report all three topic counts and both corpora; retain unfavorable comparisons.',
        '- Category relevance evaluates service-category retrieval, not shared incident identity or an analyst user study.',
        '- Frozen-predictor gain is a mechanism audit and does not guarantee coherence or retrieval improvements.',
        '- The no-downside comparison tests whether the penalty helps; its benefit must not be presumed.',
        '- Negative baseline coherence is a reported outcome under the recorded setup, not proof that published methods are generally inferior.',
        '- Corpus-size results use nested, possibly overlapping samples; their ladder SD is descriptive, not a confidence interval.',
        '- Main K=%d, B=%d and all sensitivity settings remain those fixed in the source study.' % (k, b), '']
    return '\n'.join(lines)


def write_reproduction_sources(root, bundle, destination):
    """Save reproducibility metadata without embedding datasets, private paths or raw complaint text."""
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('config.yaml', 'manifest.json', 'source_fingerprints.json', 'environment.json', 'study_plan.json',
                 'dataset_evaluation_profile.json', 'evaluation_summary.json', 'preparation_dependency.json', 'coherence_references.json'):
        source = bundle['directory'] / name
        from ..runtime.artifacts import atomic_write
        atomic_write(destination / ('study_' + name), source.read_bytes(), overwrite=True)
    current = source_identity(root)
    write_json(destination / 'reporting_source_fingerprints.json', current, overwrite=True)
    write_json(destination / 'source_result_hashes.json',
        {name: file_hash(bundle['directory'] / 'results' / name) for name in bundle['tables']}, overwrite=True)
    write_json(destination / 'acceptance_audit.json', bundle['audit'], overwrite=True)
    write_json(destination / 'method_source_check.json', method_source_check(root, bundle['data']['source_fingerprints']), overwrite=True)
    write_text(destination / 'REPRODUCE.md',
        '# Reproduce these result files\n\n'
        'Restore the delivered source code, original data and saved model/evaluation outputs into the same project tree.\n'
        'Use the Python 3.10+ environment recorded in study_environment.json.\n\n'
        'python main.py --config configs/experiments/reports.yaml --mode report --run-id ' + bundle['audit']['source_run_id'] + '\n\n'
        'Report generation reads saved outputs and does not fit models, encode posts, change seeds or retune settings. '
        'A clean numerical replication uses the locked full-study configuration and original source revisions; '
        'hardware/software differences can change floating-point training outcomes.\n\n'
        'The source data are Indian Transport SNAM grievance.csv and the Dabiri three-class training release. '
        'Their acquisition records and hashes remain in the project. Original-text evidence reports are retained locally '
        'and are omitted from compact inspection and portable numerical bundles.\n')
    archive = destination / 'reporting_and_experiment_sources.zip'
    names = set(current['files'])
    names.update(p.relative_to(root).as_posix() for p in (root / 'docs').glob('*.md'))
    names.update(p.relative_to(root).as_posix() for p in (root / 'tests/fixtures').glob('*') if p.is_file())
    for folder in ('third_party', 'vendor', 'assets/vendor'):
        target = root / folder
        if target.is_dir():
            names.update(p.relative_to(root).as_posix() for p in target.rglob('*') if p.is_file() and p.suffix in ('.py', '.md', '.txt'))
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in sorted(names):
            if (root / name).is_file():
                z.write(root / name, name)
    return archive


def portable_bundle(paper, supplement, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as z:
        for base, prefix in ((paper, 'paper'), (supplement, 'supplement')):
            for path in sorted(base.rglob('*')):
                relative = path.relative_to(base)
                if path.is_file() and relative.parts[0] != 'topic_reports' and path.suffix not in ('.aux', '.log', '.out'):
                    z.write(path, (Path(prefix) / relative).as_posix())
    with zipfile.ZipFile(destination) as z:
        if z.testzip():
            raise ValueError('The result archive failed its integrity check')
    return destination
