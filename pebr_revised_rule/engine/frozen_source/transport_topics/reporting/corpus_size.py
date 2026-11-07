"""Read-only corpus-size analysis of the locked, completed transport study.

No training, inference, parameter selection or GPU initialization takes place.
Pair with original-only NMF within the same collection and model seed, average
five seeds within each sampling ladder, and then summarize the five ladders.
"""
from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import logging
import math
from pathlib import Path
import statistics
import sys
import time
from uuid import uuid4
import zipfile


RESULT_FILES = ('corpus_size_results.csv', 'corpus_size_deltas.csv',
                'corpus_size_summary.csv')
METHODS = ('nmf_original', 'nmf_similarity', 'similarity_matched', 'proposed')
LABELS = {'nmf_original': 'Original-only NMF',
          'nmf_similarity': 'Similarity sharing',
          'similarity_matched': 'Matched-amount sharing', 'proposed': 'Proposed'}
COLORS = {'nmf_similarity': '#B86B14', 'similarity_matched': '#5B8461',
          'proposed': '#174A7E'}
MARKERS = {'nmf_similarity': 's', 'similarity_matched': '^', 'proposed': 'o'}
DATASETS = {'primary': 'Indian transport', 'external': 'Dabiri traffic'}


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False) + '\n', encoding='utf-8')


def _csv_read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def _csv_write(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError('Refusing to write an empty result table')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _inside(root, relative):
    candidate = (Path(root) / relative).resolve()
    if not candidate.is_relative_to(Path(root).resolve()):
        raise ValueError('A configured path leaves the project: ' + str(relative))
    return candidate


def _same(actual, expected, description):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-10,
                        abs_tol=1e-12):
        raise ValueError('Saved numerical values do not reconcile: ' + description)


def _average_sd(values):
    values = list(values)
    return statistics.mean(values), statistics.stdev(values)


def validate_config(cfg):
    expected = {'kind': 'saved_corpus_size_figure', 'topic_count': 12,
                'model_seeds': [0, 1, 2, 3, 4],
                'sampling_ladders': [0, 1, 2, 3, 4],
                'target_sizes': [100, 200, 400, 600],
                'datasets': ['primary', 'external'],
                'methods': list(METHODS), 'metrics': ['npmi', 'cv']}
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError('Keep the complete saved-study protocol: ' + key)
    figure = cfg['figure']
    if figure['uncertainty'] != 'sampling_ladder_sd_after_seed_averaging':
        raise ValueError('The declared error bars are sampling-ladder SDs')
    if not (5 <= figure['width_inches'] <= 12 and
            2.5 <= figure['height_inches'] <= 8 and
            150 <= figure['png_dpi'] <= 600):
        raise ValueError('Invalid figure dimensions or resolution')


def resolve_source(root, cfg):
    """Prefer the user's original run, with an exact archived fallback."""
    manifest = json.loads(_inside(root, cfg['source_manifest']).read_text(
        encoding='utf-8'))
    if manifest['source_run_id'] != cfg['source_run_id']:
        raise ValueError('Configuration and archived source run disagree')
    local = _inside(root, 'outputs/runs/' + cfg['source_run_id'] + '/results')
    archive = _inside(root, cfg['archive_directory'])
    present = [name for name in RESULT_FILES if (local / name).is_file()]
    if present and len(present) != len(RESULT_FILES):
        raise ValueError('The local source run has incomplete corpus-size CSVs')
    source = local if present else archive
    hashes = {}
    for name in RESULT_FILES:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError('Missing saved result CSV: ' + str(path))
        hashes[name] = _hash(path)
        if hashes[name] != manifest['files_sha256'][name]:
            raise ValueError('Source CSV differs from the locked completed study: '
                             + name)
    return source, {'source_run_id': cfg['source_run_id'],
                    'location': source.relative_to(root).as_posix(),
                    'using_local_run': bool(present), 'sha256': hashes,
                    'new_model_fits': 0, 'outcome_dependent_filtering': False}


def validate_and_aggregate(tables, cfg):
    """Validate every original row and reconstruct both metrics independently."""
    raw = tables['corpus_size_results.csv']
    expected = {(d, n, l, s, m) for d in cfg['datasets']
                for n in cfg['target_sizes'] for l in cfg['sampling_ladders']
                for s in cfg['model_seeds'] for m in cfg['methods']}
    indexed = {}
    references = defaultdict(set)
    for row in raw:
        key = (row['dataset'], int(row['target_size']),
               int(row['sample_index']), int(row['seed']), row['method'])
        if key in indexed:
            raise ValueError('Duplicate corpus-size model/seed row: ' + str(key))
        if row['study'] != 'corpus_size' or int(row['K']) != cfg['topic_count']:
            raise ValueError('Incorrect study or topic count in saved source')
        if str(row['development_only']).lower() != 'false':
            raise ValueError('Development/pilot rows cannot enter this figure')
        if int(row['actual_size']) < int(row['target_size']):
            raise ValueError('Actual collection is smaller than its target')
        for metric, lower in [('npmi', -1), ('cv', 0), ('diversity', 0)]:
            value = float(row[metric])
            if not math.isfinite(value) or not lower <= value <= 1:
                raise ValueError('Nonfinite or out-of-range metric: ' + metric)
        indexed[key] = row
        references[row['dataset']].add(row['reference_identity'])
    if set(indexed) != expected or len(raw) != len(expected):
        raise ValueError('The declared 800-row corpus-size experiment is incomplete')
    if any(len(ids) != 1 for ids in references.values()):
        raise ValueError('Coherence references vary within a dataset')

    paired = []
    paired_lookup = {}
    for key, row in sorted(indexed.items()):
        dataset, size, ladder, seed, method = key
        baseline = indexed[dataset, size, ladder, seed, 'nmf_original']
        for name in ('job_id', 'condition', 'actual_size', 'reference_identity'):
            if row[name] != baseline[name]:
                raise ValueError('Pairing crosses a collection boundary: ' + name)
        record = {'dataset': dataset, 'target_size': size,
                  'actual_size': int(row['actual_size']), 'sample_index': ladder,
                  'seed': seed, 'method': method, 'K': cfg['topic_count']}
        for metric in cfg['metrics']:
            record[metric] = float(row[metric])
            record[metric + '_change'] = float(row[metric]) - float(baseline[metric])
        paired.append(record)
        paired_lookup[key] = record

    delta_keys = set()
    for row in tables['corpus_size_deltas.csv']:
        key = (row['dataset'], int(row['target_size']),
               int(row['sample_index']), int(row['seed']), row['method'])
        if key in delta_keys or key not in paired_lookup or key[-1] == 'nmf_original':
            raise ValueError('Unexpected or duplicate saved paired difference')
        _same(row['npmi_change_from_no_borrowing'],
              paired_lookup[key]['npmi_change'], 'paired NPMI difference')
        delta_keys.add(key)
    if delta_keys != {key for key in expected if key[-1] != 'nmf_original'}:
        raise ValueError('Saved paired NPMI export does not cover all 600 comparisons')

    ladder_rows = []
    for dataset in cfg['datasets']:
        for size in cfg['target_sizes']:
            for ladder in cfg['sampling_ladders']:
                for method in cfg['methods']:
                    group = [paired_lookup[dataset, size, ladder, seed, method]
                             for seed in cfg['model_seeds']]
                    record = {'dataset': dataset, 'target_size': size,
                              'sample_index': ladder, 'method': method,
                              'seed_fits': len(group),
                              'actual_size_min': min(r['actual_size'] for r in group),
                              'actual_size_max': max(r['actual_size'] for r in group)}
                    for metric in cfg['metrics']:
                        record[metric + '_mean'] = statistics.mean(r[metric] for r in group)
                        record[metric + '_change_mean'] = statistics.mean(
                            r[metric + '_change'] for r in group)
                    ladder_rows.append(record)

    summaries = []
    for dataset in cfg['datasets']:
        for size in cfg['target_sizes']:
            for method in cfg['methods']:
                group = [r for r in ladder_rows if
                         (r['dataset'], r['target_size'], r['method']) ==
                         (dataset, size, method)]
                record = {'dataset': dataset, 'target_size': size, 'method': method,
                          'ladders': len(group), 'seeds_per_ladder': 5,
                          'actual_size_min': min(r['actual_size_min'] for r in group),
                          'actual_size_max': max(r['actual_size_max'] for r in group)}
                for metric in cfg['metrics']:
                    for suffix in ('', '_change'):
                        name = metric + suffix
                        values = [r[name + '_mean'] for r in group]
                        mean, sd = _average_sd(values)
                        record[name + '_mean'] = mean
                        record[name + '_ladder_sd'] = sd
                    record[metric + '_positive_ladders'] = sum(
                        r[metric + '_change_mean'] > 0 for r in group)
                summaries.append(record)

    lookup = {(r['dataset'], r['target_size'], r['method']): r for r in summaries}
    summary_keys = set()
    for row in tables['corpus_size_summary.csv']:
        key = row['dataset'], int(row['target_size']), row['method']
        if key in summary_keys or key not in lookup or key[-1] == 'nmf_original':
            raise ValueError('Unexpected or duplicate saved corpus-size summary')
        value = lookup[key]
        _same(row['mean_npmi_change'], value['npmi_change_mean'], 'summary mean')
        _same(row['ladder_mean_sd'], value['npmi_change_ladder_sd'], 'summary SD')
        for name, column in [('ladders', 'ladders'),
                             ('seed_fits_per_ladder', 'seeds_per_ladder'),
                             ('actual_size_min', 'actual_size_min'),
                             ('actual_size_max', 'actual_size_max')]:
            if int(row[name]) != value[column]:
                raise ValueError('Saved summary count differs: ' + name)
        summary_keys.add(key)
    expected_summary = {(d, n, m) for d in cfg['datasets']
                        for n in cfg['target_sizes']
                        for m in cfg['methods'] if m != 'nmf_original'}
    if summary_keys != expected_summary:
        raise ValueError('All 24 original corpus-size aggregates must reconcile')
    audit = {'passed': True, 'original_fit_rows': len(raw),
             'paired_npmi_changes_verified': len(delta_keys),
             'saved_aggregates_verified': len(summary_keys),
             'model_seeds': cfg['model_seeds'],
             'sampling_ladders': cfg['sampling_ladders'],
             'uncertainty': 'sample SD of five seed-averaged ladder differences',
             'confidence_interval': False,
             'ladders_are_independent_datasets': False,
             'new_model_fits': 0, 'new_evaluation_predictions': 0}
    return paired, ladder_rows, summaries, audit


def draw_figures(summaries, cfg, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    style = {'font.family': 'DejaVu Sans', 'font.size': 9,
             'axes.labelsize': 9, 'axes.titlesize': 10,
             'axes.spines.top': False, 'axes.spines.right': False,
             'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
             'savefig.facecolor': 'white'}
    saved = []
    with plt.rc_context(style):
        for dataset in cfg['datasets']:
            fig, axes = plt.subplots(1, 2, figsize=(cfg['figure']['width_inches'],
                                                   cfg['figure']['height_inches']))
            for ax, metric, title, ylabel in zip(
                    axes, ('npmi', 'cv'), ('(a) NPMI', r'(b) $C_v$'),
                    ('NPMI change from original-only NMF',
                     r'$C_v$ change from original-only NMF')):
                for method in METHODS[1:]:
                    group = sorted((r for r in summaries if r['dataset'] == dataset
                                    and r['method'] == method),
                                   key=lambda r: r['target_size'])
                    ax.errorbar([r['target_size'] for r in group],
                                [r[metric + '_change_mean'] for r in group],
                                yerr=[r[metric + '_change_ladder_sd'] for r in group],
                                color=COLORS[method], marker=MARKERS[method],
                                markersize=4.4 if method == 'proposed' else 4,
                                linewidth=1.9 if method == 'proposed' else 1.3,
                                elinewidth=.75, capsize=2.5, capthick=.75,
                                label=LABELS[method], zorder=4 if method == 'proposed' else 3)
                ax.axhline(0, color='#626975', linewidth=.9, linestyle='--', zorder=1)
                ax.set_title(title, loc='left', pad=8)
                ax.set_xlabel('Target posts in fitting collection')
                ax.set_ylabel(ylabel)
                ax.set_xticks(cfg['target_sizes'])
                ax.set_xlim(65, 635)
                ax.grid(axis='y', color='#E3E6EA', linewidth=.6, zorder=0)
                ax.tick_params(labelsize=8)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=3,
                       frameon=False, bbox_to_anchor=(.5, .005),
                       handlelength=2.2, columnspacing=1.6)
            fig.subplots_adjust(left=.085, right=.985, top=.9,
                                bottom=.25, wspace=.37)
            directory = output / 'figures'
            directory.mkdir(parents=True, exist_ok=True)
            for ext in ('png', 'pdf'):
                path = directory / ('corpus_size_' + dataset + '.' + ext)
                payload = io.BytesIO()
                options = {'format': ext, 'bbox_inches': 'tight', 'pad_inches': .06}
                if ext == 'png':
                    options['dpi'] = cfg['figure']['png_dpi']
                else:
                    options['metadata'] = {'Creator': 'Proposed saved corpus-size analysis',
                                           'CreationDate': None, 'ModDate': None}
                fig.savefig(payload, **options)
                path.write_bytes(payload.getvalue())
                saved.append(path.relative_to(output).as_posix())
            plt.close(fig)
    return saved


def make_readout(summaries, cfg):
    lines = ['# Corpus-size analysis of saved study results', '',
             'K=12; five fixed model seeds within each of five sampling ladders.',
             'Each point is a mean paired difference from original-only NMF.',
             'Error bars are sample SD across the five ladder means, not 95% intervals.',
             'Sampling ladders overlap and are not independent external datasets.', '']
    for dataset in cfg['datasets']:
        lines += ['## ' + DATASETS[dataset], '',
                  '| Posts | Method | NPMI gain (mean +/- SD) | C_v gain (mean +/- SD) |',
                  '|---:|---|---:|---:|']
        for size in cfg['target_sizes']:
            for method in METHODS[1:]:
                r = next(r for r in summaries if (r['dataset'], r['target_size'], r['method'])
                         == (dataset, size, method))
                lines.append(f"| {size} | {LABELS[method]} | "
                             f"{r['npmi_change_mean']:+.4f} +/- {r['npmi_change_ladder_sd']:.4f} | "
                             f"{r['cv_change_mean']:+.4f} +/- {r['cv_change_ladder_sd']:.4f} |")
        lines += ['', 'Proposed improvements across the five seed-averaged ladders:', '',
                  '| Posts | NPMI-positive ladders | C_v-positive ladders |',
                  '|---:|---:|---:|']
        for size in cfg['target_sizes']:
            r = next(r for r in summaries if (r['dataset'], r['target_size'], r['method'])
                     == (dataset, size, 'proposed'))
            lines.append(f"| {size} | {r['npmi_positive_ladders']}/5 | {r['cv_positive_ladders']}/5 |")
        lines.append('')
    lines += ['## Interpretation', '',
              'On the Indian collection, Proposed has positive mean NPMI gain at all four sizes.',
              'It does not have the largest mean NPMI gain at 100, 200 or 400 posts.',
              'Its mean C_v gain is negative at 200 posts. At 600 posts it has the largest',
              'displayed mean NPMI and C_v gains. Variation across sampling ladders is substantial.',
              'These results support a descriptive robustness analysis, not a claim of',
              'consistent superiority in the smallest-data conditions or statistical significance.',
              'The external results are retained in their own figure and numerical exports.',
              'Do not replace error bars with seed-only SD or discard unfavorable sizes.', '']
    return '\n'.join(lines)


def write_caption(output):
    text = r'''% Use the Indian panel as the main figure only if discussing its mixed size effects.
\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{corpus_size_primary.pdf}
    \caption{Corpus-size effects on the Indian transport collection at $K=12$.
    Changes in NPMI and $C_v$ are paired with original-only NMF within the same
    sampling ladder and model seed. Points average five ladder means, each
    obtained from five model seeds; error bars show the standard deviation
    across those ladder means. The dashed zero line denotes no change from
    original-only NMF. The sampling ladders overlap and are not independent
    datasets.}
    \label{fig:corpus_size}
\end{figure*}
'''
    (output / 'figure_caption.tex').write_text(text, encoding='utf-8')


class _TimeFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        return datetime.fromtimestamp(record.created).astimezone().isoformat(
            timespec='milliseconds')


def execute_report(root, args):
    """Entry called by the existing root main.py for this explicit config only."""
    root = Path(root).resolve()
    started = time.monotonic()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_corpus_size_' + uuid4().hex[:8]
    logger = logging.getLogger('transport_corpus_size.' + run_id)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    formatter = _TimeFormatter('%(asctime)s | ' + run_id + ' | %(levelname)-7s | %(message)s')
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(root / 'logs/run_log.txt', encoding='utf-8')):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    output = None
    try:
        logger.info('CORPUS-SIZE ANALYSIS | saved results only; no model fitting or GPU initialization.')
        logger.info('Interpreter: %s', sys.executable)
        if (getattr(args, 'mode', None) or 'report') != 'report':
            raise ValueError('Use --mode report for the saved-result corpus-size figure')
        import yaml
        cfg_path = Path(args.config)
        cfg_path = cfg_path if cfg_path.is_absolute() else _inside(root, cfg_path)
        cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
        validate_config(cfg)
        requested_run = getattr(args, 'run_id', None)
        if requested_run and requested_run != cfg['source_run_id']:
            raise ValueError('This report uses the configured completed source run')
        source, provenance = resolve_source(root, cfg)
        logger.info('Source: %s | %s.', cfg['source_run_id'], provenance['location'])
        tables = {name: _csv_read(source / name) for name in RESULT_FILES}
        paired, ladders, summaries, audit = validate_and_aggregate(tables, cfg)
        logger.info('SOURCE VERIFIED | %d fit rows; %d paired differences; %d saved aggregates.',
                    audit['original_fit_rows'], audit['paired_npmi_changes_verified'],
                    audit['saved_aggregates_verified'])
        output = _inside(root, cfg['output_directory']) / run_id
        output.mkdir(parents=True, exist_ok=False)
        (output / 'source_data').mkdir()
        for name in RESULT_FILES:
            (output / 'source_data' / name).write_bytes((source / name).read_bytes())
        _csv_write(output / 'paired_seed_values.csv', paired)
        _csv_write(output / 'seed_averaged_ladder_values.csv', ladders)
        _csv_write(output / 'corpus_size_summary_both_metrics.csv', summaries)
        figures = draw_figures(summaries, cfg, output)
        (output / 'RESULTS_READOUT.md').write_text(make_readout(summaries, cfg), encoding='utf-8')
        write_caption(output)
        _json(output / 'source_provenance.json', provenance)
        _json(output / 'verification.json', audit)
        _json(output / 'config.json', cfg)
        _json(output / 'environment.json', {'python': sys.version,
              'matplotlib': importlib.metadata.version('matplotlib'),
              'PyYAML': importlib.metadata.version('PyYAML'),
              'GPU_required': False, 'execution_scope': 'saved-result aggregation and figure export'})
        manifest = {'run_id': run_id, 'complete': True, 'source_run_id': cfg['source_run_id'],
                    'figures': figures, 'seconds': time.monotonic() - started,
                    'new_model_fits': 0, 'scientific_settings_changed': False,
                    'files_sha256': {p.relative_to(output).as_posix(): _hash(p)
                                     for p in sorted(output.rglob('*')) if p.is_file()}}
        _json(output / 'manifest.json', manifest)
        logger.info('INDIAN RESULT | Proposed has positive mean NPMI gain at all sizes; comparisons and C_v effects are mixed.')
        logger.info('RESULTS DIRECTORY: %s', output)
        logger.info('Read RESULTS_READOUT.md before making superiority or robustness claims.')
        archive = root / 'outputs/inspection' / (run_id + '_inspection.zip')
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for path in sorted(output.rglob('*')):
                if path.is_file():
                    z.write(path, 'results/' + path.relative_to(output).as_posix())
        logger.info('RUN COMPLETED | %.2fs | SEND THIS INSPECTION ZIP: %s',
                    time.monotonic() - started, archive)
        return 0
    except Exception:
        logger.exception('CORPUS-SIZE REPORT FAILED')
        if output:
            _json(output / 'failure.json', {'complete': False,
                  'seconds': time.monotonic() - started})
        return 1
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
