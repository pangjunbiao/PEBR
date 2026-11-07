"""Publication figures with distinct experimental axes and exact CSV sources."""
from __future__ import annotations

import io
from collections import defaultdict
import numpy as np

from ..runtime.artifacts import atomic_write
from .common import LABELS, numeric, write_csv

COLORS = {'proposed': '#174A7E', 'nmf_similarity': '#B86B14', 'similarity_matched': '#5B8461',
          'nmf_original': '#626975'}


def _plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path):
    """Write closed complete payloads through the Windows-safe atomic writer."""
    for extension, dpi in (('pdf', 300), ('png', 220)):
        stream = io.BytesIO()
        kwargs = {'format': extension, 'dpi': dpi, 'bbox_inches': 'tight', 'pad_inches': .08}
        if extension == 'pdf':
            kwargs['metadata'] = {'Creator': 'proposed: saved-result reporting', 'CreationDate': None, 'ModDate': None}
        fig.savefig(stream, **kwargs)
        atomic_write(path.with_suffix('.' + extension), stream.getvalue(), overwrite=True)


def _style(plt):
    return plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'})


def main_figures(bundle, paper):
    plt = _plotting()
    tables, config = bundle['tables'], bundle['config']
    summary = tables['corpus_size_summary.csv']
    # Both plotted summaries and the paired seed/ladder records are preserved.
    write_csv(paper / 'source_data/figure_1_corpus_size.csv', summary)
    write_csv(paper / 'source_data/figure_1_paired_ladder_seed_values.csv', tables['corpus_size_deltas.csv'])
    with _style(plt):
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.05), sharey=True, layout='constrained')
        for ax, dataset, title in zip(axes, ('primary', 'external'), ('(a) Indian transport', '(b) Dabiri traffic')):
            for method, marker in (('nmf_similarity', 's'), ('similarity_matched', '^'), ('proposed', 'o')):
                group = sorted((r for r in summary if r['dataset'] == dataset and r['method'] == method), key=lambda r: int(r['target_size']))
                x = np.array([numeric(r['target_size']) for r in group])
                y = np.array([numeric(r['mean_npmi_change']) for r in group])
                sd = np.array([numeric(r['ladder_mean_sd']) for r in group])
                ax.errorbar(x, y, yerr=sd, color=COLORS[method], marker=marker, markersize=4,
                            linewidth=1.5, elinewidth=.8, capsize=2.5, label=LABELS[method])
            ax.axhline(0, color='#72767C', linewidth=.8, linestyle='--', zorder=0)
            ax.set_title(title)
            ax.set_xticks(config['studies']['corpus_size']['sizes'])
            ax.set_xlabel('Target posts in fitting collection')
            ax.grid(axis='y', color='#E3E6EA', linewidth=.6)
        axes[0].set_ylabel('NPMI change from no borrowing')
        axes[1].legend(loc='best', frameon=False)
        _save(fig, paper / 'figures/figure_1_corpus_size')
        plt.close(fig)

    k, b = str(config['topics']['main_count']), str(config['retrieval']['main_budget'])
    rows = [r for r in tables['service_category_effects.csv'] if r['K'] == k and r['B'] == b
            and r['metric'] == 'ndcg' and r['comparator'] in ('nmf_original', 'similarity_matched')]
    write_csv(paper / 'source_data/figure_2_transportation_categories.csv', rows)
    # Alphabetic category order is independent of measured effects.
    names = sorted(bundle['data']['dataset_evaluation_profile']['query_categories'])
    display = {'Customer_interface': 'Customer interface', 'Personnel_attitude': 'Personnel attitude',
        'Physical_accessibility': 'Physical accessibility', 'Refusal_cancellation_refund': 'Refusal / cancellation / refund',
        'Unavailability_inadequacy': 'Unavailability / inadequacy', 'Vehicle_utility_condition': 'Vehicle utility / condition'}
    with _style(plt):
        fig, ax = plt.subplots(figsize=(7.1, 4.45), layout='constrained')
        for method, offset, marker in (('nmf_original', -.13, 'o'), ('similarity_matched', .13, 's')):
            group = {r['category']: r for r in rows if r['comparator'] == method}
            y = np.arange(len(names)) + offset
            # Draw endpoints directly: a percentile interval need not straddle its point estimate.
            low = [numeric(group[n]['ci_low']) for n in names]
            high = [numeric(group[n]['ci_high']) for n in names]
            color = COLORS['proposed'] if method == 'nmf_original' else COLORS['similarity_matched']
            ax.hlines(y, low, high, color=color, linewidth=1.2)
            label = 'Proposed minus no borrowing' if method == 'nmf_original' else 'Proposed minus matched sharing'
            ax.scatter([numeric(group[n]['difference']) for n in names], y, color=color,
                       marker=marker, s=24, label=label, zorder=3)
        ax.set_yticks(np.arange(len(names)), [display.get(n, n) + ' (n=%s)' % next(r['queries'] for r in rows if r['category'] == n) for n in names])
        ax.invert_yaxis()
        ax.axvline(0, color='#72767C', linestyle='--', linewidth=.8)
        ax.grid(axis='x', color='#E3E6EA', linewidth=.6)
        ax.set_xlabel('Category nDCG@%s difference (paired 95%% interval)' % b)
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1.01), frameon=False, fontsize=8)
        _save(fig, paper / 'figures/figure_2_transportation_categories')
        plt.close(fig)


def sensitivity_figure(bundle, supplement):
    plt = _plotting()
    rows, config = bundle['tables']['sensitivity_summary.csv'], bundle['config']
    write_csv(supplement / 'source_data/s1_parameter_sensitivity.csv', rows)
    defaults = {'downside_weight': numeric(config['method']['selection']['downside']['weight']),
                'borrowing_cap': numeric(config['method']['transfer']['max_weight'])}
    with _style(plt):
        fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.9), layout='constrained')
        for col, factor in enumerate(('downside_weight', 'borrowing_cap')):
            group = sorted((r for r in rows if r['factor'] == factor), key=lambda r: numeric(r['value']))
            for row, metric, label in ((0, 'npmi', 'NPMI'), (1, 'macro_ndcg', 'Macro nDCG@%d' % config['retrieval']['main_budget'])):
                ax = axes[row, col]
                x = [numeric(r['value']) for r in group]
                ax.errorbar(x, [numeric(r[metric + '_mean']) for r in group],
                    yerr=[numeric(r[metric + '_seed_sd']) for r in group],
                    marker='o', color=COLORS['proposed'], capsize=3, linewidth=1.4, markersize=4)
                ax.axvline(defaults[factor], linestyle='--', linewidth=.8, color='#868A90')
                ax.set_xticks(x)
                ax.set_ylabel(label)
                ax.set_xlabel('Downside weight' if factor == 'downside_weight' else 'Maximum borrowing weight')
                ax.grid(axis='y', color='#E3E6EA', linewidth=.6)
        _save(fig, supplement / 'figures/s1_parameter_sensitivity')
        plt.close(fig)
