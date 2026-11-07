"""Small publication helpers; numerical results are never recomputed by display code."""
from __future__ import annotations

import csv
import io
import math
from fractions import Fraction
from pathlib import Path
from statistics import mean, stdev

from ..runtime.artifacts import atomic_write

TOPIC_METHODS = ('nmf_original', 'nmf_similarity', 'btm', 'knntm', 'glocom', 'proposed')
CONTROL_METHODS = ('nmf_original', 'nmf_similarity', 'similarity_matched',
                   'full_pool_loo', 'no_downside', 'proposed')
ALL_TOPIC_METHODS = TOPIC_METHODS[:-1] + ('similarity_matched', 'full_pool_loo', 'no_downside', 'proposed')
LABELS = {'nmf_original': 'Original-only NMF', 'nmf_similarity': 'Similarity sharing',
          'btm': 'BTM', 'knntm': 'kNNTM', 'glocom': 'GloCOM', 'proposed': 'Proposed',
          'minilm_retrieval': 'Direct MiniLM', 'similarity_matched': 'Matched-amount sharing',
          'full_pool_loo': 'Full-pool LOO control', 'no_downside': 'Without downside penalty'}
VARIANTS = {'nmf_original': 'no_borrowing', 'nmf_similarity': 'similarity_maximum',
            'similarity_matched': 'similarity_matched', 'full_pool_loo': 'full_pool_loo',
            'no_downside': 'no_downside', 'proposed': 'proposed'}
RESULT_NAMES = ('topic_seed_results.csv', 'topic_quality_summary.csv', 'seed_stability.csv',
    'seed_stability_pairs.csv', 'retrieval_seed_results.csv', 'retrieval_summary.csv',
    'retrieval_category_results.csv', 'retrieval_contrasts.csv', 'service_category_effects.csv',
    'sensitivity_results.csv', 'sensitivity_summary.csv', 'corpus_size_results.csv',
    'corpus_size_deltas.csv', 'corpus_size_summary.csv', 'borrowing_diagnostics.csv', 'runtime_and_fits.csv')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        raise ValueError('A publication source cannot be an empty placeholder: ' + str(path))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(Path(path), stream.getvalue().encode('utf-8'), overwrite=True)


def write_text(path, text):
    atomic_write(Path(path), text.encode('utf-8'), overwrite=True)


def numeric(value):
    number = float(Fraction(value)) if isinstance(value, str) and '/' in value else float(value)
    if not math.isfinite(number):
        raise ValueError('Nonfinite publication value: ' + str(value))
    return number


def avg_sd(values):
    values = list(map(numeric, values))
    return mean(values), stdev(values) if len(values) > 1 else None


def escape(value):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
                    '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
                    '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(c, c) for c in str(value))


def number_cell(value, sd=None, bold=False):
    content = '%.4f' % numeric(value)
    if sd not in (None, ''):
        content += r'\,\pm\,' + ('%.4f' % numeric(sd))
    if bold:
        content = r'\mathbf{' + content + '}'
    return '$' + content + '$'


def one(rows, **conditions):
    found = [row for row in rows if all(str(row.get(k, '')) == str(v) for k, v in conditions.items())]
    if len(found) != 1:
        raise ValueError('Expected one publication row for %s; found %d' % (conditions, len(found)))
    return found[0]


def caption_table(path, caption, label, headers, rows, note='', wide=False):
    env = 'table*' if wide else 'table'
    content = ['\\begin{%s}[t]' % env, '\\centering', '\\small', r'\setlength{\tabcolsep}{4pt}',
               '\\caption{' + caption + '}', '\\label{' + label + '}',
               '\\begin{tabular}{l' + 'r' * (len(headers) - 1) + '}', '\\toprule',
               ' & '.join(headers) + r' \\', '\\midrule']
    content += [' & '.join(map(str, row)) + r' \\' for row in rows]
    content += ['\\bottomrule', '\\end{tabular}']
    if note:
        content += [r'\par\smallskip', r'\begin{minipage}{\linewidth}\footnotesize ' + note + r'\end{minipage}']
    content += ['\\end{%s}' % env, '']
    write_text(path, '\n'.join(content))
