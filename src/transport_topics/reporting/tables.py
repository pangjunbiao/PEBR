"""Main and supplementary LaTeX tables, with exact numerical CSV companions."""
from collections import defaultdict

from .common import (ALL_TOPIC_METHODS, CONTROL_METHODS, LABELS, TOPIC_METHODS, VARIANTS,
                     avg_sd, caption_table, escape, number_cell, numeric, one, write_csv, write_text)


def dataset_rows(profile):
    prep = profile['preparation']
    rows = []
    for item in prep['conditions']:
        primary = item['name'].startswith('primary')
        rows.append({'dataset': 'primary' if primary else 'external', 'collection': item['name'],
            'source_raw_rows': prep['primary_raw_rows'] if primary else prep['external']['raw_rows'],
            'source_transport_rows': prep['primary_raw_rows'] if primary else prep['external']['transport_rows'],
            'source_unique_transport_posts': prep['primary_unique_posts'] if primary else prep['external']['normalized_unique_after_overlap'],
            'retained_posts': item['posts'], 'text_families': item['families'], 'vocabulary': item['vocabulary'],
            'mean_retained_tokens': item['mean_tokens'], 'median_retained_tokens': item['median_tokens'],
            'mask_eligible_receivers': item['banks']['topic_learning']['eligible_receivers'],
            'outer_eligible_receivers': item['banks']['outer_audit']['eligible_receivers'],
            'queries': profile['queries'] if primary else None,
            'service_categories': len(profile['query_categories']) if primary else None})
    return rows


def _range(rows, name, decimals=0):
    values = [numeric(r[name]) for r in rows]
    pattern = '%.' + str(decimals) + 'f'
    a, b = pattern % min(values), pattern % max(values)
    return a if a == b else a + '--' + b


def main_tables(bundle, paper):
    data, tables, config = bundle['data'], bundle['tables'], bundle['config']
    k, b = str(config['topics']['main_count']), str(config['retrieval']['main_budget'])
    source = paper / 'source_data'
    profile = dataset_rows(data['dataset_evaluation_profile'])
    write_csv(source / 'table_1_dataset_profile.csv', profile)
    primary, external = profile[:1], profile[1:]
    profile_rows = [('Source rows', str(primary[0]['source_raw_rows']), str(external[0]['source_raw_rows'])),
        ('Transport rows before deduplication', str(primary[0]['source_transport_rows']), str(external[0]['source_transport_rows'])),
        ('Unique transport population', str(primary[0]['source_unique_transport_posts']), str(external[0]['source_unique_transport_posts'])),
        ('Fitting collections', '1', str(len(external))),
        ('Posts per collection', _range(primary, 'retained_posts'), _range(external, 'retained_posts')),
        ('Text families', _range(primary, 'text_families'), _range(external, 'text_families')),
        ('Retained vocabulary', _range(primary, 'vocabulary'), _range(external, 'vocabulary')),
        ('Mean retained tokens', _range(primary, 'mean_retained_tokens', 2), _range(external, 'mean_retained_tokens', 2)),
        ('Mask-eligible receivers', _range(primary, 'mask_eligible_receivers'), _range(external, 'mask_eligible_receivers')),
        ('Retrieval queries / categories', '%d / %d' % (primary[0]['queries'], primary[0]['service_categories']), 'Not evaluated')]
    caption_table(paper / 'tables/table_1_dataset_profile.tex', 'Data and evaluation scope.', 'tab:dataset_profile',
                  ('Quantity', 'Indian transport', 'Dabiri traffic'), profile_rows,
                  'External ranges describe five frozen, family-preserving samples of one corpus; samples are not independent datasets. '
                  'The external transport-class filter is used only to define the evaluation population.', wide=True)

    quality = [r for r in tables['topic_quality_summary.csv'] if r['method'] in TOPIC_METHODS]
    write_csv(source / 'table_2_topic_quality.csv', quality)
    rows = []
    counts = config['topics']['counts']
    for dataset, title in (('primary', 'Indian transport'), ('external', 'Dabiri traffic')):
        rows.append((r'\multicolumn{%d}{l}{\textit{%s}}' % (len(counts) + 2, title),))
        for method in TOPIC_METHODS:
            values = [one(quality, dataset=dataset, method=method, K=count) for count in counts]
            cells = [escape(LABELS[method])]
            for count, row in zip(counts, values):
                best = max(numeric(r['npmi_mean']) for r in quality if r['dataset'] == dataset and r['K'] == str(count))
                sd = row['npmi_seed_sd'] if dataset == 'primary' else row['npmi_sample_sd']
                cells.append(number_cell(row['npmi_mean'], sd, abs(numeric(row['npmi_mean']) - best) <= 1e-12))
            cells.append(number_cell(one(quality, dataset=dataset, method=method, K=k)['diversity_mean']))
            rows.append(cells)
    headers = ['Method'] + ['$K=%d$' % count for count in counts] + ['Diversity ($K=%s$)' % k]
    caption_table(paper / 'tables/table_2_topic_quality.tex', 'Topic coherence (NPMI; higher is better) on both corpora.',
        'tab:topic_quality', headers, rows,
        r'Indian: mean $\pm$ SD across five model seeds. Dabiri: mean $\pm$ SD across five sample means, '
        r'each averaging five seeds. Bold denotes the largest mean within each corpus and topic count, without a significance claim. '
        r'Per-$K$ diversity and within-sample seed variation are provided in the supplement.', wide=True)

    retrieval = [one(tables['retrieval_summary.csv'], method=m, K='' if m == 'minilm_retrieval' else k, B=b)
                 for m in TOPIC_METHODS + ('minilm_retrieval',)]
    write_csv(source / 'table_3_transportation_retrieval.csv', retrieval)
    max_ndcg = max(numeric(r['macro_ndcg_mean']) for r in retrieval)
    max_recall = max(numeric(r['macro_recall_mean']) for r in retrieval)
    rows = [(escape(LABELS[r['method']]),
             number_cell(r['macro_ndcg_mean'], r['macro_ndcg_seed_sd'], abs(numeric(r['macro_ndcg_mean']) - max_ndcg) <= 1e-12),
             number_cell(r['macro_recall_mean'], r['macro_recall_seed_sd'], abs(numeric(r['macro_recall_mean']) - max_recall) <= 1e-12)) for r in retrieval]
    caption_table(paper / 'tables/table_3_transportation_retrieval.tex',
        'Category-relevant complaint retrieval on the Indian corpus ($K=%s$, $B=%s$).' % (k, b),
        'tab:transportation_retrieval', ('Method', 'Macro nDCG@' + b, 'Macro Recall@' + b), rows,
        r'Mean $\pm$ SD across five topic-model seeds. Direct MiniLM is evaluated once and has no topic count or seed SD. '
        'All methods use the same %d queries and %d service categories. ' % (data['dataset_evaluation_profile']['queries'], len(data['dataset_evaluation_profile']['query_categories'])) +
        'Category relevance does not establish a shared specific incident.', wide=False)

    control_rows = []
    for method in CONTROL_METHODS:
        topic = one(tables['topic_quality_summary.csv'], dataset='primary', method=method, K=k)
        app = one(tables['retrieval_summary.csv'], method=method, K=k, B=b)
        audit = one(tables['borrowing_diagnostics.csv'], study='main', job_id='primary_full', K=k,
                    scope='outer_audit', variant=VARIANTS[method])
        control_rows.append({'method': method, 'K': k, 'B': b, 'npmi_mean': topic['npmi_mean'], 'npmi_seed_sd': topic['npmi_seed_sd'],
            'macro_ndcg_mean': app['macro_ndcg_mean'], 'macro_ndcg_seed_sd': app['macro_ndcg_seed_sd'],
            'outer_mean_gain_nats_per_token': audit['mean_post_gain_nats_per_token'],
            'outer_post_loss_frequency': audit['post_loss_frequency'], 'outer_eligible_posts': audit['eligible_evaluated_posts'],
            'outer_target_tokens': audit['target_tokens']})
    write_csv(source / 'table_4_borrowing_controls.csv', control_rows)
    rows = [(escape(LABELS[r['method']]), number_cell(r['npmi_mean'], r['npmi_seed_sd']),
             number_cell(r['macro_ndcg_mean'], r['macro_ndcg_seed_sd']), number_cell(r['outer_mean_gain_nats_per_token']),
             number_cell(r['outer_post_loss_frequency'])) for r in control_rows]
    caption_table(paper / 'tables/table_4_borrowing_controls.tex',
        'Controlled borrowing comparisons on the Indian corpus ($K=%s$, $B=%s$).' % (k, b),
        'tab:borrowing_controls', ('Variant', 'NPMI', 'Macro nDCG@' + b, 'Outer gain', 'Outer loss rate'), rows,
        r'Topic and retrieval outcomes are mean $\pm$ seed SD. Outer gain is the post-averaged gain in nats/token on '
        'untouched targets, scored by the common frozen predictor; it is one audit per rule configuration, not five replications. '
        'The LOO row is a matched scoring control, not a reproduction of Atlas.', wide=True)


def _longtable(path, caption, label, headers, rows, note):
    cols = 'l' + 'r' * (len(headers) - 1)
    lines = [r'{\footnotesize', r'\setlength{\tabcolsep}{4pt}', '\\begin{longtable}{' + cols + '}',
             '\\caption{' + caption + '}\\label{' + label + r'}\\', r'\toprule',
             ' & '.join(headers) + r'\\\midrule', r'\endfirsthead', r'\toprule',
             ' & '.join(headers) + r'\\\midrule', r'\endhead', r'\bottomrule\endfoot']
    lines += [' & '.join(map(str, row)) + r'\\' for row in rows]
    lines += [r'\end{longtable}', '}', '\\noindent ' + note, '']
    write_text(path, '\n'.join(lines))


def supplementary_tables(bundle, supplement):
    tables, config = bundle['tables'], bundle['config']
    k, b = str(config['topics']['main_count']), str(config['retrieval']['main_budget'])
    summaries = tables['topic_quality_summary.csv']
    rows = []
    for dataset in ('primary', 'external'):
        for count in config['topics']['counts']:
            for method in ALL_TOPIC_METHODS:
                r = one(summaries, dataset=dataset, method=method, K=count)
                sd = 'seed_sd' if dataset == 'primary' else 'sample_sd'
                rows.append((dataset, escape(LABELS[method]), count,
                    number_cell(r['npmi_mean'], r['npmi_' + sd]),
                    number_cell(r['diversity_mean'], r['diversity_' + sd]),
                    number_cell(r['cv_mean'], r['cv_' + sd])))
    _longtable(supplement / 'tables/s1_topic_quality.tex', 'All topic-quality and sharing-control results.',
        'tab:s_all_topics', ('Corpus', 'Method', '$K$', 'NPMI', 'Diversity', '$C_v$'), rows,
        'Primary SD is over five seeds; external SD is over five sample means. Exact seed/sample values and within-sample seed SDs accompany this table as CSV.')
    rows = []
    for dataset in ('primary', 'external'):
        for count in config['topics']['counts']:
            for method in ALL_TOPIC_METHODS:
                group = [r for r in tables['seed_stability.csv'] if r['study'] == 'main' and r['method'] == method
                         and r['K'] == str(count) and r['job_id'].startswith(dataset)]
                value, sd = avg_sd(r['stability'] for r in group)
                rows.append((dataset, escape(LABELS[method]), count, number_cell(value, sd), len(group)))
    _longtable(supplement / 'tables/s2_seed_stability.tex', 'Seed stability of top-word topic sets.',
        'tab:s_stability', ('Corpus', 'Method', '$K$', 'Matched Jaccard', 'Collections'), rows,
        'Each collection averages ten seed pairs after maximum-weight topic matching. External SD describes five collection-level stability values; there is no collection-level SD for the single primary collection.')
    rows = []
    for count in config['topics']['counts']:
        for budget in config['retrieval']['budgets']:
            for method in ALL_TOPIC_METHODS:
                r = one(tables['retrieval_summary.csv'], method=method, K=count, B=budget)
                rows.append((escape(LABELS[method]), count, budget, number_cell(r['macro_ndcg_mean'], r['macro_ndcg_seed_sd']),
                             number_cell(r['macro_recall_mean'], r['macro_recall_seed_sd'])))
    for budget in config['retrieval']['budgets']:
        r = one(tables['retrieval_summary.csv'], method='minilm_retrieval', K='', B=budget)
        rows.append((escape(LABELS['minilm_retrieval']), '--', budget, number_cell(r['macro_ndcg_mean']), number_cell(r['macro_recall_mean'])))
    _longtable(supplement / 'tables/s3_retrieval.tex', 'All topic counts and document budgets for complaint retrieval.',
        'tab:s_retrieval', ('Method', '$K$', '$B$', 'Macro nDCG', 'Macro Recall'), rows,
        'All 368 common queries are retained, including any zero representations. Direct MiniLM has no artificial topic count or seed variation. The category-level CSV provides every category and seed.')
    rows = [(escape(r['factor']), '%.4g' % numeric(r['value']), number_cell(r['npmi_mean'], r['npmi_seed_sd']),
             number_cell(r['macro_ndcg_mean'], r['macro_ndcg_seed_sd']), number_cell(r['diversity_mean']))
            for r in tables['sensitivity_summary.csv']]
    _longtable(supplement / 'tables/s4_sensitivity.tex', 'Prespecified one-factor sensitivity on the primary corpus.',
        'tab:s_sensitivity', ('Factor', 'Value', 'NPMI', 'Macro nDCG@' + b, 'Diversity'), rows,
        r'$K=%s$. Other settings remain fixed. Shared default configurations reuse their saved outputs; no best setting is promoted to the primary experiment.' % k)
    rows = []
    for count in config['topics']['counts']:
        for method in CONTROL_METHODS:
            r = one(tables['borrowing_diagnostics.csv'], study='main', job_id='primary_full', K=count,
                    scope='topic_learning', variant=VARIANTS[method])
            rows.append((escape(LABELS[method]), count, number_cell(r['realized_coverage']), number_cell(r['mean_offered_beta']),
                         number_cell(r['mean_actual_beta']), number_cell(r['mean_selected_donors']), number_cell(r['mean_available_donors'])))
    _longtable(supplement / 'tables/s5_borrowing.tex', 'Primary-corpus evidence-transfer diagnostics.', 'tab:s_borrowing',
        ('Variant', '$K$', 'Coverage', 'Offered mass', 'Actual mass', 'Selected', 'Available'), rows,
        'Coverage and means use all retained receivers. Selected and available are donor counts. Full external and outer-audit diagnostics, including prediction loss rates and invariants, are supplied as CSV.')
    groups = defaultdict(list)
    for row in tables['runtime_and_fits.csv']:
        groups[row['method']].append(row)
    rows = []
    for method in ALL_TOPIC_METHODS + ('minilm_retrieval',):
        group = groups[method]
        seconds = [numeric(r['source_fit_seconds']) for r in group]
        rows.append((escape(LABELS[method]), len(group), '%.2f' % sum(seconds), '%.3f' % avg_sd(seconds)[0],
                     sum(r['device'].startswith('cuda') for r in group), sum(r['converged_nmf'] == 'True' for r in group)))
    _longtable(supplement / 'tables/s6_runtime.tex', 'Saved fit accounting over distinct model outputs.', 'tab:s_runtime',
        ('Method', 'Outputs', 'Fit seconds', 'Mean seconds', 'CUDA', 'NMF converged'), rows,
        'Counts include different main, sensitivity and corpus-size scopes across methods. Totals are recorded source-fit times, including reused fits once, '
        'and are not a matched workload speed comparison or the wall time of this resumed run. BTM and neural methods use fixed iteration/epoch budgets.')
    rows = []
    for r in tables['retrieval_contrasts.csv']:
        if r['K'] == k and r['B'] == b:
            rows.append((escape(LABELS[r['comparator']]), escape(r['metric']), number_cell(r['difference']),
                         '$[%.4f, %.4f]$' % (numeric(r['ci_low']), numeric(r['ci_high']))))
    _longtable(supplement / 'tables/s7_retrieval_contrasts.tex', 'Paired retrieval contrasts for the primary configuration.',
        'tab:s_contrasts', ('Comparator', 'Metric', 'Proposed minus comparator', r'95\% interval'), rows,
        'The 2,000 query-family bootstrap resamples operate within service categories after seed averaging. Intervals are conditional on the fitted models, are not multiplicity-adjusted, and do not incorporate retraining uncertainty.')
    included = ['s1_topic_quality', 's2_seed_stability', 's3_retrieval', 's4_sensitivity', 's5_borrowing', 's6_runtime', 's7_retrieval_contrasts']
    text = [r'\documentclass[10pt]{article}', r'\usepackage[margin=0.65in]{geometry}',
            r'\usepackage{booktabs,longtable,graphicx,amsmath}', r'\begin{document}',
            r'\title{Proposed: supplementary experimental results}', r'\author{}\date{}\maketitle',
            r'These tables reproduce the fixed experiment. Complete numerical sources are in source\_data. '
            'Topic evidence reports are descriptive model outputs, not human validation.']
    for name in included:
        if name == 's6_runtime':
            text.append(r'\clearpage')
        text.append(r'\input{tables/' + name + '.tex}')
    text += [r'\clearpage\begin{figure}[p]\centering\includegraphics[width=\linewidth]{figures/s1_parameter_sensitivity.pdf}',
             r'\caption{Prespecified parameter sensitivity. Error bars are seed SD, not confidence intervals. Dashed vertical lines mark fixed primary settings.}\end{figure}',
             r'\end{document}', '']
    write_text(supplement / 'supplement_results.tex', '\n'.join(text))
