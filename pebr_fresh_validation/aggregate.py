"""Fresh-only bank summaries; never pool the inspected development bank with validation."""
import statistics
from collections import defaultdict
from transport_topics.evaluation.stability import seed_stability
from pebr_revision_step3.io import write_csv, write_json
from .protocol import ANCHOR

METRICS = ('npmi', 'cv', 'diversity')

def row_key(row):
    return row['corpus'], int(row['K']), row['policy'], str(row['mask_base']), int(row['seed'])

def summarize(rows, protocol, require_complete=False):
    unique = {row_key(r): r for r in rows}
    if len(unique) != len(rows):
        raise ValueError('Duplicate fitted-seed record')
    expected = {(c, k, p, 'not_applicable' if p == 'nmf_original' else str(m), s)
                for c in protocol['corpora'] for k in protocol['topic_counts'] for p in protocol['policies']
                for m in (['not_applicable'] if p == 'nmf_original' else protocol['selection_mask_bases'])
                for s in protocol['final_model_seeds']}
    if not set(unique).issubset(expected):
        raise ValueError('Unexpected fit, topic count, policy, seed or development mask in validation')
    groups = defaultdict(list)
    for key, r in unique.items():
        groups[key[:-1]].append(r)
    def group(c, k, p, mask):
        g = groups.get((c, k, p, 'not_applicable' if p == 'nmf_original' else str(mask)), [])
        return g if sorted(int(r['seed']) for r in g) == protocol['final_model_seeds'] else None
    banks, seed_pairs, bank_pairs, hierarchical, contrasts, matching_pairs = [], [], [], [], [], []
    masks = protocol['selection_mask_bases']
    for c in protocol['corpora']:
        for k in protocol['topic_counts']:
            for mask in masks:
                for p in protocol['policies']:
                    g = group(c, k, p, mask)
                    if g is None:
                        continue
                    row = {'corpus': c, 'K': k, 'policy': p, 'paired_context_mask_base': mask,
                           'baseline_reused': p == 'nmf_original', 'final_seeds': len(g)}
                    for metric in METRICS:
                        a = [float(r[metric]) for r in g]
                        row[metric+'_mean'] = statistics.mean(a)
                        row[metric+'_sd_final_seeds'] = statistics.stdev(a)
                    matching = seed_stability({int(r['seed']): r['top_words'] for r in g})
                    row['stability'] = matching['stability']
                    banks.append(row)
                    if p != 'nmf_original' or mask == masks[0]:
                        matching_pairs.extend(dict(x, corpus=c, K=k, policy=p,
                              mask_base='not_applicable' if p == 'nmf_original' else mask) for x in matching['pairs'])
                anchor = group(c, k, ANCHOR, mask)
                if anchor is None:
                    continue
                a = {int(r['seed']): r for r in anchor}
                for p in protocol['policies']:
                    if p == ANCHOR:
                        continue
                    g = group(c, k, p, mask)
                    if g is None:
                        continue
                    for metric in METRICS:
                        diffs = []
                        for r in g:
                            s = int(r['seed'])
                            d = float(a[s][metric])-float(r[metric])
                            diffs.append(d)
                            seed_pairs.append({'corpus': c, 'K': k, 'comparator': p, 'mask_base': mask,
                                               'final_seed': s, 'metric': metric, 'revised_minus_control': d})
                        bank_pairs.append({'corpus': c, 'K': k, 'comparator': p, 'mask_base': mask,
                                           'metric': metric, 'mean_paired_seed_difference': statistics.mean(diffs),
                                           'sd_paired_final_seeds': statistics.stdev(diffs)})
                    left = seed_stability({s: r['top_words'] for s, r in a.items()})['stability']
                    right = seed_stability({int(r['seed']): r['top_words'] for r in g})['stability']
                    bank_pairs.append({'corpus': c, 'K': k, 'comparator': p, 'mask_base': mask,
                                       'metric': 'stability', 'mean_paired_seed_difference': left-right,
                                       'sd_paired_final_seeds': None})
            for p in protocol['policies']:
                values = [r for r in banks if (r['corpus'], r['K'], r['policy']) == (c, k, p)]
                if len(values) != len(masks):
                    continue
                row = {'corpus': c, 'K': k, 'policy': p,
                       'selection_banks': 0 if p == 'nmf_original' else len(masks),
                       'unique_final_fit_records': 5 if p == 'nmf_original' else 5*len(masks),
                       'development_bank_included': False}
                for metric in METRICS:
                    a = [r[metric+'_mean'] for r in values]
                    row[metric+'_mean'] = statistics.mean(a)
                    row[metric+'_sd_bank_means'] = None if p == 'nmf_original' else statistics.stdev(a)
                    row[metric+'_mean_within_bank_fit_sd'] = statistics.mean(r[metric+'_sd_final_seeds'] for r in values)
                row['stability_mean'] = statistics.mean(r['stability'] for r in values)
                row['stability_sd_banks'] = None if p == 'nmf_original' else statistics.stdev(r['stability'] for r in values)
                hierarchical.append(row)
            for p in protocol['policies']:
                if p == ANCHOR:
                    continue
                for metric in METRICS+('stability',):
                    a = [r['mean_paired_seed_difference'] for r in bank_pairs if
                         (r['corpus'], r['K'], r['comparator'], r['metric']) == (c, k, p, metric)]
                    if len(a) != len(masks):
                        continue
                    contrasts.append({'corpus': c, 'K': k, 'comparator': p, 'metric': metric,
                                      'mean_revised_minus_control': statistics.mean(a),
                                      'sd_across_fresh_bank_means': statistics.stdev(a),
                                      'minimum_bank_difference': min(a), 'maximum_bank_difference': max(a),
                                      'positive_banks': sum(d > 1e-10 for d in a),
                                      'negative_banks': sum(d < -1e-10 for d in a),
                                      'tied_banks': sum(abs(d) <= 1e-10 for d in a),
                                      'fresh_selection_banks': len(masks),
                                      'independent_corpus_replicates': False, 'confidence_interval': False})
    if require_complete and set(unique) != expected:
        raise ValueError('Fresh validation is incomplete: missing %d fitted-seed records' % len(expected-set(unique)))
    return {'bank_summaries': banks, 'fresh_bank_summaries': hierarchical,
            'paired_seed_contrasts': seed_pairs, 'paired_bank_contrasts': bank_pairs,
            'paired_fresh_bank_contrasts': contrasts, 'stability_pairs': matching_pairs,
            'coverage': {'unique_fit_records': len(rows), 'expected': len(expected),
                         'complete': set(unique) == expected}}

def export(directory, rows, protocol, require_complete=False):
    directory.mkdir(parents=True, exist_ok=True)
    result = summarize(rows, protocol, require_complete)
    write_csv(directory/'unique_fit_results.csv', [{k: v for k, v in r.items() if k != 'top_words'} for r in rows])
    write_csv(directory/'single_seed0_by_bank.csv', [{k: v for k, v in r.items() if k != 'top_words'} for r in rows if int(r['seed']) == 0])
    write_json(directory/'topic_words.json', rows)
    for name, values in result.items():
        if name != 'coverage':
            write_csv(directory/(name+'.csv'), values)
    write_json(directory/'coverage.json', result['coverage'])
    lines = ['# Fresh-selection validation', '', 'Status: '+('COMPLETE' if result['coverage']['complete'] else 'INCOMPLETE'), '',
             'Only four NEW mask banks are summarized here; mask 2711 was used in development and is excluded.',
             'Average final seeds 0–4 inside each bank, then summarize the four bank means with sample SD.',
             'NMF without borrowing is shared across banks; its displayed SD is instead across its five final seeds (*).',
             'These banks vary masking/retrieval within the SAME corpora and fixed folds/predictors. They are not new corpora, held-out semantic tests or confidence intervals.',
             'Original PEBR is the policy named proposed; revised PEBR is preserve_forward_fixed_cap. All unfavorable outcomes remain.', '']
    for c in protocol['corpora']:
        for k in protocol['topic_counts']:
            rs = [r for r in result['fresh_bank_summaries'] if (r['corpus'], r['K']) == (c, k)]
            if not rs:
                continue
            lines += ['## '+c+', K='+str(k), '', '| Policy | NPMI | C_v | TD | Stability |', '|---|---:|---:|---:|---:|']
            for p in protocol['policies']:
                r = next((r for r in rs if r['policy'] == p), None)
                if r is None:
                    continue
                cells = []
                for m in METRICS:
                    sd = r[m+'_mean_within_bank_fit_sd'] if p == 'nmf_original' else r[m+'_sd_bank_means']
                    cells.append(f"{r[m+'_mean']:.4f} ± {sd:.4f}"+('*' if p == 'nmf_original' else ''))
                lines.append('| '+p+' | '+' | '.join(cells)+f" | {r['stability_mean']:.4f} |")
            lines.append('')
    (directory/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    return result
