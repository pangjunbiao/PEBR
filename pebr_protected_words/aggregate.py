"""Five protected-bank summaries, with adverse outcomes and correct uncertainty units."""
import statistics
from pebr_revision_step3.io import write_csv, write_json

METRICS = ('mean_post_gain', 'median_post_gain', 'token_weighted_gain', 'post_loss_fraction',
           'mean_negative_part_all_posts', 'mean_realized_mass_all_posts')

def summarize(rows, protocol, require_complete=False, include_fresh=True):
    key = lambda r: (r['corpus'], int(r['K']), int(r['outer_seed']), r['policy'])
    unique = {key(r): r for r in rows}
    expected = {(c, k, o, p) for c in protocol['corpora'] for k in protocol['topic_counts']
                for o in protocol['outer_target_bases'] for p in protocol['policies']}
    if len(unique) != len(rows) or not set(unique).issubset(expected):
        raise ValueError('Duplicate or unexpected protected-audit configuration')
    overall, contrasts = [], []
    anchor = protocol['anchor_policy']
    for c in protocol['corpora']:
        for k in protocol['topic_counts']:
            for policy in protocol['policies']:
                g = [unique.get((c, k, o, policy)) for o in protocol['outer_target_bases']]
                if any(r is None for r in g):
                    continue
                result = {'corpus': c, 'K': k, 'policy': policy, 'protected_banks': len(g),
                          'independent_corpus_replicates': False, 'confidence_interval': False}
                for metric in METRICS:
                    a = [r[metric] for r in g]
                    result[metric+'_mean'] = statistics.mean(a)
                    result[metric+'_sd_banks'] = statistics.stdev(a)
                result['eligible_posts_min'] = min(r['eligible_posts'] for r in g)
                result['eligible_posts_max'] = max(r['eligible_posts'] for r in g)
                overall.append(result)
                if policy == anchor:
                    continue
                left = [unique.get((c, k, o, anchor)) for o in protocol['outer_target_bases']]
                if any(r is None for r in left):
                    continue
                for metric in ('mean_post_gain', 'post_loss_fraction', 'mean_negative_part_all_posts'):
                    differences = []
                    for a, b in zip(left, g):
                        if a['eligible_posts'] != b['eligible_posts'] or a['protected_target_tokens'] != b['protected_target_tokens']:
                            raise ArithmeticError('Policies use different protected cohorts or targets')
                        differences.append(a[metric]-b[metric])
                    contrasts.append({'corpus': c, 'K': k, 'comparator': policy, 'metric': metric,
                        'higher_preferred': metric == 'mean_post_gain',
                        'mean_revised_minus_control': statistics.mean(differences),
                        'sd_paired_protected_bank_differences': statistics.stdev(differences),
                        'minimum_difference': min(differences), 'maximum_difference': max(differences),
                        'positive_banks': sum(d > 1e-10 for d in differences),
                        'negative_banks': sum(d < -1e-10 for d in differences),
                        'tied_banks': sum(abs(d) <= 1e-10 for d in differences),
                        'protected_banks': len(differences), 'confidence_interval': False})
    if require_complete and set(unique) != expected:
        raise ValueError('Protected-word audit is incomplete')
    result = {'coverage': {'complete': set(unique) == expected, 'bank_policy_summaries': len(rows), 'expected': len(expected)},
              'five_bank_summaries': overall, 'paired_protected_contrasts': contrasts}
    if include_fresh:
        fresh = dict(protocol, outer_target_bases=protocol['fresh_outer_target_bases'])
        values = [r for r in rows if r['outer_seed'] in protocol['fresh_outer_target_bases']]
        f = summarize(values, fresh, False, False)
        result['fresh_protected_bank_summaries'] = f['five_bank_summaries']
        result['paired_fresh_protected_contrasts'] = f['paired_protected_contrasts']
    return result

def export(directory, rows, protocol, require_complete=False):
    directory.mkdir(parents=True, exist_ok=True)
    result = summarize(rows, protocol, require_complete)
    write_json(directory/'coverage.json', result['coverage'])
    write_csv(directory/'bank_summaries.csv', rows)
    write_csv(directory/'five_bank_summaries.csv', result['five_bank_summaries'])
    write_csv(directory/'paired_protected_contrasts.csv', result['paired_protected_contrasts'])
    write_csv(directory/'fresh_protected_bank_summaries.csv', result['fresh_protected_bank_summaries'])
    write_csv(directory/'paired_fresh_protected_contrasts.csv', result['paired_fresh_protected_contrasts'])
    lines = ['# Repeated protected-word audit', '',
             'Status: '+('COMPLETE' if result['coverage']['complete'] else 'INCOMPLETE'), '',
             'Targets are reserved before retrieval/selection. All actions for a configuration are saved before protected scoring.',
             'Gain is per-post average log-probability gain in nats/token against the same outer-visible post without borrowing.',
             'Loss fraction and mean negative part use the SAME common query cohort for every policy, including zero-transfer posts.',
             'PRIMARY means/SDs summarize four FRESH protected banks (3712,4712,5712,6712); known bank 2712 is excluded.',
             'All-five-bank summaries, including known bank 2712, are secondary results in five_bank_summaries.csv.',
             'No final topic models are fitted. These SDs are not model-seed SDs or confidence intervals; protected prediction is distinct from final coherence.', '']
    for c in protocol['corpora']:
        for k in protocol['topic_counts']:
            here = [r for r in result['fresh_protected_bank_summaries'] if (r['corpus'], r['K']) == (c, k)]
            if not here:
                continue
            lines += ['## '+c+', K='+str(k), '', '| Policy | Mean gain ↑ | Posts with loss (%) ↓ | Mean negative part ↓ |',
                      '|---|---:|---:|---:|']
            for r in here:
                lines.append('| '+r['policy']+f" | {r['mean_post_gain_mean']:.5f} ± {r['mean_post_gain_sd_banks']:.5f}"
                             +f" | {100*r['post_loss_fraction_mean']:.2f} ± {100*r['post_loss_fraction_sd_banks']:.2f}"
                             +f" | {r['mean_negative_part_all_posts_mean']:.5f} ± {r['mean_negative_part_all_posts_sd_banks']:.5f} |")
            lines.append('')
    (directory/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    return result
