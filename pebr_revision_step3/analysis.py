"""Five-final-seed summaries; selection repetitions are explicitly pending."""
from __future__ import annotations
import numpy as np
from transport_topics.evaluation.stability import seed_stability
from .io import read_json, write_csv, write_json

def summarize(rows, protocol, require_complete=False):
    groups = {}
    for row in rows:
        key = (row['corpus'], int(row['K']), row['policy'])
        groups.setdefault(key, []).append(row)
    summaries, paired, single, stability = [], [], [], []
    expected_seeds = protocol['final_model_seeds']
    for (corpus,k,policy), values in sorted(groups.items()):
        values.sort(key=lambda r: int(r['seed']))
        if sorted(int(r['seed']) for r in values)!=expected_seeds:
            if require_complete: raise ValueError('Incomplete or duplicate seed group: '+str((corpus,k,policy)))
            continue
        topics = {int(r['seed']): r['top_words'] for r in values}
        matching = seed_stability(topics)
        summary = {'corpus':corpus,'K':k,'policy':policy,'mask_base':'not_applicable' if policy=='nmf_original' else 2711,'final_seeds':5,
                   'selection_banks':1,'stability':matching['stability'],'stability_seed_pairs':matching['seed_pairs']}
        for metric in ('npmi','cv','diversity'):
            a=np.asarray([r[metric] for r in values],dtype=float)
            summary[metric+'_mean']=float(a.mean());summary[metric+'_sd_final_fitting']=float(a.std(ddof=1))
        summaries.append(summary)
        one=next(r for r in values if int(r['seed'])==0)
        single.append({key:one[key] for key in ('corpus','K','policy','mask_base','seed','npmi','cv','diversity')})
        stability.extend(dict(p,corpus=corpus,K=k,policy=policy,mask_base=2711) for p in matching['pairs'])
    for corpus in protocol['corpora']:
        for k in protocol['topic_counts']:
            full=groups.get((corpus,k,'proposed'),[])
            if len(full)!=5: continue
            anchor={int(r['seed']):r for r in full}
            for policy in protocol['policies']:
                if policy=='proposed':continue
                controls=groups.get((corpus,k,policy),[])
                if len(controls)!=5:continue
                for metric in ('npmi','cv','diversity'):
                    delta=np.asarray([anchor[int(r['seed'])][metric]-r[metric] for r in controls])
                    paired.append({'corpus':corpus,'K':k,'comparator':policy,'metric':metric,
                        'mean_pebr_minus_control':float(delta.mean()),
                        'sd_paired_final_seeds':float(delta.std(ddof=1)),
                        'positive_final_seeds':int(np.sum(delta>1e-10)),
                        'negative_final_seeds':int(np.sum(delta < -1e-10)),
                        'selection_banks':1,'selection_robustness_established':False})
    if require_complete:
        expected=len(protocol['corpora'])*len(protocol['topic_counts'])*len(protocol['policies'])
        if len(summaries)!=expected:raise ValueError('Default-bank summary coverage is incomplete')
    return summaries, paired, single, stability

def export(directory, rows, protocol, require_complete=False):
    summaries,paired,single,stability=summarize(rows,protocol,require_complete)
    directory.mkdir(parents=True,exist_ok=True)
    flat=[{k:v for k,v in row.items() if k!='top_words'} for row in rows]
    write_csv(directory/'topic_seed_results.csv',flat)
    write_csv(directory/'multi_seed_summary.csv',summaries)
    write_csv(directory/'single_seed0_results.csv',single)
    write_csv(directory/'paired_pebr_contrasts.csv',paired)
    write_csv(directory/'stability_pairs.csv',stability)
    write_json(directory/'topic_words.json',rows)
    lines=['# Default-bank policy comparison','',
        'Mask base 2711; final NMF seeds 0–4. Means ± sample SD describe final fitting with one selection bank.',
        'Stability is the ten-pair aggregate; it is undefined for one final seed.',
        'These tables do not establish robustness across selection masks or independent collections.','']
    for corpus in protocol['corpora']:
        for k in protocol['topic_counts']:
            group=[r for r in summaries if r['corpus']==corpus and r['K']==k]
            if not group:continue
            lines.extend(['## '+corpus+', K='+str(k),'',
                '| Policy | NPMI | C_v | TD | Stability |','|---|---:|---:|---:|---:|'])
            for policy in protocol['policies']:
                row=next((r for r in group if r['policy']==policy),None)
                if row is None:continue
                cells=[f"{row[m+'_mean']:.4f} ± {row[m+'_sd_final_fitting']:.4f}" for m in ('npmi','cv','diversity')]
                lines.append('| '+policy+' | '+' | '.join(cells)+f" | {row['stability']:.4f} |")
            lines.append('')
    (directory/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    return summaries
