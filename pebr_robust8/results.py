"""Separate final-fit variation from reference/fold perturbations."""
from collections import defaultdict
import numpy as np
from .common import write, table
from .design import ANCHOR

def record_key(row):
    return row['corpus'],int(row['K']),row['case'],row['policy'],int(row['seed'])

def export(directory, rows, protocol, require_complete=False):
    from transport_topics.evaluation.stability import seed_stability
    directory.mkdir(parents=True,exist_ok=True)
    expected={(c,k,case['case'],p,s) for c in protocol['corpora'] for k in protocol['topic_counts']
              for case in protocol['cases'] for p in protocol['policies'] for s in protocol['final_model_seeds']}
    actual={record_key(r) for r in rows}
    if len(actual)!=len(rows) or not actual.issubset(expected):
        raise ValueError('Duplicate or undeclared sensitivity case-fit record')
    complete=actual==expected
    write(directory/'coverage.json',{'case_fit_records':len(rows),'expected':len(expected),'complete':complete,
        'new_final_fits_planned':420,'reused_case_fit_records_planned':330,'reused_unique_final_fits':150,
        'case_records_are_not_all_independent_model_fits':True})
    table(directory/'seed_metrics.csv',[{k:v for k,v in r.items() if k!='top_words'} for r in rows])
    write(directory/'topic_words.json',rows)
    groups=defaultdict(list)
    for r in rows:groups[(r['corpus'],int(r['K']),r['case'],r['policy'])].append(r)
    summaries=[];contrasts=[]
    for group,rr in sorted(groups.items()):
        if len(rr)!=5:continue
        corpus,k,case,policy=group
        item={'corpus':corpus,'K':k,'case':case,'axis':rr[0]['axis'],'reference_seed':rr[0]['reference_seed'],
              'fold_seed':rr[0]['fold_seed'],'policy':policy,'final_seeds':5}
        for metric in ('npmi','cv','diversity'):
            vals=[r[metric] for r in rr]
            item[metric+'_mean']=float(np.mean(vals));item[metric+'_sample_sd']=float(np.std(vals,ddof=1))
        item['stability']=seed_stability({r['seed']:r['top_words'] for r in rr})['stability']
        summaries.append(item)
        if policy!=ANCHOR:continue
        paired={int(r['seed']):r for r in rr}
        for comparator in protocol['policies']:
            if comparator==ANCHOR:continue
            other=groups.get((corpus,k,case,comparator),[])
            if len(other)!=5:continue
            comparison={int(r['seed']):r for r in other}
            for metric in ('npmi','cv','diversity'):
                delta=np.asarray([paired[s][metric]-comparison[s][metric] for s in range(5)])
                contrasts.append({'corpus':corpus,'K':k,'case':case,'axis':rr[0]['axis'],
                    'reference_seed':rr[0]['reference_seed'],'fold_seed':rr[0]['fold_seed'],
                    'comparator':comparator,'metric':metric,'mean_paired_difference':float(delta.mean()),
                    'sample_sd_across_five_paired_final_fits':float(delta.std(ddof=1)),
                    'positive_final_fit_pairs':int(np.sum(delta>1e-12)),
                    'negative_final_fit_pairs':int(np.sum(delta < -1e-12)),
                    'zero_final_fit_pairs':int(np.sum(np.abs(delta)<=1e-12)),
                    'significance_test_performed':False})
    table(directory/'case_summaries.csv',summaries)
    table(directory/'paired_effects.csv',contrasts)
    lines=['# Reference/fold sensitivity results','',
        'Same original corpora, vocabulary, mask 3711 and final seeds 0–4. No rule changes or tuning.',
        'Default uses reference seed 0 and fold seed 2710. r1/r2 change only reference initialization; f3710/f4710 change family folds, eligible donors and fold-excluded reference fitting.',
        'Case means average five final fits. Their SD describes final fitting only. The sensitivity levels are displayed separately; shared baseline fits are not independent replicates.','',
        '| Corpus | K | Case | PEBR NPMI | PEBR C_v | NPMI difference vs matched | C_v difference vs matched | NPMI difference vs prefix | C_v difference vs prefix |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in summaries:
        if row['policy']!=ANCHOR:continue
        def effect(comparator,metric):
            found=[x for x in contrasts if (x['corpus'],x['K'],x['case'],x['comparator'],x['metric'])==
                   (row['corpus'],row['K'],row['case'],comparator,metric)]
            return ('%.6f'%found[0]['mean_paired_difference']) if found else 'pending'
        lines.append('| %s | %d | %s | %.6f | %.6f | %s | %s | %s | %s |'%(row['corpus'],row['K'],row['case'],
            row['npmi_mean'],row['cv_mean'],effect('matched_preserve_fixed','npmi'),effect('matched_preserve_fixed','cv'),
            effect('preserve_fixed_prefix','npmi'),effect('preserve_fixed_prefix','cv')))
    lines+=['','All reversals are retained. Results are conditional on the tested collections and mask bank; these are not independent-corpus uncertainty estimates.']
    (directory/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if require_complete and not complete:
        raise ValueError('The sensitivity study has missing case-fit records')
    return {'complete':complete,'summaries':summaries,'contrasts':contrasts}
