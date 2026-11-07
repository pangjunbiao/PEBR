"""Assemble every declared development outcome and verify table arithmetic."""
from pathlib import Path
import csv,json,hashlib,math
import numpy as np


ROOT=Path(__file__).resolve().parent
OUT=ROOT/'final_report'
OUT.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def table(p,rows):
    with Path(p).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


assert read(ROOT/'run_v1/TRIAL_COMPLETE.json')['new_final_fits']==150
assert read(ROOT/'forward_controls_v1/CONTROLS_COMPLETE.json')['fresh_final_fits']==150
assert read(ROOT/'fixed_controls_v1/CONTROLS_COMPLETE.json')['evaluated_seed_conditions']==150
assert read(ROOT/'prefix_controls_v1/PREFIX_COMPLETE.json')['evaluated_seed_conditions']==60
repro=read(ROOT/'run_v1/baseline_reproduction.json');assert repro['passed'] and len(repro['checks'])==28
kernel=read(ROOT/'candidate_replay/CANDIDATE_COMPLETE.json')
assert len(kernel['cases'])==6 and all(x['transfer_matrix_identical'] for x in kernel['cases'])
old=read(ROOT/'inputs/step3/results/topic_words.json')
fresh=[]
for name in ('run_v1','forward_controls_v1','fixed_controls_v1','prefix_controls_v1'):
    rows=read(ROOT/name/'results/fresh_seed_results.json')
    assert len(rows)==(60 if name=='prefix_controls_v1' else 150)
    fresh.extend(rows)
allrows=old+fresh;assert len(allrows)==810
idx={(r['corpus'],int(r['K']),r['policy'],int(r['seed'])):r for r in allrows}
assert len(idx)==810
for r in allrows:
    assert len(r['top_words'])==r['K'] and all(len(x)==10 and len(set(x))==10 for x in r['top_words'])
    diversity=len({w for words in r['top_words'] for w in words})/(10*r['K'])
    assert abs(diversity-r['diversity'])<1e-12
    assert all(math.isfinite(r[m]) for m in ('npmi','cv','diversity'))
summary={}
for name in ('run_v1','forward_controls_v1','fixed_controls_v1','prefix_controls_v1'):
    for r in read(ROOT/name/'results/all_summary.json'):
        key=(r['corpus'],int(r['K']),r['policy'])
        if key in summary:assert summary[key]==r
        summary[key]=r
assert len(summary)==162
for key,s in summary.items():
    rows=[idx[key+(seed,)] for seed in range(5)]
    for m in ('npmi','cv','diversity'):
        a=np.asarray([r[m] for r in rows])
        assert abs(float(a.mean())-s[m+'_mean'])<1e-12
        assert abs(float(a.std(ddof=1))-s[m+'_sd'])<1e-12
candidate='preserve_forward_fixed_cap'
keys=[(c,k) for c in ('transport_grievance','chinese_pooled') for k in (8,10,12)]
compare=[];contrasts=[]
comparators=['nmf_original','nmf_similarity','similarity_matched','full_pool_loo','proposed',
    'predictively_scored_similarity_prefixes','preserve_forward','matched_preserve_fixed',
    'weighted_hidden_fixed','preserve_fixed_no_downside','preserve_fixed_individual','preserve_fixed_single_view',
    'preserve_fixed_prefix']
wins={p:[] for p in comparators}
for c,k in keys:
    new=summary[c,k,candidate];previous=summary[c,k,'proposed']
    compare.append({'corpus':c,'K':k,'old_npmi':previous['npmi_mean'],'new_npmi':new['npmi_mean'],
        'new_npmi_sd':new['npmi_sd'],'old_cv':previous['cv_mean'],'new_cv':new['cv_mean'],
        'new_cv_sd':new['cv_sd'],'old_TD':previous['diversity_mean'],'new_TD':new['diversity_mean'],
        'new_TD_sd':new['diversity_sd'],'old_stability':previous['stability'],'new_stability':new['stability']})
    for p in comparators:
        other=summary[c,k,p]
        if all(new[m+'_mean']>other[m+'_mean']+1e-12 for m in ('npmi','cv')):wins[p].append([c,k])
        delta={m:np.asarray([idx[c,k,candidate,s][m]-idx[c,k,p,s][m] for s in range(5)]) for m in ('npmi','cv','diversity')}
        row={'corpus':c,'K':k,'comparator':p,'paired_seeds_higher_both':int(np.sum((delta['npmi']>1e-10)&(delta['cv']>1e-10))),
            'selection_banks':1}
        for m,a in delta.items():row.update({m+'_delta_mean':float(a.mean()),m+'_delta_sd':float(a.std(ddof=1)),m+'_positive_seeds':int(np.sum(a>1e-10))})
        contrasts.append(row)
quantity=[]
for c,k in keys:
    a=read(ROOT/'forward_controls_v1/selection'/c/('K%d'%k)/candidate/'actions.json')
    b=read(ROOT/'fixed_controls_v1/selection'/c/('K%d'%k)/'matched_preserve_fixed/actions.json')
    assert len(a)==len(b)
    mass=max(abs(x['actual_beta']-y['actual_beta']) for x,y in zip(a,b))
    count=sum(len(x['available_donors'])!=len(y['available_donors']) for x,y in zip(a,b))
    assert mass<1e-12 and count==0
    quantity.append({'corpus':c,'K':k,'posts':len(a),'matched_mass_max_error':mass,'matched_count_violations':count,
        'mean_actual_beta':float(np.mean([x['actual_beta'] for x in a])),
        'borrowing_coverage':float(np.mean([x['actual_beta']>0 for x in a])),
        'positive_offers':sum(x['offered_beta']>0 for x in a),
        'positive_offers_at_cap':sum(x['offered_beta']>0 and abs(x['offered_beta']-1/3)<1e-12 for x in a)})
manifest=read(ROOT/'inputs/export_manifest.json')
for rec in manifest['files']:
    p=ROOT/'inputs'/rec['member']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==rec['sha256']
write(OUT/'validation.json',{'input_files_reverified':389,'baseline_fits_reproduced':28,
    'baseline_max_coherence_absolute_error':max(max(x['metric_absolute_errors'].values()) for x in repro['checks']),
    'fresh_development_fit_conditions':len(fresh),'fresh_fits_computed':sum(r['source']=='fresh_focused_development' for r in fresh),
    'declared_development_groups':102,'old_step3_groups':60,'all_declared_conditions_retained':True,
    'all_810_TD_values_checked':True,'all_summary_arithmetic_checked':True,
    'matched_new_candidate_mass_count_checks':quantity,'standalone_kernel_equal_cases':kernel['cases'],
    'selection_banks_per_configuration':1,'independent_selection_validation_complete':False,
    'development_choices_used_known_outcomes':True,'guaranteed_journal_rating':False})
write(OUT/'candidate_vs_old.json',compare);table(OUT/'candidate_vs_old.csv',compare)
write(OUT/'paired_contrasts.json',contrasts);table(OUT/'paired_contrasts.csv',contrasts)
write(OUT/'all_policy_summaries.json',list(summary.values()));table(OUT/'all_policy_summaries.csv',list(summary.values()))
write(OUT/'all_seed_results.json',allrows)
write(OUT/'win_counts.json',{p:{'higher_both_settings':len(v),'settings':v} for p,v in wins.items()})
write(OUT/'revised_candidate_seed_results.json',[idx[c,k,candidate,s] for c,k in keys for s in range(5)])
for c,k in keys:
    rows=read(ROOT/'forward_controls_v1/selection'/c/('K%d'%k)/candidate/'actions.json')
    p=OUT/'expected_actions'/c/('K%d.json'%k);p.parent.mkdir(parents=True,exist_ok=True)
    write(p,[{name:r[name] for name in ('donors','offered_beta','actual_beta','available_donors')} for r in rows])
lines=['# Focused PEBR rule revision: actual-corpus development results','',
    'The export succeeded. All 389 numerical files were verified. The 28 baseline fits reproduced with identical descriptor lists.',
    'The study computed 510 new development fits: two corpora, three topic counts, seventeen candidate/control policies, five final seeds.',
    'These are conditional on one selection bank (mask base 2711), not independent mask/corpus replications. All declared settings are retained.','',
    '## Recommended candidate to validate next','',
    'Receiver-preserving predictive selection with the existing forward search and a fixed positive offered cap of 1/3.',
    'The score rewards token-weighted hidden-word benefit and charges token-weighted visible-word prediction loss.',
    'The additional donor-swap search is not adopted. Searching offered borrowing weights is not retained in this candidate.',
    'Reference fitting, retrieval, fold/family exclusions, transfer fallback and final NMF remain unchanged.','',
    '## Comparison with original PEBR','',
    '| Collection | K | Original NPMI | Revised NPMI | Original C_v | Revised C_v | Original stability | Revised stability |',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for r in compare:
    lines.append('| '+str(r['corpus'])+' | '+str(r['K'])+' | '+' | '.join(f"{r[m]:.4f}" for m in ('old_npmi','new_npmi','old_cv','new_cv','old_stability','new_stability'))+' |')
lines+=['','## Evidence against simpler controls','',
    'Higher below means higher mean NPMI AND C_v. These counts do not establish statistical significance.','',
    '| Comparator | Higher in both coherence measures |','|---|---:|']
labels={'nmf_original':'NMF without borrowing','nmf_similarity':'Ordinary similarity sharing',
    'similarity_matched':'Original rule matched-amount control (different quantity reference)',
    'matched_preserve_fixed':'NEW matched-amount control for the revised candidate',
    'full_pool_loo':'Full-pool LOO','proposed':'Original PEBR',
    'weighted_hidden_fixed':'Same token weighting, without visible-word loss charge',
    'preserve_fixed_no_downside':'Revised candidate without downside penalty',
    'preserve_fixed_individual':'Revised candidate with individual-donor scoring',
    'preserve_fixed_single_view':'Revised candidate with single-view selection',
    'preserve_forward':'Receiver-preserving forward rule with borrowing-weight search',
    'predictively_scored_similarity_prefixes':'Original predictively scored similarity prefixes',
    'preserve_fixed_prefix':'Similarity-ranked prefixes under the SAME revised score and fixed cap'}
for p in comparators:lines.append('| '+labels[p]+' | '+str(len(wins[p]))+'/6 |')
lines+=['','## Limits and decision','',
    'This is a useful development improvement, not a completed confirmation study. New matched controls match each receiver’s realized mass and contributing donor count; all checks passed.',
    'The revised candidate improves both mean coherence measures over original PEBR in four settings. Chinese K=10 is mixed and Chinese K=12 is lower on both measures.',
    'Chinese K=12 stability also decreases substantially. The candidate is not uniformly superior, and higher coherence does not establish better service retrieval or analyst performance.',
    'Independent masking/selection repeats and a protected-word evaluation of the frozen revised rule remain necessary before making reliability claims.',
    'Do not replace manuscript numbers selectively or combine revised-method results with old-method component, sensitivity or corpus-size figures. A method change requires version-consistent reporting.',
    'The practical next step is to validate this simpler revised candidate. The old unchanged Step 4 repetition package should remain paused.','',
    'All numerical summaries, paired seed effects, borrowing checks and unfavorable comparisons are included in the accompanying CSV/JSON files.']
(OUT/'RESULTS_READ_FIRST.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'comparison':compare,'wins':{p:len(v) for p,v in wins.items()},'fresh_development_fits':len(fresh)},indent=2))
