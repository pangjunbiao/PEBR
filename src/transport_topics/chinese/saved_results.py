"""Reconcile and export the actual completed study, without refitting or selecting runs."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import json
import math
import statistics

from ..runtime.artifacts import atomic_write, file_hash, write_json

BASELINES = ('nmf_original','nmf_similarity','btm','knntm','glocom','proposed')
CONTROLS = ('nmf_original','nmf_similarity','similarity_matched','full_pool_loo','no_downside','proposed')
LABELS = {'nmf_original':'Original-only NMF','nmf_similarity':'Similarity sharing',
          'btm':'BTM','knntm':'kNNTM','glocom':'GloCOM','proposed':'Proposed',
          'similarity_matched':'Matched-amount sharing','full_pool_loo':'Full-pool LOO control',
          'no_downside':'Without downside penalty'}


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    import io
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_write(path,('\ufeff'+stream.getvalue()).encode('utf-8'),overwrite=True)


def close(left,right):
    return math.isclose(float(left),float(right),rel_tol=1e-10,abs_tol=1e-12)


def verify_tables(tables):
    main=[r for r in tables['topic_seed_results.csv'] if r['study']=='main']
    seen=set()
    for r in main:
        key=(r['condition'],r['method'],int(r['K']),int(r['seed']))
        if key in seen:raise ValueError('Duplicate archived main result: '+str(key))
        seen.add(key)
        if r['development_only'].lower()!='false':raise ValueError('Development output in main study')
    for summary in tables['topic_quality_summary.csv']:
        group=[r for r in main if (r['dataset'],r['method'],r['K'])==
               (summary['dataset'],summary['method'],summary['K'])]
        by_collection=defaultdict(list)
        for r in group:by_collection[r['condition']].append(r)
        if len(by_collection)!=int(summary['collections']):raise ValueError('Collection count mismatch')
        for values in by_collection.values():
            if sorted(int(r['seed']) for r in values)!=[0,1,2,3,4]:
                raise ValueError('The archived study must retain all five seeds')
        for metric in ('npmi','cv','diversity'):
            means=[statistics.mean(float(r[metric]) for r in values) for values in by_collection.values()]
            if not close(statistics.mean(means),summary[metric+'_mean']):
                raise ValueError('Saved aggregate does not reconcile: '+str(summary))
            if len(means)>1:
                if not close(statistics.stdev(means),summary[metric+'_sample_sd']):
                    raise ValueError('External sample SD mismatch')
            else:
                sd=statistics.stdev(float(r[metric]) for r in group)
                if not close(sd,summary[metric+'_seed_sd']):raise ValueError('Primary seed SD mismatch')
    for row in tables['seed_stability.csv']:
        pairs=[p for p in tables['seed_stability_pairs.csv']
               if all(p[k]==row[k] for k in ('study','job_id','method','K'))]
        if len(pairs)!=10 or len(pairs)!=int(row['seed_pairs']):raise ValueError('Missing seed pair')
        if not close(statistics.mean(float(p['matched_jaccard']) for p in pairs),row['stability']):
            raise ValueError('Stability does not reconcile with its ten seed pairs')
    return {'main_seed_rows_verified':len(main),'quality_summaries_verified':len(tables['topic_quality_summary.csv']),
            'stability_values_verified':len(tables['seed_stability.csv']),
            'all_five_model_seeds_retained':True,'new_fits':0}


def form_rows(tables, dataset, topics, methods, sample=None, seed=None):
    main=[r for r in tables['topic_seed_results.csv'] if r['study']=='main'
          and r['dataset']==dataset and int(r['K'])==topics
          and (sample is None or r['sample_index']==str(sample))
          and (seed is None or int(r['seed'])==seed)]
    result=[]
    for method in methods:
        values=[r for r in main if r['method']==method]
        if not values:raise ValueError('No actual saved values for '+str((dataset,topics,method,sample,seed)))
        by_condition=defaultdict(list)
        for r in values:by_condition[r['condition']].append(r)
        row={'Model':LABELS[method]}
        for metric,display in [('npmi','NPMI'),('cv','C_v'),('diversity','TD')]:
            if seed is None and sample is None:
                # Preserve the canonical saved aggregate, including rounding at display ties.
                saved=next(r for r in tables['topic_quality_summary.csv'] if
                           (r['dataset'],r['method'],int(r['K']))==(dataset,method,topics))
                row[display]=float(saved[metric+'_mean'])
                sd_key=metric+('_sample_sd' if int(saved['collections'])>1 else '_seed_sd')
                row[display+'_sd']=float(saved[sd_key])
            elif seed is not None:
                numbers=[float(r[metric]) for r in values]
                row[display]=statistics.mean(numbers)
                row[display+'_sd']=statistics.stdev(numbers) if len(numbers)>1 else None
            else:
                means=[statistics.mean(float(r[metric]) for r in g) for g in by_condition.values()]
                row[display]=statistics.mean(means)
                row[display+'_sd']=(statistics.stdev(means) if len(means)>1
                                    else statistics.stdev(float(r[metric]) for r in values))
        if seed is not None:
            row.update(Stability=None,Stability_sd=None)
        else:
            stable=[float(r['stability']) for r in tables['seed_stability.csv']
                    if r['study']=='main' and r['job_id'] in by_condition and r['method']==method and int(r['K'])==topics]
            if len(stable)!=len(by_condition):raise ValueError('Missing collection-level stability')
            row.update(Stability=statistics.mean(stable),
                       Stability_sd=statistics.stdev(stable) if len(stable)>1 else None)
        result.append(row)
    return result


def display_number(row,key,latex=False):
    value=row[key]
    if value is None:return r'---' if latex else '—'
    text=f'{value:.4f}'
    sd=row.get(key+'_sd')
    if sd is not None:text+=(r'\,\pm\,' if latex else ' ± ')+f'{sd:.4f}'
    return text


def export_table(directory,name,rows,caption,note):
    keys=['NPMI','C_v','TD','Stability']
    write_csv(directory/(name+'.csv'),rows)
    maxima={key:max(r[key] for r in rows if r[key] is not None)
            for key in keys if any(r[key] is not None for r in rows)}
    md=['# '+caption,'','| Model | NPMI ↑ | C_v ↑ | TD ↑ | Stability ↑ |',
        '|---|---:|---:|---:|---:|']
    tex=[r'\begin{table*}[t]',r'\centering',r'\small',r'\setlength{\tabcolsep}{4pt}',
         r'\caption{'+caption+'}',r'\label{tab:'+name+'}',r'\begin{tabular}{lrrrr}',r'\toprule',
         r'Model & NPMI $\uparrow$ & $C_v$ $\uparrow$ & TD $\uparrow$ & Stability $\uparrow$ \\',r'\midrule']
    for row in rows:
        cells=[];latex_cells=[]
        for key in keys:
            value=display_number(row,key)
            latex=display_number(row,key,True)
            best=row[key] is not None and close(row[key],maxima[key])
            cells.append('**'+value+'**' if best else value)
            latex_cells.append('$'+(r'\mathbf{'+latex+'}' if best else latex)+'$' if row[key] is not None else latex)
        md.append('| '+' | '.join([row['Model']]+cells)+' |')
        tex.append(' & '.join([row['Model']]+latex_cells)+r' \\')
    md+=['',note,'']
    tex += [r'\bottomrule',r'\end{tabular}',r'\par\smallskip',
            r'\begin{minipage}{0.98\textwidth}\footnotesize '+note+r'\end{minipage}',r'\end{table*}','']
    atomic_write(directory/(name+'.md'),'\n'.join(md).encode('utf-8'),overwrite=True)
    atomic_write(directory/(name+'.tex'),'\n'.join(tex).encode('utf-8'),overwrite=True)


def export_requested(root, config, output, logger):
    source_run=config['source_run_id']
    manifest_path=root/config['snapshot_directory']/'snapshot_manifest.json'
    expected=json.loads(manifest_path.read_text(encoding='utf-8'))
    local=root/'outputs/runs'/source_run/'results'
    folder=local if (local/'topic_seed_results.csv').is_file() else root/config['snapshot_directory']
    tables={}
    for name,checksum in expected['csv_sha256'].items():
        path=folder/name
        if file_hash(path)!=checksum:raise ValueError('The completed study snapshot changed: '+str(path))
        tables[name]=read_csv(path)
    audit=verify_tables(tables)
    output.mkdir(parents=True,exist_ok=True)
    source_note='Values are the actual saved completed-study results. Bold indicates the largest mean among the displayed rows.'
    primary_note='Mean plus/minus sample standard deviation over model seeds 0--4. Stability averages the ten matched topic-set Jaccard comparisons between those five seeds.'
    external_note='Five fixed samples, each fitted with model seeds 0--4. Quality values are the mean and standard deviation of the five sample means. Stability is the mean and standard deviation of five within-sample stability values.'
    single_note='One actual fit: fixed sample 0 and seed 0. Seed stability requires multiple fitted seeds and is therefore undefined for this table.'
    specs=[
        ('indian_K12_main_5seeds','primary',12,BASELINES,None,None,'Indian transport: main comparison at $K=12$.',primary_note),
        ('indian_K12_controls_seed0','primary',12,CONTROLS,None,0,'Indian transport: borrowing controls at $K=12$, seed 0.','One actual seed; stability is undefined for an individual fit.'),
        ('indian_K12_controls_5seeds','primary',12,CONTROLS,None,None,'Indian transport: borrowing controls at $K=12$.',primary_note),
        ('dabiri_K12_main_5samples_5seeds','external',12,BASELINES,None,None,'Dabiri traffic: external comparison at $K=12$.',external_note),
        ('dabiri_K16_sample0_seed0','external',16,BASELINES,0,0,'Dabiri traffic: $K=16$, sample 0, seed 0.',single_note),
        ('dabiri_K16_sample0_5seeds','external',16,BASELINES,0,None,'Dabiri traffic: $K=16$, fixed sample 0.',primary_note),
        ('dabiri_K16_main_5samples_5seeds','external',16,BASELINES,None,None,'Dabiri traffic: external comparison at $K=16$.',external_note)]
    combined=[]
    for name,dataset,k,methods,sample,seed,caption,note in specs:
        rows=form_rows(tables,dataset,k,methods,sample,seed)
        export_table(output,name,rows,caption,note+' '+source_note)
        combined += ['\n'+(output/(name+'.md')).read_text(encoding='utf-8')]
        logger.info('SAVED TABLE | %s | %d models.',name,len(rows))
    for name in ['sensitivity_summary.csv','borrowing_diagnostics.csv','runtime_and_fits.csv',
                 'corpus_size_summary.csv','seed_stability.csv']:
        write_csv(output/('full_'+name),tables[name])
    sensitivity=tables['sensitivity_summary.csv']
    s=['# Indian K=12 parameter sensitivity','', '| Setting | Value | NPMI | C_v | TD |', '|---|---:|---:|---:|---:|']
    for row in sensitivity:
        s.append('| '+ ' | '.join([row['factor'],row['value'],
                  f"{float(row['npmi_mean']):.4f} ± {float(row['npmi_seed_sd']):.4f}",
                  f"{float(row['cv_mean']):.4f} ± {float(row['cv_seed_sd']):.4f}",
                  f"{float(row['diversity_mean']):.4f} ± {float(row['diversity_seed_sd']):.4f}"])+' |')
    s+=['','All five seeds are retained. The original main setting remains lambda=1 and borrowing cap=1/3.','']
    atomic_write(output/'sensitivity_primary_K12.md','\n'.join(s).encode('utf-8'),overwrite=True)
    combined+=['\n'+'\n'.join(s)]
    behavior=[r for r in tables['borrowing_diagnostics.csv'] if r['study']=='main'
              and r['condition']=='primary_full' and r['K']=='12']
    write_csv(output/'indian_K12_borrowing_and_outer_audit.csv',behavior)
    mechanism=['# Untouched-word audit: Indian K=12','',
               '| Variant | Gain (nats/token) | Posts harmed (%) | Mean borrowed mass |',
               '|---|---:|---:|---:|']
    for method in CONTROLS:
        variant={'nmf_original':'no_borrowing','nmf_similarity':'similarity_maximum'}.get(method,method)
        row=next(r for r in behavior if r['scope']=='outer_audit' and r['variant']==variant)
        mechanism.append('| '+' | '.join([LABELS[method],f"{float(row['mean_post_gain_nats_per_token']):.4f}",
                   f"{100*float(row['post_loss_frequency']):.2f}",f"{float(row['mean_actual_beta']):.4f}"])+' |')
    mechanism+=['','One protected audit per selection configuration; not five topic-seed replications. '
                'Gain and harm frequency concern the eligible evaluated posts; borrowed mass averages all retained posts. '
                'These are prediction diagnostics, not transport-service accuracy scores.','']
    atomic_write(output/'indian_K12_untouched_prediction_audit.md','\n'.join(mechanism).encode('utf-8'),overwrite=True)
    combined+=['\n'+'\n'.join(mechanism)]
    atomic_write(output/'ALL_REQUESTED_TABLES.md','\n'.join(combined).encode('utf-8'),overwrite=True)
    audit.update(source_run_id=source_run,source_directory=str(folder),source_snapshot_sha256=file_hash(manifest_path),
                 primary_sd_unit='five model seeds',external_sd_unit='five sample means',
                 single_fit_stability='undefined',topics_16_means='sixteen topics; model seeds remain 0--4')
    write_json(output/'verification.json',audit,overwrite=True)
    return audit
