"""Aggregate declared outcomes without choosing winners or favorable settings."""
from collections import defaultdict
import csv
import io
import json
import numpy as np
from ..runtime.artifacts import atomic_write
from ..evaluation.stability import seed_stability
from ..evaluation.uncertainty import paired_family_bootstrap


def mean_sd(values):
    values=np.asarray(values,dtype=float)
    return float(values.mean()),float(values.std(ddof=1)) if len(values)>1 else None


def write_csv(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    if not fields:fields=['no_eligible_records']
    text=io.StringIO(newline='');writer=csv.DictWriter(text,fieldnames=fields)
    writer.writeheader();writer.writerows(rows)
    atomic_write(path,text.getvalue().encode('utf-8'),overwrite=True)


def logical_method(entry):
    return entry['job'].get('logical_method',entry['method'])


def topic_rows(entries):
    rows=[]
    for entry in entries:
        values=entry['topic_metrics']
        if values is None:continue
        job=entry['job'];role='primary' if entry['condition'].startswith('primary') else 'external'
        rows.append({'study':job['study'],'job_id':job['id'],'dataset':job.get('dataset',role),
                     'condition':entry['condition'],'method':logical_method(entry),'source_method':entry['method'],
                     'K':entry['topics'],'seed':entry['seed'],'sample_index':job.get('sample_index'),
                     'target_size':job.get('target_size'),'actual_size':entry['posts'],
                     'factor':job.get('factor'),'value':job.get('value'),
                     'npmi':values['npmi'],'diversity':values['diversity'],'cv':values.get('cv'),
                     'reference_identity':entry['reference_identity'],'model_identity':entry['model_identity'],
                     'evaluation_path':entry['evaluation_path'],'development_only':entry['development_only']})
    return rows


def topic_summary(rows):
    groups=defaultdict(list)
    for row in rows:
        if row['study']=='main':groups[(row['dataset'],row['method'],row['K'])].append(row)
    result=[]
    for (dataset,method,k),group in sorted(groups.items()):
        by_collection=defaultdict(list)
        for row in group:by_collection[row['condition']].append(row)
        item={'dataset':dataset,'method':method,'K':k,'collections':len(by_collection),
              'seed_fits':len(group),'seeds_per_collection':min(len(x) for x in by_collection.values())}
        for metric in ('npmi','diversity','cv'):
            if group[0][metric] is None:continue
            sample_means=[np.mean([r[metric] for r in sample]) for sample in by_collection.values()]
            item[metric+'_mean']=float(np.mean(sample_means))
            if dataset=='primary':item[metric+'_seed_sd']=mean_sd([r[metric] for r in group])[1]
            else:
                item[metric+'_sample_sd']=mean_sd(sample_means)[1]
                within=[mean_sd([r[metric] for r in sample])[1] for sample in by_collection.values()]
                item[metric+'_mean_within_sample_seed_sd']=float(np.mean(within)) if all(x is not None for x in within) else None
        result.append(item)
    return result


def retrieval_rows(entries):
    rows=[];categories=[]
    for entry in entries:
        metrics=entry['retrieval_metrics']
        if metrics is None:continue
        job=entry['job']
        base={'study':job['study'],'job_id':job['id'],'method':logical_method(entry),
              'source_method':entry['method'],'K':entry['topics'],'seed':entry['seed'],
              'factor':job.get('factor'),'value':job.get('value'),'queries':metrics['queries'],
              'zero_queries':metrics['zero_queries'],'model_identity':entry['model_identity']}
        for b,budget in enumerate(metrics['budgets']):
            rows.append(dict(base,B=budget,macro_ndcg=metrics['macro_ndcg'][b],macro_recall=metrics['macro_recall'][b]))
            for name,scores in metrics['per_category'].items():
                categories.append(dict(base,B=budget,category=name,category_queries=scores['queries'],
                                       ndcg=scores['ndcg'][b],recall=scores['recall'][b]))
    return rows,categories


def retrieval_summary(rows):
    groups=defaultdict(list)
    for row in rows:
        if row['study']=='main':groups[(row['method'],row['K'],row['B'])].append(row)
    result=[]
    for (method,k,b),group in sorted(groups.items(),key=lambda item:repr(item[0])):
        ndcg,ndcg_sd=mean_sd([r['macro_ndcg'] for r in group]);recall,recall_sd=mean_sd([r['macro_recall'] for r in group])
        result.append({'method':method,'K':k,'B':b,'model_outputs':len(group),'queries':group[0]['queries'],
                       'macro_ndcg_mean':ndcg,'macro_ndcg_seed_sd':ndcg_sd,
                       'macro_recall_mean':recall,'macro_recall_seed_sd':recall_sd,
                       'deterministic_direct_retriever':k is None})
    return result


def contrasts(root,entries,cohort,config):
    main=[e for e in entries if e['job']['study']=='main' and e['retrieval_metrics'] is not None]
    groups=defaultdict(list)
    for entry in main:groups[(entry['method'],entry['topics'])].append(entry)
    labels=[q['category'] for q in cohort.queries]
    families=[cohort.families[q['document_index']] for q in cohort.queries]
    output=[];category_output=[]
    means={}
    for key,group in groups.items():
        values=[]
        for entry in group:
            directory=root/entry['evaluation_path']
            if not np.array_equal(np.load(directory/'query_source_rows.npy',allow_pickle=False),[q['source_row'] for q in cohort.queries]):
                raise ValueError('Paired retrieval contrasts use different query cohorts')
            values.append(np.load(directory/'query_metrics.npy',allow_pickle=False))
        means[key]=np.mean(values,axis=0)
    for k in config['topics']['counts']:
        if ('proposed',k) not in means:continue
        for (method,other_k),baseline in means.items():
            if method=='proposed' or other_k not in (k,None):continue
            for b,budget in enumerate(config['retrieval']['budgets']):
                for metric,column in (('ndcg',0),('recall',1)):
                    result=paired_family_bootstrap(means[('proposed',k)][:,b,column]-baseline[:,b,column],
                         labels,families,config['evaluation']['retrieval']['bootstrap_resamples'],config['evaluation']['retrieval']['bootstrap_seed'])
                    base={'K':k,'B':budget,'metric':metric,'comparator':method,
                          'contrast':'proposed minus comparator','conditional_on_fitted_seed_averaged_models':True}
                    output.append(dict(base,**{key:val for key,val in result.items() if key!='categories'}))
                    for category in result['categories']:
                        category_output.append(dict(base,**category,
                             intended_main_figure=k==config['topics']['main_count'] and budget==config['retrieval']['main_budget'] and metric=='ndcg' and method in ('nmf_original','similarity_matched')))
    return output,category_output


def stability_rows(entries):
    groups=defaultdict(dict);meta={}
    for entry in entries:
        if entry['topic_metrics'] is None:continue
        job=entry['job'];key=(job['study'],job['id'],logical_method(entry),entry['topics'])
        groups[key][entry['seed']]=entry['topic_metrics']['top_words'];meta[key]=job
    rows=[];pairs=[]
    for (study,job,method,k),topics in groups.items():
        result=seed_stability(topics)
        base={'study':study,'job_id':job,'method':method,'K':k}
        rows.append(dict(base,stability=result['stability'],seed_pairs=result['seed_pairs']))
        pairs.extend(dict(base,**pair) for pair in result['pairs'])
    return rows,pairs


def sensitivity_rows(topic,application,main_budget):
    records=[]
    lookup={(r['job_id'],r['seed']):r for r in application if r['study']=='sensitivity' and r['B']==main_budget}
    for row in topic:
        if row['study']!='sensitivity':continue
        value=dict(row)
        app=lookup.get((row['job_id'],row['seed']),{})
        value.update(macro_ndcg=app.get('macro_ndcg'),macro_recall=app.get('macro_recall'))
        records.append(value)
    groups=defaultdict(list)
    for row in records:groups[(row['factor'],row['value'])].append(row)
    summary=[]
    for (factor,value),group in sorted(groups.items()):
        row={'factor':factor,'value':value,'seed_fits_or_reused_outputs':len(group)}
        for metric in ('npmi','diversity','cv','macro_ndcg','macro_recall'):
            if group[0][metric] is not None:row[metric+'_mean'],row[metric+'_seed_sd']=mean_sd([r[metric] for r in group])
        summary.append(row)
    return records,summary


def corpus_rows(topic):
    values=[r for r in topic if r['study']=='corpus_size']
    original={(r['job_id'],r['seed']):r['npmi'] for r in values if r['method']=='nmf_original'}
    differences=[dict(r,npmi_change_from_no_borrowing=r['npmi']-original[(r['job_id'],r['seed'])])
                 for r in values if r['method']!='nmf_original']
    groups=defaultdict(list)
    for row in differences:groups[(row['dataset'],row['target_size'],row['method'])].append(row)
    summary=[]
    for (dataset,size,method),group in sorted(groups.items()):
        ladders=defaultdict(list)
        for row in group:ladders[row['sample_index']].append(row['npmi_change_from_no_borrowing'])
        mean,sd=mean_sd([np.mean(v) for v in ladders.values()])
        summary.append({'dataset':dataset,'target_size':size,'method':method,
                        'mean_npmi_change':mean,'ladder_mean_sd':sd,'ladders':len(ladders),
                        'seed_fits_per_ladder':min(map(len,ladders.values())),
                        'actual_size_min':min(r['actual_size'] for r in group),'actual_size_max':max(r['actual_size'] for r in group)})
    return values,differences,summary


def borrowing_rows(root,controls):
    rows=[]
    for control in controls:
        directory=root/control['path'];job=control['job']
        actions=json.loads((directory/'actions.json').read_text(encoding='utf-8'))
        anchor=json.loads((root/control['anchor_path']/'summary.json').read_text(encoding='utf-8'))
        eligible=anchor['eligible_receivers'];cap=anchor['settings']['cap']
        with np.load(directory/'offered_beta.npz') as offered,np.load(directory/'actual_beta.npz') as actual,np.load(directory/'actual_donor_count.npz') as count:
            for variant in control['variants']:
                o,a,c=offered[variant],actual[variant],count[variant]
                selected=np.array([len(r['variants'][variant]['selected_source_rows']) for r in actions])
                available=np.array([len(r['variants'][variant]['available_source_rows']) for r in actions])
                if not np.array_equal(c,available):raise ValueError('Saved donor availability/count diagnostics differ')
                row={'study':job['study'],'job_id':job['id'],'condition':control['condition'],'K':control['topics'],
                     'scope':control['scope'],'variant':job.get('logical_variant',variant),'source_variant':variant,
                     'factor':job.get('factor'),'value':job.get('value'),
                     'target_size':job.get('target_size'),'sample_index':job.get('sample_index'),
                     'reused_input_bank':job.get('reused_input_bank',False),'posts':len(a),
                     'eligible_receivers':eligible,'budget_cap':cap,
                     'offered_coverage':float(np.mean(o>0)),'realized_coverage':float(np.mean(a>0)),
                     'offered_coverage_among_eligible':float(np.sum(o>0)/eligible) if eligible else None,
                     'mean_offered_beta':float(o.mean()),'mean_actual_beta':float(a.mean()),
                     'mean_missing_donor_weight':float(np.mean(o-a)),
                     'fraction_positive_offers_at_cap':float(np.sum(np.isclose(o,cap,rtol=0,atol=1e-12))/np.sum(o>0)) if np.any(o>0) else None,
                     'max_actual_beta':float(a.max()),'mean_selected_donors':float(selected.mean()),
                     'mean_available_donors':float(c.mean()),
                     'available_share_of_selected':float(available.sum()/selected.sum()) if selected.sum() else None,
                     'matched_mass_count_violations':control['matched_amount_count_violations'],
                     'bounds_violations':control['bounds_violations']}
                row.update(control['outer_prediction'].get(variant,{}))
                rows.append(row)
    return rows


def export_results(root,store,key,entries,models,controls,cohort,config):
    topics=topic_rows(entries);retrieval,categories=retrieval_rows(entries)
    contrast,category_effects=contrasts(root,entries,cohort,config)
    stability,pairs=stability_rows(entries)
    sensitivity,sensitivity_summary=sensitivity_rows(topics,retrieval,config['retrieval']['main_budget'])
    size,size_deltas,size_summary=corpus_rows(topics)
    payloads={
      'topic_seed_results.csv':('topic_quality',topics),
      'topic_quality_summary.csv':('topic_quality',topic_summary(topics)),
      'seed_stability.csv':('topic_quality',stability),'seed_stability_pairs.csv':('topic_quality',pairs),
      'retrieval_seed_results.csv':('transportation',retrieval),'retrieval_summary.csv':('transportation',retrieval_summary(retrieval)),
      'retrieval_category_results.csv':('transportation',categories),
      'retrieval_contrasts.csv':('transportation',contrast),'service_category_effects.csv':('transportation',category_effects),
      'sensitivity_results.csv':('sensitivity',sensitivity),'sensitivity_summary.csv':('sensitivity',sensitivity_summary),
      'corpus_size_results.csv':('corpus_size',size),'corpus_size_deltas.csv':('corpus_size',size_deltas),
      'corpus_size_summary.csv':('corpus_size',size_summary),
      'borrowing_diagnostics.csv':('diagnostics',borrowing_rows(root,controls))}
    unique={m['identity']:m for m in models}
    fit_rows=[]
    for model in unique.values():
        audit=model['audit'];fit=audit.get('fit',{})
        fit_rows.append({'method':model['method'],'condition':model['condition'],'K':model['topics'],
             'seed':model['seed'],'model_identity':model['identity'],'source_fit_seconds':model['seconds'],
             'device':audit.get('training_device','cpu'),'converged_nmf':fit.get('converged'),
             'nmf_iterations':fit.get('iterations'),'native_epochs':audit.get('completed_epochs'),
             'btm_iterations':audit.get('iterations'),'termination':audit.get('termination','NMF tolerance' if fit else 'frozen representation'),
             'source_model_path':model['path']})
    payloads['runtime_and_fits.csv']=('diagnostics',fit_rows)
    outputs={}
    for name,(area,rows) in payloads.items():
        path=root/'outputs/aggregated'/area/key[:20]/name
        write_csv(path,rows)
        # Compact scientific tables accompany the run inspection, without raw text.
        write_csv(store.directory/'results'/name,rows)
        outputs[name]={'path':path.relative_to(root).as_posix(),'rows':len(rows)}
    return outputs
