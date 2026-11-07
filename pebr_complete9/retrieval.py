"""Recompute the retained auxiliary audit from current frozen representations."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from .common import read,write,table,digest
from .core import shared

def verify_source(study):
    source=study.project/'data/raw/indian_transport/grievance.csv'
    if digest(source)!=study.protocol['retrieval_source_sha256']:
        raise ValueError('The released evaluation annotation source changed')
    return source

def cohort(study):
    from transport_topics.data.loaders import load_primary
    from transport_topics.application.catalogue import make_cohort
    source=verify_source(study)
    _,annotations=load_primary(source)
    c=make_cohort(study.corpora['transport_grievance']['collection'],annotations)
    if len(c.queries)!=study.protocol['retrieval_expected_queries']:
        raise ValueError('The fixed category-retrieval query count differs')
    return c

def run(study):
    from transport_topics.application.ranking import rank_queries
    from transport_topics.evaluation.retrieval import evaluate_rankings,macro_scores
    from transport_topics.evaluation.uncertainty import paired_family_bootstrap
    from pebr_revision_step3.io import complete,save_receipt
    c=cohort(study);directory=study.work/'retrieval_results';directory.mkdir(exist_ok=True)
    budgets=study.protocol['retrieval_budgets']
    records=read(study.work/'benchmark_results/seed_records.json')
    models=[r for r in records if r['corpus']=='transport_grievance' and
            (r['policy'] not in ('knntm','glocom') or r['epoch_budget']==400)]
    for k in study.protocol['topic_counts']:
        for policy in study.protocol['retrieval_controls']:
            models += [shared(study.step5,study.step3,'transport_grievance',k,policy,seed,{'epoch_budget':0}) for seed in range(5)]
    keys={(r['K'],r['policy'],r['seed']) for r in models}
    expected={(k,p,s) for k in (8,10,12) for p in ('nmf_original','nmf_similarity','btm','knntm','glocom','preserve_forward_fixed_cap',*study.protocol['retrieval_controls']) for s in range(5)}
    if len(models)!=150 or keys!=expected:raise ValueError('Incomplete current-model retrieval grid')
    means=defaultdict(list);summaries=[];seed_rows=[];category_rows=[]
    for number,r in enumerate(models,1):
        modeldir=Path(r['path'])
        if not complete(modeldir):raise ValueError('Retrieval model receipt failed')
        source=modeldir/'retrieval_features.npy';out=directory/'per_model'/('K%d_%s_s%d.npz'%(r['K'],r['policy'],r['seed']))
        original=study.corpora['transport_grievance']['collection']
        if not np.array_equal(np.load(modeldir/'source_rows.npy',allow_pickle=False),original.source_rows):
            raise ValueError('Retrieval source-row alignment failed')
        features=np.load(source,allow_pickle=False)
        if features.shape!=(len(original.source_rows),int(r['K'])) or not np.isfinite(features).all():
            raise ValueError('Invalid current original-post retrieval representation')
        ranks,scores=rank_queries(features,c,max(budgets));values=evaluate_rankings(ranks,c,budgets)
        macro,by_category=macro_scores(values,c)
        out.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out,query_values=values,ranks=ranks,scores=scores,
            source_rows=np.array([q['source_row'] for q in c.queries]),representation_sha256=np.array(digest(source)))
        means[(int(r['K']),r['policy'])].append((int(r['seed']),values))
        for j,b in enumerate(budgets):
            seed_rows.append({'K':r['K'],'policy':r['policy'],'seed':r['seed'],'budget':b,
                'ndcg':float(macro[j,0]),'recall':float(macro[j,1]),'model_path':str(modeldir),
                'features_sha256':digest(source),'cohort_identity':c.identity,
                'zero_queries':int(sum(np.linalg.norm(features[q['document_index']])==0 for q in c.queries))})
            category_rows += [{'K':r['K'],'policy':r['policy'],'seed':r['seed'],'budget':b,'category':name,
                'ndcg':float(v[j,0]),'recall':float(v[j,1])} for name,v in by_category.items()]
        if number==1 or number%10==0:study.logger.info('Auxiliary retrieval | %d/150 current fitted representations',number)
    for (k,p),rr in sorted(means.items()):
        if {s for s,_ in rr}!=set(range(5)):raise ValueError('Retrieval seed group incomplete')
        macro=np.array([macro_scores(v,c)[0] for _,v in sorted(rr)])
        for j,b in enumerate(budgets):
            summaries.append({'K':k,'policy':p,'budget':b,'final_seeds':5,
                'ndcg_mean':float(macro[:,j,0].mean()),'ndcg_sd':float(macro[:,j,0].std(ddof=1)),
                'recall_mean':float(macro[:,j,1].mean()),'recall_sd':float(macro[:,j,1].std(ddof=1))})
    categories=np.array([q['category'] for q in c.queries]);families=c.families[[q['document_index'] for q in c.queries]]
    average={p:np.mean([v for _,v in sorted(means[(12,p)])],axis=0) for k,p in means if k==12}
    contrasts=[];component_rows=[]
    for p in ('nmf_original','nmf_similarity','matched_preserve_fixed','preserve_fixed_prefix'):
        for m,name in enumerate(('ndcg','recall')):
            difference=average['preserve_forward_fixed_cap'][:,budgets.index(10),m]-average[p][:,budgets.index(10),m]
            result=paired_family_bootstrap(difference,categories,families,
                study.protocol['retrieval_bootstrap_resamples'],study.protocol['retrieval_bootstrap_seed'])
            contrasts.append(dict(K=12,budget=10,comparator=p,metric=name,**{a:v for a,v in result.items() if a!='categories'}))
            component_rows += [dict(comparator=p,metric=name,**row) for row in result['categories']]
    table(directory/'seed_metrics.csv',seed_rows);table(directory/'summaries.csv',summaries)
    table(directory/'category_seed_metrics.csv',category_rows);table(directory/'paired_contrasts.csv',contrasts)
    table(directory/'bootstrap_category_components.csv',component_rows)
    write(directory/'cohort.json',{'identity':c.identity,'queries':c.queries,'query_count':len(c.queries),
        'source_rows':c.source_rows.tolist(),'families':c.families.tolist(),'folds':c.folds.tolist(),
        'substantive_labels':c.labels,
        'source_sha256':study.protocol['retrieval_source_sha256'],'budgets':budgets,
        'label_use':'evaluation only; frozen cohort and catalogues common to all methods',
        'catalogue_rule':'exclude same source row, family and fold; zero candidates eligible with cosine zero',
        'query_policy':'exactly one substantive category and at least one eligible relevant candidate; zero query returns nothing',
        'aggregation':'mean within each category then equal mean across categories',
        'scope':study.protocol['retrieval_scope']})
    write(directory/'coverage.json',{'complete':True,'models':150,'seed_metric_rows':len(seed_rows),
        'queries':len(c.queries),'new_model_fits':0,'neural_budget':400,'old_rule_scores_reused':False,
        'direct_embedding_retrieval_included':False})
    # Per-model scores and all catalogue identities are covered, not only aggregate tables.
    save_receipt(directory)
    return True
