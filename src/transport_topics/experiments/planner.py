"""Declared study conditions only; observed scores never choose a job."""
from ..schemas import number
from collections import Counter


CORPUS_METHODS=('nmf_original','nmf_similarity','similarity_matched','proposed')


def verify_main_coverage(outputs,conditions,methods,topics,seeds):
    expected=[]
    for condition in conditions:
        for method in methods:
            if method=='minilm_retrieval':expected.append((condition,method,None,None))
            else:expected.extend((condition,method,k,seed) for k in topics for seed in seeds)
    actual=[(m['condition'],m['method'],m['topics'],m['seed']) for m in outputs]
    if Counter(actual)!=Counter(expected):
        raise ValueError('Completed main models do not cover the frozen method/K/seed/collection plan exactly')
    return len(expected)


def sensitivity_jobs(config):
    if not config['studies']['sensitivity']['enabled']:return []
    base_weight=number(config['method']['selection']['downside']['weight'],'lambda') if config['method']['selection']['downside']['enabled'] else 0.
    base_cap=number(config['method']['transfer']['max_weight'],'cap')
    result=[]
    for factor,values,baseline in (
        ('downside_weight',config['studies']['sensitivity']['downside_weights'],base_weight),
        ('borrowing_cap',config['studies']['sensitivity']['borrowing_caps'],base_cap)):
        for raw in values:
            value=number(raw,factor)
            reuse='proposed' if value==baseline else ('no_downside' if factor=='downside_weight' and value==0 else None)
            result.append({'study':'sensitivity','factor':factor,'value':value,'reuse_main_method':reuse,
                           'id':factor+'_'+str(raw).replace('/','_').replace('.','p')})
    return result


def corpus_jobs(preparation,sample_manifest,config):
    if not config['studies']['corpus_size']['enabled']:return []
    result=[]
    for role in ('primary','external'):
        ladders=sample_manifest['primary_ladders' if role=='primary' else 'external_samples']
        for index in config['studies']['corpus_size']['sample_indices']:
            for size in config['studies']['corpus_size']['sizes']:
                rows=ladders[str(index)][str(size)]
                result.append({'study':'corpus_size','dataset':role,'sample_index':index,'target_size':size,
                               'actual_size':len(rows),'source_rows':rows,
                               'id':'%s_ladder%d_n%d'%(role,index,size),'methods':list(CORPUS_METHODS)})
    return result


def plan(preparation,sample_manifest,config):
    smoke=config['integration']['scope']=='smoke'
    return {'development_only':smoke,'model_seeds':config['seeds'][:1] if smoke else config['seeds'],
            'topic_counts':[config['topics']['main_count']] if smoke else config['topics']['counts'],
            'main_count':config['topics']['main_count'],'budgets':config['retrieval']['budgets'],
            'sensitivity':sensitivity_jobs(config),'corpus_size':corpus_jobs(preparation,sample_manifest,config),
            'selection_depends_on_outcomes':False,'full_budget':not smoke,
            'external_reference':'all deduplicated transport-filtered original training posts after primary overlap exclusion',
            'coherence_probability_unit':'original document; zero cooccurrence=-1; universal pair=0',
            'reuse_is_not_an_independent_replication':True}
