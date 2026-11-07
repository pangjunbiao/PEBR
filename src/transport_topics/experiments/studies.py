"""Full fixed-settings evaluation, sensitivity and nested corpus-size studies."""
from copy import deepcopy
from pathlib import Path
import json
import time
import numpy as np
from ..config import canonical_hash
from ..schemas import number
from ..runtime.artifacts import file_hash,write_json
from ..data.prepared import preparation_dependency,load_collection
from ..data.loaders import Post
from ..data.preparation import prepare_condition
from ..application.catalogue import load_cohort
from ..evaluation.reference import prepare_references
from ..baselines.registry import fit_model,NMF_VARIANTS
from ..reporting.aggregate import export_results
from .models import run as run_main_models
from .evaluation import ModelEvaluation
from .sharing_controls import make_controls
from .planner import plan,CORPUS_METHODS


def _main_job(collection):
    name=collection.summary['name'];primary=name.startswith('primary')
    return {'study':'main','id':name,'dataset':'primary' if primary else 'external',
            'sample_index':int(name.rsplit('_',1)[1]) if name.startswith('external_sample_') else None}


def fit_condition(root,collection,config,environment,logger,methods,k,seeds):
    variants=[NMF_VARIANTS[m] for m in methods if m in NMF_VARIANTS]
    controls=[];directory=None;control_key=None
    if any(m in NMF_VARIANTS and m!='nmf_original' for m in methods):
        directory,summary=make_controls(root,collection,k,'topic_learning',config,logger,variants)
        controls.append(dict(summary,path=directory.relative_to(root).as_posix()))
        control_key=summary['identity']
        if config['evaluation']['prediction']['enabled']:
            outer,summary=make_controls(root,collection,k,'outer_audit',config,logger,variants)
            controls.append(dict(summary,path=outer.relative_to(root).as_posix()))
    models=[]
    for seed in seeds:
        for method in methods:
            path=directory/(NMF_VARIANTS[method]+'.npz') if method!='nmf_original' else None
            models.append(fit_model(root,collection,method,k,seed,config,environment,logger,path,
                                    control_key if method!='nmf_original' else None))
    return models,controls


def prepare_size_collection(root,job,parent,encoder,config,logger):
    parent_rows=parent.source_rows.tolist()
    requested=job['source_rows']
    if requested==parent_rows:return parent,True
    records=json.loads((parent.directory/'posts.json').read_text(encoding='utf-8'))
    lookup={record['source_row']:record for record in records}
    if any(row not in lookup for row in requested):raise ValueError('A frozen size ladder is not nested in its parent')
    selected=[lookup[row] for row in requested]
    posts=[Post(p['source_row'],p['source_id'],p['original_text']) for p in selected]
    texts=[p['cleaned_text'] for p in selected]
    source=json.loads((parent.directory/'identity.json').read_text(encoding='utf-8'))['source_hash']
    summary=prepare_condition(root,job['id'],posts,texts,source,encoder,config,logger)
    return load_collection(root,summary,config),False


def run(root,config,environment,logger,store):
    began=time.monotonic()
    preparation,dependency=preparation_dependency(root,config)
    smoke=config['integration']['scope']=='smoke'
    if not smoke and not preparation['complete_registered_preparation']:
        raise ValueError('Full study requires the completed full preparation, not a development corpus')
    sample_path=root/preparation['sample_manifest']
    sample=json.loads(sample_path.read_text(encoding='utf-8'))
    study_config=deepcopy(config)
    if config['experiment']['stage']=='transportation_retrieval':
        study_config['integration']['collections']='primary'
        study_config['studies']['sensitivity']['enabled']=False
        study_config['studies']['corpus_size']['enabled']=False
    if study_config['method']['name']!='proposed':
        study_config['studies']['sensitivity']['enabled']=False
        study_config['studies']['corpus_size']['enabled']=False
    specification=plan(preparation,sample,study_config)
    specification['base_collections']=[c['name'] for c in preparation['conditions'] if study_config['integration']['collections']=='all' or c['name'].startswith('primary')]
    key=canonical_hash({'configuration':study_config,'preparation':dependency['preparation_sha256'],
                        'plan':specification,'implementation':{p.relative_to(root/'src/transport_topics').as_posix():file_hash(p)
                          for folder in ('evaluation','application','experiments','reporting') for p in sorted((root/'src/transport_topics'/folder).glob('*.py'))}})
    store.json('study_plan.json',dict(specification,identity=key))
    logger.info('STUDY PLAN FROZEN | %s | base collections=%d | K=%s | seeds=%s | sensitivity points=%d | size jobs=%d.',
                'DEVELOPMENT ONLY' if smoke else 'FULL REGISTERED BUDGETS',len(specification['base_collections']),specification['topic_counts'],
                specification['model_seeds'],len(specification['sensitivity']),len(specification['corpus_size']))
    primary=load_collection(root,preparation['conditions'][0],config)
    cohort=load_cohort(root,primary,preparation)
    store.json('dataset_evaluation_profile.json',{'preparation':preparation,'queries':len(cohort.queries),
               'query_categories':{c:sum(q['category']==c for q in cohort.queries) for c in sorted({q['category'] for q in cohort.queries})},
               'cohort_identity':cohort.identity,'query_catalogue_depends_on_method':False})
    references=prepare_references(root,preparation,config,logger)
    store.json('coherence_references.json',{role:{'identity':ref.identity,'documents':ref.documents,
                 'path':ref.directory.relative_to(root).as_posix(),'evaluation_only':True} for role,ref in references.items()})
    evaluator=ModelEvaluation(root,study_config,references,cohort,logger)
    models=[];controls=[]
    def progress():
        store.json('evaluation_progress.json',{'logical_evaluations_completed':len(evaluator.entries),
                  'unique_model_evaluations':len({e['evaluation_identity'] for e in evaluator.entries}),
                  'last_condition':evaluator.entries[-1]['condition'] if evaluator.entries else None,
                  'all_results_retained':True},overwrite=True)
    def main_callback(model,collection):
        evaluator.evaluate(model,collection,_main_job(collection));progress()
    run_main_models(root,study_config,environment,logger,store,on_model=main_callback)
    models.extend(json.loads((store.directory/'models_index.json').read_text())['models'])
    main_models=list(models)
    for control in json.loads((store.directory/'sharing_controls_index.json').read_text())['banks'] if (store.directory/'sharing_controls_index.json').exists() else []:
        controls.append(dict(control,job={'study':'main','id':control['condition']}))
    main_controls=list(controls)
    seeds=specification['model_seeds'];k=config['topics']['main_count']
    for position,job in enumerate(specification['sensitivity']):
        logger.info('SENSITIVITY | point %d/%d | %s=%s | other settings fixed.',position+1,len(specification['sensitivity']),job['factor'],job['value'])
        wanted=job['reuse_main_method']
        reused=[m for m in main_models if wanted is not None and m['condition']==primary.summary['name'] and m['method']==wanted and m['topics']==k and m['seed'] in seeds]
        if len(reused)==len(seeds):
            fitted=reused;bank_results=[]
            logger.info('Reusing %s outputs for an identical sensitivity condition.',wanted)
            variant=NMF_VARIANTS[wanted]
            for bank in main_controls:
                if bank['condition']==primary.summary['name'] and bank['topics']==k and variant in bank['variants']:
                    controls.append(dict(bank,variants=[variant],job=dict(job,logical_variant='proposed',reused_input_bank=True)))
        else:
            changed=deepcopy(study_config)
            if job['factor']=='downside_weight':
                changed['method']['selection']['downside'].update(enabled=True,weight=job['value'])
            else:changed['method']['transfer']['max_weight']=job['value']
            fitted,bank_results=fit_condition(root,primary,changed,environment,logger,['proposed'],k,seeds)
            models.extend(fitted)
        logical=dict(job,logical_method='proposed')
        for model in fitted:evaluator.evaluate(model,primary,logical);progress()
        controls.extend(dict(bank,job=job) for bank in bank_results)
    encoder=None;size_conditions=[]
    for position,job in enumerate(specification['corpus_size']):
        logger.info('CORPUS SIZE | job %d/%d | %s | target=%d actual=%d | ladder=%d.',
                    position+1,len(specification['corpus_size']),job['dataset'],job['target_size'],job['actual_size'],job['sample_index'])
        parent_summary=preparation['conditions'][0] if job['dataset']=='primary' else next(
             c for c in preparation['conditions'] if c['name']=='external_sample_%d'%job['sample_index'])
        parent=load_collection(root,parent_summary,config)
        same=job['source_rows']==parent.source_rows.tolist()
        if not same and encoder is None:
            from ..representations.embeddings import FrozenEncoder
            encoder=FrozenEncoder(root,config,environment,logger,store.directory)
            store.json('size_encoder_audit.json',encoder.audit)
        changed=deepcopy(study_config);changed['evaluation']['prediction']['enabled']=False
        collection,reused_parent=prepare_size_collection(root,job,parent,encoder,changed,logger)
        size_conditions.append(dict(job,condition=collection.summary['name'],condition_identity=collection.summary['identity_sha256'],reused_parent=reused_parent))
        reused=[m for m in models if m['condition']==collection.summary['name'] and m['topics']==k and m['method'] in CORPUS_METHODS and m['seed'] in seeds]
        if reused_parent and len(reused)==len(CORPUS_METHODS)*len(seeds):
            fitted=reused;bank_results=[]
            logger.info('Size condition equals its frozen parent; reusing its fitted models without counting a new replication.')
            for bank in main_controls:
                if bank['condition']==collection.summary['name'] and bank['topics']==k and bank['scope']=='topic_learning':
                    controls.append(dict(bank,variants=[NMF_VARIANTS[m] for m in CORPUS_METHODS],job=dict(job,reused_input_bank=True)))
        else:
            fitted,bank_results=fit_condition(root,collection,changed,environment,logger,CORPUS_METHODS,k,seeds)
            models.extend(fitted)
        controls.extend(dict(bank,job=job) for bank in bank_results)
        for model in fitted:evaluator.evaluate(model,collection,job,retrieval=False);progress()
        store.json('size_conditions_index.json',size_conditions,overwrite=True)
    unique_models={m['identity']:m for m in models}
    compact=[{name:m[name] for name in ('identity','method','condition','topics','seed','path','posts','vocabulary','development_only')}
             for m in unique_models.values()]
    store.json('study_models_index.json',{'models':compact,'unique_fitted_or_reused_models':len(compact)})
    store.json('study_evaluations_index.json',{'evaluations':[{'job':e['job'],'model_identity':e['model_identity'],
                    'evaluation_identity':e['evaluation_identity'],'path':e['evaluation_path']} for e in evaluator.entries]})
    store.json('study_controls_index.json',{'banks':controls})
    outputs=export_results(root,store,key,evaluator.entries,models,controls,cohort,study_config)
    store.json('evaluation_outputs.json',outputs)
    registered=(not smoke and study_config['method']['name']=='proposed' and
                 config['seeds']==list(range(5)) and config['topics']['counts']==[8,12,16] and
                 study_config['integration']['collections']=='all' and all(config['baselines'].values()) and
                 config['studies']['ablations']['enabled'] and all(config['studies']['ablations']['variants'].values()) and
                 k==12 and config['topics']['top_words']==10 and config['retrieval']['main_budget']==10 and
                 config['retrieval']['budgets']==[5,10,20] and config['method']['borrowing']['enabled'] and
                 config['method']['selection']['strategy']=='predictive' and config['method']['transfer']['budget_mode']=='grid' and
                 number(config['method']['transfer']['max_weight'],'cap')==1/3 and
                 config['method']['selection']['downside']['enabled'] and number(config['method']['selection']['downside']['weight'],'lambda')==1 and
                 config['evaluation']['topic']['supplementary_cv'] and config['evaluation']['retrieval']['bootstrap_resamples']==2000 and
                 config['model_settings']['glocom']['epochs']==200 and config['model_settings']['knntm']['epochs']==200 and
                 config['model_settings']['btm']['iterations']==1000 and
                 config['studies']['corpus_size']['sizes']==[100,200,400,600] and
                 [number(v,'lambda') for v in config['studies']['sensitivity']['downside_weights']]==[0,.5,1,2] and
                 [number(v,'cap') for v in config['studies']['sensitivity']['borrowing_caps']]==[1/6,1/3,1/2] and
                 len(specification['sensitivity'])==7 and len(size_conditions)==40)
    summary={'identity':key,'complete_requested_study':True,'complete_registered_study':registered,'development_only':smoke,
             'models_unique':len(unique_models),'logical_evaluations':len(evaluator.entries),
             'query_count':len(cohort.queries),'categories':len({q['category'] for q in cohort.queries}),
             'sensitivity_points_completed':len(specification['sensitivity']),
             'corpus_size_jobs_completed':len(size_conditions),
             'dataset_count':len({'primary' if e['condition'].startswith('primary') else 'external' for e in evaluator.entries}),
             'reference_dataset_count':len(references),'base_collection_count':len(specification['base_collections']),
             'topic_counts':specification['topic_counts'],'model_seeds':seeds,
             'comparative_metrics_computed':True,'proposed_rule_changed':False,
             'zero_queries_removed':False,'settings_selected_from_results':False,
             'result_files':outputs,'seconds':time.monotonic()-began,
             'next_phase':'Paper Results and Reproducibility'}
    store.json('evaluation_summary.json',summary)
    logger.info('EVALUATION COMPLETE | %d unique models, %d logical evaluations; queries=%d categories=%d; development_only=%s.',
                len(unique_models),len(evaluator.entries),len(cohort.queries),summary['categories'],smoke)
    return summary,dependency
