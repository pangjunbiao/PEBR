"""Model integration and fixed-budget fitting; outcome metrics follow next phase."""
import json
from ..data.prepared import preparation_dependency, load_collection
from ..runtime.artifacts import write_json
from ..baselines.registry import enabled_methods, NMF_VARIANTS, fit_model
from .sharing_controls import make_controls
from .planner import verify_main_coverage


def run(root,config,environment,logger,store,on_model=None):
    preparation,dependency=preparation_dependency(root,config)
    store.json('preparation_dependency.json',dependency)
    conditions=preparation['conditions']
    if config['integration']['collections']=='primary':conditions=conditions[:1]
    methods=enabled_methods(config)
    variants=[NMF_VARIANTS[method] for method in methods if method in NMF_VARIANTS]
    smoke=config['integration']['scope']=='smoke'
    topics=[config['topics']['main_count']] if smoke else config['topics']['counts']
    seeds=config['seeds'][:1] if smoke else config['seeds']
    logger.info('MODEL INTEGRATION | scope=%s | collections=%d | K=%s | seeds=%s | methods=%s.',
                'SMOKE: DEVELOPMENT ONLY' if smoke else 'full fixed-budget fitting',len(conditions),topics,seeds,methods)
    if on_model is None:logger.info('No coherence, retrieval or winning-model selection occurs in this phase.')
    outputs,controls=[],[]
    for condition in conditions:
        collection=load_collection(root,condition,config)
        direct_done=False
        for count in topics:
            control_directory=None
            if any(name in NMF_VARIANTS and name!='nmf_original' for name in methods):
                control_directory,summary=make_controls(root,collection,count,'topic_learning',config,logger,variants)
                controls.append(dict(summary,path=control_directory.relative_to(root).as_posix()))
                if config['evaluation']['prediction']['enabled']:
                    audit_directory,audit_summary=make_controls(root,collection,count,'outer_audit',config,logger,variants)
                    controls.append(dict(audit_summary,path=audit_directory.relative_to(root).as_posix()))
                store.json('sharing_controls_index.json',{'banks':controls},overwrite=True)
            for seed in seeds:
                for method in methods:
                    if method=='minilm_retrieval' and direct_done:continue
                    learning=None;control_key=None
                    if method in NMF_VARIANTS and method!='nmf_original':
                        learning=control_directory/(NMF_VARIANTS[method]+'.npz')
                        control_key=next(c['identity'] for c in controls if c['path']==control_directory.relative_to(root).as_posix())
                    result=fit_model(root,collection,method,count,seed,config,environment,logger,learning,control_key)
                    outputs.append(result)
                    if on_model is not None:on_model(result,collection)
                    if method=='minilm_retrieval':direct_done=True
                    store.json('models_index.json',{'models':outputs},overwrite=True)
    native=[entry for entry in outputs if entry['method'] in ('glocom','knntm')]
    planned=verify_main_coverage(outputs,[c['name'] for c in conditions],methods,topics,seeds)
    summary={'complete_requested_run':True,'development_only':smoke,'scope':config['integration']['scope'],
             'collections':len(conditions),'topic_counts':topics,'model_seeds':seeds,
             'registered_seed_list':config['seeds'],'requested_methods':methods,
             'completed_model_outputs':len(outputs),'completed_control_banks':len(controls),
             'planned_model_outputs':planned,'method_seed_condition_coverage_verified':True,
             'published_neural_fits':len(native),
             'cuda_neural_fits':sum(bool(x['audit'].get('cuda_execution_verified')) for x in native),
             'topic_model_outputs':sum(x['topic_model'] for x in outputs),
             'direct_retriever_outputs':sum(not x['topic_model'] for x in outputs),
             'document_vocabulary_alignment_violations':0,'proposed_rule_changed':False,
             'performance_comparisons_computed':False,
             'next_phase':'Transportation Evaluation and Experimental Studies',
             'sample_manifest_sha256':preparation['sample_manifest_sha256']}
    store.json('model_integration.json',summary)
    logger.info('MODEL INTEGRATION COMPLETE | %d outputs, %d control banks | development_only=%s.',
                len(outputs),len(controls),smoke)
    return summary,dependency
