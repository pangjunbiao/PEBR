"""Finite completion plan; no score-based choice of rule, corpus, K or budget."""
import copy
POLICIES=['nmf_original','nmf_similarity','matched_preserve_fixed','preserve_fixed_prefix','preserve_forward_fixed_cap']
def build(fresh):
    p=copy.deepcopy(fresh)
    if p['topic_counts']!=[8,10,12] or p['cap']!=1/3 or p['downside_weight']!=1:raise ValueError('Revised rule/defaults changed')
    for name in ['new_final_fits','reused_no_borrowing_fits','fresh_mask_results_not_known_at_freeze','later_validation_pending']:
        p.pop(name,None)
    p.update(stage='final_revised_results_completion',known_selection_mask=3711,selection_mask_bases=[3711],
        final_model_seeds=list(range(5)),main_topic_count=12,main_neural_epoch_budget=400,
        neural_budget_check=[200,400],neural_settings_source='verified historical settings; only epoch budget extends',
        new_benchmark_fits=90,benchmark_records=240,main_benchmark_records=180,
        parameter_cases=[{'case':'default','cap':1/3,'downside_weight':1.},
                         {'case':'cap6','cap':1/6,'downside_weight':1.},
                         {'case':'cap2','cap':1/2,'downside_weight':1.},
                         {'case':'lambda0','cap':1/3,'downside_weight':0.},
                         {'case':'lambda05','cap':1/3,'downside_weight':.5},
                         {'case':'lambda2','cap':1/3,'downside_weight':2.}],
        parameter_records=60,new_parameter_fits=40,
        size_targets=[100,200,400,600],size_ladders=list(range(5)),size_mask_base=3711,size_K=12,
        size_namespace='pebr_revised_size_v1',size_minimum_document_frequency=2,
        size_policies=POLICIES,size_records=1000,size_reference_fits=200,
        new_performance_model_fits=1130,new_timing_model_replays=2,new_predictive_reference_fits=210,
        retrieval_corpus='transport_grievance',retrieval_budgets=[5,10,20],retrieval_expected_queries=368,
        retrieval_source_sha256='d0da254163384f6e1276f6fcb1cd3ca8cf4bd515a7e8817d9766b2f2bd7c9eeb',
        retrieval_bootstrap_resamples=2000,retrieval_bootstrap_seed=2713,
        retrieval_records=150,retrieval_neural_epoch_budget=400,
        retrieval_controls=['matched_preserve_fixed','preserve_fixed_prefix','full_pool_loo','preserve_fixed_no_downside'],
        retrieval_scope='within-collection, frozen models containing query posts; labels enter evaluation only; no direct-embedding reference; all adverse findings retained',
        size_reference='same full original-text coherence reference across all sample sizes',
        size_preparation='nested full-family samples; refit minimum-DF2 vocabulary within each sample; remove zero representations and retain exact manifests',
        size_fold_assignment='original family-fold labels restricted to retained sample rows',
        runtime_scope='one predefined seed0 PEBR pipeline per corpus; cold audited encoder load, full/masked encoding, retrieval, five references, standalone PEBR selection, final NMF and original-post inference; topic evaluation excluded',
        rule_changed=False,settings_selected_from_outcomes=False,
        main_benchmark_selection_bank=3711,main_benchmark_aggregation='five final seeds conditional on preselected first fresh bank',
        supplemental_robustness='all four fresh selection banks and all reference/fold settings retained separately')
    return p
