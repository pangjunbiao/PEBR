"""Freeze separate sensitivity axes; no outcome-dependent parameter choices."""
import copy

ANCHOR = 'preserve_forward_fixed_cap'
POLICIES = ('nmf_original', 'nmf_similarity', 'matched_preserve_fixed', 'preserve_fixed_prefix', ANCHOR)
CASES = (
    {'case': 'default', 'axis': 'default', 'reference_seed': 0, 'fold_seed': 2710},
    {'case': 'r1', 'axis': 'reference_initialization', 'reference_seed': 1, 'fold_seed': 2710},
    {'case': 'r2', 'axis': 'reference_initialization', 'reference_seed': 2, 'fold_seed': 2710},
    {'case': 'f3710', 'axis': 'family_folds', 'reference_seed': 0, 'fold_seed': 3710},
    {'case': 'f4710', 'axis': 'family_folds', 'reference_seed': 0, 'fold_seed': 4710},
)

def build(frozen):
    p = copy.deepcopy(frozen)
    if p['topic_counts'] != [8, 10, 12] or p['final_model_seeds'] != list(range(5)):
        raise ValueError('Expected the accepted K=8/10/12 and five final seeds')
    required = {'cap': 1/3, 'views': 20, 'candidates_per_view': 20,
                'max_donors': 3, 'downside_weight': 1., 'admission_margin': 1e-8}
    if any(p.get(k) != v for k, v in required.items()):
        raise ValueError('The revised scientific rule changed')
    for name in ('new_final_fits','reused_no_borrowing_fits','fixed_reference_seed','fixed_family_folds',
                 'fresh_mask_results_not_known_at_freeze','selection_mask_bases'):
        p.pop(name,None)
    p.update(stage='reference_and_family_fold_sensitivity', policies=list(POLICIES), cases=list(CASES),
             selection_mask_base=3711, selection_mask_bases=[3711], reference_initialization_seeds=[1,2], alternate_fold_seeds=[3710,4710],
             default_reference_seed=0, default_fold_seed=2710, selection_mask_changes=False,
             encoder_calls=0, new_reference_fits=120, new_final_topic_fits=420,
             reused_unique_final_fits=150, reused_case_fit_records=330,
             expected_case_fit_records=750, expected_case_policy_summaries=150,
             new_nondefault_corpus_K_cases=24, main_topic_count=12,
             rule_and_defaults_unchanged=True, outcomes_used_for_tuning=False,
             known_mask_bank=True, positive_offered_weights=[1/3],
             inferential_scope='Same two collections and fixed mask 3711; reference initialization and family-fold protocol perturbations are studied separately.')
    p['aggregation'] = {
        'fitting': 'Within each corpus/K/case/policy, mean and sample SD across five final NMF seeds; ten-pair topic stability.',
        'paired': 'PEBR minus comparator for the same corpus/K/case/final seed.',
        'sensitivity': 'Default and two alternative levels displayed separately for each axis; no pooled twenty-five-seed SD.',
        'reused_models': 'No-borrowing fits are shared across all cases; ordinary-sharing fits are also shared across reference-seed cases.',
        'not_claimed': 'No independent-corpus uncertainty, significance or universal ranking guarantee.'}
    p['fold_axis_scope'] = 'New fold-excluded predictive references AND newly eligible masked/full donor lists; not an isolated predictor-seed perturbation.'
    p['control_definitions']={name:frozen['control_definitions'][name] for name in POLICIES}
    p['later_validation_pending']=['genuine K10 BTM/neural benchmarks','baseline budget review',
                                   'updated corpus-size/parameter/runtime assets','final table/figure/claim ledger']
    return p
