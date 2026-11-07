"""Frozen before fresh-mask results: no benchmark-dependent selection or tuning."""
import copy

ANCHOR = 'preserve_forward_fixed_cap'
POLICIES = ('nmf_original', 'nmf_similarity', 'full_pool_loo', 'proposed', ANCHOR,
            'matched_preserve_fixed', 'preserve_fixed_prefix',
            'weighted_hidden_fixed', 'preserve_fixed_no_downside',
            'preserve_fixed_individual', 'preserve_fixed_single_view')
MASKS = (3711, 4711, 5711, 6711)

def build_protocol(original):
    p = copy.deepcopy(original)
    required = {'views': 20, 'candidates_per_view': 20, 'cap': 1/3,
                'max_donors': 3, 'downside_weight': 1., 'admission_margin': 1e-8}
    for name, value in required.items():
        if p.get(name) != value:
            raise ValueError('The accepted development protocol differs: '+name)
    if p['topic_counts'] != [8, 10, 12] or p['final_model_seeds'] != list(range(5)):
        raise ValueError('Topic counts or final seeds differ from the frozen design')
    if set(p['corpora']) != {'transport_grievance', 'chinese_pooled'}:
        raise ValueError('Expected both complete audited corpora')
    p.update(policies=list(POLICIES), selection_mask_bases=list(MASKS),
             anchor_policy=ANCHOR, development_mask_base=2711,
             stage='fresh_mask_validation_of_revised_fixed_cap_rule',
             new_final_fits=1200, reused_no_borrowing_fits=30,
             fixed_reference_seed=0, fixed_family_folds=True,
             fresh_mask_results_not_known_at_freeze=True,
             inferential_scope='same collections and fixed predictors; mask variation only',
             selection_score='hidden-token-share times hidden gain minus visible-token-share times visible loss',
             positive_offered_weights=[1/3],
             development_bank_in_validation_summary=False,
             changes_allowed_during_run=False)
    p['scoring'] = p['selection_score']
    p['control_definitions'] = {
        'nmf_original': 'No borrowing; five final seeds per corpus/K reused, not refitted per mask.',
        'nmf_similarity': 'Original full-receiver cosine ranking of the masked-view candidate union; top three at cap without admission.',
        'full_pool_loo': 'Original hidden-word full-pool leave-one-out removal utility, top three at cap without positive threshold.',
        'proposed': 'ORIGINAL PEBR: hidden-word gain with downside penalty and four positive budget fractions; complete-group forward search.',
        ANCHOR: 'REVISED PEBR: receiver-preserving score, downside penalty and complete-group forward search; only cap is a positive offer.',
        'matched_preserve_fixed': 'Similarity-ranked available donors match revised PEBR actual transfer mass and contributing donor count for EACH receiver.',
        'preserve_fixed_prefix': 'Same revised score, cap and no-borrowing option; score reciprocal-rank prefixes of lengths 1..3.',
        'weighted_hidden_fixed': 'Remove only the visible-loss term; retain hidden-token-share weighting, downside utility and fixed cap.',
        'preserve_fixed_no_downside': 'Only downside lambda becomes zero; revised score, fixed cap and group search retained.',
        'preserve_fixed_individual': 'Average singleton revised gains at the same offered cap; actual-mixture audit without reranking/rejection.',
        'preserve_fixed_single_view': 'View 0 and its list only; changes BOTH retrieval coverage and predictive scoring views.'}
    p['aggregation'].update(selection_variation='Average seeds inside each fresh bank, then summarize FOUR fresh bank means; exclude development mask 2711.',
                            comparison='Revised PEBR minus comparator, paired by corpus/K/mask/final seed.')
    p['single_display_run'] = {'selection_mask_base': MASKS[0], 'final_model_seed': 0}
    # Unexecuted stages are not silently inherited as designs for the old rule.
    p.pop('protected_word_audit', None)
    p.pop('targeted_sensitivity_later_phase', None)
    p['later_validation_pending'] = ['protected-word replication for revised rule',
                                     'reference initialization and fold sensitivity',
                                     'neural baseline fidelity/tuning checks',
                                     'method-dependent table, figure and case regeneration']
    p['runner_implementation_pending'] = False
    return p
