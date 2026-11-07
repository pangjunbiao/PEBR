"""Prespecified core design. Historical outcomes remain visible in provenance."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import hashlib


POLICIES = ("nmf_original", "nmf_similarity", "similarity_matched", "full_pool_loo", "proposed",
            "fixed_cap_pebr", "predictively_scored_similarity_prefixes", "no_downside",
            "individual_donor", "single_view")
CORPORA = ("transport_grievance", "chinese_pooled")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def build_protocol():
    return {
        "schema": "pebr_revision_core_v1", "status": "prespecified_design_not_a_performance_result",
        "purpose": "Estimate repeatable added value over simpler borrowing and justify retained decisions",
        "corpora": list(CORPORA), "dataset_roles": "equal core evaluation scope; no primary corpus chosen from outcome rankings",
        "topic_counts": [8, 10, 12], "main_topic_count": 12,
        "final_model_seeds": [0, 1, 2, 3, 4],
        "selection_mask_bases": [2711, 3711, 4711, 5711, 6711],
        "default_fold_seed": 2710, "default_reference_seed": 0, "family_folds": 5,
        "views": 20, "candidates_per_view": 20, "max_donors": 3,
        "masking": {"unit": "retained word type", "hide_fraction": .5, "minimum_types": 4,
                    "query_removal": "all occurrences of every hidden type"},
        "reference_predictor": {"topics_equal_final_K": True, "folds": 5, "seed": 0,
                                "smoothing": .1, "unigram_smoothing": .1,
                                "nmf_tolerance": .0001, "nmf_max_iterations": 2000},
        "final_nmf": {"tolerance": .0001, "max_iterations": 2000,
                      "inference": "original posts with learned dictionary fixed"},
        "corpus_preparation": {"minimum_document_frequency": 2, "family_threshold": .9,
                               "vocabulary_and_stopwords": "audited language-specific preparation retained"},
        "cap": 1/3, "downside_weight": 1., "admission_margin": 1e-8,
        "positive_budget_cap_fractions": [.25, .5, .75, 1.],
        "borrowed_evidence": "complete normalized donor word distributions; equal offered shares; unavailable shares return to receiver",
        "scoring": "audited complete-mixture predictor; receiver-only reference; current fold excluded",
        "preprocessing": "unchanged audited counts, vocabulary and original post records; language-specific tokenizers and frozen encoders retained",
        "policies": list(POLICIES),
        "control_definitions": {
            "nmf_original": "No borrowing; five final fits per corpus/K, reused across selection repeats",
            "nmf_similarity": "Existing similarity policy: rank masked-view candidate union by currently available full-receiver cosine, top three, offer cap, no predictive admission",
            "similarity_matched": "Existing matched policy: similarity donors with per-receiver realized mass and contributing donor count matched to PEBR",
            "full_pool_loo": "Existing full-pool removal utility at offered cap, top three donors, no positive admission threshold; a complete-policy comparator",
            "proposed": "Audited complete-group forward search over borrowing budgets, with no borrowing admissible",
            "fixed_cap_pebr": "Identical predictive forward search with only cap as a positive offered budget; no borrowing remains admissible",
            "predictively_scored_similarity_prefixes": "Mean reciprocal rank across ordered visible-view lists; absent=0; source-order ties; evaluate all prefixes 1..3 and all budgets with actual-mixture utility against zero",
            "no_downside": "Only lambda is zero; same complete-group search",
            "individual_donor": "Mean of singleton predictive gains at the same offered beta; same utility/search; actual-mixture audit after selection without reranking/veto",
            "single_view": "Fixed view 0 and its candidate list only; changes both coverage and predictive assessment"},
        "metrics": {"coherence": ["npmi", "cv"], "complementary": ["diversity", "seed_pair_stability"],
                    "descriptor_words": 10, "npmi_unit": "presence in original posts", "cv_window": 110,
                    "reference": "same original unaugmented corpus within each language; borrowed counts excluded"},
        "aggregation": {"comparison": "paired differences within corpus/K/selection bank/final seed",
                        "selection_variation": "average paired seed differences within each bank, then summarize five bank means",
                        "fitting_variation": "report separately within each bank",
                        "stability": "optimal top-word Jaccard matching over ten seed pairs in each bank; no single-seed stability",
                        "corpus_uncertainty": "selection repeats are not independent corpora; do not present them as such"},
        "single_display_run": {"selection_mask_base": 2711, "final_model_seed": 0},
        "protected_word_audit": {"outer_bases": [2712, 3712, 4712, 5712, 6712],
                                 "inner_mask_base": 2711, "outer_type_fraction": .2,
                                 "outer_minimum_types": 1, "final_model_seeds_not_replicates": True,
                                 "policies": list(POLICIES),
                                 "outcomes": ["mean_post_gain", "median_post_gain", "loss_fraction", "mean_negative_part_all_posts"],
                                 "scope": "reserve outer targets before retrieval/selection; same targets and common eligibility across policies; references exclude receiving fold"},
        "targeted_sensitivity_later_phase": {"K": 12, "fold_seeds": [2710, 3710, 4710],
                                              "reference_seeds": [0, 1, 2], "mask_base": 2711,
                                              "one_factor_at_a_time": True},
        "timing": "separate reference fitting, masked retrieval, selection, final fitting and reporting; include total cost",
        "runtime": {"download_or_install": False, "missing_assets": "fail without modifying environment",
                    "cpu_threads": 4, "gpu_index": 0, "embedding_batch_size": 32,
                    "embedding_precision": "float32", "tf32": False, "run_corpora_sequentially": True,
                    "encoder_max_lengths": {"transport_grievance": 256, "chinese_pooled": 128}},
        "baseline_configuration_review": "separate later-stage study; source preservation and finite training do not establish adequate tuning",
        "annotations": "not used in borrowing or topic fitting; no synthetic Chinese service labels",
        "historical_K16": "retained with original version; not relabelled and not part of the new core grid",
        "disclosure": "Existing corpus outcomes at K=8/12 have been examined; this is a prespecified revision, not blind confirmation",
        "outcome_policy": "retain every declared condition; no score-based seed/corpus deletion; simplify or narrow claims when warranted",
        "runner_implementation_pending": True,
        "guaranteed_performance_or_journal_rating": False}


def validate_protocol(protocol):
    if protocol.get("topic_counts") != [8, 10, 12]:
        raise ValueError("The revised topic-count grid must remain 8,10,12")
    if protocol.get("corpora") != list(CORPORA):
        raise ValueError("Both audited corpora must receive the core study")
    if protocol.get("final_model_seeds") != [0, 1, 2, 3, 4]:
        raise ValueError("Final-model seeds must be the five declared seeds")
    masks = protocol.get("selection_mask_bases", [])
    if len(masks) != 5 or len(set(masks)) != 5:
        raise ValueError("Require five distinct selection masking repeats")
    if protocol.get("policies") != list(POLICIES):
        raise ValueError("Declared policies changed")
    if protocol.get("single_display_run") != {"selection_mask_base": 2711, "final_model_seed": 0}:
        raise ValueError("The displayed seed cannot be selected from outcomes")
    if protocol.get("borrowed_evidence", "").startswith("residual"):
        raise ValueError("Audited PEBR transfers full donor distributions, not residual-only words")
    if not 0 < protocol["cap"] < 1 or protocol["downside_weight"] < 0:
        raise ValueError("Invalid borrowing settings")
    return canonical_hash(protocol)


def experiment_plan(protocol):
    validate_protocol(protocol)
    rows = []
    for corpus in protocol["corpora"]:
        for k in protocol["topic_counts"]:
            for seed in protocol["final_model_seeds"]:
                rows.append({"stage": "core_final_fit", "corpus": corpus, "K": k, "policy": "nmf_original",
                             "mask_base": "not_applicable", "model_seed": seed, "status": "planned_not_run"})
            for mask in protocol["selection_mask_bases"]:
                for policy in protocol["policies"]:
                    if policy == "nmf_original": continue
                    rows.append({"stage": "core_selection", "corpus": corpus, "K": k, "policy": policy,
                                 "mask_base": mask, "model_seed": "not_applicable", "status": "planned_not_run"})
                    for seed in protocol["final_model_seeds"]:
                        rows.append({"stage": "core_final_fit", "corpus": corpus, "K": k, "policy": policy,
                                     "mask_base": mask, "model_seed": seed, "status": "planned_not_run"})
            for outer in protocol["protected_word_audit"]["outer_bases"]:
                for policy in protocol["policies"]:
                    rows.append({"stage": "protected_word_audit", "corpus": corpus, "K": k, "policy": policy,
                                 "mask_base": protocol["protected_word_audit"]["inner_mask_base"],
                                 "outer_base": outer, "model_seed": "not_applicable", "status": "planned_not_run"})
    return rows


def write_plan(path, rows):
    fields = ["stage", "corpus", "K", "policy", "mask_base", "outer_base", "model_seed", "status"]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
