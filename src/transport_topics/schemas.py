"""Strict, readable configuration contracts; no model execution in this module."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any


class ConfigurationError(ValueError):
    pass


def number(value: Any, path: str) -> float:
    if isinstance(value, bool):
        raise ConfigurationError(path + " must be a number, not a boolean")
    try:
        result = float(Fraction(value)) if isinstance(value, str) else float(value)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as error:
        raise ConfigurationError(path + " must be a finite number or fraction") from error
    if not math.isfinite(result):
        raise ConfigurationError(path + " must be finite")
    return result


def _validate(value: Any, rule: Any, path: str, partial: bool = False) -> None:
    if isinstance(rule, dict):
        if not isinstance(value, dict):
            raise ConfigurationError(path + " must be a mapping")
        unknown = set(value) - set(rule)
        if unknown:
            raise ConfigurationError(path + ": unknown field(s): " + ", ".join(sorted(map(str, unknown))))
        if not partial and set(rule) - set(value):
            raise ConfigurationError(path + ": missing field(s): " + ", ".join(sorted(set(rule) - set(value))))
        for key, item in value.items():
            _validate(item, rule[key], path + "." + key, partial=partial)
    elif isinstance(rule, list):
        if not isinstance(value, list):
            raise ConfigurationError(path + " must be a list")
        for index, item in enumerate(value):
            _validate(item, rule[0], "%s[%d]" % (path, index), partial=False)
    elif isinstance(rule, set):
        if not isinstance(value, str) or value not in rule:
            raise ConfigurationError(path + " must be one of " + str(sorted(rule)))
    elif rule == "number":
        number(value, path)
    elif rule is int:
        if type(value) is not int:
            raise ConfigurationError(path + " must be an integer")
    elif not isinstance(value, rule):
        raise ConfigurationError(path + " must have type " + rule.__name__)


METHOD_SCHEMA = {
    "name": {"proposed", "nmf_original", "nmf_similarity", "btm", "knntm", "glocom", "minilm_retrieval"},
    "borrowing": {"enabled": bool},
    "selection": {"strategy": {"predictive", "similarity", "full_pool_loo", "none"},
                  "max_donors": int, "margin": "number",
                  "downside": {"enabled": bool, "weight": "number"}},
    "transfer": {"budget_mode": {"grid", "maximum", "matched", "none"},
                 "max_weight": "number", "grid_fractions": ["number"]},
    "views": {"count": int, "min_word_types": int, "hide_fraction": "number"},
    "candidates": {"per_view": int},
    "predictor": {"folds": int, "seed": int, "fold_seed": int, "mask_seed": int,
                  "outer_seed": int, "smoothing": "number", "unigram_smoothing": "number",
                  "tolerance": "number", "max_iterations": int},
    "encoder": {"name": str, "revision": str, "backend": {"onnx"},
                "max_sequence_length": int, "precision": {"float32"},
                "normalize": bool},
}

SCHEMA = {
    "schema_version": int,
    "project": {"name": str, "method_label": {"proposed"}, "protocol_id": str},
    "experiment": {"name": str, "stage": {
        "execution_foundation", "corpus_preparation", "borrowing_rule", "model_smoke",
        "topic_benchmark", "transportation_retrieval", "borrowing_audit", "full_study", "reports"},
        "mode": {"check", "run", "report"}},
    "runtime": {"device": {"auto", "cpu", "cuda"}, "gpu_index": int,
                "cpu_threads": int, "probe_timeout_seconds": int,
                "deterministic": bool, "precision": {"float32"},
                "allow_tf32": bool, "concurrent_gpu_jobs": int},
    "seeds": [int],
    "datasets": {"primary": str, "external": str},
    "baselines": {"nmf_original": bool, "nmf_similarity": bool, "btm": bool,
                  "knntm": bool, "glocom": bool, "minilm_retrieval": bool},
    "method": METHOD_SCHEMA,
    "topics": {"counts": [int], "main_count": int, "top_words": int},
    "retrieval": {"budgets": [int], "main_budget": int},
    "evaluation": {"topic": {"primary": {"npmi"}, "diversity": bool, "supplementary_cv": bool,
                              "original_text_reference": bool},
                   "retrieval": {"primary": {"macro_ndcg"}, "recall": bool,
                                 "bootstrap_resamples": int, "bootstrap_seed": int},
                   "prediction": {"enabled": bool, "outer_fraction": "number"}},
    "studies": {"ablations": {"enabled": bool, "variants": {
                    "no_borrowing": bool, "similarity_maximum": bool, "similarity_matched": bool,
                    "full_pool_loo": bool, "no_downside": bool, "proposed": bool}},
                "sensitivity": {"enabled": bool, "downside_weights": ["number"], "borrowing_caps": ["number"]},
                "corpus_size": {"enabled": bool, "sizes": [int], "sample_indices": [int]}},
    "output": {"resume": bool, "inspection_zip": bool},
    "preparation": {"scope": {"full", "smoke"}, "download": bool,
                    "embedding_batch_size": int, "gpu_memory_mb": int,
                    "min_document_frequency": int, "family_threshold": "number",
                    "family_block_size": int, "external_target_posts": int, "smoke_posts": int},
    "borrowing_run": {"collections": {"all", "primary"},
                      "banks": [{"topic_learning", "outer_audit"}],
                      "receiver_limit": int, "proposal_batch_size": int,
                      "progress_seconds": int},
    "model_settings": {
        "btm": {"iterations": int, "alpha_mass": "number", "beta": "number", "window": int},
        "glocom": {"epochs": int, "batch_size": int, "learning_rate": "number", "hidden_units": int,
                   "clusters": int, "cluster_seed": int, "augmentation": "number",
                   "prior_variance": "number", "ecr_weight": "number", "dropout": "number",
                   "beta_temperature": "number", "sinkhorn_alpha": "number", "sinkhorn_max_iterations": int},
        "knntm": {"epochs": int, "batch_size": int, "learning_rate": "number", "hidden_units": int,
                  "warmup_epochs": int, "neighbors": int, "alpha": "number", "eta": "number", "rho": "number"}},
    "integration": {"scope": {"smoke", "full"}, "collections": {"primary", "all"},
                    "smoke_epochs": int, "smoke_btm_iterations": int,
                    "checkpoint_epochs": int, "download_sources": bool},
}


def validate_config(data: dict) -> dict:
    _validate(data, SCHEMA, "config")
    integration = data['integration']
    if any(integration[k] < 1 for k in ('smoke_epochs','smoke_btm_iterations','checkpoint_epochs')):
        raise ConfigurationError('Model integration epoch/iteration/checkpoint settings must be positive')
    for name, settings in data['model_settings'].items():
        for key in ('epochs','iterations','batch_size','hidden_units','clusters','neighbors','window','sinkhorn_max_iterations'):
            if key in settings and settings[key] < (2 if key in ('batch_size','window') else 1):
                raise ConfigurationError('Invalid model setting: '+name+'.'+key)
        for key in ('learning_rate','alpha_mass','beta','prior_variance','beta_temperature','sinkhorn_alpha'):
            if key in settings and number(settings[key],name+'.'+key) <= 0:
                raise ConfigurationError('Positive model parameter required: '+name+'.'+key)
        for key in ('augmentation','ecr_weight','alpha'):
            if key in settings and number(settings[key],name+'.'+key) < 0:
                raise ConfigurationError('Nonnegative model parameter required: '+name+'.'+key)
    knn = data['model_settings']['knntm']
    if knn['warmup_epochs'] < 0 or knn['warmup_epochs'] >= knn['epochs'] or integration['smoke_epochs'] <= knn['warmup_epochs']:
        raise ConfigurationError('Both full and smoke kNNTM training must include epochs beyond warmup')
    if any(not 0 <= number(knn[key],key) <= 1 for key in ('eta','rho')):
        raise ConfigurationError('kNNTM fusion parameters must be in [0,1]')
    if data['model_settings']['glocom']['cluster_seed'] not in range(5):
        raise ConfigurationError('Use a declared seed for fixed global clustering')
    if not 0 <= number(data['model_settings']['glocom']['dropout'],'glocom.dropout') < 1:
        raise ConfigurationError('GloCOM dropout must be in [0,1)')
    rule_run = data['borrowing_run']
    if not rule_run['banks'] or len(set(rule_run['banks'])) != len(rule_run['banks']):
        raise ConfigurationError('Borrowing bank scopes must be nonempty and distinct')
    if rule_run['receiver_limit'] < 0 or rule_run['proposal_batch_size'] < 1 or rule_run['progress_seconds'] < 1:
        raise ConfigurationError('Borrowing receiver limit is nonnegative; batching and progress settings are positive')
    prep=data['preparation']
    if any(prep[k]<1 for k in ('embedding_batch_size','gpu_memory_mb','family_block_size')):
        raise ConfigurationError('Preparation batch and memory settings must be positive')
    if prep['min_document_frequency']!=2 or number(prep['family_threshold'],'family_threshold')!=.9:
        raise ConfigurationError('The locked preparation uses min_df=2 and family threshold 0.9')
    if prep['external_target_posts']!=600 or prep['smoke_posts']<8:
        raise ConfigurationError('Registered external target is 600; development smoke needs at least eight posts')
    if 600 not in data['studies']['corpus_size']['sizes']:
        raise ConfigurationError('The frozen sampling ladder must include the 600-post target')
    if data["schema_version"] != 1:
        raise ConfigurationError("Only schema_version 1 is supported")
    if data["project"]["name"] != "proposed_transport_topics":
        raise ConfigurationError("project.name must be proposed_transport_topics")
    if not data["project"]["protocol_id"].strip():
        raise ConfigurationError("project.protocol_id cannot be empty")
    seeds = data["seeds"]
    if not seeds or len(seeds) > 5 or len(set(seeds)) != len(seeds) or any(s not in range(5) for s in seeds):
        raise ConfigurationError("seeds must be unique members of [0,1,2,3,4], with at most five entries")
    for key in ("gpu_index",):
        if data["runtime"][key] < 0:
            raise ConfigurationError("runtime." + key + " cannot be negative")
    for key in ("cpu_threads", "probe_timeout_seconds", "concurrent_gpu_jobs"):
        if data["runtime"][key] < 1:
            raise ConfigurationError("runtime." + key + " must be positive")
    if data["runtime"]["concurrent_gpu_jobs"] != 1:
        raise ConfigurationError("The current execution policy permits one GPU model job at a time")
    if data["runtime"]["allow_tf32"] or not data["runtime"]["deterministic"]:
        raise ConfigurationError("The registered runtime requires deterministic settings and TF32 disabled")
    method = data["method"]
    selection, transfer, predictor = method["selection"], method["transfer"], method["predictor"]
    if selection["max_donors"] < 1 or number(selection["margin"], "selection.margin") <= 0:
        raise ConfigurationError("max_donors and admission margin must be positive")
    weight = number(selection["downside"]["weight"], "selection.downside.weight")
    if weight < 0:
        raise ConfigurationError("downside weight cannot be negative")
    cap = number(transfer["max_weight"], "transfer.max_weight")
    fractions = [number(x, "transfer.grid_fractions") for x in transfer["grid_fractions"]]
    if not 0 < cap < 1:
        raise ConfigurationError("transfer.max_weight must be strictly between zero and one")
    if (not fractions or fractions != sorted(set(fractions)) or fractions[0] <= 0
            or fractions[-1] != 1.0):
        raise ConfigurationError("grid_fractions must increase strictly in (0,1] and include 1")
    if method["borrowing"]["enabled"] and (selection["strategy"] == "none" or transfer["budget_mode"] == "none"):
        raise ConfigurationError("Enabled borrowing needs an explicit selector and budget policy")
    if method["views"]["count"] < 1 or method["views"]["min_word_types"] < 4:
        raise ConfigurationError("Use positive view count and at least four types for masking")
    if not 0 < number(method["views"]["hide_fraction"], "views.hide_fraction") < 1:
        raise ConfigurationError("views.hide_fraction must be in (0,1)")
    if method["candidates"]["per_view"] < 1 or predictor["folds"] < 2:
        raise ConfigurationError("Positive neighbor count and at least two family folds are required")
    if predictor["seed"] != 0:
        raise ConfigurationError("The frozen reference uses its specified fixed seed 0")
    for key in ("fold_seed", "mask_seed", "outer_seed"):
        if predictor[key] < 0:
            raise ConfigurationError("Predictor construction seeds cannot be negative")
    if not 0 < number(predictor["smoothing"], "predictor.smoothing") < 1:
        raise ConfigurationError("Predictor smoothing must be in (0,1)")
    for key in ("unigram_smoothing", "tolerance"):
        if number(predictor[key], "predictor." + key) <= 0:
            raise ConfigurationError("predictor." + key + " must be positive")
    if predictor["max_iterations"] < 1:
        raise ConfigurationError("predictor.max_iterations must be positive")
    if method["encoder"]["max_sequence_length"] < 1 or not method["encoder"]["normalize"]:
        raise ConfigurationError("The frozen encoder needs a positive token limit and normalized embeddings")
    if len(method["encoder"]["revision"]) != 40 or any(c not in "0123456789abcdef" for c in method["encoder"]["revision"]):
        raise ConfigurationError("encoder.revision must be an explicit 40-character commit")
    for group, key, main in (("topics", "counts", "main_count"), ("retrieval", "budgets", "main_budget")):
        values = data[group][key]
        if not values or values != sorted(set(values)) or min(values) < 1:
            raise ConfigurationError(group + "." + key + " must be strictly increasing positive integers")
        if data[group][main] not in values:
            raise ConfigurationError(group + "." + main + " must occur in " + key)
    if data["topics"]["top_words"] < 2:
        raise ConfigurationError("At least two top words are required for coherence")
    if not data["evaluation"]["topic"]["original_text_reference"]:
        raise ConfigurationError("Topic evaluation must use original, unaugmented reference text")
    if not data['evaluation']['topic']['diversity'] or not data['evaluation']['retrieval']['recall']:
        raise ConfigurationError('The registered evaluation retains both topic diversity and retrieval recall as companion outcomes')
    retrieval = data["evaluation"]["retrieval"]
    if retrieval["bootstrap_resamples"] < 1 or retrieval["bootstrap_seed"] not in range(5):
        raise ConfigurationError("Use a positive bootstrap count and a seed from 0 to 4")
    if not 0 < number(data["evaluation"]["prediction"]["outer_fraction"], "outer_fraction") < 1:
        raise ConfigurationError("Outer hiding fraction must be in (0,1)")
    sensitivity = data["studies"]["sensitivity"]
    for key, lower, upper in (("downside_weights", 0, math.inf), ("borrowing_caps", 0, 1)):
        values = [number(v, "studies.sensitivity." + key) for v in sensitivity[key]]
        if not values or values != sorted(set(values)):
            raise ConfigurationError("Sensitivity lists must be nonempty and strictly increasing")
        if any(v < lower or v >= upper or (key == "borrowing_caps" and v == 0) for v in values):
            raise ConfigurationError("Invalid sensitivity value in " + key)
    corpus = data["studies"]["corpus_size"]
    if not corpus["sizes"] or corpus["sizes"] != sorted(set(corpus["sizes"])) or min(corpus["sizes"]) < 1:
        raise ConfigurationError("Corpus sizes must increase strictly and be positive")
    if corpus["sample_indices"] != list(range(5)):
        raise ConfigurationError("The registered sample/ladder indices are [0,1,2,3,4]")
    if data["datasets"] != {"primary": "indian_transport", "external": "dabiri_traffic"}:
        raise ConfigurationError("Dataset identities must match the registered two-source study")
    return {"model_seeds": seeds[:], "borrowing_enabled": method["borrowing"]["enabled"],
            "selection_strategy": selection["strategy"] if method["borrowing"]["enabled"] else "none",
            "effective_downside_weight": weight if selection["downside"]["enabled"] else 0.0,
            "borrowing_grid": [0.0] + [cap * f for f in fractions],
            "maximum_borrowing_weight": cap}


@dataclass(frozen=True)
class LoadedConfiguration:
    values: dict
    effective: dict
    source_files: tuple[str, ...]
    sha256: str
