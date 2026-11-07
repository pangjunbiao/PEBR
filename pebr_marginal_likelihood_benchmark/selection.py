"""Cross-fitted REALM-inspired retrieval and PEBR's singleton admission."""
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import logsumexp

from inputs import CAP, LIMIT, N_VIEWS, TOPICS
from pebr_core.numerics.nmf import fit_nmf
from pebr_core.proposed.actions import BorrowingAction, NO_BORROWING
from pebr_core.proposed.reference import PredictiveReference
from pebr_core.proposed.scoring import downside_utility
from pebr_core.proposed.search import preferred
from pebr_core.proposed.transfer import normalized_counts
from pebr_core.representations.tfidf import idf_weights, transform

MARGIN = 1e-8
DOWNSIDE = 1.0
REGULARIZATION = 1e-3
STARTS = ((0.0, 0.0), (-1.0, 4.0), (1.0, -4.0))


def fit_inner_reference(data, excluded):
    """Compute training likelihoods with both held-out folds removed."""
    indices = np.flatnonzero(~np.isin(data.folds, excluded))
    counts = sparse.csr_matrix(data.counts[indices])
    idf = idf_weights(counts)
    _, dictionary, audit = fit_nmf(transform(counts, idf), TOPICS, seed=0)
    if not audit["converged"]:
        raise RuntimeError(f"Inner reference failed to converge for excluded folds {excluded}")
    prior = np.asarray(counts.sum(axis=0)).ravel() + 0.1
    return PredictiveReference(dictionary, idf, prior), {
        "excluded_folds": list(excluded), "fit_indices": indices.tolist(), **audit}


def masked_log_gains(reference, visible, hidden, counts, candidate_ids,
                     donor_rows, donor_projection, with_visible):
    """Return summed hidden and visible log ratios for each available singleton."""
    word_ids = np.flatnonzero(counts)
    log_baseline = np.log(reference.predict(visible, word_ids))
    view, column = np.nonzero(candidate_ids >= 0)
    donors = candidate_ids[view, column]
    hidden_gain = np.zeros(candidate_ids.shape, dtype=np.float64)
    visible_gain = np.zeros_like(hidden_gain)
    if len(donors) == 0:
        return hidden_gain, visible_gain

    base_projection = reference.projection(visible)
    mixed_projection = (1 - CAP) * base_projection[view] + CAP * donor_projection[donors]

    def fallback(indices):
        return ((1 - CAP) * visible[view[indices]] + CAP * donor_rows[donors[indices]]) * reference.idf

    mixed = reference.from_projection(mixed_projection, word_ids, input_factory=fallback)
    log_ratio = np.log(mixed) - log_baseline[view]
    hidden_gain[view, column] = (log_ratio * hidden[view][:, word_ids]).sum(axis=1)
    if with_visible:
        visible_counts = counts[word_ids] - hidden[view][:, word_ids]
        visible_gain[view, column] = (log_ratio * visible_counts).sum(axis=1)
    if not np.isfinite(hidden_gain).all() or not np.isfinite(visible_gain).all():
        raise ArithmeticError("Nonfinite masked predictive log ratios")
    return hidden_gain, visible_gain


def features_for_view(similarities, lists):
    """Column zero is the null donor, with a fixed zero logit."""
    features = np.zeros((len(lists), lists.shape[1] + 1, 2), dtype=np.float64)
    features[:, 1:, 0] = lists >= 0
    scores = np.take_along_axis(similarities, np.maximum(lists, 0), axis=1)
    features[:, 1:, 1] = np.where(lists >= 0, scores, 0)
    allowed = np.column_stack((np.ones(len(lists), dtype=bool), lists >= 0))
    return features, allowed


def loss_and_gradient(weights, features, allowed, log_gain):
    logits = np.where(allowed, features @ weights, -np.inf)
    joint = logits + log_gain
    log_prior_norm = logsumexp(logits, axis=1, keepdims=True)
    log_joint_norm = logsumexp(joint, axis=1, keepdims=True)
    loss = np.mean(log_prior_norm - log_joint_norm)
    prior = np.exp(logits - log_prior_norm)
    posterior = np.exp(joint - log_joint_norm)
    gradient = ((prior - posterior)[..., None] * features).sum(axis=1).mean(axis=0)
    return (float(loss + REGULARIZATION * (weights @ weights) / 2),
            gradient + REGULARIZATION * weights)


def fit_retriever(features, allowed, log_gain):
    attempts = [minimize(loss_and_gradient, np.asarray(start), jac=True, method="L-BFGS-B",
                         args=(features, allowed, log_gain),
                         options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
                for start in STARTS]
    converged = [r for r in attempts if r.success and np.isfinite(r.fun) and np.isfinite(r.x).all()]
    if not converged:
        raise RuntimeError(f"Marginal retriever failed: {[str(r.message) for r in attempts]}")
    best = min(converged, key=lambda r: r.fun)
    return best.x, [{"success": bool(r.success), "message": str(r.message),
                     "objective": float(r.fun), "iterations": int(r.nit)}
                    for r in attempts]


def choose_retrieval_action(weights, similarities, lists, corpus_size):
    """Infer from visible-query retrieval priors only, averaging over all views."""
    features, allowed = features_for_view(similarities, lists)
    logits = np.where(allowed, features @ weights, -np.inf)
    probability = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    masses = np.bincount((lists + 1).ravel(), weights=probability[:, 1:].ravel(),
                         minlength=corpus_size + 1)
    masses[0] += probability[:, 0].sum()
    masses /= len(lists)
    if not np.isclose(masses.sum(), 1, atol=1e-12, rtol=0):
        raise ArithmeticError("Document-level retrieval probabilities fail to sum to one")
    best = int(np.argmax(masses))
    return (() if best == 0 else (best - 1,)), float(masses[best]), float(masses[0])


def choose_pebr_action(lists, hidden_log_gain, visible_log_gain, tokens):
    # Token-weighted hidden gain plus the nonpositive visible contribution.
    scores = (hidden_log_gain + np.minimum(visible_log_gain, 0)) / tokens
    donors = np.unique(lists[lists >= 0])
    by_donor = np.zeros((len(donors), len(lists)), dtype=np.float64)
    view, column = np.nonzero(lists >= 0)
    by_donor[np.searchsorted(donors, lists[view, column]), view] = scores[view, column]
    utilities = downside_utility(by_donor, DOWNSIDE)
    best, value = NO_BORROWING, 0.0
    for donor, score in zip(donors, utilities):
        action = BorrowingAction((int(donor),), CAP)
        if preferred(action, float(score), best, value, 0.0, MARGIN):
            best, value = action, float(score)
    return best.donors, value


def train_lists(similarities, eligible_ids, limit):
    # Filter before ranking so the excluded evaluation fold cannot displace a donor.
    order = np.argsort(-similarities[:, eligible_ids], axis=1, kind="stable")[:, :limit]
    lists = np.full((len(similarities), limit), -1, dtype=np.int32)
    lists[:, :order.shape[1]] = eligible_ids[order]
    return lists


def run_selection(data, bank, frozen, smoke, save_audit):
    n = len(data.counts)
    donor_rows = normalized_counts(data.counts)
    # Float32 dot products match the archived ranking pipeline.
    similarities = np.empty((n * N_VIEWS, n), dtype=bank.vectors.dtype)
    for start in range(0, len(bank.positions), 256):
        sl = slice(start, start + 256)
        similarities[bank.positions[sl]] = bank.vectors[sl] @ data.embeddings.T
    similarities = similarities.reshape(n, N_VIEWS, n)
    eligible = {f: np.flatnonzero((data.folds == f) & bank.eligible) for f in range(5)}
    evaluate = {f: ids[:4] if smoke else ids for f, ids in eligible.items()}
    active = set(np.concatenate(list(evaluate.values())).tolist())
    inner_references = {}
    choices = {"realm_inspired": [()] * n, "pebr_one_donor": [()] * n}
    decisions = []
    training_audit = []
    for outer in range(5):
        train_features, train_masks, train_log_gain = [], [], []
        train_counts = []
        print(f"Fold {outer + 1}/5 | Fit marginal retriever outside receiver fold {outer}", flush=True)
        for inner in range(5):
            if inner == outer:
                continue
            pair = tuple(sorted((outer, inner)))
            if pair not in inner_references:
                ref, audit = fit_inner_reference(data, pair)
                inner_references[pair] = (ref, ref.projection(donor_rows))
                save_audit(f"inner_reference_f{pair[0]}_f{pair[1]}", audit)
            ref, projections = inner_references[pair]
            ids = eligible[inner][:8] if smoke else eligible[inner]
            for i in ids:
                sl = slice(i * N_VIEWS, (i + 1) * N_VIEWS)
                permitted = np.flatnonzero((data.folds != outer) & (data.folds != inner)
                                           & (data.families != data.families[i]))
                lists = train_lists(similarities[i], permitted, LIMIT)
                features, allowed = features_for_view(similarities[i], lists)
                hg, _ = masked_log_gains(ref, bank.visible[sl].toarray(), bank.hidden[sl].toarray(),
                                         data.counts[i], lists, donor_rows, projections, False)
                train_features.append(features)
                train_masks.append(allowed)
                train_log_gain.append(np.column_stack((np.zeros(N_VIEWS), hg)))
            train_counts.append({"inner_fold": inner, "training_receivers": len(ids),
                                 "training_views": len(ids) * N_VIEWS})
        weights, attempts = fit_retriever(np.concatenate(train_features), np.concatenate(train_masks),
                                          np.concatenate(train_log_gain))
        training_audit.append({"evaluation_fold": outer, "training_by_fold": train_counts,
                               "weights": {"donor_intercept": float(weights[0]),
                                           "cosine_scale": float(weights[1])},
                               "optimizer_attempts": attempts, "evaluation_receivers": len(evaluate[outer])})
        save_audit("retrievers", training_audit)

        ref = frozen[outer]
        projections = ref.projection(donor_rows)
        for i in evaluate[outer]:
            lists = bank.candidates[i]
            choice, mass, null_mass = choose_retrieval_action(weights, similarities[i], lists, n)
            choices["realm_inspired"][i] = choice
            sl = slice(i * N_VIEWS, (i + 1) * N_VIEWS)
            hidden_gain, visible_gain = masked_log_gains(
                ref, bank.visible[sl].toarray(), bank.hidden[sl].toarray(), data.counts[i],
                lists, donor_rows, projections, True)
            pebr, utility = choose_pebr_action(lists, hidden_gain, visible_gain, data.counts[i].sum())
            choices["pebr_one_donor"][i] = pebr
            decisions.append({"receiver": int(i), "source_row": int(data.source[i]), "fold": outer,
                              "retrieval_donor": choice[0] if choice else -1,
                              "retrieval_choice_probability": mass, "retrieval_null_probability": null_mass,
                              "pebr_singleton_donor": pebr[0] if pebr else -1, "pebr_selection_utility": utility})
        print(f"Fold {outer + 1}/5 | Selected on {len(evaluate[outer])} held-out receivers", flush=True)
    return choices, active, decisions
