"""Conventional NMF on transferred inputs, followed by original-post inference."""
from ..baselines.base import ModelOutput
from ..numerics.nmf import fit_nmf
from ..representations.tfidf import idf_weights, transform
from ..schemas import number
from .memberships import infer_original_memberships


def fit_topics(original_counts, learning_rows, topics, seed, settings):
    weights = idf_weights(original_counts)
    original = transform(original_counts, weights)
    augmented = transform(learning_rows, weights)
    coefficients, dictionary, fit = fit_nmf(
        augmented, topics, seed, number(settings['tolerance'],'tolerance'), settings['max_iterations'])
    memberships, features, inference = infer_original_memberships(original, dictionary)
    return ModelOutput(dictionary, memberships, features,
                       {'fit': fit, 'original_membership_inference': inference,
                        'idf_estimated_from': 'original unaugmented collection',
                        'retrieval_representation': 'W_ik times topic-row L1 mass, normalized per original post',
                        'topic_score_kind': 'nonnegative NMF weights',
                        'training_coefficients_used_for_retrieval': False},
                       {'augmented_training_coefficients': coefficients, 'final_idf': weights})
