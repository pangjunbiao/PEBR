"""The common NMF backbone for original, similarity and rule controls."""
from ..topics.learning import fit_topics


def fit(original_counts, learning_rows, topics, seed, settings):
    return fit_topics(original_counts, learning_rows, topics, seed, settings)
