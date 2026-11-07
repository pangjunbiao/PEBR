"""Aligned model outputs; a direct encoder retriever is not a topic model."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ModelOutput:
    topic_scores: np.ndarray | None
    doc_topics: np.ndarray | None
    retrieval_features: np.ndarray
    audit: dict
    extra_arrays: dict = field(default_factory=dict)

    def validate(self, posts, vocabulary, topics, direct_retriever=False):
        features = np.asarray(self.retrieval_features)
        if features.ndim != 2 or features.shape[0] != posts or not np.isfinite(features).all():
            raise ValueError('Model retrieval features are not finite/aligned with original posts')
        if direct_retriever:
            if self.topic_scores is not None or self.doc_topics is not None:
                raise ValueError('A direct retriever must not fabricate topics or topic memberships')
        else:
            scores, theta = np.asarray(self.topic_scores), np.asarray(self.doc_topics)
            if scores.shape != (topics, vocabulary) or theta.shape != (posts, topics):
                raise ValueError('Topic model outputs have inconsistent document/vocabulary dimensions')
            if not np.isfinite(scores).all() or not np.isfinite(theta).all() or (theta < -1e-10).any():
                raise ValueError('Topic scores or nonnegative memberships are invalid')
            if features.shape[1] != topics:
                raise ValueError('Topic retrieval features must use the same topic coordinates')
        return self
