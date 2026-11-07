"""Direct frozen-encoder retrieval features, with no fabricated topic output."""
import numpy as np
from .base import ModelOutput


def prepare(embeddings):
    values = np.asarray(embeddings, dtype=np.float32)
    length = np.linalg.norm(values, axis=1, keepdims=True)
    features = np.divide(values, length, out=np.zeros_like(values), where=length > 0)
    return ModelOutput(None, None, features,
                       {'kind': 'direct_retrieval', 'stochastic': False,
                        'topic_model': False, 'source': 'same frozen MiniLM vectors from verified preparation'})
