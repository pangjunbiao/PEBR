"""BTM through the pinned Cython bitermplus implementation."""
from importlib import metadata
import numpy as np
from ..data.preprocessing import WORD
from .base import ModelOutput


def ordered_documents(posts, vocabulary):
    mapping = {str(word): i for i, word in enumerate(vocabulary)}
    return [np.array([mapping[token] for token in WORD.findall(post['cleaned_text']) if token in mapping],
                     dtype=np.int32) for post in posts]


def fit(counts, posts, vocabulary, topics, seed, settings, iterations=None):
    import bitermplus as btm
    docs = ordered_documents(posts, vocabulary)
    biterms = btm.get_biterms(docs, win=settings['window'], as_array=True)
    epochs = settings['iterations'] if iterations is None else iterations
    model = btm.BTM(counts, np.asarray(vocabulary), T=topics,
                    M=min(20, len(vocabulary)), alpha=settings['alpha_mass']/topics,
                    beta=settings['beta'], seed=int(seed), win=settings['window'],
                    has_background=False)
    model.fit(biterms, iterations=epochs, verbose=False)
    theta = np.asarray(model.transform(docs, infer_type='sum_b', verbose=False), dtype=np.float64)
    beta = np.asarray(model.matrix_topics_words_, dtype=np.float64)
    if theta.shape != (len(docs), topics):
        raise ValueError('BTM inference changed original document alignment')
    return ModelOutput(beta, theta, theta.copy(),
                       {'implementation': 'bitermplus', 'version': metadata.version('bitermplus'),
                        'native_seed': int(seed), 'iterations': epochs,
                        'alpha': settings['alpha_mass']/topics, 'beta': settings['beta'],
                        'window': settings['window'], 'inference': 'sum_b',
                        'biterms': int(sum(len(row) for row in biterms)),
                        'one_token_documents': int(sum(len(row)==1 for row in docs)),
                        'empty_documents': int(sum(len(row)==0 for row in docs)),
                        'short_document_inference': 'native sum_b: word posterior for one token; uniform topics for no tokens',
                        'topic_score_kind': 'topic-word probabilities',
                        'training_device': 'cpu', 'termination': 'fixed Gibbs iterations'})
