"""Independent evaluation-only equations from the pinned native neural decoders.

No fitting, gradient steps, parameter changes or normalization of signed logits.
The GloCOM decoder normalizes over topics (axis 0), as its released code does.
"""
import numpy as np

def softmax(x, axis):
    x = x - np.max(x, axis=axis, keepdims=True)
    p = np.exp(x)
    return p/p.sum(axis=axis, keepdims=True)

def linear(x, state, name):
    return x@state[name+'.weight'].T + state[name+'.bias']

def batch_norm(x, state, name):
    return ((x-state[name+'.running_mean']) / np.sqrt(state[name+'.running_var']+np.float32(1e-5))
            *state[name+'.weight']+state[name+'.bias'])

def encode(x, state, suffix='', norm='mean_bn'):
    h = np.logaddexp(np.float32(0), linear(x, state, 'fc11'+suffix))
    h = np.logaddexp(np.float32(0), linear(h, state, 'fc12'+suffix))
    return batch_norm(linear(h, state, 'fc21'+suffix), state, norm)

def beta(state, method, temperature=.2):
    if method == 'knntm':
        return state['fcd1.weight'].T.copy()
    if method != 'glocom':
        raise ValueError('Only the pinned neural methods have a decoder replay')
    topics, words = state['topic_embeddings'], state['word_embeddings']
    distance = (np.sum(topics**2, axis=1, keepdims=True)+np.sum(words**2, axis=1)[None, :]
                -np.float32(2)*topics@words.T)
    return softmax(-distance/np.float32(temperature), axis=0)

def theta(counts, state, method, contexts=None, batch=200):
    output = []
    for start in range(0, counts.shape[0], batch):
        x = counts[start:start+batch]
        x = x.toarray() if hasattr(x, 'toarray') else np.asarray(x)
        x = x.astype(np.float32, copy=False)
        if method == 'knntm':
            value = softmax(encode(x, state), axis=1)
        elif method == 'glocom':
            if contexts is None:
                raise ValueError('GloCOM requires its prescribed global context')
            global_theta = softmax(encode(contexts[start:start+batch], state), axis=1)
            local = encode(x, state, '_adapter', 'adapter_mean_bn')
            value = softmax(global_theta*local, axis=1)
        else:
            raise ValueError('Unknown native method')
        output.append(value)
    return np.vstack(output)

def normalized_contributions(w, h):
    if np.any(w < 0) or np.any(h < 0) or not np.isfinite(w).all() or not np.isfinite(h).all():
        raise ValueError('NMF contributions require finite nonnegative factors')
    raw = np.asarray(w, dtype=float)*np.sum(np.abs(h), axis=1)[None, :]
    total = raw.sum(axis=1, keepdims=True)
    return np.divide(raw, total, out=np.zeros_like(raw), where=total > 0)
