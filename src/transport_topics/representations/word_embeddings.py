"""Exact pretrained GloVe lookups, with fixed per-word OOV initialization."""
import json
import hashlib
import numpy as np
from ..runtime.artifacts import file_hash
from ..runtime.reproducibility import stable_seed

BUNDLE_HASHES = {
    200: 'b2652cc9fb8b026a6636191ba7deb396dc36d3c1de5c1f5076c63dbf39f6d94f',
    300: 'b67b9ae00be2a9968f7782b4c1a4229c5303339308f808c364a7e8ccf522bbbc',
}


def load_word_vectors(root, vocabulary, dimension):
    path = root/'assets/pretrained/word_embeddings'/('glove_6b_%dd_transport_lookup.npz' % dimension)
    if not path.is_file() or file_hash(path) != BUNDLE_HASHES[dimension]:
        raise ValueError('Missing or changed pretrained lookup asset: '+str(path))
    receipt = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    with np.load(path, allow_pickle=False) as data:
        mapping = {str(word): i for i, word in enumerate(data['words'])}
        unknown = [str(word) for word in vocabulary if str(word) not in mapping]
        if unknown:
            raise ValueError('The pretrained lookup must be rebuilt for new vocabulary terms: '+', '.join(unknown[:5]))
        ids = [mapping[str(word)] for word in vocabulary]
        values = data['vectors'][ids].astype(np.float32, copy=True)
        found = data['found'][ids].copy()
    for i in np.flatnonzero(~found):
        rng = np.random.default_rng(stable_seed(0, 'glove_oov', str(vocabulary[i]), dimension))
        values[i] = rng.normal(0, .6, dimension)
    if not np.isfinite(values).all() or (np.linalg.norm(values, axis=1) == 0).any():
        raise ValueError('Pretrained/OOV vectors must be finite and nonzero')
    audit = {'dimension': dimension, 'found': int(found.sum()), 'oov': int((~found).sum()),
             'lookup_sha256': BUNDLE_HASHES[dimension], 'source_sha256': receipt['source_sha256'],
             'aligned_values_sha256': hashlib.sha256(values.tobytes()).hexdigest(),
             'license': receipt['license'], 'oov_initialization': 'fixed SHA256 per word/dimension; normal(0,0.6)',
             'fitted_to_transport_text': False, 'annotations_used': False}
    return values, audit
