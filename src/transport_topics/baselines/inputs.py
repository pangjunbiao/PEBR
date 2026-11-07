"""Published baseline inputs, constructed from original text evidence only."""
from pathlib import Path
import json
import time
import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from ..config import canonical_hash
from ..data.preparation import check_receipt, complete, save_array
from ..runtime.artifacts import file_hash, write_json


def global_context(counts, embeddings, clusters, seed):
    number = min(int(clusters), len(embeddings))
    if number < 1:
        raise ValueError('Global contexts require at least one original post')
    with threadpool_limits(limits=1):
        labels = KMeans(n_clusters=number, random_state=seed, n_init=10, algorithm='lloyd').fit_predict(embeddings)
    x = counts.toarray() if sparse.issparse(counts) else np.asarray(counts)
    sums = np.zeros((number, x.shape[1]), dtype=np.float32)
    np.add.at(sums, labels, x)
    return sums[labels], labels


def word_costs(counts, word_vectors):
    unit = word_vectors / np.linalg.norm(word_vectors, axis=1, keepdims=True)
    cosine = np.clip((1.0 - unit.astype(np.float64) @ unit.astype(np.float64).T)/2.0, 0, 1)
    presence = (counts > 0).astype(np.float64)
    co = (presence.T @ presence).toarray() if sparse.issparse(presence) else presence.T @ presence
    df = np.asarray(presence.sum(axis=0)).ravel()
    conditional = np.divide(co, df[None, :], out=np.zeros_like(co), where=df[None, :] > 0)
    cooccurrence = np.clip(1.0 - (conditional + conditional.T)/2.0, 0, 1)
    np.fill_diagonal(cosine, 0); np.fill_diagonal(cooccurrence, 0)
    return cosine, cooccurrence


def transport_distances(root, collection, word_vectors, vector_audit, logger):
    import ot
    spec = {'schema': 1, 'condition': collection.summary['identity_sha256'],
            'source_vectors': vector_audit, 'ot_version': ot.__version__,
            'algorithm': 'unregularized exact balanced transport, normalized original BOW, symmetric pairwise distances',
            'implementation': file_hash(Path(__file__))}
    key = canonical_hash(spec)
    directory = root/'cache/baseline_inputs/knntm'/key[:20]
    directory.mkdir(parents=True, exist_ok=True)
    if check_receipt(directory):
        return directory, json.loads((directory/'audit.json').read_text(encoding='utf-8'))
    costs = word_costs(collection.counts, word_vectors)
    rows = collection.counts.tocsr()
    support = [rows[i].indices for i in range(rows.shape[0])]
    weights = [rows[i].data/rows[i].data.sum() if rows[i].nnz else np.empty(0) for i in range(rows.shape[0])]
    size = rows.shape[0]
    matrices = [np.zeros((size, size), dtype=np.float64) for _ in range(2)]
    progress = directory/'progress.npz'
    start_row = 0
    if progress.exists():
        with np.load(progress, allow_pickle=False) as data:
            if str(data['identity'].item()) != key:
                raise ValueError('OT distance checkpoint identity differs')
            start_row = int(data['next_row']); matrices = [data['cosine'].copy(), data['cooccurrence'].copy()]
    tick = time.monotonic()
    for i in range(start_row, size):
        for j in range(i+1, size):
            for kind, cost in enumerate(costs):
                if not len(support[i]) or not len(support[j]):
                    distance = float(bool(len(support[i]) or len(support[j])))
                else:
                    distance = float(ot.emd2(weights[i], weights[j], cost[np.ix_(support[i], support[j])],
                                             numItermax=100000, numThreads=1, check_marginals=True))
                if not np.isfinite(distance) or distance < -1e-8:
                    raise ArithmeticError('Invalid exact document transport distance')
                matrices[kind][i, j] = matrices[kind][j, i] = max(distance, 0.0)
        if (i+1) % 20 == 0 or i+1 == size:
            temporary = progress.with_suffix('.partial.npz')
            np.savez_compressed(temporary, identity=np.array(key), next_row=np.array(i+1),
                                cosine=matrices[0], cooccurrence=matrices[1])
            temporary.replace(progress)
        if i == start_row or time.monotonic()-tick > 10 or i+1 == size:
            logger.info('kNNTM transport preparation | %s | rows %d/%d.', collection.summary['name'], i+1, size)
            tick = time.monotonic()
    for name, matrix in zip(('M_cos', 'M_coo'), matrices):
        np.savez_compressed(directory/(name+'.npz'), matrix.astype(np.float32))
    audit = {'identity': key, 'posts': size, 'algorithm': spec['algorithm'],
             'cosine_cost': '(1-cosine(GloVe300))/2',
             'cooccurrence_cost': '1-(df(i,j)/df(i)+df(i,j)/df(j))/2',
             'empty_document_convention': 'zero to another empty row; maximum cost one to nonempty rows',
             'published_reproduction_input': True, 'source_vectors': vector_audit}
    write_json(directory/'identity.json', spec, overwrite=True)
    write_json(directory/'audit.json', audit, overwrite=True)
    complete(directory)
    return directory, audit
