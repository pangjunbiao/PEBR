"""Rebuild fold eligibility from verified cached embeddings; never encode text."""
from dataclasses import replace
import numpy as np
from .common import read, digest, key, write

def cached_queries(step5, corpus, collection, bank, records, protocol):
    from pebr_fresh_validation.banks import masked_evidence
    from pebr_revision_step3.io import complete
    view = masked_evidence(corpus, collection, records, protocol, 3711)
    for name in ('base', 'visible', 'targets'):
        delta = (view[name]-getattr(bank, name)).tocsr()
        if delta.nnz and np.max(np.abs(delta.data)) > 2e-12:
            raise ValueError('Known mask 3711 cannot be reconstructed: '+name)
    if not np.array_equal(view['eligible'], bank.eligible):
        raise ValueError('Known-mask eligibility changed')
    hashes = np.asarray([key(q) for q in view['queries']], dtype='U64')
    if not np.array_equal(hashes, np.load(bank.directory/'query_hashes.npy', allow_pickle=False)):
        raise ValueError('Known-mask query strings changed')
    if not np.array_equal(view['positions'], np.load(bank.directory/'query_positions.npy', allow_pickle=False)):
        raise ValueError('Known-mask query order changed')
    encoder = read(bank.directory/'summary.json')['encoder_identity']
    if corpus == 'transport_grievance':
        identity = key({'encoder': encoder, 'texts': view['queries']})
        path = step5/'cache/embeddings'/('vectors_'+identity+'.npy')
        receipt = read(path.with_suffix('.json'))
        if receipt['identity'] != identity or digest(path) != receipt['sha256']:
            raise ValueError('Cached English query embeddings changed')
    else:
        identity = key({'encoder': encoder, 'ordered_texts': list(view['queries'])})
        directory = step5/'cache/embeddings/chinese'/identity[:24]
        if not complete(directory):
            raise ValueError('Cached Chinese query embeddings are incomplete')
        if read(directory/'identity.json')['identity'] != identity:
            raise ValueError('Cached Chinese embedding identity differs')
        path = directory/'vectors.npy'
    vectors = np.load(path, allow_pickle=False, mmap_mode='r')
    dimension = np.load(collection.directory/'embeddings.npy', allow_pickle=False, mmap_mode='r').shape[1]
    if vectors.shape != (len(view['queries']), dimension) or not np.isfinite(vectors).all():
        raise ValueError('Invalid cached visible-query embeddings')
    return vectors, view['positions'], {'path': str(path), 'sha256': digest(path), 'queries': len(vectors),
                                       'identity': identity, 'encoder_called': False}

def retrieve_lists(collection, bank, query_vectors, positions, folds, protocol):
    from transport_topics.proposed.candidates import retrieve
    n = collection.counts.shape[0]
    m, limit = protocol['views'], protocol['candidates_per_view']
    embeddings = np.load(collection.directory/'embeddings.npy', allow_pickle=False)
    full_vectors = np.load(bank.directory/'full_receiver_embeddings.npy', allow_pickle=False)
    positive = np.asarray(collection.counts.sum(axis=1)).ravel() > 0
    lists = np.full((n*m, limit), -1, dtype=np.int32)
    lists[positions] = retrieve(query_vectors, positions//m, embeddings, collection.families, folds, positive, limit)
    full = retrieve(full_vectors, np.arange(n), embeddings, collection.families, folds, positive, limit)
    return lists.reshape(n, m, limit), full

def original_bank_check(collection, bank, vectors, positions, protocol):
    lists, full = retrieve_lists(collection, bank, vectors, positions, collection.folds, protocol)
    if not np.array_equal(lists, bank.candidates) or not np.array_equal(full, bank.full_candidates):
        raise ValueError('Cached embeddings do not reproduce the original eligible donor lists exactly')
    return {'known_mask': 3711, 'original_masked_and_full_candidate_lists_reproduced': True,
            'encoder_calls': 0, 'queries': len(vectors)}

def alternate_collection(collection, fold_seed):
    from transport_topics.data.splits import family_folds
    folds = family_folds(collection.families, folds=5, seed=fold_seed)
    if np.array_equal(folds, collection.folds):
        raise ValueError('The alternate fold assignment did not change')
    for family in np.unique(collection.families):
        if len(np.unique(folds[collection.families == family])) != 1:
            raise ValueError('An alternate fold splits a text family')
    identity = {'original_collection': collection.summary['identity_sha256'], 'family_fold_seed': fold_seed,
                'folds': folds.tolist(), 'count_vocabulary_and_source_order_unchanged': True}
    summary = dict(collection.summary, name=('zh' if collection.summary['name']=='chinese_pooled' else 'tg')+'_f'+str(fold_seed),
                   identity_sha256=key(identity))
    return replace(collection, summary=summary, folds=folds), identity

def validate_both_lists(collection,bank):
    n=collection.counts.shape[0]
    for i in range(n):
        ids=np.concatenate((bank.candidates[i].ravel(),bank.full_candidates[i]))
        ids=ids[ids>=0]
        if len(ids) and (np.any(ids>=n) or np.any(collection.folds[ids]==collection.folds[i]) or
                         np.any(collection.families[ids]==collection.families[i])):
            raise ValueError('Masked or full-receiver donors violate the current family/fold exclusions')
    return bank

def save_alternate_bank(directory, collection, original, vectors, positions, protocol):
    from scipy import sparse
    from transport_topics.data.prepared import SelectionBank
    from pebr_revision_step3.io import complete, save_receipt
    from pebr_fresh_validation.banks import load_bank
    if complete(directory):
        identity=read(directory/'identity.json')
        if (identity['collection']!=collection.summary or identity['frozen_rule_protocol_hash']!=key(protocol) or
            not np.array_equal(np.load(directory/'folds.npy',allow_pickle=False),collection.folds)):
            raise ValueError('Cached alternate-fold bank identity changed')
        return validate_both_lists(collection,load_bank(directory, collection, protocol))
    lists, full = retrieve_lists(collection, original, vectors, positions, collection.folds, protocol)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ('base', 'visible', 'targets'):
        sparse.save_npz(directory/(name+'.npz'), getattr(original, name))
    for name, values in (('eligible', original.eligible), ('candidates', lists), ('full_candidates', full),
                         ('full_receiver_embeddings', np.load(original.directory/'full_receiver_embeddings.npy', allow_pickle=False)),
                         ('query_positions', positions), ('folds', collection.folds)):
        np.save(directory/(name+'.npy'), values, allow_pickle=False)
    write(directory/'identity.json', {'collection': collection.summary, 'mask_base': 3711,
                                     'frozen_rule_protocol_hash': key(protocol), 'encoder_calls': 0})
    save_receipt(directory)
    return validate_both_lists(collection,load_bank(directory, collection, protocol))
