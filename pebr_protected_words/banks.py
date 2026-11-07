"""Reserve outer targets before any query construction, then build inner views."""
from __future__ import annotations
import time
import numpy as np
from scipy import sparse
from transport_topics.data.prepared import SelectionBank
from transport_topics.proposed.masks import build_views
from transport_topics.proposed.candidates import retrieve
from transport_topics.chinese.data import visible_queries
from transport_topics.data.preprocessing import WORD
from pebr_revision_step3.io import read_json, write_json, save_receipt, complete, canonical_hash
from pebr_fresh_validation.banks import view_configuration

def masked_inputs(corpus, collection, records, protocol, outer_seed):
    words = np.load(collection.directory/'vocabulary.npy', allow_pickle=False)
    cfg = view_configuration(collection, protocol, protocol['inner_mask_base'])
    cfg['method']['predictor']['outer_seed'] = int(outer_seed)
    cfg['evaluation']['prediction']['outer_fraction'] = protocol['outer_type_fraction']
    texts = [r['cleaned_text'] for r in records] if corpus == 'transport_grievance' else ['']*len(records)
    view = build_views(collection.counts, words, texts, collection.source_rows, cfg, True)
    blocked = [set(words[view['outer'][i].indices]) for i in range(len(records))]
    if corpus == 'chinese_pooled':
        sanitized = []
        full = []
        for i, record in enumerate(records):
            segments = [' ' if token in blocked[i] else token for token in record['segments']]
            full.append(''.join(segments))
            sanitized.append({'segments': segments, 'text': full[-1],
                              'tokens': [w for w in record['tokens'] if w not in blocked[i]]})
            if set(segments) & blocked[i]:
                raise ArithmeticError('A protected Chinese word entered the outer-visible query')
        visible, targets, eligible, queries, positions = visible_queries(
            view['base'], words, sanitized, collection.source_rows, cfg)
        for a, b in ((visible, view['visible']), (targets, view['targets'])):
            delta = (a-b).tocsr()
            if delta.nnz and np.max(np.abs(delta.data)) > 2e-12:
                raise ValueError('Chinese and original numerical mask streams disagree')
        if not np.array_equal(eligible, view['eligible']) or not np.array_equal(positions, view['positions']):
            raise ValueError('Chinese protected masking eligibility/positions differ')
        view.update(visible=visible, targets=targets, eligible=eligible, queries=queries,
                    positions=positions, full_queries=full)
        for query, pos in zip(queries, positions):
            i = int(pos)//protocol['views']
            hidden = set(words[targets[int(pos)].indices])
            expected = ''.join(' ' if token in hidden else token for token in sanitized[i]['segments'])
            kept = [token for token in sanitized[i]['segments'] if token not in hidden]
            if query != expected or set(kept) & (blocked[i] | hidden):
                raise ArithmeticError('A protected/inner-hidden Chinese segment entered retrieval')
    else:
        for i, text in enumerate(view['full_queries']):
            if set(WORD.findall(text)) & blocked[i]:
                raise ArithmeticError('A protected English word entered the outer-visible query')
        for query, position in zip(view['queries'], view['positions']):
            i = int(position)//protocol['views']
            hidden = set(words[view['targets'][int(position)].indices])
            if set(WORD.findall(query)) & (blocked[i] | hidden):
                raise ArithmeticError('A protected/inner-hidden English word entered retrieval')
    difference = (view['base']+view['outer']-collection.counts).tocsr()
    if difference.nnz or view['base'].multiply(view['outer']).nnz:
        raise ArithmeticError('Outer-visible and protected counts do not partition the original')
    for start in range(0, collection.counts.shape[0], 100):
        end = min(start+100, collection.counts.shape[0])
        held = view['outer'][np.repeat(np.arange(start, end), protocol['views'])]
        if held.multiply(view['visible'][start*protocol['views']:end*protocol['views']]).nnz or held.multiply(view['targets'][start*protocol['views']:end*protocol['views']]).nnz:
            raise ArithmeticError('A protected word entered an inner numeric view')
    return view

def load_bank(directory, collection, protocol):
    if not complete(directory):
        raise ValueError('Protected-word evidence bank is incomplete')
    bank = SelectionBank('outer_audit', directory,
        *[sparse.load_npz(directory/(n+'.npz')).tocsr() for n in ('base', 'visible', 'targets')],
        *[np.load(directory/(n+'.npy'), allow_pickle=False) for n in ('eligible', 'candidates', 'full_candidates')])
    outer = sparse.load_npz(directory/'outer.npz').tocsr()
    n, v = collection.counts.shape
    if bank.visible.shape != (n*protocol['views'], v) or outer.shape != (n, v):
        raise ValueError('Protected-word bank shape differs')
    if (bank.base+outer-collection.counts).nnz or bank.base.multiply(outer).nnz:
        raise ValueError('Protected-word bank changed the original count partition')
    if bank.visible.multiply(bank.targets).nnz:
        raise ValueError('Inner visible and target supports overlap')
    if bank.candidates.shape != (n, protocol['views'], protocol['candidates_per_view']) or bank.full_candidates.shape != (n, protocol['candidates_per_view']):
        raise ValueError('Protected retrieval-list shape differs')
    for i in range(n):
        ids = np.concatenate((bank.candidates[i].ravel(), bank.full_candidates[i]))
        ids = ids[ids >= 0]
        if len(ids) and (np.any(collection.folds[ids] == collection.folds[i]) or np.any(collection.families[ids] == collection.families[i])):
            raise ValueError('A protected query uses an excluded family/fold donor')
    return bank

def prepare_bank(corpus, collection, records, protocol, outer_seed, encoder, directory, logger):
    if complete(directory):
        return load_bank(directory, collection, protocol)
    started = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    view = masked_inputs(corpus, collection, records, protocol, outer_seed)
    logger.info('Protected retrieval | %s outer=%d | %d inner queries', corpus, outer_seed, len(view['queries']))
    full = encoder.encode(view['full_queries'], label=corpus+'_outer'+str(outer_seed)+'_visible_receiver')
    masked = encoder.encode(view['queries'], label=corpus+'_outer'+str(outer_seed)+'_inner_visible')
    embeddings = np.load(collection.directory/'embeddings.npy', allow_pickle=False)
    n, v = collection.counts.shape
    m, limit = protocol['views'], protocol['candidates_per_view']
    if full.shape != (n, embeddings.shape[1]) or masked.shape != (len(view['queries']), embeddings.shape[1]) or not np.isfinite(full).all() or not np.isfinite(masked).all():
        raise ValueError('Protected-query embedding shape or values differ')
    positive = np.asarray(collection.counts.sum(axis=1)).ravel() > 0
    candidates = np.full((n*m, limit), -1, dtype=np.int32)
    if len(view['positions']):
        candidates[view['positions']] = retrieve(masked, view['positions']//m, embeddings, collection.families, collection.folds, positive, limit)
    # Both full-receiver ranking and transfer availability use OUTER-VISIBLE text only.
    available = retrieve(full, np.arange(n), embeddings, collection.families, collection.folds, positive, limit)
    for name in ('base', 'outer', 'visible', 'targets'):
        sparse.save_npz(directory/(name+'.npz'), view[name])
    for name, value in [('eligible', view['eligible']), ('candidates', candidates.reshape(n, m, limit)),
                        ('full_candidates', available), ('full_receiver_embeddings', full),
                        ('query_positions', view['positions']),
                        ('query_hashes', np.asarray([canonical_hash(q) for q in view['queries']], dtype='U64')),
                        ('full_query_hashes', np.asarray([canonical_hash(q) for q in view['full_queries']], dtype='U64'))]:
        np.save(directory/(name+'.npy'), value, allow_pickle=False)
    write_json(directory/'summary.json', {'corpus': corpus, 'outer_seed': int(outer_seed),
        'inner_mask_base': protocol['inner_mask_base'], 'retained_posts': n,
        'audit_eligible_posts': int(view['eligible'].sum()),
        'protected_target_tokens_in_common_cohort': int(view['outer'][view['eligible']].sum()),
        'queries': len(view['queries']), 'encoder_identity': encoder.identity,
        'outer_targets_removed_before_queries_and_selection': True,
        'full_query_ranking_and_transfer_availability_recomputed': True,
        'donor_vectors_use_other_posts_original_text_with_fold_family_exclusions': True,
        'seconds': time.perf_counter()-started})
    save_receipt(directory)
    return load_bank(directory, collection, protocol)
