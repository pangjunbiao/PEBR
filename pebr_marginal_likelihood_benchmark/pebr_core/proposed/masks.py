"""Original word-type masking seeds and ordered, leakage-checked queries."""
import hashlib
import numpy as np
from scipy import sparse
from ..data.preprocessing import query_without, WORD


def mask_seed(row, base):
    return int.from_bytes(hashlib.sha256(f'{base}:{row}'.encode()).digest()[:8],'little')


def build_views(counts, words, texts, source_rows, settings, audit=False):
    method=settings['method']; n,V=counts.shape; M=method['views']['count']
    base=counts.copy().tolil(); outer=sparse.lil_matrix(counts.shape,dtype=np.float64)
    forbidden=[]
    for i in range(n):
        ids=counts[i].indices
        hidden=np.array([],dtype=int)
        if audit and len(ids):
            rng=np.random.default_rng(mask_seed(int(source_rows[i]),method['predictor']['outer_seed']))
            hidden=rng.permutation(ids)[:max(1,int(np.ceil(settings['evaluation']['prediction']['outer_fraction']*len(ids))))]
            outer[i,hidden]=counts[i,hidden].toarray(); base[i,hidden]=0
        forbidden.append(set(words[hidden]))
    base=base.tocsr(); base.eliminate_zeros(); base.sort_indices(); outer=outer.tocsr()
    eligible=np.diff(base.indptr)>=method['views']['min_word_types']
    visible=sparse.lil_matrix((n*M,V),dtype=np.float64)
    targets=sparse.lil_matrix((n*M,V),dtype=np.float64)
    queries=[]; positions=[]; full_queries=[]
    for i in range(n):
        full_queries.append(query_without(texts[i],forbidden[i]))
        if not eligible[i]: continue
        ids=base[i].indices
        rng=np.random.default_rng(mask_seed(int(source_rows[i]),method['predictor']['mask_seed']))
        for m in range(M):
            permutation=rng.permutation(ids)
            cut=max(2,int(np.floor(len(ids)*(1-method['views']['hide_fraction']))))
            if len(ids)-cut<2: raise ValueError('Mask fraction leaves fewer than two hidden types')
            v,h=permutation[:cut],permutation[cut:]
            values=base[i,v].toarray().ravel()
            visible[i*M+m,v]=values/values.sum()
            targets[i*M+m,h]=base[i,h].toarray()
            blocked=forbidden[i]|set(words[h]); query=query_without(texts[i],blocked)
            if set(WORD.findall(query)) & blocked: raise AssertionError('Hidden type leaked into retrieval query')
            queries.append(query); positions.append(i*M+m)
    return {'base':base,'outer':outer,'visible':visible.tocsr(),'targets':targets.tocsr(),
            'eligible':eligible,'queries':queries,'positions':np.asarray(positions,dtype=np.int32),
            'full_queries':full_queries}
