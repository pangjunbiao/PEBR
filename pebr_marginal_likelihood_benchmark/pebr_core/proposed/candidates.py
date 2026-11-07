"""Stable cosine retrieval with receiving-fold/family and zero-count exclusions."""
import numpy as np


def retrieve(queries, receivers, donors, families, folds, positive, limit=20, batch_size=256):
    result=np.full((len(queries),limit),-1,dtype=np.int32)
    for start in range(0,len(queries),batch_size):
        scores=queries[start:start+batch_size] @ donors.T
        for offset, receiver in enumerate(receivers[start:start+batch_size]):
            allowed=positive & (folds!=folds[receiver]) & (families!=families[receiver])
            ids=np.flatnonzero(allowed)
            order=np.argsort(-scores[offset,ids],kind='stable')[:limit]
            result[start+offset,:len(order)]=ids[order]
    return result
