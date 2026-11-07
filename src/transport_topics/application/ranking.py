"""Cosine ranking has no annotation input and never restricts to donors."""
import numpy as np


def rank_one(features, query, eligible, source_rows, budget):
    values=np.asarray(features,dtype=np.float64)
    ids=np.asarray(eligible,dtype=np.int64)
    if values.ndim!=2 or not np.isfinite(values).all():raise ValueError('Invalid retrieval features')
    if budget<1 or len(set(ids.tolist()))!=len(ids) or query in ids:
        raise ValueError('Invalid fixed retrieval catalogue or budget')
    norm=np.linalg.norm(values[query])
    if norm==0 or not len(ids):return np.empty(0,dtype=np.int64),np.empty(0)
    candidate_norm=np.linalg.norm(values[ids],axis=1)
    score=np.divide(values[ids]@values[query],candidate_norm*norm,
                    out=np.zeros(len(ids)),where=candidate_norm>0)
    order=np.lexsort((np.asarray(source_rows)[ids],-score))[:budget]
    return ids[order],score[order]


def rank_queries(features, cohort, budget):
    ranks=np.full((len(cohort.queries),budget),-1,dtype=np.int64)
    scores=np.zeros((len(cohort.queries),budget),dtype=np.float64)
    for row,query in enumerate(cohort.queries):
        ids,values=rank_one(features,query['document_index'],cohort.eligible(query['document_index']),
                            cohort.source_rows,budget)
        ranks[row,:len(ids)],scores[row,:len(ids)]=ids,values
    return ranks,scores
