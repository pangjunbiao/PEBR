"""Binary category relevance; equal category weighting after query averages."""
import numpy as np


def query_scores(ranked_relevance, relevant_count, budget):
    if relevant_count<1 or budget<1:raise ValueError('Retrieval metrics require a positive fixed relevance denominator and budget')
    relevance=np.asarray(ranked_relevance,dtype=np.float64)[:budget]
    discount=1/np.log2(np.arange(budget)+2)
    dcg=float(relevance@discount[:len(relevance)])
    ideal=float(discount[:min(budget,relevant_count)].sum())
    return dcg/ideal,float(relevance.sum()/relevant_count)


def evaluate_rankings(ranks,cohort,budgets):
    values=np.zeros((len(cohort.queries),len(budgets),2),dtype=np.float64)
    for row,query in enumerate(cohort.queries):
        ids=np.asarray(ranks[row]);ids=ids[ids>=0]
        eligible=set(cohort.eligible(query['document_index']).tolist())
        if len(set(ids.tolist()))!=len(ids) or any(int(i) not in eligible for i in ids):
            raise ValueError('A ranking violates the fixed catalogue')
        relevant=[query['category'] in cohort.labels[int(i)] for i in ids]
        for b,budget in enumerate(budgets):
            values[row,b]=query_scores(relevant,query['relevant_candidates'],budget)
    return values


def macro_scores(values,cohort):
    labels=np.array([q['category'] for q in cohort.queries])
    categories=sorted(set(labels))
    if not categories:raise ValueError('The fixed primary query cohort is empty')
    per_category={name:np.mean(values[labels==name],axis=0) for name in categories}
    return np.mean(list(per_category.values()),axis=0),per_category
