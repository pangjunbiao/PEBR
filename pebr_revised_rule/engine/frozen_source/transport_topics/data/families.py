"""Exact character-TFIDF threshold components with bounded working memory."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Components:
    def __init__(self, n): self.parent = np.arange(n)
    def find(self, i):
        while self.parent[i]!=i:
            self.parent[i]=self.parent[self.parent[i]]; i=int(self.parent[i])
        return i
    def join(self, a, b):
        a,b=self.find(a),self.find(b)
        if a!=b: self.parent[max(a,b)]=min(a,b)
    def labels(self):
        roots=[self.find(i) for i in range(len(self.parent))]
        mapping={v:i for i,v in enumerate(sorted(set(roots)))}
        return np.array([mapping[v] for v in roots],dtype=np.int32)


def text_families(texts, threshold=.9, block_size=128, logger=None):
    if not texts: return np.empty(0,dtype=np.int32)
    components=Components(len(texts))
    try:
        matrix=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),dtype=np.float64).fit_transform(texts)
    except ValueError:
        return components.labels()
    for start in range(0,len(texts),block_size):
        similarity=(matrix[start:start+block_size] @ matrix.T).tocoo()
        keep=similarity.data>=threshold
        for a,b in zip(similarity.row[keep]+start, similarity.col[keep]):
            if a<b: components.join(int(a),int(b))
        if logger and (start==0 or (start//block_size)%20==0):
            logger.info('Family audit: %d/%d posts.',min(start+block_size,len(texts)),len(texts))
    return components.labels()


def cross_source_exclusions(primary, external, families, threshold=.9, block_size=128):
    # Joint IDF is used only for the documented cross-source overlap audit.
    if not primary or not external: return np.zeros(len(external),dtype=bool)
    transform=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),dtype=np.float64)
    matrix=transform.fit_transform(primary+external)
    p=matrix[:len(primary)]; e=matrix[len(primary):]
    direct=np.zeros(len(external),dtype=bool)
    exact=set(primary)
    for start in range(0,len(external),block_size):
        scores=e[start:start+block_size] @ p.T
        direct[start:start+scores.shape[0]]=np.asarray(scores.max(axis=1).toarray()).ravel()>=threshold
    direct |= np.array([text in exact for text in external])
    bad=np.unique(families[direct])
    return np.isin(families,bad)
