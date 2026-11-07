import numpy as np
from sklearn.preprocessing import normalize


def idf_weights(counts):
    return 1+np.log((1+counts.shape[0])/(1+np.asarray((counts>0).sum(axis=0)).ravel()))


def transform(counts, idf):
    return normalize(counts.multiply(idf),norm='l2').tocsr()
