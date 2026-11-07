"""Common top-word ranking, document NPMI, diversity and supplementary C_v."""
from itertools import combinations
from importlib import metadata
import numpy as np


def top_words(scores,vocabulary,topn=10):
    scores=np.asarray(scores)
    if scores.ndim!=2 or scores.shape[1]!=len(vocabulary) or not np.isfinite(scores).all():
        raise ValueError('Invalid topic-word score/vocabulary alignment')
    if topn<2 or topn>scores.shape[1]:raise ValueError('The fixed displayed word count must fit the vocabulary')
    return np.asarray(vocabulary)[np.argsort(-scores,axis=1,kind='stable')[:,:topn]]


def npmi_topics(topics,reference):
    mapping={str(word):i for i,word in enumerate(reference.vocabulary)}
    result=[]
    for words in topics:
        if any(str(word) not in mapping for word in words):raise ValueError('Topic words are outside the fixed reference')
        ids=[mapping[str(word)] for word in words]
        values=[]
        for left,right in combinations(ids,2):
            count=float(reference.cooccurrence[left,right])
            if count==0:values.append(-1.)
            elif count==reference.documents:values.append(0.)
            else:
                value=np.log(count*reference.documents/(reference.df[left]*reference.df[right]))/-np.log(count/reference.documents)
                values.append(float(np.clip(value,-1,1)))
        result.append(float(np.mean(values)))
    return np.asarray(result)


def evaluate_topics(scores,vocabulary,reference,topn=10,with_cv=True):
    words=top_words(scores,vocabulary,topn)
    per_topic=npmi_topics(words,reference)
    result={'npmi':float(per_topic.mean()),'diversity':len(set(words.ravel()))/words.size,
            'npmi_per_topic':per_topic.tolist(),'top_words':words.tolist(),
            'constant_score_topics':int(np.sum(np.ptp(np.asarray(scores),axis=1)==0)),
            'reference_identity':reference.identity,'reference_documents':reference.documents,
            'top_word_ties':'stable saved vocabulary order','npmi_zero_cooccurrence':-1.,
            'npmi_universal_pair':0.,'evaluation_only_original_reference':True}
    if with_cv:
        from gensim.models import CoherenceModel
        model=CoherenceModel(topics=words.tolist(),texts=reference.tokens,dictionary=reference.dictionary(),
                             coherence='c_v',window_size=110,processes=1,topn=topn)
        cv=np.asarray(model.get_coherence_per_topic(),dtype=float)
        if not np.isfinite(cv).all():raise ArithmeticError('Supplementary C_v returned nonfinite values')
        result.update(cv=float(cv.mean()),cv_per_topic=cv.tolist(),cv_implementation='gensim '+metadata.version('gensim'),cv_window=110)
    return result
