import hashlib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS


def count_representation(texts, min_df=2):
    stopwords=sorted(set(ENGLISH_STOP_WORDS)-{'no','not','nor','never'})
    vectorizer=CountVectorizer(stop_words=stopwords,min_df=min_df,token_pattern=r'(?u)\b[a-z][a-z]+\b')
    counts=vectorizer.fit_transform(texts).astype(np.float64).tocsr()
    counts.sort_indices()
    vocabulary=vectorizer.get_feature_names_out().astype(str)
    return counts,vocabulary,hashlib.sha256('\n'.join(stopwords).encode()).hexdigest()
