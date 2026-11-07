"""Original-text coherence references, never supplied to fitting or selection."""
from dataclasses import dataclass,field
from pathlib import Path
import json
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer,ENGLISH_STOP_WORDS
from ..config import canonical_hash
from ..data.sources import SOURCES,obtain
from ..data.loaders import load_primary,load_external
from ..data.preprocessing import deduplicate,WORD
from ..data.families import cross_source_exclusions
from ..data.preparation import parent_pool,check_receipt,complete,save_array,save_matrix,frozen_json
from ..runtime.artifacts import file_hash


@dataclass
class TextReference:
    identity: str
    name: str
    vocabulary: np.ndarray
    documents: int
    df: np.ndarray
    cooccurrence: sparse.csr_matrix
    tokens: list
    directory: Path
    _dictionary: object=field(default=None,repr=False)

    def dictionary(self):
        if self._dictionary is None:
            from gensim.corpora import Dictionary
            self._dictionary=Dictionary(self.tokens)
        return self._dictionary


def tokenize_reference(texts):
    stop=set(ENGLISH_STOP_WORDS)-{'no','not','nor','never'}
    return [[word for word in WORD.findall(text) if word not in stop] for text in texts]


def make_reference(root,name,texts,vocabulary,source_identity,logger):
    vocabulary=np.array(sorted(set(map(str,vocabulary))))
    spec={'schema':1,'source':source_identity,'name':name,'texts':canonical_hash(texts),
          'vocabulary':vocabulary.tolist(),'implementation':file_hash(Path(__file__)),
          'probability_unit':'one original retained document, not augmented text',
          'npmi_unseen_pair':-1,'npmi_universal_pair':0,'cv_window':110}
    key=canonical_hash(spec)
    directory=root/'cache/coherence'/key[:24]
    directory.mkdir(parents=True,exist_ok=True)
    if not check_receipt(directory):
        tokens=tokenize_reference(texts)
        vectorizer=CountVectorizer(vocabulary={word:i for i,word in enumerate(vocabulary)},
                                  analyzer=lambda row:row,binary=True,dtype=np.int64)
        binary=vectorizer.fit_transform(tokens).tocsr()
        df=np.asarray(binary.sum(axis=0)).ravel()
        if not len(texts) or (df==0).any():raise ValueError('Reference lacks a fitting vocabulary word or documents')
        save_matrix(directory/'cooccurrence.npz',(binary.T@binary).tocsr())
        save_array(directory/'df.npy',df);save_array(directory/'vocabulary.npy',vocabulary)
        frozen_json(directory/'tokens.json',tokens)
        frozen_json(directory/'identity.json',spec)
        frozen_json(directory/'reference.json',{'identity':key,'name':name,'documents':len(texts),
                                               'vocabulary':len(vocabulary),'evaluation_only':True})
        complete(directory)
    info=json.loads((directory/'reference.json').read_text())
    logger.info('Coherence reference verified | %s | %d original documents; fixed across model/sample/size conditions.',name,info['documents'])
    return TextReference(key,name,np.load(directory/'vocabulary.npy',allow_pickle=False),info['documents'],
                         np.load(directory/'df.npy',allow_pickle=False),sparse.load_npz(directory/'cooccurrence.npz').tocsr(),
                         json.loads((directory/'tokens.json').read_text(encoding='utf-8')),directory)


def prepare_references(root,preparation,config,logger):
    # Source labels are consumed solely by the already declared external transport filter.
    primary_source=obtain(root,SOURCES['primary'],logger,False)
    external_source=obtain(root,SOURCES['external'],logger,False)
    primary,_=load_primary(root/primary_source['path'])
    primary,texts,_=deduplicate(primary)
    external,_=load_external(root/external_source['path'])
    external,external_texts,_=deduplicate(external)
    parents=parent_pool(root,external,external_texts,'dabiri_traffic',external_source['sha256'],config,logger)
    drop=cross_source_exclusions(texts,external_texts,parents,config['preparation']['family_threshold'],
                                config['preparation']['family_block_size'])
    external_texts=[t for t,reject in zip(external_texts,drop) if not reject]
    if len(external_texts)!=preparation['external']['normalized_unique_after_overlap']:
        raise ValueError('The external evaluation population differs from the frozen assembly audit')
    words={'primary':set(),'external':set()}
    for condition in preparation['conditions']:
        role='primary' if condition['name'].startswith('primary') else 'external'
        words[role].update(np.load(root/condition['path']/'vocabulary.npy',allow_pickle=False).tolist())
    refs={'primary':make_reference(root,'primary_original',texts,words['primary'],primary_source['sha256'],logger)}
    if words['external']:
        refs['external']=make_reference(root,'external_original_transport_population',external_texts,words['external'],
                                         {'source':external_source['sha256'],'primary_overlap_reference':primary_source['sha256']},logger)
    return refs
