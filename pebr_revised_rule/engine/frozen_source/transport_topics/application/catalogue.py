"""Freeze query eligibility before reading method-specific representations."""
from dataclasses import dataclass
import json
import numpy as np
from ..config import canonical_hash
from ..data.preparation import frozen_json
from ..runtime.artifacts import file_hash


@dataclass
class QueryCohort:
    queries: list
    labels: list
    families: np.ndarray
    folds: np.ndarray
    source_rows: np.ndarray
    identity: str

    def eligible(self, query):
        keep=(self.families!=self.families[query]) & (self.folds!=self.folds[query])
        keep[query]=False
        return np.flatnonzero(keep)


def make_cohort(collection, annotations):
    labels=[tuple(annotations[str(int(row))] if str(int(row)) in annotations else annotations[int(row)])
            for row in collection.source_rows]
    queries=[]
    for q, categories in enumerate(labels):
        if len(categories)!=1:continue
        eligible=(collection.families!=collection.families[q]) & (collection.folds!=collection.folds[q])
        eligible[q]=False
        relevant=sum(bool(eligible[i]) and categories[0] in labels[i] for i in range(len(labels)))
        if relevant:
            queries.append({'document_index':q,'source_row':int(collection.source_rows[q]),
                            'category':categories[0],'eligible_candidates':int(eligible.sum()),
                            'relevant_candidates':int(relevant)})
    identity=canonical_hash({'condition':collection.summary['identity_sha256'],'queries':queries,
                             'labels':labels,'families':collection.families.tolist(),
                             'folds':collection.folds.tolist(),'source_rows':collection.source_rows.tolist(),
                             'policy':'all retained posts outside query family/fold; zero query stays in cohort'})
    return QueryCohort(queries,labels,collection.families,collection.folds,collection.source_rows,identity)


def load_cohort(root, collection, preparation):
    from ..data.loaders import load_primary
    from ..data.sources import SOURCES
    path=root/preparation['annotation_file']
    annotation=json.loads(path.read_text(encoding='utf-8'))
    if annotation['source_sha256']!=json.loads((collection.directory/'identity.json').read_text())['source_hash']:
        raise ValueError('Evaluation annotations refer to a different primary source')
    raw=root/SOURCES['primary'][0]
    if file_hash(raw)!=annotation['source_sha256']:raise ValueError('Primary annotation source bytes changed')
    _,published=load_primary(raw)
    if {str(k):v for k,v in published.items()}!=annotation['categories_by_source_row']:
        raise ValueError('Stored evaluation categories differ from the pinned released annotations')
    cohort=make_cohort(collection,annotation['categories_by_source_row'])
    original=json.loads((root/preparation['retrieval_cohort']['path']).read_text(encoding='utf-8'))
    if original['condition_identity']!=collection.summary['identity_sha256'] or original['queries']!=cohort.queries:
        raise ValueError('The evaluation query cohort differs from frozen preparation')
    frozen_json(root/'data/manifests/queries'/('evaluation_'+cohort.identity[:20]+'.json'),
                {'identity':cohort.identity,'annotation_sha256':file_hash(path),
                 'queries':cohort.queries,'query_count':len(cohort.queries),
                 'candidate_rule':'fixed family/fold exclusions, independent of model outputs and labels',
                 'zero_query_policy':'empty response, unchanged catalogue and common cohort'})
    return cohort
