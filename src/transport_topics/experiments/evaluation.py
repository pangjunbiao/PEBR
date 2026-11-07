"""Evaluate immutable model outputs and save recoverable per-model results."""
from pathlib import Path
import json
import numpy as np
from ..config import canonical_hash
from ..data.preparation import check_receipt,complete,save_array
from ..runtime.artifacts import file_hash,write_json
from ..application.ranking import rank_queries
from ..evaluation.retrieval import evaluate_rankings,macro_scores
from ..evaluation.topic_quality import evaluate_topics


class ModelEvaluation:
    def __init__(self,root,config,references,cohort,logger):
        self.root,self.config,self.references,self.cohort,self.logger=root,config,references,cohort,logger
        self.entries=[]

    def evaluate(self,model,collection,job,retrieval=True):
        root=self.root;directory=root/model['path']
        if not check_receipt(directory):raise ValueError('Cannot evaluate an incomplete model')
        expected=np.load(directory/'source_rows.npy',allow_pickle=False)
        vocabulary=np.load(directory/'vocabulary.npy',allow_pickle=False)
        if not np.array_equal(expected,collection.source_rows) or not np.array_equal(vocabulary,np.load(collection.directory/'vocabulary.npy',allow_pickle=False)):
            raise ValueError('Model evaluation row/vocabulary order changed')
        role='primary' if collection.summary['name'].startswith('primary') else 'external'
        reference=self.references[role]
        application=bool(retrieval and role=='primary')
        if application and not np.array_equal(expected,self.cohort.source_rows):
            raise ValueError('Primary retrieval model is outside the fixed query collection')
        implementation={p:file_hash(root/'src/transport_topics'/p) for p in (
            'experiments/evaluation.py','evaluation/topic_quality.py','evaluation/retrieval.py',
            'application/catalogue.py','application/ranking.py','evaluation/reference.py')}
        from importlib import metadata
        spec={'schema':1,'model':model['identity'],'reference':reference.identity,
              'cohort':self.cohort.identity if application else None,
              'budgets':self.config['retrieval']['budgets'] if application else [],
              'top_words':self.config['topics']['top_words'],
              'cv':self.config['evaluation']['topic']['supplementary_cv'],
              'gensim':metadata.version('gensim'),'implementation':implementation}
        key=canonical_hash(spec)
        target=root/'outputs/evaluation'/model['scope']/model['condition']/model['method']/key[:20]
        target.mkdir(parents=True,exist_ok=True)
        if check_receipt(target):
            metrics=json.loads((target/'metrics.json').read_text(encoding='utf-8'))
        else:
            metrics={'evaluation_identity':key,'model_identity':model['identity'],
                     'condition':model['condition'],'method':model['method'],'topics':model['topics'],
                     'seed':model['seed'],'posts':model['posts'],'vocabulary':model['vocabulary'],
                     'development_only':model['development_only'],'reference_identity':reference.identity,
                     'model_path':model['path'],'topic_metrics':None,'retrieval_metrics':None,
                     'annotations_used_for_fitting_or_ranking':False}
            if model['topic_model']:
                scores=np.load(directory/'topic_scores.npy',allow_pickle=False)
                metrics['topic_metrics']=evaluate_topics(scores,vocabulary,reference,self.config['topics']['top_words'],
                                                   self.config['evaluation']['topic']['supplementary_cv'])
            if application:
                features=np.load(directory/'retrieval_features.npy',allow_pickle=False)
                ranks,similarities=rank_queries(features,self.cohort,max(self.config['retrieval']['budgets']))
                per_query=evaluate_rankings(ranks,self.cohort,self.config['retrieval']['budgets'])
                macro,categories=macro_scores(per_query,self.cohort)
                save_array(target/'query_metrics.npy',per_query)
                save_array(target/'ranking_indices.npy',ranks)
                original_rows=np.full_like(ranks,-1);keep=ranks>=0;original_rows[keep]=collection.source_rows[ranks[keep]]
                save_array(target/'ranking_source_rows.npy',original_rows)
                save_array(target/'ranking_scores.npy',similarities)
                save_array(target/'query_source_rows.npy',[q['source_row'] for q in self.cohort.queries])
                metrics['retrieval_metrics']={'cohort_identity':self.cohort.identity,'queries':len(self.cohort.queries),
                     'categories':len(categories),'budgets':self.config['retrieval']['budgets'],
                     'query_metric_columns':['ndcg','recall'],
                     'macro_ndcg':macro[:,0].tolist(),'macro_recall':macro[:,1].tolist(),
                     'per_category':{c:{'queries':sum(q['category']==c for q in self.cohort.queries),
                                        'ndcg':v[:,0].tolist(),'recall':v[:,1].tolist()} for c,v in categories.items()},
                     'zero_queries':int(sum(np.linalg.norm(features[q['document_index']])==0 for q in self.cohort.queries)),
                     'zero_candidates_remain_eligible':True,'retrieved_original_rows_only':True,
                     'zero_candidate_score':0.0,'zero_query_response':'empty, fixed catalogue and cohort unchanged'}
            write_json(target/'identity.json',spec,overwrite=True)
            write_json(target/'metrics.json',metrics,overwrite=True)
            complete(target)
        entry=dict(metrics,job={k:v for k,v in job.items() if k not in ('source_rows','methods')},
                   evaluation_path=target.relative_to(root).as_posix())
        self.entries.append(entry)
        topic=metrics['topic_metrics'];application_metrics=metrics['retrieval_metrics']
        quality=('NPMI=%.5f diversity=%.4f'%(topic['npmi'],topic['diversity'])) if topic else 'direct retrieval'
        app=(' macro nDCG@%d=%.5f'%(self.config['retrieval']['main_budget'],application_metrics['macro_ndcg'][self.config['retrieval']['budgets'].index(self.config['retrieval']['main_budget'])])) if application_metrics else ''
        self.logger.info('EVALUATED | %s | %s %s K=%s seed=%s | %s%s',job['study'],model['condition'],model['method'],model['topics'],model['seed'],quality,app)
        return entry
