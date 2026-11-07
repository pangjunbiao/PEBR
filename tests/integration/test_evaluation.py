from pathlib import Path
import json
import shutil
import tempfile
import unittest
from unittest.mock import Mock,patch
from types import SimpleNamespace
from copy import deepcopy
import numpy as np
from scipy import sparse
from transport_topics.config import load_configuration
from transport_topics.data.prepared import PreparedCollection
from transport_topics.data.preparation import complete,save_array
from transport_topics.application.catalogue import make_cohort
from transport_topics.evaluation.reference import make_reference
from transport_topics.experiments.evaluation import ModelEvaluation
from transport_topics.experiments.studies import prepare_size_collection
from transport_topics.experiments.studies import run as run_studies
from transport_topics.runtime.artifacts import RunStore

ROOT=Path(__file__).resolve().parents[2]


class EvaluationIntegrationTests(unittest.TestCase):
    def test_saved_model_evaluation_reuses_checksums_and_retains_zero_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for folder in ('evaluation','application','experiments'):
                shutil.copytree(ROOT/'src/transport_topics'/folder,root/'src/transport_topics'/folder)
            directory=root/'fixture';directory.mkdir()
            words=np.array(['bus','delay','fare','taxi'])
            save_array(directory/'vocabulary.npy',words)
            rows=np.arange(6)+10
            collection=PreparedCollection({'name':'primary_fixture','identity_sha256':'toy'},directory,
                  sparse.csr_matrix([[1.,1.,0.,0.]]*6),np.array([0,1,0,1,0,1]),np.arange(6),rows)
            labels={int(rows[i]):v for i,v in enumerate([['Delay'],['Delay'],['Safety'],['Safety'],['Delay','Safety'],['Pricing']])}
            cohort=make_cohort(collection,labels)
            features=np.array([[1.,0.],[1.,0.],[0.,0.],[0.,1.],[.5,.5],[0.,1.]])
            model_dir=root/'model';model_dir.mkdir()
            save_array(model_dir/'source_rows.npy',rows);save_array(model_dir/'vocabulary.npy',words)
            save_array(model_dir/'retrieval_features.npy',features)
            save_array(model_dir/'topic_scores.npy',[[2.,1.,0.,0.],[0.,0.,1.,2.]])
            complete(model_dir)
            info={'path':'model','identity':'toy_model','scope':'smoke','condition':'primary_fixture',
                  'method':'proposed','topics':2,'seed':0,'posts':6,'vocabulary':4,'development_only':True,'topic_model':True}
            config=load_configuration(ROOT).values;config['topics']['top_words']=2
            config['evaluation']['topic']['supplementary_cv']=False
            reference=make_reference(root,'toy',['bus delay','bus delay','taxi fare','taxi fare','bus fare','taxi'],words,'toy',Mock())
            evaluator=ModelEvaluation(root,config,{'primary':reference},cohort,Mock())
            a=evaluator.evaluate(info,collection,{'study':'main','id':'fixture'})
            b=evaluator.evaluate(info,collection,{'study':'main','id':'fixture'})
            self.assertEqual(a['evaluation_identity'],b['evaluation_identity'])
            self.assertEqual(a['retrieval_metrics']['zero_queries'],1)
            self.assertEqual(a['retrieval_metrics']['queries'],4)
            target=root/a['evaluation_path']
            actual=np.load(target/'ranking_source_rows.npy')
            self.assertTrue(np.all((actual==-1)|np.isin(actual,rows)))
            save_array(model_dir/'retrieval_features.npy',np.ones_like(features))
            with self.assertRaises(ValueError):evaluator.evaluate(info,collection,{'study':'main','id':'fixture'})

    def test_identical_size_parent_reuses_without_encoder_or_refit(self):
        parent=Mock();parent.source_rows=np.array([1,2,3])
        same,reused=prepare_size_collection(Path('.'),{'source_rows':[1,2,3]},parent,None,{},Mock())
        self.assertIs(same,parent);self.assertTrue(reused)

    def test_sensitivity_aliases_use_main_outputs_after_other_points_are_fitted(self):
        # A growing model list must not confuse a previous lambda/cap condition
        # with the main proposed setting when an exact reuse is requested.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'samples.json').write_text('{}')
            config=deepcopy(load_configuration(ROOT).values)
            config['studies']['corpus_size']['enabled']=False
            preparation={'complete_registered_preparation':True,'sample_manifest':'samples.json',
                         'conditions':[{'name':'primary_full'}]}
            collection=SimpleNamespace(summary={'name':'primary_full'})
            cohort=SimpleNamespace(identity='cohort',queries=[{'category':'Delay'}])
            refs={'primary':SimpleNamespace(identity='ref',documents=10,directory=root/'reference')}
            def model(method,seed,identity):
                return {'identity':identity,'method':method,'condition':'primary_full','topics':12,
                        'seed':seed,'path':'model/'+identity,'posts':10,'vocabulary':20,'development_only':False}
            base=[model(method,seed,method+'_'+str(seed)) for method in ('proposed','no_downside') for seed in range(5)]
            class Evaluation:
                def __init__(self,*args):self.entries=[]
                def evaluate(self,info,collection,job,retrieval=True):
                    self.entries.append({'evaluation_identity':info['identity'],'model_identity':info['identity'],
                         'condition':'primary_full','job':job,'evaluation_path':'evaluation/'+info['identity']})
            def main(root,config,env,logger,store,on_model=None):
                store.json('models_index.json',{'models':base})
                for info in base:on_model(info,collection)
            fitted=[]
            def additional(root,collection,changed,env,logger,methods,k,seeds):
                weight=changed['method']['selection']['downside']['weight']
                cap=changed['method']['transfer']['max_weight']
                fitted.append((float(weight),float(cap) if not isinstance(cap,str) else 1/3))
                return [model('proposed',s,'additional_%s_%s_%s'%(weight,cap,s)) for s in seeds],[]
            store=RunStore(root)
            module='transport_topics.experiments.studies'
            with patch(module+'.preparation_dependency',return_value=(preparation,{'preparation_sha256':'prep'})), \
                 patch(module+'.load_collection',return_value=collection),patch(module+'.load_cohort',return_value=cohort), \
                 patch(module+'.prepare_references',return_value=refs),patch(module+'.ModelEvaluation',Evaluation), \
                 patch(module+'.file_hash',return_value='fixture'),patch(module+'.run_main_models',side_effect=main), \
                 patch(module+'.fit_condition',side_effect=additional),patch(module+'.export_results',return_value={}):
                summary,_=run_studies(root,config,{},Mock(),store)
            self.assertEqual(len(fitted),4)
            self.assertEqual(summary['models_unique'],30)
            self.assertEqual(summary['logical_evaluations'],45)
            index=json.loads((store.directory/'study_evaluations_index.json').read_text())['evaluations']
            for entry in index:
                job=entry['job']
                if job.get('reuse_main_method'):
                    self.assertTrue(entry['model_identity'].startswith(job['reuse_main_method']+'_'))
