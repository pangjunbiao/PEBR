from copy import deepcopy
from pathlib import Path
import unittest
from transport_topics.config import load_configuration
from transport_topics.experiments.planner import sensitivity_jobs,corpus_jobs,CORPUS_METHODS,verify_main_coverage

ROOT=Path(__file__).resolve().parents[2]


class StudyPlanTests(unittest.TestCase):
    def setUp(self):self.config=deepcopy(load_configuration(ROOT).values)

    def test_sensitivity_declares_seven_points_and_three_exact_reuses(self):
        jobs=sensitivity_jobs(self.config)
        self.assertEqual(len(jobs),7)
        self.assertEqual(sum(j['reuse_main_method'] is not None for j in jobs),3)
        self.assertEqual(next(j for j in jobs if j['factor']=='downside_weight' and j['value']==0)['reuse_main_method'],'no_downside')

    def test_disabled_studies_have_no_hidden_jobs(self):
        self.config['studies']['sensitivity']['enabled']=False
        self.config['studies']['corpus_size']['enabled']=False
        self.assertEqual(sensitivity_jobs(self.config),[])
        self.assertEqual(corpus_jobs({}, {},self.config),[])

    def test_size_jobs_use_frozen_membership_for_all_five_ladders(self):
        ladders={str(i):{str(n):list(range(n)) for n in (100,200,400,600)} for i in range(5)}
        jobs=corpus_jobs({}, {'primary_ladders':ladders,'external_samples':ladders},self.config)
        self.assertEqual(len(jobs),40)
        self.assertEqual(len({j['id'] for j in jobs}),40)
        self.assertEqual({j['sample_index'] for j in jobs},set(range(5)))
        for job in jobs:
            self.assertEqual(job['actual_size'],job['target_size'])
            self.assertEqual(job['methods'],list(CORPUS_METHODS))

    def test_smoke_and_full_model_budgets_are_distinct(self):
        smoke=load_configuration(ROOT,'configs/experiments/evaluation_smoke.yaml').values
        self.assertEqual(smoke['integration']['scope'],'smoke')
        self.assertFalse(smoke['studies']['sensitivity']['enabled'])
        self.assertFalse(smoke['studies']['corpus_size']['enabled'])
        self.assertEqual(self.config['integration']['scope'],'full')
        self.assertEqual(self.config['seeds'],[0,1,2,3,4])

    def test_coverage_counts_deterministic_retriever_once(self):
        outputs=[{'condition':'primary','method':'proposed','topics':k,'seed':s} for k in (8,12) for s in (0,1)]
        outputs.append({'condition':'primary','method':'minilm_retrieval','topics':None,'seed':None})
        self.assertEqual(verify_main_coverage(outputs,['primary'],['proposed','minilm_retrieval'],[8,12],[0,1]),5)

    def test_missing_seed_and_duplicated_seed_are_not_complete_coverage(self):
        one={'condition':'primary','method':'proposed','topics':12,'seed':0}
        for values in ([one],[one,one]):
            with self.assertRaises(ValueError):verify_main_coverage(values,['primary'],['proposed'],[12],[0,1])
