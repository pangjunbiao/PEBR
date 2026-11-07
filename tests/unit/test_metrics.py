from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock
import numpy as np
from transport_topics.evaluation.reference import make_reference
from transport_topics.evaluation.topic_quality import npmi_topics,top_words,evaluate_topics
from transport_topics.evaluation.retrieval import query_scores,macro_scores
from transport_topics.evaluation.stability import matched_jaccard,seed_stability
from transport_topics.evaluation.uncertainty import paired_family_bootstrap
from transport_topics.reporting.aggregate import topic_summary,corpus_rows


class MetricTests(unittest.TestCase):
    def test_document_npmi_matches_hand_calculation_and_zero_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref=make_reference(Path(tmp),'toy',['bus delay','bus fare','taxi fare'],['bus','delay','fare','taxi'],'toy',Mock())
            result=npmi_topics([['bus','delay'],['bus','taxi']],ref)
            np.testing.assert_allclose(result,[np.log(1.5)/np.log(3),-1.])

    def test_universal_pair_is_finite_zero_npmi_but_high_cv(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref=make_reference(Path(tmp),'toy',['bus delay']*3,['bus','delay'],'toy',Mock())
            result=evaluate_topics([[2.,1.]],np.array(['bus','delay']),ref,topn=2,with_cv=True)
            self.assertEqual(result['npmi'],0.)
            self.assertGreater(result['cv'],.99)

    def test_top_word_ties_and_diversity_are_common_for_signed_scores(self):
        scores=np.array([[1.,1.,-2.],[-1.,2.,2.]])
        np.testing.assert_array_equal(top_words(scores,np.array(['bus','fare','taxi']),2),[['bus','fare'],['fare','taxi']])

    def test_ndcg_and_recall_use_fixed_relevance_count(self):
        ndcg,recall=query_scores([True,False,True],4,3)
        self.assertAlmostEqual(ndcg,1.5/(1.+1/np.log2(3)+.5))
        self.assertEqual(recall,.5)
        self.assertEqual(query_scores([],4,3),(0.,0.))
        self.assertEqual(query_scores([True,False,True],4,1),(1.,.25))

    def test_macro_average_gives_categories_equal_weight(self):
        cohort=SimpleNamespace(queries=[{'category':'a'}]*9+[{'category':'b'}])
        values=np.array([[[.5,.2]]]*9+[[[1.,.8]]])
        macro,_=macro_scores(values,cohort)
        np.testing.assert_allclose(macro,[[.75,.5]])

    def test_topic_matching_is_invariant_to_permutation(self):
        a=[['bus','delay'],['fare','taxi']]
        self.assertEqual(matched_jaccard(a,list(reversed(a))),1.)
        self.assertEqual(matched_jaccard(a,[['boat','river'],['rail','station']]),0.)
        self.assertIsNone(seed_stability({0:a})['stability'])
        self.assertEqual(seed_stability({i:a for i in range(5)})['seed_pairs'],10)

    def test_cluster_bootstrap_never_splits_query_families(self):
        result=paired_family_bootstrap([1.,-1.,2.,-2.],['a']*4,[0,0,1,1],2000,0)
        self.assertEqual((result['difference'],result['ci_low'],result['ci_high']),(0.,0.,0.))

    def test_paired_bootstrap_uses_category_macro_and_is_reproducible(self):
        a=paired_family_bootstrap([.2,.2,-.1],['a','a','b'],[0,1,2],200,0)
        b=paired_family_bootstrap([.2,.2,-.1],['a','a','b'],[0,1,2],200,0)
        self.assertEqual(a,b)
        self.assertAlmostEqual(a['difference'],.05)
        self.assertAlmostEqual(a['ci_low'],.05)

    def test_external_summary_separates_samples_from_seed_variation(self):
        rows=[{'study':'main','dataset':'external','method':'proposed','K':12,'condition':sample,
               'npmi':value,'diversity':.5,'cv':None} for sample,values in [('one',[0.,0.]),('two',[1.,1.])]
              for value in values]
        result=topic_summary(rows)[0]
        self.assertEqual(result['npmi_mean'],.5)
        self.assertEqual(result['collections'],2)
        self.assertAlmostEqual(result['npmi_sample_sd'],np.sqrt(.5))
        self.assertEqual(result['npmi_mean_within_sample_seed_sd'],0.)

    def test_corpus_differences_pair_the_same_seed_and_ladder(self):
        rows=[]
        for ladder in range(2):
            for seed in range(2):
                for method,extra in [('nmf_original',0.),('proposed',.1)]:
                    rows.append({'study':'corpus_size','dataset':'primary','target_size':100,
                      'actual_size':100,'method':method,'seed':seed,'job_id':str(ladder),
                      'sample_index':ladder,'npmi':float(seed)+extra})
        _,_,result=corpus_rows(rows)
        self.assertAlmostEqual(result[0]['mean_npmi_change'],.1)
        self.assertEqual(result[0]['ladders'],2)
        self.assertEqual(result[0]['seed_fits_per_ladder'],2)
