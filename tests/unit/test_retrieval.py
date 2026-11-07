from copy import deepcopy
from pathlib import Path
import unittest
import numpy as np
from scipy import sparse
from transport_topics.application.catalogue import make_cohort
from transport_topics.application.ranking import rank_one,rank_queries
from transport_topics.data.prepared import PreparedCollection
from transport_topics.evaluation.retrieval import evaluate_rankings


def fixture():
    collection=PreparedCollection({'identity_sha256':'toy'},Path('.'),sparse.eye(4,format='csr'),
                np.array([0,1,0,1]),np.arange(4),np.array([10,20,30,40]))
    labels={10:['Delay'],20:['Delay','Pricing'],30:['Safety'],40:['Pricing']}
    return collection,make_cohort(collection,labels)


class RetrievalTests(unittest.TestCase):
    def test_cohort_accepts_multilabel_relevant_candidates_but_single_label_queries(self):
        collection,cohort=fixture()
        self.assertEqual(len(cohort.queries),1)
        self.assertEqual(cohort.queries[0]['document_index'],0)
        self.assertEqual(cohort.queries[0]['relevant_candidates'],1)
        np.testing.assert_array_equal(cohort.eligible(0),[1,3])

    def test_zero_query_keeps_fixed_catalogue_and_scores_zero(self):
        _,cohort=fixture();features=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
        before=cohort.eligible(0).copy()
        ranks,_=rank_queries(features,cohort,3)
        np.testing.assert_array_equal(ranks,[[-1,-1,-1]])
        np.testing.assert_array_equal(cohort.eligible(0),before)
        np.testing.assert_array_equal(evaluate_rankings(ranks,cohort,[1,3]),np.zeros((1,2,2)))
        self.assertEqual(len(cohort.queries),1)

    def test_zero_candidate_is_eligible_with_score_zero_even_against_negative_cosine(self):
        features=np.array([[1.,0.],[0.,0.],[-1.,0.],[1.,0.]])
        ranks,scores=rank_one(features,0,[1,2,3],np.array([0,10,20,30]),3)
        np.testing.assert_array_equal(ranks,[3,1,2]);np.testing.assert_allclose(scores,[1,0,-1])

    def test_exact_ties_use_source_order_and_catalogue_is_not_donor_pool(self):
        features=np.ones((4,2));rows=np.array([0,30,20,10])
        ranks,_=rank_one(features,0,[1,2,3],rows,3)
        np.testing.assert_array_equal(ranks,[3,2,1])

    def test_ranking_does_not_read_service_annotations(self):
        _,cohort=fixture();features=np.array([[1.,0.],[1.,0.],[0.,1.],[1.,1.]])
        a,_=rank_queries(features,cohort,2)
        changed=deepcopy(cohort);changed.labels=[() for _ in changed.labels]
        changed.queries[0]['category']='different'
        b,_=rank_queries(features,changed,2)
        np.testing.assert_array_equal(a,b)

    def test_ineligible_or_duplicate_returned_candidates_are_rejected(self):
        _,cohort=fixture()
        for ranks in (np.array([[0,1]]),np.array([[1,1]]),np.array([[2,1]])):
            with self.assertRaises(ValueError):evaluate_rankings(ranks,cohort,[2])
