from pathlib import Path
import unittest
import numpy as np
from scipy import sparse
from transport_topics.config import load_configuration
from transport_topics.proposed.masks import build_views,mask_seed
from transport_topics.proposed.candidates import retrieve

ROOT=Path(__file__).resolve().parents[2]


class MaskTests(unittest.TestCase):
    def setUp(self):
        self.config=load_configuration(ROOT).values
        self.words=np.array(['bus','late','no','fare','refund','train'])
        self.X=sparse.csr_matrix([[2,3,1,1,0,0],[1,0,0,2,3,1],[1,2,1,1,2,1]],dtype=float)
        self.texts=['bus late no bus late fare late','bus fare refund refund refund train fare','bus late late no fare refund refund train']
        self.rows=np.array([40,90,125])

    def test_view_reconstructs_exact_original_counts(self):
        bank=build_views(self.X,self.words,self.texts,self.rows,self.config)
        M=self.config['method']['views']['count']
        for i in range(3):
            for m in range(M):
                hidden=bank['targets'][i*M+m]
                visible=bank['visible'][i*M+m]
                self.assertEqual(hidden.multiply(visible).nnz,0)
                np.testing.assert_array_equal((hidden>0).toarray()+(visible>0).toarray(),(self.X[i]>0).toarray())
                self.assertAlmostEqual(visible.sum(),1)

    def test_outer_targets_do_not_enter_inner_views(self):
        bank=build_views(self.X,self.words,self.texts,self.rows,self.config,True)
        self.assertEqual(bank['base'].multiply(bank['outer']).nnz,0)
        M=self.config['method']['views']['count']
        self.assertTrue(bank['eligible'][2])
        self.assertGreater(bank['targets'].nnz,0)
        outer=bank['outer'][np.repeat(np.arange(3),M)]
        self.assertEqual(outer.multiply(bank['visible']+bank['targets']).nnz,0)

    def test_too_short_receiver_has_no_views(self):
        short=sparse.csr_matrix([[1,1,1,0,0,0]],dtype=float)
        bank=build_views(short,self.words,['bus late no'],[3],self.config)
        self.assertFalse(bank['eligible'][0]);self.assertEqual(bank['queries'],[])

    def test_mask_seed_matches_original_pilot_formula(self):
        import hashlib
        expected=int.from_bytes(hashlib.sha256(b'2711:40').digest()[:8],'little')
        self.assertEqual(mask_seed(40,2711),expected)

    def test_retrieval_respects_fold_family_and_positive_counts(self):
        E=np.array([[1.,0],[1.,0],[1.,0],[1.,0],[1.,0]])
        indices=retrieve(E[[0]],[0],E,np.arange(5),np.array([0,0,1,2,3]),np.array([1,1,0,1,1],bool),4)
        np.testing.assert_array_equal(indices,[[3,4,-1,-1]])

    def test_similarity_ties_use_retained_source_order(self):
        E=np.ones((4,2))/np.sqrt(2)
        ids=retrieve(E[[0]],[0],E,np.arange(4),np.arange(4),np.ones(4,bool),3)
        np.testing.assert_array_equal(ids,[[1,2,3]])
