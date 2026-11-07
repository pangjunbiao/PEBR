import unittest
import numpy as np
from transport_topics.data.families import text_families,cross_source_exclusions
from transport_topics.data.splits import family_folds,nested_family_samples


class FamilyTests(unittest.TestCase):
    def test_identical_texts_join_but_different_texts_do_not(self):
        families=text_families(['bus late again','bus late again','refund never received'])
        self.assertEqual(families[0],families[1]);self.assertNotEqual(families[0],families[2])

    def test_block_size_preserves_exact_family_graph(self):
        texts=['bus late again','bus late again','refund not received','road closed','road closed']
        np.testing.assert_array_equal(text_families(texts,block_size=1),text_families(texts,block_size=4))

    def test_family_members_never_split_across_folds(self):
        families=np.array([0,0,1,2,3,4,5,5])
        folds=family_folds(families)
        self.assertEqual(folds[0],folds[1]);self.assertEqual(folds[-1],folds[-2])
        np.testing.assert_array_equal(folds,family_folds(families))

    def test_sampling_is_nested_and_keeps_whole_families(self):
        families=np.repeat(np.arange(10),2)
        samples=nested_family_samples(families,[3,8,15],0,'test')
        self.assertTrue(set(samples[3])<=set(samples[8])<=set(samples[15]))
        for ids in samples.values():
            for family in np.unique(families[ids]):
                self.assertTrue(set(np.flatnonzero(families==family))<=set(ids))

    def test_insufficient_parent_is_not_silently_resampled(self):
        with self.assertRaises(ValueError):nested_family_samples(np.arange(4),[10],0,'test')

    def test_overlap_removes_entire_external_family(self):
        excluded=cross_source_exclusions(['bus late'],['bus late','something else','refund refused'],np.array([0,0,1]))
        np.testing.assert_array_equal(excluded,[True,True,False])
