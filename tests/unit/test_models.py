from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import numpy as np
from scipy import sparse
from scipy.optimize import nnls

from transport_topics.baselines.base import ModelOutput
from transport_topics.baselines.btm import fit as fit_btm, ordered_documents
from transport_topics.baselines.inputs import global_context, word_costs, transport_distances
from transport_topics.baselines.minilm import prepare as direct_features
from transport_topics.baselines.neural import training_batches, _save_checkpoint
from transport_topics.baselines.registry import enabled_methods
from transport_topics.config import load_configuration
from transport_topics.data.prepared import PreparedCollection
from transport_topics.experiments.sharing_controls import full_pool_marginals, matched_action
from transport_topics.proposed.reference import PredictiveReference
from transport_topics.proposed.scoring import ActionEvaluator
from transport_topics.proposed.transfer import transfer_action
from transport_topics.representations.word_embeddings import load_word_vectors
from transport_topics.topics.memberships import contribution_features, infer_original_memberships

ROOT=Path(__file__).resolve().parents[2]


class ModelTests(unittest.TestCase):
    def test_original_memberships_match_independent_nnls(self):
        rng=np.random.default_rng(0)
        dictionary=rng.random((3,9));original=sparse.csr_matrix(rng.random((7,9)))
        actual,features,_=infer_original_memberships(original,dictionary)
        expected=np.array([nnls(dictionary.T,row)[0] for row in original.toarray()])
        np.testing.assert_allclose(actual,expected,atol=2e-9)
        expected_mass=expected*dictionary.sum(axis=1)
        expected_mass/=expected_mass.sum(axis=1,keepdims=True)
        np.testing.assert_allclose(features,expected_mass,atol=2e-9)

    def test_contribution_features_ignore_reciprocal_topic_scaling(self):
        h=np.array([[1.,2.,0.],[0.,1.,3.]])
        w=np.array([[.2,.8],[1.,0.],[0.,0.]])
        scale=np.array([7.,.3])
        np.testing.assert_allclose(contribution_features(w,h),contribution_features(w/scale,h*scale[:,None]))
        np.testing.assert_array_equal(contribution_features(w,h)[-1],np.zeros(2))

    def test_direct_encoder_is_not_a_topic_model(self):
        result=direct_features([[1.,2.],[0.,0.]])
        result.validate(2,9,3,direct_retriever=True)
        self.assertIsNone(result.topic_scores);self.assertIsNone(result.doc_topics)
        with self.assertRaises(ValueError):result.validate(2,9,3,direct_retriever=False)

    def test_signed_native_decoder_scores_are_valid_but_negative_memberships_are_not(self):
        good=ModelOutput(np.array([[-1.,2.],[3.,-2.]]),np.eye(2),np.eye(2),{})
        good.validate(2,2,2)
        good.doc_topics[0,0]=-1
        with self.assertRaises(ValueError):good.validate(2,2,2)

    def test_btm_preserves_token_order_counts_and_seed_zero(self):
        words=np.array(['bus','delay','fare','taxi'])
        posts=[{'cleaned_text':text} for text in ('bus delay bus','taxi fare taxi','bus delay','taxi fare')]
        docs=ordered_documents(posts,words)
        counts=sparse.csr_matrix(np.array([np.bincount(row,minlength=4) for row in docs],dtype=float))
        settings={'iterations':5,'window':15,'alpha_mass':50.,'beta':.01}
        a=fit_btm(counts,posts,words,2,0,settings)
        b=fit_btm(counts,posts,words,2,0,settings)
        a.validate(4,4,2)
        np.testing.assert_array_equal(a.topic_scores,b.topic_scores)
        np.testing.assert_array_equal(a.doc_topics,b.doc_topics)
        np.testing.assert_allclose(a.topic_scores.sum(axis=1),1)

    def test_global_context_is_original_cluster_sum(self):
        counts=sparse.csr_matrix([[1.,0.],[2.,1.],[0.,3.]])
        contexts,labels=global_context(counts,np.array([[1.,0.],[1.,.1],[0.,1.]]),2,0)
        for i in range(3):
            expected=counts[np.flatnonzero(labels==labels[i])].sum(axis=0)
            np.testing.assert_allclose(contexts[i],np.asarray(expected).ravel())

    def test_btm_native_inference_retains_short_and_empty_rows(self):
        words=np.array(['bus','delay','fare','taxi'])
        posts=[{'cleaned_text':s} for s in ('bus delay','taxi fare','bus','')]
        counts=sparse.csr_matrix([[1.,1.,0.,0.],[0.,0.,1.,1.],[1.,0.,0.,0.],[0.,0.,0.,0.]])
        settings={'iterations':5,'window':15,'alpha_mass':50.,'beta':.01}
        result=fit_btm(counts,posts,words,2,0,settings)
        result.validate(4,4,2)
        np.testing.assert_allclose(result.doc_topics.sum(axis=1),1)
        np.testing.assert_allclose(result.doc_topics[-1],[.5,.5])
        self.assertEqual(result.audit['one_token_documents'],1)
        self.assertEqual(result.audit['empty_documents'],1)

    def test_paper_word_costs_and_exact_single_word_transport(self):
        counts=sparse.csr_matrix([[2.,0.],[0.,3.]])
        cosine,co=word_costs(counts,np.eye(2))
        np.testing.assert_allclose(cosine,[[0.,.5],[.5,0.]])
        np.testing.assert_allclose(co,[[0.,1.],[1.,0.]])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            collection=PreparedCollection({'identity_sha256':'toy','name':'toy'},root,counts,np.arange(2),np.arange(2),np.arange(2))
            directory,_=transport_distances(root,collection,np.eye(2),{'kind':'toy'},Mock())
            with np.load(directory/'M_cos.npz') as data:np.testing.assert_allclose(data['arr_0'],cosine)
            with np.load(directory/'M_coo.npz') as data:np.testing.assert_allclose(data['arr_0'],co)

    def test_pretrained_assets_are_aligned_and_not_random_replacements(self):
        for dim in (200,300):
            vectors,audit=load_word_vectors(ROOT,np.array(['bus','train','delay']),dim)
            self.assertEqual(vectors.shape,(3,dim))
            self.assertEqual(audit['found'],3)
            self.assertFalse(audit['fitted_to_transport_text'])

    def test_singleton_merge_keeps_every_training_document(self):
        import torch
        chunks=training_batches(5,2,torch.arange(5))
        self.assertEqual([len(x) for x in chunks],[2,3])
        self.assertEqual(torch.cat(chunks).tolist(),list(range(5)))

    def test_checkpoint_pointer_commits_after_tensor_file(self):
        import torch,json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=torch.nn.Linear(2,2);optimizer=torch.optim.Adam(model.parameters())
            old=_save_checkpoint(root,model,optimizer,1,[{'epoch':1,'loss':1.}], 'same',torch.device('cpu'))
            # An interrupted next write leaves the previous committed pointer usable.
            torch.save({'incomplete':True},root/'epoch_0002.pt')
            pointer=json.loads((root/'latest.json').read_text())
            self.assertEqual(pointer['filename'],old.name)
            state=torch.load(old,weights_only=True)
            self.assertEqual(state['epoch'],1)
            new=_save_checkpoint(root,model,optimizer,3,[{'epoch':3,'loss':.5}], 'same',torch.device('cpu'))
            self.assertEqual(json.loads((root/'latest.json').read_text())['filename'],new.name)

    def test_full_pool_marginal_does_not_reward_redundancy(self):
        reference=PredictiveReference(np.eye(2),np.ones(2),np.ones(2))
        evaluator=ActionEvaluator(reference,[[.1,.9]],[[1.,0.]],[[0,1]],[[.9,.1],[.9,.1]])
        np.testing.assert_allclose(full_pool_marginals(evaluator,1/3,1.),0,atol=1e-12)

    def test_matched_action_preserves_actual_mass_and_count(self):
        donors=np.array([[.1,.9],[.3,.7],[.8,.2],[.9,.1]])
        action=matched_action((1,2),1/6,[3,1,2],{1,3})
        mixed,alpha,actual=transfer_action(np.array([.2,.8]),action,{1,3},donors)
        self.assertEqual(action.donors,(3,));self.assertEqual(alpha,1/6);self.assertEqual(len(actual),1)

    def test_registered_comparisons_and_control_switches(self):
        config=deepcopy(load_configuration(ROOT).values)
        methods=enabled_methods(config)
        self.assertEqual(len(methods),10)
        config['baselines']['btm']=False
        config['studies']['ablations']['enabled']=False
        self.assertNotIn('btm',enabled_methods(config))
        self.assertNotIn('no_downside',enabled_methods(config))
