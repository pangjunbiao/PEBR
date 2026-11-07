from pathlib import Path
import unittest,tempfile,logging,copy,json
from unittest.mock import patch
import numpy as np
from scipy import sparse
from pebr_revision_step3.io import write_json,read_json,save_receipt,complete
from pebr_fresh_validation.protocol import build_protocol,ANCHOR
from pebr_revision_step2.protocol import build_protocol as original_protocol
from transport_topics.data.prepared import PreparedCollection,SelectionBank
from transport_topics.proposed.reference import PredictiveReference
from transport_topics.proposed.candidates import retrieve
from pebr_protected_words.gate import audit_protocol
from pebr_protected_words.banks import masked_inputs,prepare_bank
from pebr_protected_words.selection import select_actions
from pebr_protected_words.outcomes import gain_from_probabilities,summarize_posts,audit_configuration
from pebr_protected_words.aggregate import summarize

def protocol():return audit_protocol(build_protocol(original_protocol()))

class ProtectedInputChecks(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  self.p=protocol();rng=np.random.default_rng(9)
  self.counts=sparse.csr_matrix(rng.integers(1,4,(7,12)).astype(float));self.words=np.asarray(['term'+chr(97+i) for i in range(12)])
  self.posts=[{'cleaned_text':' '.join(str(w) for j,w in enumerate(self.words) for _ in range(int(self.counts[i,j])))} for i in range(7)]
  self.data=self.root/'data';self.data.mkdir();np.save(self.data/'vocabulary.npy',self.words)
  write_json(self.data/'identity.json',{'views':{'count':20,'min_word_types':4,'hide_fraction':.5}})
  emb=rng.normal(size=(7,8));emb/=np.linalg.norm(emb,axis=1,keepdims=True);np.save(self.data/'embeddings.npy',emb)
  self.c=PreparedCollection({'name':'fixture','identity_sha256':'fixed-fixture'},self.data,self.counts,np.arange(7)%5,np.arange(7),np.arange(7))
  self.logger=logging.getLogger('protected-fixture');self.logger.addHandler(logging.NullHandler())
  self.refs=[PredictiveReference(rng.uniform(.1,1,(3,12)),np.ones(12),np.ones(12)) for _ in range(5)]

 def bank(self,view):
  directory=self.root/'bank';directory.mkdir(exist_ok=True)
  sparse.save_npz(directory/'outer.npz',view['outer'])
  np.save(directory/'full_receiver_embeddings.npy',np.load(self.data/'embeddings.npy'))
  candidates=np.full((7,20,20),-1,dtype=int);full=np.full((7,20),-1,dtype=int)
  for i in range(7):
   ids=np.flatnonzero(self.c.folds!=self.c.folds[i]);candidates[i,:,:len(ids)]=ids;full[i,:len(ids)]=ids
  return SelectionBank('outer_audit',directory,view['base'],view['visible'],view['targets'],view['eligible'],candidates,full)

 def test_outer_and_inner_counts_are_disjoint_and_reconstruct(self):
  v=masked_inputs('transport_grievance',self.c,self.posts,self.p,2712)
  self.assertEqual((v['base']+v['outer']-self.counts).nnz,0)
  self.assertEqual(v['base'].multiply(v['outer']).nnz,0)
  self.assertEqual(v['visible'].multiply(v['targets']).nnz,0)
  for i in range(7):
   ids=set(v['outer'][i].indices);self.assertEqual(len(ids),3)
   for row in range(i*20,(i+1)*20):
    self.assertFalse(ids&set(v['visible'][row].indices));self.assertFalse(ids&set(v['targets'][row].indices))

 def test_english_queries_remove_every_protected_occurrence(self):
  v=masked_inputs('transport_grievance',self.c,self.posts,self.p,2712)
  for i,q in enumerate(v['full_queries']):self.assertFalse(set(q.split())&set(self.words[v['outer'][i].indices]))
  for q,pos in zip(v['queries'],v['positions']):
   i=int(pos)//20;blocked=set(self.words[v['outer'][i].indices])|set(self.words[v['targets'][int(pos)].indices])
   self.assertFalse(set(q.split())&blocked)

 def test_chinese_queries_remove_every_protected_segment(self):
  words=np.asarray(['公交','晚点','司机','收费','乘客','投诉','车站','等待','服务','问题','今天','线路'])
  np.save(self.data/'vocabulary.npy',words)
  records=[]
  for i in range(7):
   terms=[str(w) for j,w in enumerate(words) for _ in range(int(self.counts[i,j]))]
   records.append({'segments':terms,'tokens':terms,'text':''.join(terms)})
  v=masked_inputs('chinese_pooled',self.c,records,self.p,2712)
  for i,q in enumerate(v['full_queries']):self.assertFalse(any(w in q for w in words[v['outer'][i].indices]))
  for q,pos in zip(v['queries'],v['positions']):
   i=int(pos)//20;blocked=set(words[v['outer'][i].indices])|set(words[v['targets'][int(pos)].indices])
   self.assertFalse(any(w in q for w in blocked))

 def test_new_outer_seeds_change_targets_deterministically(self):
  a=masked_inputs('transport_grievance',self.c,self.posts,self.p,2712)
  b=masked_inputs('transport_grievance',self.c,self.posts,self.p,2712)
  d=masked_inputs('transport_grievance',self.c,self.posts,self.p,3712)
  self.assertEqual((a['outer']-b['outer']).nnz,0);self.assertNotEqual((a['outer']-d['outer']).nnz,0)
  np.testing.assert_array_equal(a['eligible'],d['eligible'])

 def test_outer_hidden_counts_cannot_change_selection(self):
  v=masked_inputs('transport_grievance',self.c,self.posts,self.p,2712);b=self.bank(v)
  a=select_actions(self.c,b,self.refs,self.p,2712,self.root/'selection1',self.logger)
  changed=v['outer'].copy();changed.data*=37;sparse.save_npz(b.directory/'outer.npz',changed)
  d=select_actions(self.c,b,self.refs,self.p,2712,self.root/'selection2',self.logger)
  self.assertEqual(a,d)

 def test_selection_receives_only_outer_visible_receiver(self):
  import pebr_protected_words.selection as module
  real=module.choose;seen=[]
  v=masked_inputs('transport_grievance',self.c,self.posts,self.p,2712);b=self.bank(v)
  def inspect(*args):
   shown=np.asarray(args[5]);seen.append(shown.copy())
   self.assertTrue(any(np.array_equal(shown,b.base[i].toarray()) for i in range(7)))
   return real(*args)
  with patch.object(module,'choose',inspect):select_actions(self.c,b,self.refs,self.p,2712,self.root/'selection',self.logger)
  self.assertEqual(len(seen),7)

 def test_protected_outcomes_start_only_after_complete_actions(self):
  v=masked_inputs('transport_grievance',self.c,self.posts,self.p,2712);b=self.bank(v)
  with self.assertRaises(ValueError):audit_configuration('transport_grievance',3,2712,self.c,b,self.refs,{},self.p,self.root/'missing',self.root/'outcomes',self.logger)

 def test_common_cohort_zero_reference_and_checkpoint_resume(self):
  v=masked_inputs('transport_grievance',self.c,self.posts,self.p,2712);b=self.bank(v);selection=self.root/'selection'
  a=select_actions(self.c,b,self.refs,self.p,2712,selection,self.logger)
  summaries=audit_configuration('transport_grievance',3,2712,self.c,b,self.refs,a,self.p,selection,self.root/'outcomes',self.logger)
  self.assertEqual(len(summaries),11);self.assertEqual({r['eligible_posts'] for r in summaries},{7})
  self.assertEqual(len({r['protected_target_tokens'] for r in summaries}),1)
  base=next(r for r in summaries if r['policy']=='nmf_original');self.assertEqual(base['mean_post_gain'],0);self.assertEqual(base['post_loss_fraction'],0)
  with patch('pebr_protected_words.selection.choose',side_effect=AssertionError('must reuse')):
   self.assertEqual(a,select_actions(self.c,b,self.refs,self.p,2712,selection,self.logger))
  self.assertEqual(summaries,audit_configuration('transport_grievance',3,2712,self.c,b,self.refs,a,self.p,selection,self.root/'outcomes',self.logger))

 def test_protected_full_query_embeddings_drive_availability(self):
  calls=[];rng=np.random.default_rng(11)
  class Encoder:
   identity={'fixture':True}
   def encode(obj,texts,label):
    calls.append(list(texts));x=rng.normal(size=(len(texts),8));return x/np.linalg.norm(x,axis=1,keepdims=True)
  b=prepare_bank('transport_grievance',self.c,self.posts,self.p,2712,Encoder(),self.root/'prepared',self.logger)
  self.assertEqual(len(calls),2)
  q=np.load(b.directory/'full_receiver_embeddings.npy');emb=np.load(self.data/'embeddings.npy')
  expected=retrieve(q,np.arange(7),emb,self.c.families,self.c.folds,np.ones(7,dtype=bool),20)
  np.testing.assert_array_equal(expected,b.full_candidates)
  v=masked_inputs('transport_grievance',self.c,self.posts,self.p,2712)
  self.assertEqual(calls[0],v['full_queries']);self.assertEqual(calls[1],v['queries'])

class OutcomeChecks(unittest.TestCase):
 def test_log_gain_uses_token_weights(self):
  p=np.asarray([[.25,.75],[.5,.5]]);scores,gains=gain_from_probabilities(p,[2,1])
  self.assertEqual(gains[0],0);self.assertAlmostEqual(gains[1],(2*np.log(2)+np.log(2/3))/3)

 def test_negative_part_denominator_is_all_audited_posts(self):
  rows=[{'gain_nats_per_token':g,'target_tokens':1,'offered_beta':0,'actual_beta':0,'selection_utility':0} for g in [0,.2,-.4,0]]
  s=summarize_posts(rows,1e-10);self.assertAlmostEqual(s['post_loss_fraction'],.25);self.assertAlmostEqual(s['mean_negative_part_all_posts'],.1)

 def test_undefined_correlations_not_invented(self):
  rows=[{'gain_nats_per_token':0,'target_tokens':1,'offered_beta':0,'actual_beta':0,'selection_utility':0} for _ in range(5)]
  s=summarize_posts(rows,1e-10);self.assertIsNone(s['inner_outer_spearman_all_utility_posts'])

 def test_actual_negative_protected_gain_is_valid(self):
  p=np.asarray([[.8,.2],[.2,.8]]);_,g=gain_from_probabilities(p,[3,1]);self.assertLess(g[1],0)

 def test_no_final_model_seeds_as_audit_replicates(self):
  p=protocol();self.assertEqual(p['new_final_fits'],0);self.assertNotIn('final_model_seeds',p)
  self.assertEqual(p['inner_mask_base'],2711);self.assertEqual(p['outer_target_bases'],[2712,3712,4712,5712,6712])

class AggregationChecks(unittest.TestCase):
 def rows(self):
  result=[]
  for p in ['nmf_original',ANCHOR,'nmf_similarity']:
   for i,o in enumerate(protocol()['outer_target_bases']):
    g=0 if p=='nmf_original' else .1+i*.01+(0 if p==ANCHOR else .02)
    result.append({'corpus':'transport_grievance','K':12,'outer_seed':o,'policy':p,'eligible_posts':7,'protected_target_tokens':15,
      'mean_post_gain':g,'median_post_gain':g,'token_weighted_gain':g,'post_loss_fraction':.2,'mean_negative_part_all_posts':.05,'mean_realized_mass_all_posts':.2})
  return result

 def test_hierarchical_aggregation_and_adverse_results(self):
  s=summarize(self.rows(),protocol());r=next(r for r in s['five_bank_summaries'] if r['policy']==ANCHOR)
  self.assertAlmostEqual(r['mean_post_gain_mean'],.12)
  self.assertAlmostEqual(r['mean_post_gain_sd_banks'],np.std([.1,.11,.12,.13,.14],ddof=1))
  c=next(r for r in s['paired_protected_contrasts'] if r['comparator']=='nmf_similarity' and r['metric']=='mean_post_gain')
  self.assertAlmostEqual(c['mean_revised_minus_control'],-.02);self.assertEqual(c['negative_banks'],5)
  self.assertFalse(c['confidence_interval'])

 def test_known_bank_is_excluded_from_primary_fresh_summary(self):
  rows=self.rows()
  for r in rows:
   if r['outer_seed']==2712:r['mean_post_gain']=1000000
  s=summarize(rows,protocol())
  r=next(r for r in s['fresh_protected_bank_summaries'] if r['policy']==ANCHOR)
  self.assertAlmostEqual(r['mean_post_gain_mean'],.125)
  self.assertEqual(r['protected_banks'],4)

 def test_incomplete_duplicate_and_cohort_mismatch_are_rejected(self):
  rows=self.rows();self.assertFalse(summarize(rows,protocol())['coverage']['complete'])
  with self.assertRaises(ValueError):summarize(rows,protocol(),True)
  with self.assertRaises(ValueError):summarize(rows+[rows[0]],protocol())
  rows[0]['eligible_posts']=8
  with self.assertRaises(ArithmeticError):summarize(rows,protocol())

if __name__=='__main__':unittest.main()
