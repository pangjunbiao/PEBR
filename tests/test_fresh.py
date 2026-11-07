from pathlib import Path
import copy,json,logging,sys,tempfile,unittest
from unittest.mock import patch
import numpy as np
from scipy import sparse
from pebr_revision_step3.io import write_json,save_receipt
from pebr_fresh_validation.protocol import build_protocol as revised_protocol, ANCHOR
from pebr_revision_step2.protocol import build_protocol as original_protocol
from pebr_fresh_validation.aggregate import summarize
from pebr_fresh_validation.banks import masked_evidence,validate_default_reconstruction,make_encoder
from transport_topics.proposed.reference import PredictiveReference
from transport_topics.data.prepared import PreparedCollection,SelectionBank
from transport_topics.proposed.transfer import normalized_counts

def build_protocol():return revised_protocol(original_protocol())

class MaskAndSelectionChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.p=build_protocol();rng=np.random.default_rng(22)
        self.counts=sparse.csr_matrix(rng.integers(1,4,(7,12)).astype(float));self.words=np.asarray(['terma'+chr(97+i) for i in range(12)])
        self.posts=[{'cleaned_text':' '.join(str(w) for j,w in enumerate(self.words) for _ in range(int(self.counts[i,j])))} for i in range(7)]
        d=self.root/'data';d.mkdir();self.directory=d
        write_json(d/'identity.json',{'views':{'count':20,'min_word_types':4,'hide_fraction':.5}})
        np.save(d/'vocabulary.npy',self.words,allow_pickle=False)
        emb=rng.normal(size=(7,8));emb/=np.linalg.norm(emb,axis=1,keepdims=True);np.save(d/'embeddings.npy',emb,allow_pickle=False)
        self.c=PreparedCollection({'name':'primary_full','identity_sha256':'fixture'},d,self.counts,np.arange(7)%5,np.arange(7),np.arange(7))
        self.logger=logging.getLogger('test_repeats');self.logger.addHandler(logging.NullHandler())

    def test_mask_repeats_are_deterministic_and_different(self):
        a=masked_evidence('transport_grievance',self.c,self.posts,self.p,3711)
        b=masked_evidence('transport_grievance',self.c,self.posts,self.p,3711)
        c=masked_evidence('transport_grievance',self.c,self.posts,self.p,4711)
        self.assertEqual((a['visible']-b['visible']).nnz,0)
        self.assertNotEqual((a['targets']-c['targets']).nnz,0)
        self.assertEqual(a['visible'].multiply(a['targets']).nnz,0)

    def test_hidden_english_types_are_removed_from_queries(self):
        a=masked_evidence('transport_grievance',self.c,self.posts,self.p,3711)
        for q,pos in zip(a['queries'],a['positions']):
            hidden=set(self.words[a['targets'][pos].indices])
            self.assertFalse(set(q.split())&hidden)

    def test_default_mask_reconstruction_is_checked(self):
        view=masked_evidence('transport_grievance',self.c,self.posts,self.p,2711)
        bank=SelectionBank('topic_learning',self.directory,view['base'],view['visible'],view['targets'],view['eligible'],None,None)
        self.assertTrue(validate_default_reconstruction('transport_grievance',self.c,bank,self.posts,self.p)['passed'])
        bank.targets=bank.targets.copy();bank.targets.data[0]+=1
        with self.assertRaises(ValueError):validate_default_reconstruction('transport_grievance',self.c,bank,self.posts,self.p)

    def test_chinese_hidden_segments_are_removed(self):
        words=np.asarray(['公交','晚点','司机','收费','乘客','投诉','车站','等待','服务','问题','今天','线路'])
        np.save(self.directory/'vocabulary.npy',words,allow_pickle=False)
        records=[{'segments':words.tolist(),'tokens':words.tolist(),'text':''.join(words)} for _ in range(7)]
        view=masked_evidence('chinese_pooled',self.c,records,self.p,3711)
        for q,pos in zip(view['queries'],view['positions']):
            self.assertFalse(any(w in q for w in words[view['targets'][pos].indices]))

    def test_chinese_default_query_identity_must_match(self):
        from pebr_revision_step3.io import canonical_hash
        records=[{'segments':str(r['cleaned_text']).split(),'text':r['cleaned_text'],'tokens':str(r['cleaned_text']).split()} for r in self.posts]
        view=masked_evidence('chinese_pooled',self.c,records,self.p,2711)
        bank=SelectionBank('topic_learning',self.directory,view['base'],view['visible'],view['targets'],view['eligible'],None,None)
        encoder={'fixture':True};write_json(self.directory/'encoder_identity.json',{'encoder':encoder})
        identity=canonical_hash({'encoder':encoder,'ordered_texts':list(view['queries'])})
        self.assertTrue(validate_default_reconstruction('chinese_pooled',self.c,bank,records,self.p,identity)['passed'])
        with self.assertRaises(ValueError):validate_default_reconstruction('chinese_pooled',self.c,bank,records,self.p,'incorrect')

    def test_encoder_asset_root_and_cache_root_are_distinct(self):
        expected={'provider':'CPUExecutionProvider','max_sequence_length':256,'batch_size':32}
        write_json(self.directory/'identity.json',{'encoder':expected})
        class Fake:
            def __init__(obj,root,config,environment,logger,run_directory):
                obj.identity=copy.deepcopy(expected);obj.root=root
                self.assertEqual(root,self.root/'frozen');self.assertEqual(config['preparation']['embedding_batch_size'],32)
        with patch('transport_topics.representations.embeddings.FrozenEncoder',Fake):
            encoder=make_encoder('transport_grievance',self.root/'frozen',self.root/'new',self.c,self.p,self.logger)
        self.assertEqual(encoder.root,self.root/'new')


    def test_checkpoint_reuse_and_matched_transfer(self):
        from pebr_fresh_validation.selection import select_bank
        rng=np.random.default_rng(4);v=masked_evidence('transport_grievance',self.c,self.posts,self.p,3711)
        candidates=np.full((7,20,20),-1,dtype=int);full=np.full((7,20),-1,dtype=int)
        for i in range(7):
            ids=np.flatnonzero(self.c.folds!=self.c.folds[i]);candidates[i,:,:len(ids)]=ids;full[i,:len(ids)]=ids
        bankdir=self.directory/'bank';bankdir.mkdir()
        np.save(bankdir/'full_receiver_embeddings.npy',np.load(self.directory/'embeddings.npy'),allow_pickle=False)
        bank=SelectionBank('topic_learning',bankdir,v['base'],v['visible'],v['targets'],v['eligible'],candidates,full)
        refs=[PredictiveReference(rng.uniform(.1,1,(3,12)),np.ones(12),np.ones(12)) for _ in range(5)]
        directory=self.root/'selection/corpus/mask3711/K3'
        paths=select_bank(self.c,bank,refs,self.p,3711,directory,self.logger)
        a=read_json_local(paths[ANCHOR]/'actions.json');b=read_json_local(paths['matched_preserve_fixed']/'actions.json')
        for x,y in zip(a,b):
            self.assertAlmostEqual(x['actual_beta'],y['actual_beta'])
            self.assertEqual(len(x['available_donors']),len(y['available_donors']))
        with patch('pebr_fresh_validation.selection.choose',side_effect=AssertionError('must reuse')):
            self.assertEqual(paths,select_bank(self.c,bank,refs,self.p,3711,directory,self.logger))
        # A modified completed artifact must fail rather than silently be used.
        p=paths[ANCHOR]/'summary.json';p.write_text('{}')
        with self.assertRaises(ValueError):select_bank(self.c,bank,refs,self.p,3711,directory,self.logger)

class AggregationChecks(unittest.TestCase):
    def rows(self):
        out=[]
        for policy in ('nmf_original',ANCHOR,'preserve_fixed_prefix'):
            masks=['not_applicable'] if policy=='nmf_original' else [3711,4711,5711,6711]
            for i,mask in enumerate(masks):
                for seed in range(5):
                    value=.1 if policy=='nmf_original' else .2+i*.01+(0 if policy==ANCHOR else -.02)
                    out.append({'corpus':'transport_grievance','K':12,'policy':policy,'mask_base':mask,
                        'seed':seed,'npmi':value,'cv':value,'diversity':.7,'top_words':[['a','b']], 'path':'fixture'})
        return out

    def test_fresh_mean_and_sd_are_separate_from_fit_sd(self):
        r=summarize(self.rows(),build_protocol())
        p=next(x for x in r['fresh_bank_summaries'] if x['policy']==ANCHOR)
        self.assertAlmostEqual(p['npmi_mean'],.215)
        self.assertAlmostEqual(p['npmi_sd_bank_means'],np.std([.2,.21,.22,.23],ddof=1))
        self.assertEqual(p['npmi_mean_within_bank_fit_sd'],0)
        self.assertEqual(p['unique_final_fit_records'],20)
        self.assertFalse(p['development_bank_included'])

    def test_baseline_fit_count_not_inflated(self):
        r=summarize(self.rows(),build_protocol())
        n=next(x for x in r['fresh_bank_summaries'] if x['policy']=='nmf_original')
        self.assertEqual(n['unique_final_fit_records'],5)
        self.assertEqual(n['selection_banks'],0)
        self.assertIsNone(n['npmi_sd_bank_means'])

    def test_paired_fresh_contrasts(self):
        r=summarize(self.rows(),build_protocol())
        p=next(x for x in r['paired_fresh_bank_contrasts'] if x['comparator']=='preserve_fixed_prefix' and x['metric']=='npmi')
        self.assertAlmostEqual(p['mean_revised_minus_control'],.02)
        self.assertEqual(p['positive_banks'],4)
        self.assertFalse(p['confidence_interval'])
        self.assertFalse(p['independent_corpus_replicates'])

    def test_incomplete_banks_not_summarized_as_complete(self):
        rows=[r for r in self.rows() if r['mask_base']!=6711]
        r=summarize(rows,build_protocol())
        self.assertFalse(any(x['policy']==ANCHOR for x in r['fresh_bank_summaries']))
        self.assertFalse(r['coverage']['complete'])
        with self.assertRaises(ValueError):summarize(rows,build_protocol(),True)

    def test_duplicates_and_development_bank_rejected(self):
        rows=self.rows();rows.append(rows[0])
        with self.assertRaises(ValueError):summarize(rows,build_protocol())
        rows=self.rows();rows[-1]=dict(rows[-1],mask_base=2711)
        with self.assertRaises(ValueError):summarize(rows,build_protocol())

    def test_negative_outcomes_retained(self):
        rows=self.rows()
        for r in rows:
            if r['policy']=='preserve_fixed_prefix':r['npmi']+=.04
        result=summarize(rows,build_protocol())
        p=next(x for x in result['paired_fresh_bank_contrasts'] if x['comparator']=='preserve_fixed_prefix' and x['metric']=='npmi')
        self.assertEqual(p['negative_banks'],4)
        self.assertLess(p['mean_revised_minus_control'],0)

    def test_frozen_rule_protocol_and_controls(self):
        p=build_protocol();self.assertEqual(p['new_final_fits'],1200)
        self.assertEqual(set(p['policies']),set(p['control_definitions']))
        self.assertEqual(p['positive_offered_weights'],[1/3])
        self.assertEqual(p['development_mask_base'],2711)
        self.assertEqual(len(p['selection_mask_bases']),4)
        self.assertNotIn(2711,p['selection_mask_bases'])
        self.assertNotIn('protected_word_audit',p)


def read_json_local(path):return json.loads(Path(path).read_text())
if __name__=='__main__':unittest.main()
