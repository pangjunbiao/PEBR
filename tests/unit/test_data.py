import csv
import io
from pathlib import Path
import tempfile
import unittest
import numpy as np
from transport_topics.data.loaders import Post, load_primary, load_external, SUBSTANTIVE
from transport_topics.data.preprocessing import clean_text, deduplicate, query_without
from transport_topics.data.sources import obtain
from transport_topics.representations.counts import count_representation
from transport_topics.representations.tfidf import idf_weights, transform
from transport_topics.runtime.logging import RunLogger


class DataTests(unittest.TestCase):
    def test_normalization_preserves_negation(self):
        text=clean_text('No &amp; NOT late @user httpurl https://example.org a12 never')
        self.assertEqual(text,'no & not late never')

    def test_dedup_keeps_earliest_original_source(self):
        posts=[Post(3,'a','Bus delayed'),Post(7,'b','BUS delayed'),Post(9,'c','No refund')]
        kept,texts,mapping=deduplicate(posts)
        self.assertEqual([p.source_row for p in kept],[3,9])
        self.assertEqual(mapping,{7:3})
        self.assertEqual(kept[0].original_text,'Bus delayed')

    def test_external_headerless_first_tweet_is_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'external.csv'
            p.write_text('2,s100,First transport tweet\n0,s101,Other text\n1,s102,Crash reported\n',encoding='utf-8')
            posts,audit=load_external(p)
            self.assertEqual([p.source_row for p in posts],[0,2])
            self.assertEqual(posts[0].source_id,'dabiri_traffic:100')
            self.assertEqual(audit['raw_rows'],3)
            self.assertFalse(hasattr(posts[0],'label'))

    def test_external_bad_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'external.csv';p.write_text('class,id,text\n',encoding='utf-8')
            with self.assertRaises(ValueError):load_external(p)

    def test_primary_annotations_are_separate_from_posts(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'primary.csv'
            fields=['text',*SUBSTANTIVE,'Miscellaneous','Not_applicable']
            with p.open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
                row={name:'0' for name in fields};row.update(text='bus late',Delay='1');w.writerow(row)
            posts,ann=load_primary(p)
            self.assertEqual(ann,{0:['Delay']})
            self.assertEqual(set(posts[0].__dict__),{'source_row','source_id','original_text'})

    def test_counts_preserve_no_not_and_document_frequency(self):
        counts,words,_=count_representation(['no bus not late never','no bus not safe never','singleton'],2)
        self.assertTrue({'no','not','never','bus'}<=set(words))
        self.assertNotIn('singleton',words)
        self.assertTrue(counts.has_sorted_indices)

    def test_tfidf_idf_and_zero_row(self):
        from scipy import sparse
        counts=sparse.csr_matrix([[1,1],[0,1],[0,0]],dtype=float)
        weights=idf_weights(counts)
        self.assertAlmostEqual(weights[0],1+np.log(4/2))
        matrix=transform(counts,weights).toarray()
        self.assertAlmostEqual(np.linalg.norm(matrix[0]),1)
        self.assertEqual(matrix[2].sum(),0)

    def test_all_hidden_occurrences_removed_in_order(self):
        self.assertEqual(query_without('late bus late no fare late',{'late'}),'bus no fare')

    def test_existing_bad_source_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'file.csv';p.write_bytes(b'preserve')
            log=RunLogger(root/'run_log.txt','test',io.StringIO())
            try:
                with self.assertRaises(ValueError):obtain(root,('file.csv','https://example.org','0'*64),log)
                self.assertEqual(p.read_bytes(),b'preserve')
            finally:log.close()
