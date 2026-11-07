import unittest
import numpy as np
from transport_topics.data.loaders import Post
from transport_topics.data.preprocessing import deduplicate
from transport_topics.representations.counts import count_representation
from transport_topics.data.families import text_families


class BoundaryTests(unittest.TestCase):
    def test_post_representation_has_no_annotation_field(self):
        self.assertEqual(set(Post.__dataclass_fields__),{'source_row','source_id','original_text'})

    def test_changing_separate_annotations_cannot_change_counts_or_families(self):
        posts=[Post(0,'a','bus late no refund'),Post(1,'b','bus delayed no refund'),Post(2,'c','train late no refund')]
        kept,texts,_=deduplicate(posts)
        before,words,_=count_representation(texts)
        categories={0:['Delay'],1:['Safety']}
        categories[0]=['Pricing']
        after,other_words,_=count_representation(texts)
        np.testing.assert_array_equal(before.toarray(),after.toarray())
        np.testing.assert_array_equal(words,other_words)
        np.testing.assert_array_equal(text_families(texts),text_families(texts))

    def test_sampler_interface_does_not_accept_labels(self):
        import inspect
        from transport_topics.data.splits import nested_family_samples
        self.assertNotIn('labels',inspect.signature(nested_family_samples).parameters)
