"""Path-limit regression and actual frozen encoder cache-I/O tests (inference stubbed)."""
from pathlib import Path, PureWindowsPath
import logging
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from pebr_protected_words.storage import workspace_path, verify_cache_write

USER_PROJECT = PureWindowsPath(r'C:\Users\Sabir\PycharmProjects\proposed')
OLD_NAME = '20261101T095053677062Z_d25536e042_b3f523_8f044d9c431e_7c370336b1ed_6d5bfcf0df61'

class WindowsPathChecks(unittest.TestCase):
    def windows_work(self, short=True):
        name = workspace_path(Path('.'), 'a'*64).name if short else OLD_NAME
        return USER_PROJECT/'outputs/revision_rule_lab/protected_words'/name

    def test_reported_error_path_is_263_characters(self):
        path = self.windows_work(False)/'cache/embeddings'/('vectors_'+'a'*64+'.partial.npy')
        self.assertEqual(len(str(path)), 263)

    def test_short_workspace_reduces_cache_path_to_199(self):
        root = self.windows_work()
        path = root/'cache/embeddings'/('vectors_'+'a'*64+'.partial.npy')
        self.assertEqual(len(str(path)), 199)
        self.assertLess(len(str(path)), 260)
        self.assertLess(len(str(root/'cache/embeddings/chinese'/('a'*24)/'vectors.partial.npy')), 260)

    def test_cache_probe_writes_reads_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as d:
            result = verify_cache_write(Path(d)/'s6_aaaaaaaaaaaa')
            self.assertTrue(result['passed'])
            self.assertTrue(result['probe_removed'])
            self.assertFalse(result['model_or_encoder_inference_performed'])
            self.assertFalse(list(Path(d).rglob('*.partial.npy')))

    def test_workspace_requires_a_complete_valid_identity(self):
        for value in ('short', 'g'*64, 'A'*64):
            with self.assertRaises(ValueError):
                workspace_path(Path('.'), value)

    def actual_cache(self, chinese=False, short=True):
        from transport_topics.representations.embeddings import FrozenEncoder
        from transport_topics.chinese.embeddings import FrozenChineseEncoder
        cls = FrozenChineseEncoder if chinese else FrozenEncoder
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/('new' if short else 'old')
            encoder = cls.__new__(cls)
            encoder.root = root
            encoder.identity = {'scope': 'cache-I/O regression test; model inference is stubbed'}
            encoder.logger = logging.getLogger('cache-test')
            encoder.session = None
            encoder.batch_size = 2
            encoder.cfg = {'runtime': {'embedding_batch_size': 2}}
            calls = []
            def inference(texts, session):
                calls.append(list(texts))
                out = np.zeros((len(texts), 384), dtype=np.float32)
                out[:, 0] = 1.
                return out
            encoder._encode = inference
            encoder._batch = inference
            real_memmap = np.lib.format.open_memmap
            projected_paths = []
            def legacy_windows_memmap(filename, *args, **kwargs):
                relative = Path(filename).relative_to(root)
                projected = self.windows_work(short).joinpath(*relative.parts)
                projected_paths.append(str(projected))
                if len(str(projected)) >= 260:
                    raise FileNotFoundError(2, 'Simulated legacy Windows path limit', str(projected))
                return real_memmap(filename, *args, **kwargs)
            with patch('numpy.lib.format.open_memmap', legacy_windows_memmap):
                if not short and not chinese:
                    with self.assertRaises(FileNotFoundError):
                        encoder.encode(['bus late', 'taxi fare', 'coach water'], 'cache-probe')
                    self.assertEqual(len(projected_paths[0]), 263)
                    return
                first = encoder.encode(['bus late', 'taxi fare', 'coach water'], 'cache-probe')
                np.testing.assert_array_equal(first[:, 0], np.ones(3, dtype=np.float32))
                count = len(calls)
                second = encoder.encode(['bus late', 'taxi fare', 'coach water'], 'cache-probe-again')
                self.assertEqual(len(calls), count, 'A valid cache should be reused without inference')
                np.testing.assert_array_equal(first, second)
                self.assertTrue(all(len(p) < 260 for p in projected_paths))
                self.assertFalse(list(root.rglob('*.partial.npy')))
                del first, second

    def test_real_frozen_english_cache_reproduces_legacy_failure(self):
        self.actual_cache(False, False)

    def test_real_frozen_english_cache_write_and_reuse_with_short_paths(self):
        self.actual_cache(False, True)

    def test_real_frozen_chinese_cache_write_and_reuse_with_short_paths(self):
        self.actual_cache(True, True)

if __name__ == '__main__':
    unittest.main()
