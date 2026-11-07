"""Filesystem recovery checks, including real Windows open-reader behavior."""
import errno
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from transport_topics.runtime.artifacts import (
    ArtifactWriteBlockedError, RunStore, atomic_write, file_hash,
)


def windows_error(code):
    error = PermissionError(errno.EACCES, 'simulated Windows replacement denial')
    error.winerror = code
    return error


class ArtifactRecoveryTests(unittest.TestCase):
    def test_transient_access_denied_retries_and_keeps_previous_bytes_until_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'models_index.json'
            target.write_bytes(b'previous complete index')
            real_replace = os.replace
            attempts = []
            def replace(source, destination):
                attempts.append(1)
                self.assertEqual(Path(source).read_bytes(), b'new complete index')
                if len(attempts) < 3:
                    self.assertEqual(target.read_bytes(), b'previous complete index')
                    raise windows_error(5)
                return real_replace(source, destination)
            with patch('transport_topics.runtime.artifacts.os.replace', side_effect=replace), \
                 patch('transport_topics.runtime.artifacts.time.sleep') as delay:
                atomic_write(target, b'new complete index', overwrite=True)
            self.assertEqual(len(attempts), 3)
            self.assertEqual(delay.call_count, 2)
            self.assertEqual(target.read_bytes(), b'new complete index')
            self.assertFalse(list(target.parent.glob('.writing_*')))

    def test_sharing_and_lock_violations_have_the_same_recovery(self):
        for code in (32, 33):
            with self.subTest(winerror=code), tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / 'index.json'; target.write_bytes(b'old')
                real_replace = os.replace
                calls = []
                def replace(source, destination):
                    calls.append(1)
                    if len(calls) == 1: raise windows_error(code)
                    return real_replace(source, destination)
                with patch('transport_topics.runtime.artifacts.os.replace', side_effect=replace), \
                     patch('transport_topics.runtime.artifacts.time.sleep'):
                    atomic_write(target, b'new', overwrite=True)
                self.assertEqual(target.read_bytes(), b'new')

    def test_exhausted_retries_preserve_old_file_and_fully_written_pending_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'index.json'; target.write_bytes(b'old')
            with patch('transport_topics.runtime.artifacts._WINDOWS_REPLACE_DELAYS', (0., 0.)), \
                 patch('transport_topics.runtime.artifacts.os.replace', side_effect=windows_error(5)) as replace, \
                 patch('transport_topics.runtime.artifacts.time.sleep'):
                with self.assertRaises(ArtifactWriteBlockedError) as caught:
                    atomic_write(target, b'complete pending bytes', overwrite=True)
            error = caught.exception
            self.assertEqual(replace.call_count, 3)
            self.assertEqual(error.attempts, 3)
            self.assertEqual(error.winerror, 5)
            self.assertEqual(target.read_bytes(), b'old')
            self.assertEqual(error.pending_path.read_bytes(), b'complete pending bytes')
            self.assertIn(str(error.pending_path), str(error))

    def test_nonwindows_permission_error_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'index.json'; target.write_bytes(b'old')
            error = PermissionError(errno.EACCES, 'not a Windows sharing error')
            with patch('transport_topics.runtime.artifacts.os.replace', side_effect=error) as replace, \
                 patch('transport_topics.runtime.artifacts.time.sleep') as delay:
                with self.assertRaises(PermissionError):atomic_write(target, b'new', overwrite=True)
            self.assertEqual(replace.call_count, 1); delay.assert_not_called()
            self.assertEqual(target.read_bytes(), b'old')
            self.assertFalse(list(target.parent.glob('.writing_*')))

    def test_unrelated_windows_disk_error_is_not_treated_as_a_temporary_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'index.json'; target.write_bytes(b'old')
            error = OSError(errno.ENOSPC, 'disk full'); error.winerror = 112
            with patch('transport_topics.runtime.artifacts.os.replace', side_effect=error) as replace, \
                 patch('transport_topics.runtime.artifacts.time.sleep') as delay:
                with self.assertRaises(OSError):atomic_write(target, b'new', overwrite=True)
            self.assertEqual(replace.call_count, 1); delay.assert_not_called()
            self.assertEqual(target.read_bytes(), b'old')

    def test_failure_before_payload_is_fsynced_does_not_preserve_an_incomplete_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'index.json'; target.write_bytes(b'old')
            with patch('transport_topics.runtime.artifacts.os.fsync', side_effect=OSError(errno.ENOSPC, 'flush failed')):
                with self.assertRaises(OSError):atomic_write(target, b'new', overwrite=True)
            self.assertEqual(target.read_bytes(), b'old')
            self.assertFalse(list(target.parent.glob('.writing_*')))

    def test_open_reader_released_by_another_thread_allows_update(self):
        # On Windows this exercises a real sharing denial until the reader closes.
        # POSIX permits the rename immediately; neither platform skips the check.
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'folder with spaces'; folder.mkdir()
            target = folder / '\u7ed3\u679c.json'; target.write_bytes(b'old')
            reader = target.open('rb')
            release = threading.Timer(.15, reader.close)
            release.start()
            try:atomic_write(target, '{"method":"proposed"}'.encode(), overwrite=True)
            finally:
                release.join(timeout=2)
                reader.close()
            self.assertEqual(json.loads(target.read_text()), {'method':'proposed'})

    def test_new_run_rebuilds_index_without_touching_saved_model_or_failed_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            saved = root / 'outputs/models/saved.bin'; saved.parent.mkdir(parents=True)
            saved.write_bytes(b'previously committed model bytes')
            before = file_hash(saved)
            failed = RunStore(root)
            failed.json('models_index.json', {'models':['one']})
            with patch('transport_topics.runtime.artifacts._WINDOWS_REPLACE_DELAYS', ()), \
                 patch('transport_topics.runtime.artifacts.os.replace', side_effect=windows_error(5)):
                with self.assertRaises(ArtifactWriteBlockedError):
                    failed.json('models_index.json', {'models':['one','two']}, overwrite=True)
            resumed = RunStore(root)
            resumed.json('models_index.json', {'models':['one']})
            resumed.json('models_index.json', {'models':['one','two']}, overwrite=True)
            self.assertEqual(json.loads((failed.directory/'models_index.json').read_text())['models'], ['one'])
            self.assertEqual(json.loads((resumed.directory/'models_index.json').read_text())['models'], ['one','two'])
            self.assertEqual(file_hash(saved), before)
