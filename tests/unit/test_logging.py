import io
from pathlib import Path
import re
import tempfile
import unittest

from transport_topics.runtime.logging import RunLogger, run_log_snapshot


class LoggingTests(unittest.TestCase):
    def test_append_preserves_old_runs_and_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_log.txt"
            for name in ("old", "new"):
                logger = RunLogger(path, name, io.StringIO())
                logger.info("hello %s", name)
                logger.close()
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn(" | old | ", lines[0])
            self.assertIn(" | new | ", lines[1])
            self.assertRegex(lines[1], r"^\d{4}-\d\d-\d\dT.*[+-]\d\d:\d\d \|")

    def test_each_multiline_error_keeps_run_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_log.txt"
            logger = RunLogger(path, "runA", io.StringIO())
            logger.error("first\nsecond")
            logger.close()
            self.assertEqual(run_log_snapshot(path, "runA").decode().count(" | runA | "), 2)

    def test_snapshot_excludes_other_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_log.txt"
            for name in ("A", "B"):
                logger = RunLogger(path, name, io.StringIO())
                logger.info("message")
                logger.close()
            self.assertNotIn(" | B | ", run_log_snapshot(path, "A").decode())
