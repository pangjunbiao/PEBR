from pathlib import Path
import tempfile
import unittest

from transport_topics.runtime.artifacts import RunStore
from transport_topics.runtime.checkpointing import verify_resume


class ResumeTests(unittest.TestCase):
    def fixture(self, root):
        store = RunStore(root)
        store.json("status.json", {"state": "completed", "operation": "check"})
        store.json("environment.json", {"test": True})
        identity = {"configuration_sha256": "a", "source_sha256": "b", "input_sha256": "c"}
        store.manifest(identity)
        return store, identity

    def test_matching_completed_run_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, identity = self.fixture(root)
            self.assertEqual(verify_resume(root, store.run_id, identity)["validated_artifacts"], 2)

    def test_changed_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, identity = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                verify_resume(root, store.run_id, dict(identity, configuration_sha256="different"))

    def test_tampered_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, identity = self.fixture(root)
            (store.directory / "environment.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                verify_resume(root, store.run_id, identity)

    def test_interrupted_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, identity = self.fixture(root)
            store.json("status.json", {"state": "interrupted"}, overwrite=True)
            with self.assertRaisesRegex(ValueError, "completed"):
                verify_resume(root, store.run_id, identity)
