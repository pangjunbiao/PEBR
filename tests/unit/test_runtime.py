import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from transport_topics.config import canonical_hash
from transport_topics.experiments.preflight import RESULTS_MAP, validate_profiles
from transport_topics.runtime.artifacts import RunStore, atomic_write, make_inspection, safe_path, source_identity
from transport_topics.runtime.bootstrap import ensure_foundation
from transport_topics.runtime.checkpointing import ProjectLock, validate_run_id
from transport_topics.runtime.device import select_devices
from transport_topics.runtime.logging import RunLogger
from transport_topics.runtime.reproducibility import stable_seed

ROOT = Path(__file__).resolve().parents[2]


class RuntimeTests(unittest.TestCase):
    def test_safe_artifacts_reject_traversal_and_absolute_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            for value in ("../outside.json", Path(tmp).resolve() / "absolute.json"):
                with self.assertRaises(ValueError):
                    safe_path(Path(tmp), value)

    def test_atomic_artifacts_cannot_be_accidentally_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "file.json"
            atomic_write(path, b"first")
            with self.assertRaises(FileExistsError):
                atomic_write(path, b"second")
            self.assertEqual(path.read_bytes(), b"first")

    def test_run_directories_are_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = RunStore(Path(tmp)), RunStore(Path(tmp))
            self.assertNotEqual(a.run_id, b.run_id)

    def test_run_id_rejects_paths(self):
        for value in ("../run", "a/b", "a\\b", "", "/root", "a..b"):
            with self.assertRaises(ValueError):
                validate_run_id(value)

    def test_source_fingerprint_changes_with_code_not_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.py").write_text("pass\n", encoding="utf-8")
            before = source_identity(root)["sha256"]
            (root / "data.txt").write_text("not code", encoding="utf-8")
            self.assertEqual(before, source_identity(root)["sha256"])
            (root / "main.py").write_text("x=1\n", encoding="utf-8")
            self.assertNotEqual(before, source_identity(root)["sha256"])

    def test_os_lock_prevents_a_second_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = ProjectLock(Path(tmp) / "execution.lock")
            b = ProjectLock(Path(tmp) / "execution.lock")
            a.acquire()
            try:
                with self.assertRaises(RuntimeError):
                    b.acquire()
            finally:
                a.release()
            b.acquire()
            b.release()

    def test_stable_seed_is_repeatable_and_namespace_specific(self):
        self.assertEqual(stable_seed(0, "masks", 41), stable_seed(0, "masks", 41))
        self.assertNotEqual(stable_seed(0, "masks", 41), stable_seed(0, "samples", 41))

    def test_cpu_policy_does_not_claim_gpu_use(self):
        selected = select_devices({"device": "cpu", "gpu_index": 0}, {
            "torch": {"import_ok": True, "cuda_available": True, "cuda_smoke": {"status": "passed"}},
            "onnxruntime": {"cuda_provider_advertised": True}})
        self.assertEqual(selected["torch_device_selected"], "cpu")
        self.assertFalse(selected["torch_cuda_execution_verified"])

    def test_explicit_cuda_fails_without_execution_evidence(self):
        with self.assertRaises(RuntimeError):
            select_devices({"device": "cuda", "gpu_index": 0}, {"torch": {"cuda_available": True}})

    def test_advertised_onnx_cuda_is_not_claimed_verified(self):
        selected = select_devices({"device": "auto", "gpu_index": 0}, {
            "onnxruntime": {"cuda_provider_advertised": True}})
        self.assertEqual(selected["encoder_provider_candidate"], "CUDAExecutionProvider")
        self.assertFalse(selected["encoder_provider_execution_verified"])

    def test_verified_cuda_is_used_in_auto_policy(self):
        selected = select_devices({"device": "auto", "gpu_index": 1}, {
            "torch": {"import_ok": True, "cuda_available": True, "cuda_smoke": {"status": "passed"}}})
        self.assertEqual(selected["torch_device_selected"], "cuda:1")

    def test_missing_torch_is_not_reported_as_verified_cpu(self):
        selected = select_devices({"device": "auto", "gpu_index": 0}, {})
        self.assertIsNone(selected["torch_device_selected"])
        self.assertFalse(selected["torch_cpu_execution_verified"])

    def test_invalid_gpu_index_is_not_silently_changed(self):
        with self.assertRaisesRegex(RuntimeError, "gpu_index"):
            select_devices({"device": "auto", "gpu_index": 9}, {"torch": {
                "import_ok": True, "cuda_available": True,
                "cuda_smoke": {"status": "failed", "error": "Configured GPU index is not available"}}})

    def test_all_declared_profiles_validate(self):
        report = validate_profiles(ROOT)
        self.assertEqual(report["validated_overlay_count"], 32)
        self.assertEqual(len(report["datasets"]), 2)
        self.assertFalse(report["model_performance_computed"])

    def test_transportation_has_table_and_figure(self):
        transportation = [item for item in RESULTS_MAP if item["area"] == "transportation"]
        self.assertEqual({x["id"] for x in transportation}, {"table_3", "figure_2"})

    def test_existing_foundation_packages_are_not_installed_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = RunLogger(Path(tmp) / "run_log.txt", "test", io.StringIO())
            try:
                with patch("transport_topics.runtime.bootstrap._install") as installer:
                    records = ensure_foundation(logger)
                installer.assert_not_called()
                self.assertTrue(all(not item["installed_now"] for item in records))
            finally:
                logger.close()

    def test_missing_requirement_respects_no_install(self):
        from importlib import metadata
        real_version = metadata.version
        def version(name):
            if name == "PyYAML":
                raise metadata.PackageNotFoundError(name)
            return real_version(name)
        with tempfile.TemporaryDirectory() as tmp:
            logger = RunLogger(Path(tmp) / "run_log.txt", "test", io.StringIO())
            try:
                with patch("transport_topics.runtime.bootstrap.metadata.version", side_effect=version), \
                     patch("transport_topics.runtime.bootstrap._install") as installer:
                    with self.assertRaises(RuntimeError):
                        ensure_foundation(logger, install_missing=False)
                    installer.assert_not_called()
            finally:
                logger.close()

    def test_missing_only_installer_does_not_reinstall_other_packages(self):
        from importlib import metadata
        real_version = metadata.version
        state = {"installed": False}
        def version(name):
            if name == "PyYAML" and not state["installed"]:
                raise metadata.PackageNotFoundError(name)
            return real_version(name)
        def install(requirement, logger):
            self.assertEqual(requirement, "PyYAML>=6,<7")
            state["installed"] = True
        with tempfile.TemporaryDirectory() as tmp:
            logger = RunLogger(Path(tmp) / "run_log.txt", "test", io.StringIO())
            try:
                with patch("transport_topics.runtime.bootstrap.metadata.version", side_effect=version), \
                     patch("transport_topics.runtime.bootstrap._install", side_effect=install) as installer:
                    records = ensure_foundation(logger, install_missing=True)
                self.assertEqual(installer.call_count, 1)
                self.assertFalse(records[0]["installed_now"])
                self.assertTrue(records[1]["installed_now"])
            finally:
                logger.close()

    def test_incompatible_existing_package_is_not_upgraded(self):
        from importlib import metadata
        real_version = metadata.version
        def version(name):
            return "5.0" if name == "PyYAML" else real_version(name)
        with tempfile.TemporaryDirectory() as tmp:
            logger = RunLogger(Path(tmp) / "run_log.txt", "test", io.StringIO())
            try:
                with patch("transport_topics.runtime.bootstrap.metadata.version", side_effect=version), \
                     patch("transport_topics.runtime.bootstrap._install") as installer:
                    with self.assertRaisesRegex(RuntimeError, "preserved"):
                        ensure_foundation(logger)
                    installer.assert_not_called()
            finally:
                logger.close()

    def test_inspection_has_valid_checksums_and_excludes_raw_data(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RunStore(root)
            store.json("status.json", {"state": "completed"})
            store.json("environment.json", {"safe_inventory": True})
            store.json("complaint_text.json", {"text": "must not enter the diagnostic ZIP"})
            raw = root / "data/raw"
            raw.mkdir(parents=True)
            (raw / "private.txt").write_text("not for inspection", encoding="utf-8")
            logger = RunLogger(root / "logs/run_log.txt", store.run_id, io.StringIO())
            logger.info("finished")
            logger.close()
            store.manifest({"stage": "execution_foundation"})
            inspection = make_inspection(root, store.run_id)
            with zipfile.ZipFile(inspection) as archive:
                self.assertFalse(any("private" in name for name in archive.namelist()))
                self.assertNotIn("run/complaint_text.json", archive.namelist())
                checksums = json.loads(archive.read("inspection_manifest.json"))
                for name, digest in checksums.items():
                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), digest)
