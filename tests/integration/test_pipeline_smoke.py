import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class PipelineIntegrationTests(unittest.TestCase):
    def test_help_works_outside_project_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            child = subprocess.run([sys.executable, str(ROOT / "main.py"), "--help"], cwd=tmp,
                                   capture_output=True, text=True, encoding="utf-8", timeout=20)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertIn("--install-missing", child.stdout)

    def test_reports_stage_rejects_fitting_mode_with_inspection_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project with spaces"
            root.mkdir()
            for name in ("src", "configs"):
                shutil.copytree(ROOT / name, root / name, ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copy2(ROOT / "main.py", root / "main.py")
            (root/'configs/experiments/future_test.yaml').write_text('experiment:\n  name: future_test\n  stage: reports\n  mode: run\n')
            child = subprocess.run([sys.executable, str(root / "main.py"), "--no-install", "--config",
                                    "configs/experiments/future_test.yaml"], cwd=tmp,
                                   capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
            self.assertEqual(child.returncode, 1, child.stdout + child.stderr)
            latest = json.loads((root / "outputs/manifests/latest_reports.json").read_text(encoding="utf-8"))
            self.assertEqual(latest["state"], "failed")
            self.assertTrue((root / latest["inspection_zip"]).is_file())
            status = json.loads((root / latest["run_directory"] / "status.json").read_text(encoding="utf-8"))
            self.assertIn("requires --mode report", status["error"])
            self.assertFalse(status["research_performance_computed"])
