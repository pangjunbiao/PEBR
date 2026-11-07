"""Prepare a separate, byte-verified revision workspace. Standard library only."""
from __future__ import annotations
import ast
import collections
import csv
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import zipfile

from .protocol import build_protocol, canonical_hash, experiment_plan, validate_protocol, write_plan

MIB = 1024 * 1024
PREPARATION_VERSION = "1.0.1"
REQUIRED_PASSES = {"canonical_run_coverage": 1, "canonical_run_status": 2, "canonical_bank_coverage": 1,
                   "prepared_alignment": 2, "tokens_to_counts": 2, "predictor_fold_idf_prior": 30,
                   "borrowing_mass_and_matching": 9, "top_word_and_td_reproduction": 270,
                   "document_npmi_reproduction": 270, "stability_reproduction": 54,
                   "cv_reproduction": 8, "chinese_masked_query_cache": 1, "core_tests": 1}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(MIB), b""):
            digest.update(block)
    return digest.hexdigest()


def within(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def validate_closeout(directory):
    directory = Path(directory)
    report = read_json(directory / "audit_report.json")
    if report.get("audit_execution") != "completed":
        raise ValueError("Step 1 closeout did not finish")
    findings = report.get("all_findings", [])
    unresolved = [r for r in findings if r.get("status") in {"FAIL", "WARN", "INCOMPLETE", "SKIP"}]
    if unresolved:
        raise ValueError("Step 1 closeout has unresolved findings; do not bypass this gate")
    passes = collections.Counter(r.get("check") for r in findings if r.get("status") == "PASS")
    missing = {key: minimum - passes[key] for key, minimum in REQUIRED_PASSES.items() if passes[key] < minimum}
    if missing:
        raise ValueError("Step 1 coverage is insufficient: " + json.dumps(missing))
    expected = read_json(directory / "canonical_sources.json")
    if len(expected.get("model_directories", [])) != 270 or len(expected.get("action_directories", [])) != 9:
        raise ValueError("Unexpected Step 1 main-run coverage")
    return report, expected


def latest_closeout(root):
    paths = sorted((Path(root) / "outputs/revision_audit/step1_closeout").glob("*/audit_report.json"), reverse=True)
    if not paths:
        raise FileNotFoundError("No step-1 closeout found in this project")
    directory = paths[0].parent
    validate_closeout(directory)
    return directory


def check_source_unchanged(root, closeout):
    root, closeout = Path(root), Path(closeout)
    with (closeout / "source_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    expected = {r["path"]: r["sha256"] for r in records if r["path"].startswith("src/transport_topics/") and r["path"].endswith(".py")}
    actual = {p.relative_to(root).as_posix() for p in (root / "src/transport_topics").rglob("*.py") if "__pycache__" not in p.parts}
    if not expected or actual != set(expected):
        raise ValueError("Scientific source file inventory changed after closeout")
    changed = [relative for relative, digest in expected.items() if file_hash(root / relative) != digest]
    if changed:
        raise ValueError("Scientific source changed after closeout: " + ", ".join(changed))
    return expected


def prepared_records(directory):
    """Read individual numbered corpus records, never the aggregate index list."""
    selected = []
    seen = set()
    expected_corpora = {"primary_full": "transport_grievance", "chinese_pooled": "chinese"}
    paths = sorted(p for p in Path(directory).glob("prepared_*.json")
                   if re.fullmatch(r"prepared_[0-9]+\.json", p.name))
    for path in paths:
        record = read_json(path)
        if not isinstance(record, dict) or not isinstance(record.get("summary"), dict):
            raise ValueError("Numbered prepared record must contain an object and summary object: " + path.name)
        name = record["summary"].get("name")
        if name not in expected_corpora:
            continue
        if name in seen:
            raise ValueError("Duplicate prepared full-corpus record: " + name)
        if not isinstance(record.get("path"), str) or not record["path"]:
            raise ValueError("Prepared record has no valid source path: " + path.name)
        if record.get("corpus") != expected_corpora[name]:
            raise ValueError("Prepared corpus name and language role disagree: " + path.name)
        seen.add(name)
        selected.append((path, record))
    if seen != set(expected_corpora):
        raise ValueError("Expected exactly the two full prepared corpus records")
    return selected


def checked_copy(source, target, progress=None):
    """Create a new independent file, hash both streams and never overwrite."""
    source, target = Path(source), Path(target)
    if source.is_symlink():
        raise ValueError("Snapshot input must be a regular file: " + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    copied, tick = 0, time.monotonic()
    with source.open("rb") as incoming, target.open("xb") as outgoing:
        for block in iter(lambda: incoming.read(MIB), b""):
            outgoing.write(block); digest.update(block)
            copied += len(block)
            if progress is not None and time.monotonic() - tick >= 10:
                progress(copied); tick = time.monotonic()
    expected = digest.hexdigest()
    if file_hash(target) != expected:
        raise IOError("Snapshot copy did not preserve bytes: " + str(source))
    return expected


class Preparation:
    def __init__(self, root, closeout):
        self.root = Path(root).resolve()
        self.closeout = Path(closeout).resolve()
        self.protocol = build_protocol()
        self.protocol_id = validate_protocol(self.protocol)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.run_id = stamp + "_" + self.protocol_id[:10] + "_" + uuid.uuid4().hex[:6]
        self.directory = self.root / "outputs/revision_study/step2" / self.run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.copied = []
        self.copied_paths = set()
        self.report = {"stage": "step2_protocol_and_isolation", "run_id": self.run_id,
                       "preparation_code_version": PREPARATION_VERSION,
                       "status": "preparing", "new_performance_results": False, "experiments_trained": False,
                       "legacy_project_files_replaced": False, "runner_implementation_pending": True,
                       "protocol_sha256": self.protocol_id, "closeout_directory": str(self.closeout)}

    def log(self, text):
        now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec="seconds")
        line = now + " | " + text
        print(line, flush=True)
        with (self.directory / "console_log.txt").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def copy(self, source, relative=None, expected=None):
        source = Path(source)
        if relative is None: relative = source.relative_to(self.root)
        relative = Path(relative)
        target = self.directory / relative
        if not within(target, self.directory):
            raise ValueError("Snapshot target escapes the revision workspace")
        if target in self.copied_paths:
            return next(r["sha256"] for r in self.copied if r["destination"] == relative.as_posix())
        callback = (lambda count: self.log("Copying %s: %.1f MiB" % (relative.as_posix(), count/MIB))) if source.stat().st_size >= 32*MIB else None
        digest = checked_copy(source, target, callback)
        if expected is not None and digest != expected:
            raise ValueError("Input differs from its declared identity: " + str(source))
        self.copied_paths.add(target)
        self.copied.append({"source": str(source), "destination": relative.as_posix(),
                            "bytes": target.stat().st_size, "sha256": digest})
        return digest

    def copy_folder(self, source, relative=None, extensions=None):
        source = Path(source)
        if not source.is_dir(): raise FileNotFoundError(str(source))
        relative = Path(relative) if relative is not None else source.relative_to(self.root)
        for path in sorted(source.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and (extensions is None or path.suffix.lower() in extensions):
                self.copy(path, relative / path.relative_to(source))

    def freeze_source(self):
        expected = check_source_unchanged(self.root, self.closeout)
        for relative, digest in expected.items():
            self.copy(self.root / relative, relative, digest)
        for name in ("configs", "requirements"):
            folder = self.root / name
            if folder.is_dir(): self.copy_folder(folder, Path("provenance/original_configuration") / name,
                                                  {".yaml", ".yml", ".json", ".txt", ".toml"})
        if (self.root / "main.py").is_file():
            self.copy(self.root / "main.py", "provenance/original_main.py")
        own_package = Path(__file__).resolve().parent
        self.copy_folder(own_package, "revision_code/pebr_revision_step2", {".py"})
        entry = own_package.parent / "pebr_step2_prepare.py"
        if entry.is_file(): self.copy(entry, "provenance/pebr_step2_prepare.py")
        self.log("Frozen %d scientific source files; existing main.py/src are unchanged." % len(expected))

    def freeze_inputs(self):
        self.collections = []
        for path, record in prepared_records(self.closeout):
            source = self.root / record["path"]
            required = ["counts.npz", "vocabulary.npy", "source_rows.npy", "families.npy", "folds.npy",
                        "posts.json", "identity.json", "complete.json", "embeddings.npy"]
            bank_files = ["complete.json", "base.npz", "visible.npz", "targets.npz", "candidates.npy",
                          "full_candidates.npy", "eligible.npy", "full_receiver_embeddings.npy"]
            required += ["topic_learning/" + name for name in bank_files]
            if record["corpus"] == "transport_grievance":
                required += ["outer_audit/" + name for name in bank_files] + ["outer_audit/outer.npz"]
            missing = [name for name in required if not (source / name).is_file()]
            if missing: raise FileNotFoundError("Audited prepared inputs are missing: " + ", ".join(missing))
            identity = read_json(source / "identity.json")
            view_settings = identity.get("masked_views", identity.get("views", {}))
            if view_settings.get("count") != self.protocol["views"] or identity.get("neighbors") != self.protocol["candidates_per_view"]:
                raise ValueError("Prepared retrieval settings differ from the locked design")
            seeds = identity.get("predictor_seeds", {})
            expected_seeds = {"folds": 5, "fold_seed": 2710, "mask_seed": 2711, "outer_seed": 2712}
            if seeds != expected_seeds:
                raise ValueError("Prepared default seeds differ from the audited revision starting point")
            self.log("Copying audited original counts, post records and existing masks: " + record["summary"]["name"])
            self.copy_folder(source)
            copied_directory = self.directory / record["path"]
            copied_hashes = {r["destination"]: r["sha256"] for r in self.copied}
            for receipt in copied_directory.rglob("complete.json"):
                for relative, digest in read_json(receipt).get("files", {}).items():
                    target = receipt.parent / relative
                    if not within(target, copied_directory):
                        raise ValueError("Prepared receipt points outside its immutable collection")
                    key = target.relative_to(self.directory).as_posix()
                    if copied_hashes.get(key) != digest:
                        raise ValueError("A prepared input changed after its completion receipt: " + key)
            self.collections.append({"corpus": "chinese_pooled" if record["corpus"] == "chinese" else "transport_grievance",
                                     "relative_directory": record["path"], "summary": record["summary"]})
        if {c["corpus"] for c in self.collections} != {"transport_grievance", "chinese_pooled"} or len(self.collections) != 2:
            raise ValueError("Both full audited collections are required")
        for name in ("audit_report.json", "canonical_sources.json", "environment.json", "source_manifest.csv",
                     "cv_reproduction.csv", "stability_reproduction.csv", "metric_reproduction.csv"):
            self.copy(self.closeout / name, Path("provenance/step1_closeout") / name)
        source_record = read_json(self.closeout / "canonical_sources.json")
        for index, relative in enumerate(source_record["seed_files"]):
            self.copy(self.root / relative, Path("provenance/historical_results") / ("source_%d_seed_results.csv" % index))
        write_json(self.directory / "provenance/collections.json", self.collections)

    def freeze_assets(self):
        primary = next(c for c in self.collections if c["corpus"] == "transport_grievance")
        pidentity = read_json(self.root / primary["relative_directory"] / "identity.json")
        encoder = pidentity.get("encoder", {})
        canonical = read_json(self.closeout / "canonical_sources.json")
        dependency_path = self.root / "outputs/runs" / canonical["primary_run"] / "preparation_dependency.json"
        audit = read_json(dependency_path).get("encoder_audit", {})
        if not encoder.get("model_sha256") or not audit.get("tokenizer_sha256"):
            raise ValueError("Pinned English model/tokenizer hashes are required")
        required = [("assets/pretrained/encoder/onnx/model.onnx", encoder.get("model_sha256")),
                    ("assets/pretrained/encoder/tokenizer.json", audit["tokenizer_sha256"])]
        self.copy(dependency_path, "provenance/original_preparation_dependency.json")
        chinese = next(c for c in self.collections if c["corpus"] == "chinese_pooled")
        cidentity = read_json(self.root / chinese["relative_directory"] / "identity.json")
        manifest_path = "assets/chinese_study/encoder_files.json"
        if file_hash(self.root / manifest_path) != cidentity["encoder_manifest_sha256"]:
            raise ValueError("Chinese encoder manifest changed after the audited preparation")
        pinned = read_json(self.root / manifest_path)
        directory = cidentity["encoder"]["directory"]
        required += [(str(Path(directory) / r["file"]), r["sha256"]) for r in pinned["files"]]
        self.log("Copying frozen encoders into the new workspace; no models are loaded or downloaded.")
        for relative, expected in required:
            self.copy(self.root / relative, relative, expected)
        self.copy_folder(self.root / "assets/chinese_study", extensions={".json", ".txt"})
        for relative in ("assets/pretrained/word_embeddings", "assets/pretrained/chinese_word_embeddings"):
            if (self.root / relative).is_dir():
                self.copy_folder(self.root / relative, extensions={".npz", ".npy", ".json"})
        if (self.root / "third_party/published").is_dir():
            self.copy_folder(self.root / "third_party/published", extensions={".py"})
        self.log("Snapshot bytes copied and verified so far: %.1f MiB" % (sum(r["bytes"] for r in self.copied) / MIB))

    def run_control_checks(self):
        worker = self.directory / "revision_code/pebr_revision_step2/control_checks.py"
        env = dict(os.environ)
        env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        command = [sys.executable, "-B", str(worker), str(self.directory / "src")]
        process = subprocess.run(command, cwd=self.directory, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, timeout=180)
        (self.directory / "control_tests_console.txt").write_bytes(process.stdout)
        result_path = self.directory / "control_tests.json"
        if not result_path.is_file(): raise RuntimeError("Control checks produced no result file; inspect their console log")
        result = read_json(result_path)
        if process.returncode != 0 or not result.get("passed") or result.get("tests_run", 0) < 24:
            raise RuntimeError("New policy control checks did not pass; no training has started")
        self.report["control_checks"] = result
        self.log("New control checks passed: %d tests." % result["tests_run"])

    def freeze_plan(self):
        config_path = self.directory / "configs/revision_protocol.json"
        write_json(config_path, self.protocol)
        if validate_protocol(read_json(config_path)) != self.protocol_id:
            raise ValueError("Protocol changed while being saved")
        rows = experiment_plan(self.protocol)
        write_plan(self.directory / "experiment_plan.csv", rows)
        counts = dict(collections.Counter(r["stage"] for r in rows))
        self.report["plan_counts"] = counts
        self.report["stage2_preparation_complete"] = True
        self.report["frozen_source_and_input_files"] = len(self.copied)
        self.report["copied_bytes"] = sum(r["bytes"] for r in self.copied)
        self.report["status"] = "prepared_for_runner_implementation"
        write_json(self.directory / "snapshot_manifest.json", {"protocol_sha256": self.protocol_id,
                   "files": self.copied, "snapshot_hash": canonical_hash(self.copied),
                   "old_project_modified": False, "hardlinks_or_symlinks_created": False})
        environment = {}
        for name in ("numpy", "scipy", "scikit-learn", "gensim", "torch", "tokenizers", "jieba", "onnxruntime-gpu"):
            try: environment[name] = metadata.version(name)
            except metadata.PackageNotFoundError: environment[name] = None
        write_json(self.directory / "environment.json", {"python": sys.version, "interpreter": sys.executable,
                   "packages": environment, "packages_installed_or_updated": False})
        text = (
            "# Step 2 preparation complete\n\n"
            "The core design and a separate source/data/asset snapshot are ready. No corpus experiment has run.\n\n"
            "Use configs/revision_protocol.json and experiment_plan.csv for the next runner implementation.\n"
            "Every listed job is planned_not_run. The single display run is mask base 2711, model seed 0.\n"
            "The new controls passed synthetic implementation checks; their scientific benefit remains untested.\n\n"
            "Original project files, trained models and existing caches remain unchanged.\n"
            "New masks, predictors, final fits and reports must be written within this revision workspace.\n"
            "Inputs, source and protocol are hash recorded. Do not edit this snapshot after performance runs begin.\n\n"
            "The next step implements the two-corpus comparison runner and validates its default-bank reproduction before the full study.\n"
            "Neural-baseline tuning adequacy and practical transportation benefit are not established by these preparation checks.\n")
        (self.directory / "READ_ME_FIRST.md").write_text(text, encoding="utf-8")
        self.log("Plan saved: %d core final-fit jobs, %d selection cases, %d protected-word cases; none executed." %
                 (counts["core_final_fit"], counts["core_selection"], counts["protected_word_audit"]))

    def finish(self, error=None):
        if error is not None:
            self.report["status"] = "blocked"
            self.report["error"] = type(error).__name__ + ": " + str(error)
            self.log("STEP 2 BLOCKED | " + self.report["error"])
        else:
            self.log("STEP 2 PREPARATION COMPLETE | protocol and isolated snapshot saved; no experiment training")
        write_json(self.directory / "step2_report.json", self.report)
        archive = self.directory / (self.run_id + "_step2_inspection.zip")
        self.log("REVISION WORKSPACE: " + str(self.directory))
        self.log("SEND THIS INSPECTION ZIP: " + str(archive))
        with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for name in ("step2_report.json", "snapshot_manifest.json", "environment.json", "control_tests.json",
                         "control_tests_console.txt", "experiment_plan.csv", "READ_ME_FIRST.md", "console_log.txt", "exception.txt"):
                p = self.directory / name
                if p.is_file(): z.write(p, "step2/" + name)
            p = self.directory / "configs/revision_protocol.json"
            if p.is_file(): z.write(p, "step2/revision_protocol.json")
            for p in (self.directory / "revision_code").rglob("*.py") if (self.directory / "revision_code").is_dir() else []:
                z.write(p, "revision_code/" + p.relative_to(self.directory / "revision_code").as_posix())
            p = self.directory / "provenance/collections.json"
            if p.is_file(): z.write(p, "step2/collections.json")
        self.log("Inspection ZIP saved: %.2f MiB" % (archive.stat().st_size / MIB))
        return archive


def prepare(root, closeout=None):
    root = Path(root).resolve()
    if not (root / "main.py").is_file(): raise ValueError("Run beside the existing main.py")
    closeout = Path(closeout).resolve() if closeout is not None else latest_closeout(root)
    validate_closeout(closeout)
    check_source_unchanged(root, closeout)
    prep = Preparation(root, closeout)
    try:
        prep.log("PEBR STEP 2 | preparation v%s | freeze design and create isolated revision workspace" % PREPARATION_VERSION)
        prep.freeze_source()
        prep.freeze_inputs()
        prep.freeze_assets()
        prep.run_control_checks()
        prep.freeze_plan()
    except Exception as error:
        (prep.directory / "exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
        prep.finish(error)
        return 1
    prep.finish()
    return 0
