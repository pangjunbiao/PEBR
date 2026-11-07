#!/usr/bin/env python3
"""PEBR step-1 audit. Python 3.10+, no installation or experiment execution.

Writes only to a unique outputs/revision_audit/step1/<id> directory.
Existing source, data, model outputs, cache receipts and logs are read only.
Core tests execute against a copied source tree with offline environment flags.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import importlib.metadata as metadata
import importlib.util
import io
import itertools
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import unittest
import uuid
import zipfile

VERSION = "1.0.1-closeout"
MIB = 1024 * 1024
SKIP_DIRS = {".git", ".idea", ".venv", "venv", "env", "site-packages",
             "__pycache__", "node_modules", ".pytest_cache", ".mypy_cache"}
CORE_TESTS = ("test_transfer.py", "test_reference.py", "test_nnls.py",
              "test_search.py", "test_action_scoring.py")
MAIN_STUDIES = {"main", "chinese_full"}
ALIASES = {"proposed": "pebr", "pebr": "pebr", "nmf_original": "nmf_original",
           "nmf_similarity": "similarity_sharing", "similarity_maximum": "similarity_sharing",
           "similarity_sharing": "similarity_sharing", "similarity_matched": "matched_amount",
           "matched_amount": "matched_amount"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def resolve_quality_file(directory):
    directory = Path(directory)
    for filename in ("quality.json", "metrics.json"):
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def topic_quality_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("Topic-quality record is not an object")
    if "topic_metrics" in payload:
        if not isinstance(payload["topic_metrics"], dict):
            raise ValueError("Evaluation envelope has no topic metrics")
        return payload["topic_metrics"]
    return payload


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                          encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(MIB), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def int_value(value):
    result = number(value)
    return int(result) if result is not None and result == int(result) else None


def corpus_name(value):
    text = str(value).lower().replace("\\", "/")
    if any(t in text for t in ("chinese", "beijing", "shanghai", "xiamen")):
        return "chinese"
    if any(t in text for t in ("dabiri", "external_sample", "external_original", "traffic_population")):
        return "legacy_other"
    if text == "external":
        return "legacy_other"
    if any(t in text for t in ("primary", "indian", "grievance")):
        return "transport_grievance"
    return "unknown"


def row_corpus(row, fallback=""):
    for key in ("collection", "condition", "dataset", "job_id"):
        if row.get(key):
            result = corpus_name(row[key])
            if result != "unknown":
                return result
    return corpus_name(fallback)


def csv_rows(path, max_bytes=32 * MIB):
    if Path(path).stat().st_size > max_bytes:
        raise ValueError("CSV exceeds audit read limit; file has not been partially accepted")
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    return rows, list(reader.fieldnames or [])


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple))
                             else v for k, v in row.items()})


def quartiles(values):
    values = sorted(values)
    if not values:
        return {"n": 0}
    def quantile(p):
        pos = (len(values) - 1) * p
        left, right = math.floor(pos), math.ceil(pos)
        return values[left] + (values[right] - values[left]) * (pos - left)
    return {"n": len(values), "min": values[0], "q1": quantile(.25),
            "median": quantile(.5), "q3": quantile(.75), "max": values[-1],
            "mean": statistics.mean(values)}


def inside(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def group_key(row):
    # Never pool corpus-size, parameter, collection or variant conditions.
    keys = ("study", "job_id", "dataset", "collection", "condition", "method", "variant",
            "K", "sample_index", "target_size", "factor", "value", "selection_seed",
            "mask_seed", "reference_seed", "fold_seed")
    return tuple((k, str(row.get(k, ""))) for k in keys if str(row.get(k, "")) != "")


def topic_stability(topics_by_seed):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    scores = []
    for left, right in itertools.combinations(sorted(topics_by_seed), 2):
        a, b = topics_by_seed[left], topics_by_seed[right]
        if len(a) != len(b) or not len(a):
            raise ValueError("Stability requires equal nonzero topic counts")
        jaccard = np.array([[len(set(x) & set(y)) / len(set(x) | set(y)) if set(x) | set(y) else 0.
                            for y in b] for x in a], dtype=float)
        rows, cols = linear_sum_assignment(-jaccard)
        scores.append(float(jaccard[rows, cols].mean()))
    return statistics.mean(scores) if scores else None, len(scores)


def reconcile_summary(seed_rows, summary_rows, tolerance=1e-10):
    """Only reconcile direct seed summaries; sample hierarchies are explicit skips."""
    checks = []
    for index, summary in enumerate(summary_rows):
        corpus = row_corpus(summary)
        if corpus == "legacy_other" or int_value(summary.get("collections", 1)) not in (None, 1):
            continue
        method = summary.get("variant") or summary.get("method")
        chosen = []
        for row in seed_rows:
            if str(row.get("K")) != str(summary.get("K")):
                continue
            rmethod = row.get("variant") or row.get("method")
            if rmethod != method:
                continue
            if corpus != "unknown" and row_corpus(row) != corpus:
                continue
            if row.get("study") and row["study"] not in MAIN_STUDIES and not summary.get("variant"):
                continue
            if any(summary.get(k) and row.get(k) and str(summary[k]) != str(row[k])
                   for k in ("collection", "condition", "dataset", "job_id", "factor", "value")):
                continue
            chosen.append(row)
        if not chosen:
            checks.append({"row": index, "method": method, "K": summary.get("K"),
                           "status": "SKIP", "reason": "No unambiguous matching seed records"})
            continue
        if len({group_key(r) for r in chosen}) != 1:
            checks.append({"row": index, "method": method, "K": summary.get("K"),
                           "status": "SKIP", "reason": "Several sampling/parameter conditions; not pooled"})
            continue
        seeds = [int_value(r.get("seed")) for r in chosen]
        if None in seeds or len(set(seeds)) != len(seeds):
            checks.append({"row": index, "method": method, "K": summary.get("K"),
                           "status": "FAIL", "reason": "Missing or duplicate final-model seeds"})
            continue
        declared = int_value(summary.get("seed_fits", summary.get("seeds")))
        if declared is not None and declared != len(chosen):
            checks.append({"row": index, "status": "FAIL", "reason": "Declared seed count differs",
                           "declared": declared, "found": len(chosen)})
        for metric in ("npmi", "cv", "diversity"):
            values = [number(r.get(metric)) for r in chosen]
            if None in values:
                checks.append({"row": index, "status": "FAIL", "metric": metric,
                               "reason": "Missing or nonfinite seed metric"})
                continue
            for suffix, expected in (("mean", statistics.mean(values)),
                                     ("seed_sd", statistics.stdev(values) if len(values) > 1 else None),
                                     ("sd", statistics.stdev(values) if len(values) > 1 else None)):
                column = metric + "_" + suffix
                if column not in summary or summary[column] == "" or expected is None:
                    continue
                saved = number(summary[column])
                error = abs(saved - expected) if saved is not None else None
                checks.append({"row": index, "method": method, "K": summary.get("K"),
                               "corpus": corpus, "metric": column, "seeds": sorted(seeds),
                               "saved": saved, "recalculated": expected, "absolute_error": error,
                               "status": "PASS" if error is not None and error <= tolerance else "FAIL"})
    return checks


class Audit:
    def __init__(self, root, options):
        self.root = Path(root).resolve()
        self.options = options
        self.run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
        self.output = self.root / "outputs/revision_audit/step1_closeout" / self.run_id
        self.output.mkdir(parents=True, exist_ok=False)
        self.findings = []
        self.inventory = []
        self.paths = []
        self.bundle_candidates = {}
        self.seed_records = []
        self.model_words = []
        self.collections = []
        self.references = {}
        self.json_cache = {}
        self.hasher = {}
        self.hash_budget = options.hash_budget_mb * MIB
        self.predictors = []
        self.started = time.monotonic()
        self.log("PEBR STEP 1 CLOSEOUT | completed main runs only; no experiment fitting")
        self.log("Project: " + str(self.root))
        self.log("Audit directory: " + str(self.output))

    def rel(self, path):
        return Path(path).relative_to(self.root).as_posix()

    def log(self, text):
        line = dt.datetime.now().astimezone().isoformat(timespec="seconds") + " | " + text
        print(line, flush=True)
        with (self.output / "console_log.txt").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def finding(self, status, check, detail, path=None, **data):
        record = {"status": status, "check": check, "detail": detail}
        if path is not None:
            record["path"] = self.rel(path) if inside(path, self.root) else str(path)
        record.update(data)
        self.findings.append(record)
        return record

    def get_json(self, path):
        path = Path(path)
        if path not in self.json_cache:
            if path.stat().st_size > 32 * MIB:
                raise ValueError("JSON exceeds 32 MiB audit limit")
            self.json_cache[path] = read_json(path)
        return self.json_cache[path]

    def digest(self, path):
        path = Path(path)
        stat = path.stat()
        stamp = (stat.st_size, stat.st_mtime_ns)
        cached = self.hasher.get(path)
        if cached is None or cached[:2] != stamp:
            self.hasher[path] = (stamp[0], stamp[1], sha256(path))
        return self.hasher[path][2]

    def queue(self, path, priority=50):
        path = Path(path)
        if path.is_file() and inside(path, self.root):
            self.bundle_candidates[path] = min(priority, self.bundle_candidates.get(path, priority))

    def phase(self, title, function):
        self.log(title)
        start = time.monotonic()
        try:
            function()
        except Exception as error:
            self.finding("INCOMPLETE", title, type(error).__name__ + ": " + str(error))
            with (self.output / "exceptions.txt").open("a", encoding="utf-8") as stream:
                stream.write(title + "\n" + traceback.format_exc() + "\n")
            self.log("This phase could not finish; its exception was recorded.")
        self.log("Phase finished in %.1f s" % (time.monotonic() - start))

    def discover(self):
        for directory, folders, files in os.walk(self.root, followlinks=False):
            here = Path(directory)
            folders[:] = [f for f in sorted(folders) if f not in SKIP_DIRS
                          and not (here / f).is_symlink()
                          and not (here == self.root / "outputs" and f == "revision_audit")]
            for name in sorted(files):
                path = here / name
                if path.is_symlink() or not path.is_file():
                    continue
                try:
                    stat = path.stat()
                except OSError as error:
                    self.finding("INCOMPLETE", "file_inventory", str(error), path)
                    continue
                self.paths.append(path)
                self.inventory.append({"path": self.rel(path), "bytes": stat.st_size,
                                       "mtime_ns": stat.st_mtime_ns})
        write_csv(self.output / "file_inventory.csv", self.inventory)
        self.finding("PASS", "file_inventory", "Read-only metadata inventory completed",
                     files=len(self.paths), bytes=sum(r["bytes"] for r in self.inventory))
        self.log("Inventoried %d files; large checkpoints and pretrained weights will not be packaged." % len(self.paths))

    def source_snapshot(self):
        manifest = []
        for path in self.paths:
            relative = Path(self.rel(path))
            first = relative.parts[0]
            wanted = ((first in {"src", "configs", "requirements", "tests"} and path.suffix.lower() in
                       {".py", ".yaml", ".yml", ".json", ".txt", ".toml", ".cfg"}) or
                      (len(relative.parts) == 1 and (path.suffix == ".py" or path.name in
                       {"requirements.txt", "pyproject.toml", "setup.cfg", "setup.py"})) or
                      (first == "third_party" and path.suffix == ".py") or
                      (first == "assets" and path.suffix.lower() in {".json", ".txt"}
                       and path.stat().st_size < MIB))
            if not wanted or path == Path(__file__).resolve():
                continue
            manifest.append({"path": self.rel(path), "bytes": path.stat().st_size,
                             "sha256": self.digest(path)})
            self.queue(path, 0 if first != "third_party" else 5)
            if path.suffix == ".py":
                try:
                    import ast
                    ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path), feature_version=(3, 10))
                except (SyntaxError, UnicodeError) as error:
                    self.finding("FAIL", "python_310_syntax", str(error), path)
        write_csv(self.output / "source_manifest.csv", manifest)
        self.finding("PASS" if manifest else "INCOMPLETE", "source_snapshot", "Source/configuration files identified",
                     files=len(manifest), content_hash=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest())
        package_names = ("numpy", "scipy", "scikit-learn", "gensim", "torch", "bitermplus", "jieba",
                         "PyYAML", "transformers", "sentence-transformers", "onnxruntime", "onnxruntime-gpu",
                         "tokenizers", "POT", "matplotlib", "threadpoolctl")
        versions = {}
        for name in package_names:
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = None
        write_json(self.output / "environment.json", {"audit_version": VERSION, "python": sys.version,
                   "audit_script_sha256": sha256(Path(__file__).resolve()),
                   "interpreter": sys.executable, "platform": platform.platform(), "packages": versions,
                   "free_bytes": shutil.disk_usage(self.root).free,
                   "environment_variables_collected": False, "packages_installed_or_updated": False})
        # Only syntax checks are automated. Source presence cannot establish a scientific property.
        self.finding("INFO", "source_review_scope", "Current source is collected for review; keyword matches are not treated as proof of correct implementation")

    def raw_data(self):
        results = []
        target_names = {"grievance.csv": ("transport_grievance", "text"),
                        "beijing.csv": ("chinese", "正文"), "shanghai.csv": ("chinese", "正文"),
                        "xiamen.csv": ("chinese", "正文")}
        for path in self.paths:
            if path.name.lower() not in target_names or "outputs" in path.relative_to(self.root).parts:
                continue
            corpus, preferred = target_names[path.name.lower()]
            if path.stat().st_size > 64 * MIB:
                self.finding("INCOMPLETE", "raw_data", "Raw file exceeds 64 MiB audit read limit", path)
                continue
            decoded = None
            for encoding in ("utf-8-sig", "gb18030", "cp1252"):
                try:
                    decoded = path.read_text(encoding=encoding)
                    break
                except UnicodeError:
                    continue
            if decoded is None:
                self.finding("INCOMPLETE", "raw_data", "Cannot decode using known release encodings", path)
                continue
            reader = csv.DictReader(io.StringIO(decoded, newline=""))
            columns = reader.fieldnames or []
            if preferred not in columns:
                self.finding("INCOMPLETE", "raw_text_column", "Expected original text field not found; no column guessed", path,
                             expected=preferred, columns=columns)
                continue
            texts = [(row.get(preferred) or "") for row in reader]
            nonblank = [t for t in texts if t.strip()]
            result = {"path": self.rel(path), "corpus": corpus, "encoding": encoding,
                      "text_column": preferred, "columns": columns, "records": len(texts),
                      "nonblank": len(nonblank), "exact_unique_nonblank": len(set(nonblank)),
                      "characters": quartiles([len(t) for t in nonblank]), "sha256": self.digest(path)}
            results.append(result)
        write_json(self.output / "raw_data_audit.json", results)
        for corpus in ("transport_grievance", "chinese"):
            found = [r for r in results if r["corpus"] == corpus]
            self.finding("PASS" if found else "INCOMPLETE", "raw_data_" + corpus,
                         "Local raw text statistics collected; raw posts are not copied to the ZIP" if found else
                         "No expected raw files found; inspect custom paths in configurations", files=len(found))

    def result_tables(self):
        seed_files = [p for p in self.paths if p.name == "topic_seed_results.csv" or
                      (p.name == "seed_results.csv" and "component" in self.rel(p))]
        summary_checks = []
        inventory = []
        for path in seed_files:
            try:
                rows, columns = csv_rows(path)
                if not {"K", "seed"}.issubset(columns):
                    continue
                target = [r for r in rows if row_corpus(r, self.rel(path)) != "legacy_other"]
                for row in target:
                    if row_corpus(row, self.rel(path)) == "unknown" and "component" in self.rel(path):
                        row = dict(row, collection="primary_full")
                    self.seed_records.append((path, row))
                grouped = collections.defaultdict(list)
                for row in target:
                    grouped[group_key(row)].append(row)
                duplicate_groups = 0
                for key, group in grouped.items():
                    seeds = [str(r.get("seed")) for r in group]
                    duplicate_groups += len(seeds) != len(set(seeds))
                    if sorted(int_value(s) for s in seeds if int_value(s) is not None) != [0, 1, 2, 3, 4]:
                        self.finding("WARN", "final_seed_coverage", "Group does not contain exactly final-model seeds 0-4; inspect whether this is an incomplete or development run", path,
                                     condition=dict(key), seeds=seeds)
                    for row in group:
                        for metric in ("npmi", "cv", "diversity"):
                            value = number(row.get(metric))
                            if value is None or (metric == "npmi" and not -1.00000001 <= value <= 1.00000001) or (
                                    metric == "diversity" and not 0 <= value <= 1):
                                self.finding("FAIL", "seed_metric", "Missing, nonfinite or invalid-range metric", path,
                                             metric=metric, K=row.get("K"), seed=row.get("seed"))
                            if metric == "diversity" and value is not None and int_value(row.get("K")):
                                positions = 10 * int_value(row["K"])
                                if abs(value * positions - round(value * positions)) > 1e-7:
                                    self.finding("WARN", "single_fit_td_arithmetic", "TD is not an integer multiple of 1/(10K); check descriptor count and row type", path,
                                                 value=value, K=row["K"], seed=row.get("seed"))
                self.finding("FAIL" if duplicate_groups else "PASS", "seed_record_uniqueness",
                             "Checked distinct seeds within complete sampling/parameter conditions", path,
                             rows=len(target), groups=len(grouped), duplicate_groups=duplicate_groups)
                inventory.append({"path": self.rel(path), "rows": len(rows), "target_rows": len(target),
                                  "groups": len(grouped), "sha256": self.digest(path)})
                self.queue(path, 10)
                candidates = [path.with_name("topic_quality_summary.csv")]
                if path.name == "seed_results.csv":
                    candidates += list(path.parent.glob("*_five_seeds.csv"))
                for summary_path in candidates:
                    if not summary_path.is_file():
                        continue
                    summaries, _ = csv_rows(summary_path)
                    for check in reconcile_summary(rows, summaries):
                        summary_checks.append(dict(check, seed_file=self.rel(path), summary_file=self.rel(summary_path)))
                    self.queue(summary_path, 10)
            except Exception as error:
                self.finding("INCOMPLETE", "result_table", str(error), path)
        write_csv(self.output / "result_file_index.csv", inventory)
        write_csv(self.output / "summary_reconciliation.csv", summary_checks)
        tested = [r for r in summary_checks if r["status"] != "SKIP"]
        bad = sum(r["status"] == "FAIL" for r in tested)
        self.finding("FAIL" if bad else "PASS" if tested else "INCOMPLETE", "summary_arithmetic",
                     "Direct seed means/SDs checked; ambiguous or hierarchical summaries are not guessed",
                     checked_statistics=len(tested), failures=bad, skipped=sum(r["status"] == "SKIP" for r in summary_checks))
        coverage = collections.defaultdict(set)
        for path, row in self.seed_records:
            if row.get("study", "main") in MAIN_STUDIES:
                coverage[row_corpus(row, self.rel(path))].add(int_value(row.get("K")))
        write_json(self.output / "topic_count_coverage.json", {corpus: {"saved_K": sorted(k for k in values if k is not None),
                   "new_protocol_K": [8, 10, 12], "missing_new_K": [k for k in (8, 10, 12) if k not in values]}
                   for corpus, values in coverage.items()})
        self.finding("INFO", "new_topic_counts", "Missing K=10 is expected before the revision; no experiment is launched by this audit")

    def collect_records(self):
        metadata_names = {"manifest.json", "status.json", "configuration.json", "config.json", "identity.json",
                          "condition.json", "study_summary.json", "progress.json", "model.json", "fit.json",
                          "quality.json", "metrics.json", "audit.json", "coherence_reference.json", "reference.json",
                          "sharing_controls_index.json", "borrowing_controls_index.json", "models_index.json",
                          "model_fit_audits.json", "corpus_preparation.json", "preparation_dependency.json",
                          "all_topic_words.json", "complete.json", "selection.json", "summary.json",
                          "chinese_collections.json", "resolved_protocol.json", "environment.json"}
        for path in self.paths:
            rel = self.rel(path)
            if not (rel.startswith(("outputs/", "cache/", "data/manifests/", "assets/"))):
                continue
            if path.name in metadata_names or (path.suffix.lower() in {".csv", ".json", ".yaml", ".yml"}
                    and any(t in path.name.lower() for t in ("audit", "stability", "summary", "loss_history", "diagnostic", "index", "outer_prediction"))):
                if corpus_name(rel) == "legacy_other":
                    continue
                self.queue(path, 15 if path.name in {"model.json", "quality.json", "fit.json"} else 25)
        # Tail the append-only project log without changing it or bundling its full history.
        log = self.root / "logs/run_log.txt"
        if log.is_file():
            with log.open("rb") as stream:
                size = stream.seek(0, os.SEEK_END)
                stream.seek(max(0, size - 256 * 1024))
                tail = stream.read().decode("utf-8", errors="replace")
            (self.output / "existing_log_tail.txt").write_text(tail, encoding="utf-8")

    def numerical_dependencies(self):
        try:
            import numpy as np
            from scipy import sparse
            self.np, self.sp = np, sparse
            self.finding("PASS", "numerical_dependencies", "Existing NumPy/SciPy imported; nothing installed")
        except Exception as error:
            self.np, self.sp = None, None
            self.finding("INCOMPLETE", "numerical_dependencies", type(error).__name__ + ": " + str(error))

    def array(self, path):
        path = Path(path)
        if path.stat().st_size > 128 * MIB:
            raise ValueError("Array exceeds 128 MiB audit read limit")
        return self.np.load(path, allow_pickle=False, mmap_mode="r")

    def matrix(self, path):
        path = Path(path)
        if path.stat().st_size > 128 * MIB:
            raise ValueError("Sparse array exceeds 128 MiB audit read limit")
        value = self.sp.load_npz(path).tocsr()
        if value.nnz > 40_000_000:
            raise ValueError("Sparse array exceeds 40 million stored entries")
        return value

    def receipt(self, directory):
        path = Path(directory) / "complete.json"
        if not path.is_file():
            self.finding("INCOMPLETE", "artifact_receipt", "No completion receipt; not assumed complete", directory)
            return
        receipt = self.get_json(path)
        entries = receipt.get("files", {}) if isinstance(receipt, dict) else {}
        if not isinstance(entries, dict) or not entries:
            self.finding("INCOMPLETE", "artifact_receipt", "No nonempty file-hash map in receipt", path)
            return
        verified, missing, mismatches, skipped = 0, [], [], []
        for name, expected in entries.items():
            target = Path(directory) / name
            if not inside(target, directory):
                mismatches.append(name)
                continue
            if not target.is_file():
                missing.append(name)
                continue
            expected = expected.get("sha256") if isinstance(expected, dict) else expected
            if not isinstance(expected, str) or not re.fullmatch("[0-9a-fA-F]{64}", expected):
                skipped.append(name)
                continue
            size = target.stat().st_size
            if target not in self.hasher and size > self.hash_budget:
                skipped.append(name)
                continue
            if target not in self.hasher:
                self.hash_budget -= size
            if self.digest(target).lower() != expected.lower():
                mismatches.append(name)
            else:
                verified += 1
        state = "FAIL" if missing or mismatches else "INCOMPLETE" if skipped else "PASS"
        self.finding(state, "artifact_receipt", "Compared recorded file hashes to current bytes", path,
                     verified=verified, missing=missing, mismatches=mismatches, skipped=skipped)

    def prepared_collections(self):
        if self.np is None:
            self.finding("SKIP", "prepared_collections", "Numerical dependencies unavailable")
            return
        np = self.np
        summary_by_path = {}
        def remember(value):
            if isinstance(value, dict):
                if all(key in value for key in ("path", "name", "identity_sha256", "posts", "vocabulary")):
                    candidate = self.root / str(value["path"])
                    if inside(candidate, self.root):
                        summary_by_path[candidate.resolve()] = value
                for item in value.values():
                    remember(item)
            elif isinstance(value, list):
                for item in value:
                    remember(item)
        for path in self.paths:
            if path.name in {"chinese_collections.json", "corpus_preparation.json", "preparation.json"}:
                try:
                    remember(self.get_json(path))
                except Exception as error:
                    self.finding("INCOMPLETE", "prepared_collection_manifest", str(error), path)
        candidates = []
        for path in self.paths:
            if path.name != "counts.npz" or not (path.parent / "vocabulary.npy").is_file():
                continue
            directory = path.parent
            if not all((directory / (name + ".npy")).is_file() for name in ("folds", "families", "source_rows")):
                continue
            summary_path = directory / "condition.json"
            try:
                summary = self.get_json(summary_path) if summary_path.is_file() else summary_by_path.get(directory.resolve(), {})
                if not summary and (directory / "identity.json").is_file():
                    identity = self.get_json(directory / "identity.json")
                    key = canonical_hash(identity)
                    if identity.get("language") == "zh" and directory.name.endswith(key[:16]):
                        shape = identity.get("count_shape", [])
                        if len(shape) == 2:
                            summary = {"name": "chinese_" + str(identity.get("collection", "unknown")),
                                       "identity_sha256": key, "posts": shape[0], "vocabulary": shape[1],
                                       "path": self.rel(directory), "recovered_from_identity": True}
            except Exception:
                summary = {}
            corpus = corpus_name(summary.get("name", self.rel(directory)))
            if corpus not in {"transport_grievance", "chinese"}:
                continue
            full = summary.get("name") in {"primary_full", "chinese_pooled"}
            candidates.append((not full, -path.stat().st_mtime_ns, directory, summary, corpus))
        candidates.sort(key=lambda r: (r[0], r[1], str(r[2])))
        write_json(self.output / "prepared_collection_index.json", [
            {"path": self.rel(p), "corpus": c, "name": s.get("name"), "selected": i < self.options.max_collections}
            for i, (_, _, p, s, c) in enumerate(candidates)])
        for index, (_, _, directory, summary, corpus) in enumerate(candidates[:self.options.max_collections]):
            self.log("Prepared data %d/%d | %s" % (index + 1, min(len(candidates), self.options.max_collections), self.rel(directory)))
            try:
                counts = self.matrix(directory / "counts.npz")
                vocabulary = self.array(directory / "vocabulary.npy")
                folds, families, rows = (self.array(directory / (name + ".npy")) for name in ("folds", "families", "source_rows"))
                n, v = counts.shape
                bad = []
                if vocabulary.shape != (v,) or any(a.shape != (n,) for a in (folds, families, rows)):
                    bad.append("row/vocabulary dimensions")
                if not np.isfinite(counts.data).all() or (counts.data < 0).any():
                    bad.append("finite nonnegative counts")
                if len(set(map(str, vocabulary))) != v:
                    bad.append("unique vocabulary")
                if len(set(map(int, rows))) != n:
                    bad.append("distinct source rows")
                if summary.get("posts") not in (None, n) or summary.get("vocabulary") not in (None, v):
                    bad.append("condition metadata dimensions")
                if not np.issubdtype(folds.dtype, np.integer) or (folds < 0).any():
                    bad.append("integer nonnegative folds")
                if len(families) == len(folds):
                    family_folds = collections.defaultdict(set)
                    for family, fold in zip(families.tolist(), folds.tolist()):
                        family_folds[str(family)].add(int(fold))
                    crossings = sum(len(values) != 1 for values in family_folds.values())
                    if crossings:
                        bad.append("families crossing folds")
                else:
                    crossings = None
                self.finding("FAIL" if bad else "PASS", "prepared_alignment", "Counts, vocabulary, rows, families and folds checked", directory,
                             violations=bad, posts=n, vocabulary=v, family_fold_crossings=crossings)
                if bad:
                    continue
                lengths = np.asarray(counts.sum(axis=1)).ravel()
                collection = {"directory": directory, "summary": summary, "corpus": corpus, "counts": counts,
                              "vocabulary": np.asarray(vocabulary), "folds": np.asarray(folds),
                              "families": np.asarray(families), "source_rows": np.asarray(rows), "tokens": None,
                              "cleaned_texts": None}
                posts_path = directory / "posts.json"
                if posts_path.is_file():
                    posts = self.get_json(posts_path)
                    if isinstance(posts, list) and len(posts) == n:
                        if all(isinstance(r, dict) and isinstance(r.get("cleaned_text"), str) for r in posts):
                            collection["cleaned_texts"] = [r["cleaned_text"] for r in posts]
                        if all(isinstance(r, dict) and isinstance(r.get("tokens"), list) for r in posts):
                            tokens = [r["tokens"] for r in posts]
                        elif all(isinstance(r, dict) and isinstance(r.get("cleaned_text"), str) for r in posts):
                            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
                            stop = set(ENGLISH_STOP_WORDS) - {"no", "not", "nor", "never"}
                            tokens = [[w for w in re.findall(r"\b[a-z][a-z]+\b", r["cleaned_text"]) if w not in stop] for r in posts]
                        else:
                            tokens = None
                        if tokens is not None:
                            collection["tokens"] = tokens
                            mapping = {str(word): i for i, word in enumerate(vocabulary)}
                            rr, cc, vv = [], [], []
                            for row_id, words in enumerate(tokens):
                                for word, count in collections.Counter(map(str, words)).items():
                                    if word in mapping:
                                        rr.append(row_id); cc.append(mapping[word]); vv.append(count)
                            rebuilt = self.sp.csr_matrix((vv, (rr, cc)), shape=counts.shape, dtype=float)
                            delta = (rebuilt - counts).tocsr()
                            error = float(np.max(np.abs(delta.data))) if delta.nnz else 0.0
                            self.finding("PASS" if error <= 1e-10 else "FAIL", "tokens_to_counts",
                                         "Rebuilt vocabulary-aligned counts from stored original-post tokens", posts_path,
                                         max_absolute_error=error)
                    else:
                        self.finding("FAIL", "prepared_posts", "Post-record count differs from matrix rows", posts_path)
                else:
                    self.finding("INCOMPLETE", "tokens_to_counts", "No stored original-post token records", directory)
                write_json(self.output / ("prepared_%02d.json" % index), {
                    "path": self.rel(directory), "corpus": corpus, "summary": summary,
                    "counts_shape": [n, v], "nonzero_entries": counts.nnz,
                    "tokens_per_post": quartiles(lengths.tolist()), "zero_count_posts": int((lengths == 0).sum()),
                    "family_count": len(set(map(str, families))),
                    "fold_sizes": dict(collections.Counter(map(str, folds.tolist()))),
                    "negation_vocabulary": [w for w in ("no", "not", "nor", "never", "不", "没", "没有", "不能") if w in set(map(str, vocabulary))]})
                self.receipt(directory)
                self.collections.append(collection)
                for scope in ("topic_learning", "outer_audit"):
                    bank_dir = directory / scope
                    if bank_dir.is_dir():
                        self.check_bank(collection, bank_dir)
                    else:
                        self.finding("INFO", "missing_" + scope, "No stored bank for this scope; not created by audit", directory)
            except Exception as error:
                self.finding("INCOMPLETE", "prepared_collection", type(error).__name__ + ": " + str(error), directory)
        if not candidates:
            self.finding("INCOMPLETE", "prepared_collections", "No known prepared-count/fold layout found")
        if len(candidates) > self.options.max_collections:
            self.finding("INCOMPLETE", "prepared_collection_limit", "Only the recorded selection was audited; increase --max-collections if needed",
                         selected=self.options.max_collections, discovered=len(candidates))

    def check_bank(self, collection, directory):
        np = self.np
        n, v = collection["counts"].shape
        required = ("base.npz", "visible.npz", "targets.npz", "candidates.npy", "full_candidates.npy", "eligible.npy")
        if not all((directory / name).is_file() for name in required):
            self.finding("INCOMPLETE", "mask_bank", "Known bank files are missing", directory,
                         missing=[name for name in required if not (directory / name).is_file()])
            return
        base, visible, targets = (self.matrix(directory / (name + ".npz")) for name in ("base", "visible", "targets"))
        candidates = self.array(directory / "candidates.npy")
        full = self.array(directory / "full_candidates.npy")
        eligible = self.array(directory / "eligible.npy")
        if candidates.ndim != 3 or candidates.shape[0] != n:
            self.finding("FAIL", "mask_bank_dimensions", "Candidate array does not have (posts,views,neighbors) shape", directory)
            return
        m = candidates.shape[1]
        bad = []
        if base.shape != (n, v) or visible.shape != (n * m, v) or targets.shape != visible.shape or eligible.shape != (n,) or full.ndim != 2 or full.shape[0] != n:
            self.finding("FAIL", "mask_bank_dimensions", "Prepared mask dimensions disagree", directory)
            return
        for name, value in (("base", base), ("visible", visible), ("targets", targets)):
            if not np.isfinite(value.data).all() or (value.data < 0).any():
                bad.append(name + " nonfinite/negative")
        if visible.multiply(targets).nnz:
            bad.append("visible and hidden word types overlap")
        repeated_base = base[np.repeat(np.arange(n), m)]
        active = np.repeat(np.asarray(eligible, dtype=bool), m)
        observed_support = (visible > 0).astype(np.int8)
        hidden_support = (targets > 0).astype(np.int8)
        support_error = observed_support[active] + hidden_support[active] - (repeated_base[active] > 0).astype(np.int8)
        support_error.eliminate_zeros()
        if support_error.nnz:
            bad.append("inner masks do not partition eligible receiver word types")
        if visible[~active].nnz or targets[~active].nnz:
            bad.append("mask-ineligible receivers have nonempty stored views")
        hidden_error = targets - repeated_base.multiply(hidden_support)
        hidden_error.eliminate_zeros()
        if hidden_error.nnz and np.max(np.abs(hidden_error.data)) > 1e-10:
            bad.append("hidden target counts differ from receiver counts")
        expected_visible = repeated_base.multiply(observed_support).tocsr()
        visible_mass = np.asarray(expected_visible.sum(axis=1)).ravel()
        inverse = np.divide(1., visible_mass, out=np.zeros_like(visible_mass, dtype=float), where=visible_mass > 0)
        expected_visible = self.sp.diags(inverse) @ expected_visible
        visible_error = (visible - expected_visible).tocsr()
        if visible_error.nnz and np.max(np.abs(visible_error.data)) > 1e-10:
            bad.append("visible distributions differ from normalized observed receiver counts")
        if (visible_mass[active] <= 0).any() or (np.asarray(targets.sum(axis=1)).ravel()[active] <= 0).any():
            bad.append("eligible receivers have an empty scoring view")
        positive = np.asarray(collection["counts"].sum(axis=1)).ravel() > 0
        violations = 0
        duplicates = 0
        for array in (candidates, full):
            if not np.issubdtype(array.dtype, np.integer):
                bad.append("noninteger donor indices")
                continue
            valid = array >= 0
            if (array < -1).any() or (array[valid] >= n).any():
                bad.append("donor indices outside index universe")
                continue
            receiver_shape = (n,) + (1,) * (array.ndim - 1)
            receivers = np.broadcast_to(np.arange(n).reshape(receiver_shape), array.shape)[valid]
            donors = array[valid]
            violations += int(np.sum((donors == receivers) |
                              (collection["families"][donors] == collection["families"][receivers]) |
                              (collection["folds"][donors] == collection["folds"][receivers]) | ~positive[donors]))
            sorted_rows = np.sort(array.reshape(-1, array.shape[-1]), axis=1)
            duplicates += int(np.sum((sorted_rows[:, 1:] == sorted_rows[:, :-1]) & (sorted_rows[:, 1:] >= 0)))
        if violations:
            bad.append("self/family/fold/empty donor exclusions")
        if duplicates:
            bad.append("duplicate donors within a candidate list")
        if (directory / "outer.npz").is_file():
            outer = self.matrix(directory / "outer.npz")
            if outer.shape != (n, v):
                bad.append("outer target dimensions")
            else:
                if base.multiply(outer).nnz or (visible + targets).multiply(outer[np.repeat(np.arange(n), m)]).nnz:
                    bad.append("untouched word types leak into inner evidence")
                if directory.name == "outer_audit":
                    reconstruction = base + outer - collection["counts"]
                    if reconstruction.nnz and np.max(np.abs(reconstruction.data)) > 1e-10:
                        bad.append("outer bank does not reconstruct original counts")
        self.finding("FAIL" if bad else "PASS", "mask_and_donor_boundaries",
                     "Checked saved arrays; source review must also verify encoder query construction", directory,
                     violations=bad, donor_boundary_entries=violations, duplicate_entries=duplicates,
                     views=m, neighbors=candidates.shape[2], eligible=int(np.asarray(eligible, dtype=bool).sum()))
        hash_path, positions_path = directory / "query_hashes.npy", directory / "query_positions.npy"
        if collection["corpus"] == "transport_grievance" and collection.get("cleaned_texts") is not None and hash_path.is_file() and positions_path.is_file():
            hashes = self.array(hash_path)
            positions = self.array(positions_path)
            mismatches = 0
            if hashes.shape != positions.shape or positions.ndim != 1:
                mismatches += 1
            else:
                outer = self.matrix(directory / "outer.npz") if (directory / "outer.npz").is_file() else None
                for saved_hash, position in zip(hashes, positions):
                    receiver, view = divmod(int(position), m)
                    if not 0 <= receiver < n:
                        mismatches += 1; continue
                    blocked = set(map(str, collection["vocabulary"][targets[int(position)].indices]))
                    if outer is not None:
                        blocked.update(map(str, collection["vocabulary"][outer[receiver].indices]))
                    text = re.sub(r"\b[a-z][a-z]+\b", lambda match: " " if match[0] in blocked else match[0], collection["cleaned_texts"][receiver])
                    text = re.sub(r"\s+", " ", text).strip()
                    mismatches += str(saved_hash) != canonical_hash(text)
            self.finding("PASS" if not mismatches else "FAIL", "saved_masked_query_hashes",
                         "Rebuilt English retrieval queries after removing all hidden-type occurrences and protected outer words", directory,
                         queries=len(hashes), mismatches=mismatches)
        elif collection["corpus"] == "chinese":
            self.finding("INCOMPLETE", "saved_masked_query_hashes", "Chinese bank has no saved per-query text hashes; segmentation/query construction needs collected-source review", directory)
        self.receipt(directory)

    def predictor_artifacts(self):
        if self.np is None:
            self.finding("SKIP", "predictor_artifacts", "Numerical dependencies unavailable")
            return
        np = self.np
        by_identity = {c["summary"].get("identity_sha256"): c for c in self.collections if c["summary"].get("identity_sha256")}
        checked = 0
        for path in self.paths:
            if path.name != "identity.json" or "predictors" not in path.parts:
                continue
            try:
                identity = self.get_json(path)
                c = by_identity.get(identity.get("condition"))
                if c is None or "excluded_fold" not in identity:
                    continue
                directory = path.parent
                expected = np.flatnonzero(c["folds"] != int(identity["excluded_fold"]))
                actual = self.array(directory / "training_indices.npy")
                good = np.array_equal(actual, expected)
                if good:
                    counts = c["counts"][expected]
                    idf = self.array(directory / "idf.npy")
                    recomputed = 1 + np.log((1 + len(expected)) / (1 + np.asarray((counts > 0).sum(axis=0)).ravel()))
                    good = idf.shape == recomputed.shape and np.allclose(idf, recomputed, atol=1e-11, rtol=1e-11)
                    prior = self.array(directory / "prior.npy")
                    smoothing = number(identity.get("settings", {}).get("unigram_smoothing"))
                    if smoothing is not None:
                        expected_prior = np.asarray(counts.sum(axis=0)).ravel() + smoothing
                        expected_prior = expected_prior / expected_prior.sum()
                        good = good and prior.shape == expected_prior.shape and np.allclose(prior, expected_prior, atol=1e-11, rtol=1e-11)
                    else:
                        self.finding("INCOMPLETE", "predictor_prior", "Unigram smoothing missing; exact prior not reconstructed", path)
                self.finding("PASS" if good else "FAIL", "predictor_fold_idf_prior",
                             "Compared training indices and IDF/prior to fold-excluded original counts", directory,
                             excluded_fold=identity["excluded_fold"], training_posts=len(expected), topics=identity.get("topics"))
                self.predictors.append({"identity": identity, "directory": directory, "collection": c})
                checked += 1
            except Exception as error:
                self.finding("INCOMPLETE", "predictor_artifact", str(error), path)
        if not checked:
            self.finding("INCOMPLETE", "predictor_artifacts", "No cached predictors matched the audited prepared collections")

    def borrowing_artifacts(self):
        if self.np is None:
            self.finding("SKIP", "borrowing_artifacts", "Numerical dependencies unavailable")
            return
        np = self.np
        checked = []
        directories = sorted({p.parent for p in self.paths if p.name in {"actual_beta.npz", "actual_beta.npy"}
                              and corpus_name(self.rel(p)) in {"transport_grievance", "chinese"}})
        for directory in directories:
            try:
                if (directory / "actual_beta.npz").is_file():
                    with np.load(directory / "actual_beta.npz", allow_pickle=False) as z:
                        actual = {k: z[k].copy() for k in z.files}
                    with np.load(directory / "offered_beta.npz", allow_pickle=False) as z:
                        offered = {k: z[k].copy() for k in z.files}
                    with np.load(directory / "actual_donor_count.npz", allow_pickle=False) as z:
                        donor_count = {k: z[k].copy() for k in z.files}
                    identity = self.get_json(directory / "identity.json") if (directory / "identity.json").is_file() else {}
                    cap = number(identity.get("settings", {}).get("cap"))
                    bad = []
                    for name, values in actual.items():
                        if name not in offered or values.shape != offered[name].shape:
                            bad.append(name + " offered/actual shape")
                            continue
                        if not np.isfinite(values).all() or not np.isfinite(offered[name]).all() or (values < -1e-12).any() or (values > offered[name] + 1e-12).any():
                            bad.append(name + " realized mass")
                        if cap is not None and (offered[name] > cap + 1e-12).any():
                            bad.append(name + " cap")
                    if "proposed" in actual and "similarity_matched" in actual:
                        if actual["proposed"].shape != actual["similarity_matched"].shape or not np.allclose(actual["proposed"], actual["similarity_matched"], atol=1e-12, rtol=0):
                            bad.append("matched realized masses")
                        if "proposed" not in donor_count or "similarity_matched" not in donor_count or not np.array_equal(donor_count["proposed"], donor_count["similarity_matched"]):
                            bad.append("matched contributing donor counts")
                    else:
                        self.finding("INCOMPLETE", "matched_control", "Both PEBR and matched arrays are required", directory)
                    result = {"path": self.rel(directory), "cap": cap, "violations": bad,
                              "policies": {name: {"posts": len(values), "mean_actual_beta": float(values.mean()),
                                           "coverage": float(np.mean(values > 0))} for name, values in actual.items()}}
                    checked.append(result)
                    self.finding("FAIL" if bad else "PASS", "borrowing_mass_and_matching", "Checked saved per-receiver arrays", directory, violations=bad)
                    if cap is None:
                        self.finding("INCOMPLETE", "borrowing_cap", "No exact saved cap located; cap comparison not guessed", directory)
                elif (directory / "offered_beta.npy").is_file():
                    actual = self.array(directory / "actual_beta.npy")
                    offered = self.array(directory / "offered_beta.npy")
                    good = actual.shape == offered.shape and np.isfinite(actual).all() and np.isfinite(offered).all() and not (actual < -1e-12).any() and not (actual > offered + 1e-12).any()
                    self.finding("PASS" if good else "FAIL", "pebr_realized_mass", "Checked realized versus offered mass", directory)
            except Exception as error:
                self.finding("INCOMPLETE", "borrowing_artifact", type(error).__name__ + ": " + str(error), directory)
        write_json(self.output / "borrowing_array_audit.json", checked)
        if not directories:
            self.finding("INCOMPLETE", "borrowing_artifacts", "No known saved mass-array layout found")

    def real_action_checks(self):
        if self.np is None:
            self.finding("SKIP", "real_action_checks", "Numerical dependencies unavailable")
            return
        import numpy as np
        from scipy.optimize import lsq_linear
        by_id = {c["summary"].get("identity_sha256"): c for c in self.collections if c["summary"].get("identity_sha256")}
        checked_banks = 0
        rows_out = []
        for path in sorted(p for p in self.paths if p.name == "selected_donors.npy"):
            directory = path.parent
            if not (directory / "identity.json").is_file() or not (directory / "transferred.npz").is_file():
                continue
            try:
                identity = self.get_json(directory / "identity.json")
                c = by_id.get(identity.get("condition"))
                if c is None:
                    continue
                if checked_banks >= self.options.max_action_banks:
                    self.finding("INCOMPLETE", "real_action_bank_limit", "Remaining stored action banks were not numerically sampled; increase --max-action-banks", directory)
                    continue
                checked_banks += 1
                settings = identity.get("settings", {})
                cap, lam = number(settings.get("cap")), number(settings.get("downside_weight"))
                if cap is None or lam is None:
                    self.finding("INCOMPLETE", "real_action_settings", "Cap or downside weight missing; no default guessed", directory)
                    continue
                bank_dir = c["directory"] / identity.get("scope", "topic_learning")
                base = self.matrix(bank_dir / "base.npz")
                lengths = np.asarray(base.sum(axis=1)).ravel()
                inverse = np.divide(1., lengths, out=np.zeros_like(lengths, dtype=float), where=lengths > 0)
                base = self.sp.diags(inverse) @ base
                transferred = self.matrix(directory / "transferred.npz")
                actual = self.array(directory / "actual_beta.npy")
                offered = self.array(directory / "offered_beta.npy")
                selected = self.array(directory / "selected_donors.npy")
                utility = self.array(directory / "utility.npy")
                gains = self.array(directory / "view_gains.npy")
                if transferred.shape != base.shape or actual.shape != (len(lengths),) or offered.shape != actual.shape:
                    self.finding("FAIL", "real_transfer_dimensions", "Saved transfer and receiver rows disagree", directory)
                    continue
                tv = .5 * np.asarray(abs(transferred - base).sum(axis=1)).ravel()
                mass_error = np.abs(np.asarray(transferred.sum(axis=1)).ravel() - np.asarray(base.sum(axis=1)).ravel())
                good = np.isfinite(transferred.data).all() and not (transferred.data < -1e-12).any() and not (tv > actual + 1e-10).any() and not (mass_error > 1e-10).any() and not (offered > cap + 1e-12).any()
                self.finding("PASS" if good else "FAIL", "real_count_space_transfer_bound",
                             "Checked every stored receiver row in this bank; bound concerns normalized count space", directory,
                             posts=len(lengths), max_tv=float(np.max(tv)), cap=cap, max_mass_error=float(np.max(mass_error)))
                if selected.ndim != 2 or selected.shape[0] != len(lengths) or gains.ndim != 2 or gains.shape[0] != len(lengths) or utility.shape != (len(lengths),):
                    self.finding("FAIL", "stored_action_shapes", "Saved donors, gains or utilities have incompatible dimensions", directory)
                    continue
                full_candidates = self.array(bank_dir / "full_candidates.npy")
                per_view_candidates = self.array(bank_dir / "candidates.npy")
                expected_alpha = np.zeros(len(lengths), dtype=float)
                selection_errors = 0
                for receiver in range(len(lengths)):
                    donor_ids = [int(j) for j in selected[receiver] if j >= 0]
                    union = set(map(int, per_view_candidates[receiver].ravel())) - {-1}
                    if len(set(donor_ids)) != len(donor_ids) or any(j not in union for j in donor_ids):
                        selection_errors += 1
                    available = set(map(int, full_candidates[receiver])) - {-1}
                    if donor_ids:
                        expected_alpha[receiver] = float(offered[receiver]) * sum(j in available for j in donor_ids) / len(donor_ids)
                    elif abs(float(offered[receiver])) > 1e-12:
                        selection_errors += 1
                alpha_error = float(np.max(np.abs(expected_alpha - actual)))
                self.finding("PASS" if not selection_errors and alpha_error <= 1e-10 else "FAIL", "stored_action_availability",
                             "Checked selected donor union membership and realized mass under unavailable-donor fallback", directory,
                             invalid_actions=selection_errors, max_realized_mass_error=alpha_error)
                nonempty = np.any(selected >= 0, axis=1)
                recomputed_utility = gains.mean(axis=1) - lam * np.maximum(-gains, 0).mean(axis=1)
                uerr = float(np.max(np.abs(recomputed_utility - utility)))
                margin = number(settings.get("margin"))
                admitted_good = np.isfinite(gains).all() and np.isfinite(utility).all() and uerr <= 1e-8
                if margin is not None:
                    admitted_good = admitted_good and bool(np.all(utility[nonempty] > margin))
                self.finding("PASS" if admitted_good else "FAIL", "stored_selection_utility",
                             "Recomputed utility from all saved view gains; this is not an untouched-word guarantee", directory,
                             max_utility_error=uerr, admitted=int(nonempty.sum()), margin=margin)
                if self.options.score_rows == 0:
                    self.finding("SKIP", "direct_saved_action_prediction", "Disabled by --score-rows 0", directory)
                    continue
                visible = self.matrix(bank_dir / "visible.npz")
                targets = self.matrix(bank_dir / "targets.npz")
                candidate_lists = self.array(bank_dir / "candidates.npy")
                m = gains.shape[1]
                if candidate_lists.shape[:2] != (len(lengths), m):
                    self.finding("FAIL", "real_action_views", "Candidate and saved-gain view counts disagree", directory)
                    continue
                donor_lengths = np.asarray(c["counts"].sum(axis=1)).ravel()
                donor_inverse = np.divide(1., donor_lengths, out=np.zeros_like(donor_lengths, dtype=float), where=donor_lengths > 0)
                donor_rows = self.sp.diags(donor_inverse) @ c["counts"]
                sample = np.flatnonzero(nonempty)[:self.options.score_rows]
                if not len(sample):
                    self.finding("SKIP", "direct_saved_action_prediction", "No admitted action exists to sample in this bank", directory)
                for i in sample:
                    fold = int(c["folds"][i])
                    keys = identity.get("predictor_keys", [])
                    key = keys[fold] if fold < len(keys) else None
                    refs = [r for r in self.predictors if r["identity"].get("condition") == identity["condition"]
                            and int_value(r["identity"].get("topics")) == int_value(identity.get("topics"))
                            and int_value(r["identity"].get("excluded_fold")) == fold
                            and (key is None or r["directory"].name.endswith(key[:16]))]
                    if len(refs) != 1:
                        self.finding("INCOMPLETE", "direct_saved_action_prediction", "Exact saved predictor could not be identified uniquely", directory,
                                     receiver=int(i), reference_matches=len(refs))
                        continue
                    refdir, refid = refs[0]["directory"], refs[0]["identity"]
                    H = np.asarray(self.array(refdir / "dictionary.npy"), dtype=float)
                    idf = np.asarray(self.array(refdir / "idf.npy"), dtype=float)
                    prior = np.asarray(self.array(refdir / "prior.npy"), dtype=float)
                    prior = prior / prior.sum()
                    smoothing = number(refid.get("settings", {}).get("smoothing"))
                    if smoothing is None:
                        self.finding("INCOMPLETE", "direct_saved_action_prediction", "Saved predictive smoothing missing", refdir)
                        continue
                    words = H / idf[None, :]
                    def predict(p):
                        fitted = lsq_linear(H.T, np.asarray(p) * idf, bounds=(0, np.inf),
                                            method="bvls", lsq_solver="exact", tol=1e-12)
                        if not fitted.success:
                            raise ArithmeticError("Direct dictionary least-squares solver did not converge")
                        reconstructed = fitted.x @ words
                        total = reconstructed.sum()
                        return prior if total <= 0 else (1 - smoothing) * reconstructed / total + smoothing * prior
                    donors = [int(j) for j in selected[i] if j >= 0]
                    final_available = [j for j in donors if j in set(map(int, full_candidates[i]))]
                    share = float(offered[i]) / len(donors)
                    final_expected = (1 - share * len(final_available)) * base[int(i)].toarray().ravel()
                    if final_available:
                        final_expected += share * np.asarray(donor_rows[final_available].sum(axis=0)).ravel()
                    transfer_error = float(np.max(np.abs(final_expected - transferred[int(i)].toarray().ravel())))
                    self.finding("PASS" if transfer_error <= 2e-10 else "FAIL", "direct_saved_final_mixture",
                                 "Rebuilt saved final representation from original counts and selected available donors", directory,
                                 receiver=int(i), max_mixture_error=transfer_error)
                    current = []
                    for view in range(m):
                        x = visible[int(i) * m + view].toarray().ravel()
                        target = targets[int(i) * m + view].toarray().ravel()
                        if x.sum() <= 0 or target.sum() <= 0:
                            raise ValueError("Selected receiver has an empty inner scoring view")
                        x = x / x.sum()
                        available = [j for j in donors if j in set(map(int, candidate_lists[i, view]))]
                        if not available:
                            current.append(0.)
                            continue
                        share = float(offered[i]) / len(donors)
                        mixed = (1 - share * len(available)) * x + share * np.asarray(donor_rows[available].sum(axis=0)).ravel()
                        p0, p1 = predict(x), predict(mixed)
                        if not np.isfinite(p0).all() or not np.isfinite(p1).all() or (p0 <= 0).any() or (p1 <= 0).any():
                            raise ArithmeticError("Invalid independently reconstructed predictive probabilities")
                        current.append(float(target @ (np.log(p1) - np.log(p0)) / target.sum()))
                    error = float(np.max(np.abs(np.asarray(current) - gains[i])))
                    row = {"bank": self.rel(directory), "receiver_index": int(i), "source_row": int(c["source_rows"][i]),
                           "K": identity.get("topics"), "scope": identity.get("scope"), "corpus": c["corpus"],
                           "max_direct_gain_error": error, "max_final_mixture_error": transfer_error,
                           "tolerance": 2e-7, "status": "PASS" if error <= 2e-7 and transfer_error <= 2e-10 else "FAIL"}
                    rows_out.append(row)
                    self.finding(row["status"], "direct_saved_action_prediction",
                                 "Recomputed saved masked-view gains using independent direct dictionary NNLS; no model fitting", directory,
                                 receiver=int(i), max_gain_error=error, tolerance=2e-7)
                self.log("Stored action bank %d | %s" % (checked_banks, self.rel(directory)))
            except Exception as error:
                self.finding("INCOMPLETE", "real_action_checks", type(error).__name__ + ": " + str(error), directory)
        write_csv(self.output / "direct_saved_action_checks.csv", rows_out)
        if not checked_banks:
            self.finding("INCOMPLETE", "real_action_checks", "No complete saved action bank matched the prepared collection identities")

    def untouched_word_records(self):
        output = []
        for path in self.paths:
            if not (path.name.endswith("_outer_prediction.csv") or path.name in {"outer_prediction.csv", "prediction_audit.csv"}):
                continue
            if corpus_name(self.rel(path)) == "legacy_other":
                continue
            try:
                rows, columns = csv_rows(path)
                if "gain_nats_per_token" not in columns:
                    self.finding("INCOMPLETE", "untouched_word_records", "Unrecognized saved gain schema; file collected for review", path)
                    self.queue(path, 15)
                    continue
                gains = [number(row.get("gain_nats_per_token")) for row in rows]
                if not rows or None in gains:
                    self.finding("FAIL", "untouched_word_records", "Empty or nonfinite prediction-gain records", path)
                    continue
                action_column = "action_log_score" if "action_log_score" in columns else "proposed_log_score"
                bad = 0
                for row, gain in zip(rows, gains):
                    before, after = number(row.get("no_borrowing_log_score")), number(row.get(action_column))
                    if before is None or after is None or abs((after - before) - gain) > 1e-10:
                        bad += 1
                keys = [(row.get("document_index"), row.get("source_row")) for row in rows]
                if len(set(keys)) != len(keys):
                    bad += 1
                result = {"path": self.rel(path), "corpus": corpus_name(self.rel(path)), "posts": len(rows),
                          "K_values": sorted(set(row.get("topics", "") for row in rows)),
                          "mean_post_gain": statistics.mean(gains), "median_post_gain": statistics.median(gains),
                          "fraction_with_loss": sum(g < -1e-10 for g in gains) / len(gains),
                          "mean_negative_part_all_posts": statistics.mean(max(-g, 0) for g in gains),
                          "cohort_sha256": hashlib.sha256(json.dumps(keys, ensure_ascii=False).encode()).hexdigest(),
                          "gain_arithmetic_errors": bad}
                output.append(result)
                self.finding("FAIL" if bad else "PASS", "untouched_word_saved_arithmetic",
                             "Verified saved gain differences and summarized loss magnitudes; targets/queries still require boundary review", path,
                             posts=len(rows), arithmetic_errors=bad)
                self.queue(path, 15)
            except Exception as error:
                self.finding("INCOMPLETE", "untouched_word_records", str(error), path)
        write_csv(self.output / "untouched_word_saved_summary.csv", output)
        self.finding("INFO", "untouched_word_scope", "Existing protected-word records are summarized, not repeated; selection randomness is a later experiment",
                     saved_audits=len(output))

    def model_artifacts(self):
        if self.np is None:
            self.finding("SKIP", "model_artifacts", "Numerical dependencies unavailable")
            return
        np = self.np
        models = []
        for path in self.paths:
            if path.name != "model.json":
                continue
            try:
                info = self.get_json(path)
                if not isinstance(info, dict) or not info.get("topic_model", True) or info.get("development_only"):
                    continue
                corpus = corpus_name(info.get("condition", self.rel(path)))
                if corpus not in {"transport_grievance", "chinese"}:
                    continue
                models.append((path, info, corpus))
            except Exception as error:
                self.finding("INCOMPLETE", "model_metadata", str(error), path)
        # Round-robin over corpus/method/K, newest artifacts first within a stratum.
        strata = collections.defaultdict(list)
        for record in models:
            strata[(record[2], str(record[1].get("method")), str(record[1].get("topics")))].append(record)
        for values in strata.values():
            values.sort(key=lambda r: (int_value(r[1].get("seed")) != 0, -r[0].stat().st_mtime_ns, str(r[0])))
        ordered = []
        for depth in range(max((len(v) for v in strata.values()), default=0)):
            for key in sorted(strata):
                if depth < len(strata[key]):
                    ordered.append(strata[key][depth])
        selected = ordered[:self.options.max_models]
        selected_paths = {p for p, _, _ in selected}
        write_csv(self.output / "model_artifact_index.csv", [
            {"path": self.rel(p), "corpus": corpus, "method": info.get("method"), "K": info.get("topics"),
             "seed": info.get("seed"), "identity": info.get("identity"), "selected": p in selected_paths}
            for p, info, corpus in models])
        collections_by_id = {c["summary"].get("identity_sha256"): c for c in self.collections if c["summary"].get("identity_sha256")}
        self.quality_by_model = collections.defaultdict(list)
        for source, row in self.seed_records:
            if row.get("model_identity") and row.get("evaluation_path"):
                path = resolve_quality_file(self.root / row["evaluation_path"])
                if path is not None and inside(path, self.root):
                    self.quality_by_model[row["model_identity"]].append(path)
        metric_rows, provenance, baseline_rows = [], [], []
        for index, (path, info, corpus) in enumerate(selected):
            try:
                directory = path.parent
                linked_rows = [r for _, r in self.seed_records if r.get("model_identity") == info.get("identity")]
                bad_links = [r for r in linked_rows if int_value(r.get("K")) != int_value(info.get("topics"))
                             or int_value(r.get("seed")) != int_value(info.get("seed"))]
                if bad_links:
                    self.finding("FAIL", "seed_to_model_identity", "Saved seed rows identify a model with a different seed or topic count", path,
                                 mismatched_links=len(bad_links))
                vocabulary = self.array(directory / "vocabulary.npy")
                scores = self.array(directory / "topic_scores.npy")
                k = int_value(info.get("topics"))
                expected_v = int_value(info.get("vocabulary"))
                expected_n = int_value(info.get("posts"))
                if scores.ndim != 2 or scores.shape != (k, len(vocabulary)) or expected_v != len(vocabulary) or not np.isfinite(scores).all():
                    self.finding("FAIL", "topic_score_alignment", "Topic scores/vocabulary/metadata disagree or contain nonfinite values", path)
                    continue
                if len(vocabulary) < 10:
                    self.finding("INCOMPLETE", "top_ten_words", "Vocabulary smaller than the declared ten-word descriptor", path)
                    continue
                if len(set(map(str, vocabulary))) != len(vocabulary):
                    self.finding("FAIL", "model_vocabulary", "Model vocabulary has duplicate entries", path)
                c = collections_by_id.get(info.get("condition_identity"))
                if c is not None and not np.array_equal(vocabulary, c["vocabulary"]):
                    self.finding("FAIL", "model_prepared_vocabulary", "Model word order differs from its identified prepared collection", path)
                descriptor = np.asarray(vocabulary)[np.argsort(-scores, axis=1, kind="stable")[:, :10]]
                td = len(set(map(str, descriptor.ravel()))) / (10 * k)
                flat = int(np.sum(np.ptp(scores, axis=1) == 0))
                self.finding("WARN" if flat else "PASS", "topic_score_alignment", "Checked finite K-by-vocabulary arrays; signed neural logits are allowed", path,
                             flat_topics=flat, diversity_recalculated=td)
                for name in ("doc_topics", "retrieval_features"):
                    array_path = directory / (name + ".npy")
                    if not array_path.is_file():
                        self.finding("INCOMPLETE", "document_features", "Expected saved array missing", array_path)
                        continue
                    features = self.array(array_path)
                    valid = features.ndim == 2 and features.shape[0] == expected_n and features.shape[1] == k and np.isfinite(features).all() and not (features < -1e-10).any()
                    self.finding("PASS" if valid else "FAIL", "document_features", "Checked shape and finite nonnegative values", array_path)
                identity_path = directory / "identity.json"
                identity = self.get_json(identity_path) if identity_path.is_file() else {}
                for name, expected in identity.get("implementation", {}).items():
                    current = self.root / "src/transport_topics" / name
                    if current.is_file() and isinstance(expected, str):
                        provenance.append({"model": self.rel(directory), "source": self.rel(current),
                                           "saved_sha256": expected, "current_sha256": self.digest(current),
                                           "same": expected == self.digest(current)})
                quality_paths = sorted(set(self.quality_by_model.get(info.get("identity"), [])))
                if not quality_paths:
                    self.finding("INCOMPLETE", "saved_quality_link", "Neither supported topic-quality file is linked by a seed record", path)
                for quality_path in quality_paths:
                    quality = topic_quality_payload(self.get_json(quality_path))
                    same_words = quality.get("top_words") == descriptor.tolist()
                    saved_td = number(quality.get("diversity"))
                    td_ok = saved_td is not None and abs(saved_td - td) <= 1e-10
                    self.finding("PASS" if same_words and td_ok else "FAIL", "top_word_and_td_reproduction",
                                 "Re-extracted top words from saved scores using stable vocabulary-order ties", quality_path,
                                 words_equal=same_words, td_saved=saved_td, td_recalculated=td)
                    record = {"model": self.rel(directory), "quality": self.rel(quality_path), "corpus": corpus,
                              "method": info.get("method"), "K": k, "seed": info.get("seed"),
                              "td_saved": saved_td, "td_recomputed": td, "top_words_equal": same_words}
                    if c is not None and quality.get("reference_documents") == c["counts"].shape[0] and np.array_equal(vocabulary, c["vocabulary"]):
                        binary = (c["counts"] > 0).astype(np.int64).tocsc()
                        values = []
                        mapping = {str(w): j for j, w in enumerate(vocabulary)}
                        n = binary.shape[0]
                        for words in descriptor:
                            ids = [mapping[str(w)] for w in words]
                            local = binary[:, ids]
                            df = np.asarray(local.sum(axis=0)).ravel()
                            pairs = (local.T @ local).toarray()
                            terms = []
                            for left, right in itertools.combinations(range(10), 2):
                                count = int(pairs[left, right])
                                terms.append(-1. if count == 0 else 0. if count == n else
                                             max(-1., min(1., math.log(count * n / (df[left] * df[right])) / -math.log(count / n))))
                            values.append(statistics.mean(terms))
                        recomputed = statistics.mean(values)
                        saved = number(quality.get("npmi"))
                        ok = saved is not None and abs(saved - recomputed) <= 1e-9
                        record.update(npmi_saved=saved, npmi_recomputed=recomputed)
                        self.finding("PASS" if ok else "FAIL", "document_npmi_reproduction",
                                     "Independently recomputed document-presence NPMI from identified original counts", quality_path,
                                     saved=saved, recalculated=recomputed)
                    else:
                        self.finding("INCOMPLETE", "document_npmi_reproduction", "Exact matching original-count reference not identified; no substitute reference used", quality_path)
                    metric_rows.append(record)
                    self.model_words.append({"path": path, "info": info, "corpus": corpus,
                                             "words": descriptor.tolist(), "collection": c, "quality_path": quality_path,
                                             "quality": quality})
                audit = info.get("audit", {})
                fit = audit.get("fit", {})
                if fit.get("converged") is False:
                    self.finding("FAIL", "fit_completion", "Saved NMF fit declares nonconvergence", path)
                if info.get("method") in {"knntm", "glocom"}:
                    complete = int_value(audit.get("completed_epochs"))
                    epochs = int_value(audit.get("epochs"))
                    if complete is None or epochs is None:
                        self.finding("INCOMPLETE", "neural_epoch_budget", "Completed/requested epochs not both recorded", path)
                    elif complete != epochs:
                        self.finding("FAIL", "neural_epoch_budget", "Saved training did not reach its declared epoch budget", path,
                                     completed=complete, declared=epochs)
                    else:
                        self.finding("PASS", "neural_epoch_budget", "Saved metadata declares full training budget", path,
                                     completed=complete)
                    inputs = audit.get("input_preparation", {})
                    vector_info = inputs.get("word_vectors", {})
                    baseline_rows.append({"model": self.rel(directory), "corpus": corpus, "method": info.get("method"),
                                          "K": k, "seed": info.get("seed"), "epochs": epochs, "completed_epochs": complete,
                                          "word_vector_audit": vector_info, "topic_score_kind": audit.get("topic_score_kind"),
                                          "implementation": audit.get("implementation"), "settings": audit.get("settings")})
                self.receipt(directory)
            except Exception as error:
                self.finding("INCOMPLETE", "model_artifact", type(error).__name__ + ": " + str(error), path)
            if index == 0 or (index + 1) % 20 == 0 or index + 1 == len(selected):
                self.log("Saved models checked: %d/%d" % (index + 1, len(selected)))
        write_csv(self.output / "metric_reproduction.csv", metric_rows)
        write_csv(self.output / "model_source_provenance.csv", provenance)
        write_json(self.output / "neural_baseline_audit.json", baseline_rows)
        drift = sum(not r["same"] for r in provenance)
        self.finding("INFO", "source_provenance", "Source differences since fitting are recorded; differences alone do not prove saved results are invalid",
                     comparisons=len(provenance), differing_files=drift)
        if len(models) > len(selected):
            self.finding("INCOMPLETE", "model_sampling_limit", "Model artifact audit was limited to the recorded stratified selection",
                         discovered=len(models), checked=len(selected))
        if not models:
            self.finding("INCOMPLETE", "model_artifacts", "No known live model.json layout found")

    def cv_reproduction(self):
        if not self.options.cv_checks:
            self.finding("SKIP", "cv_reproduction", "Disabled by --cv-checks 0")
            return
        try:
            from gensim.corpora import Dictionary
            from gensim.models import CoherenceModel
        except Exception as error:
            self.finding("INCOMPLETE", "cv_reproduction", "Existing Gensim could not import: " + str(error))
            return
        preferred_methods = {"proposed": 0, "nmf_original": 1, "knntm": 2, "glocom": 3, "nmf_similarity": 4}
        candidates = [r for r in self.model_words if r["collection"] is not None and r["collection"].get("tokens") is not None]
        candidates.sort(key=lambda r: (int_value(r["info"].get("topics")) != 12,
                                       int_value(r["info"].get("seed")) != 0,
                                       preferred_methods.get(r["info"].get("method"), 99), r["corpus"], str(r["path"])))
        selected, seen = [], set()
        for row in candidates:
            key = (row["corpus"], row["info"].get("method"))
            if key not in seen:
                selected.append(row); seen.add(key)
            if len(selected) >= self.options.cv_checks:
                break
        results = []
        for index, row in enumerate(selected):
            self.log("C_v check %d/%d | %s %s K=%s seed=%s" %
                     (index + 1, len(selected), row["corpus"], row["info"].get("method"),
                      row["info"].get("topics"), row["info"].get("seed")))
            try:
                c = row["collection"]
                if row["quality"].get("reference_documents") != len(c["tokens"]):
                    self.finding("INCOMPLETE", "cv_reproduction", "Original token reference size differs; no alternative guessed", row["quality_path"])
                    continue
                dictionary = Dictionary(c["tokens"])
                model = CoherenceModel(topics=row["words"], texts=c["tokens"], dictionary=dictionary,
                                       coherence="c_v", window_size=110, processes=1, topn=10)
                score = float(model.get_coherence())
                saved = number(row["quality"].get("cv"))
                difference = abs(score - saved) if saved is not None and math.isfinite(score) else None
                tolerance = 2e-6
                status = "PASS" if difference is not None and difference <= tolerance else "WARN"
                result = {"quality": self.rel(row["quality_path"]), "corpus": row["corpus"], "method": row["info"].get("method"),
                          "saved": saved, "recomputed": score if math.isfinite(score) else None,
                          "absolute_difference": difference, "tolerance": tolerance, "status": status}
                results.append(result)
                self.finding(status, "cv_reproduction", "Representative C_v recalculation; differences require token/version review", row["quality_path"],
                             saved=saved, recalculated=result["recomputed"], tolerance=tolerance)
            except Exception as error:
                self.finding("INCOMPLETE", "cv_reproduction", type(error).__name__ + ": " + str(error), row["quality_path"])
        write_csv(self.output / "cv_reproduction.csv", results)
        if not selected:
            self.finding("INCOMPLETE", "cv_reproduction", "No model had an identified complete original token reference")

    def stability_reproduction(self):
        if self.np is None:
            self.finding("SKIP", "stability_reproduction", "Numerical dependencies unavailable")
            return
        words = {r["info"].get("identity"): r["words"] for r in self.model_words}
        groups = collections.defaultdict(list)
        for source, row in self.seed_records:
            if row.get("study", "main") not in MAIN_STUDIES:
                continue
            groups[(source, group_key(row))].append(row)
        output = []
        for (source, key), records in groups.items():
            if sorted(int_value(r.get("seed")) for r in records if int_value(r.get("seed")) is not None) != [0, 1, 2, 3, 4]:
                continue
            if not all(r.get("model_identity") in words for r in records):
                continue
            saved_path = source.with_name("seed_stability.csv")
            if not saved_path.is_file():
                continue
            sample = records[0]
            saved_rows, _ = csv_rows(saved_path)
            matches = [r for r in saved_rows if r.get("method") == sample.get("method") and str(r.get("K")) == str(sample.get("K"))
                       and all(not r.get(k) or not sample.get(k) or r[k] == sample[k] for k in ("study", "job_id", "collection"))]
            if len(matches) != 1:
                self.finding("INCOMPLETE", "stability_reproduction", "Saved stability row not uniquely identified", saved_path)
                continue
            value, pairs = topic_stability({int(r["seed"]): words[r["model_identity"]] for r in records})
            saved = number(matches[0].get("stability"))
            good = saved is not None and value is not None and abs(saved - value) <= 1e-10 and pairs == 10
            result = {"seed_file": self.rel(source), "corpus": row_corpus(sample), "method": sample.get("method"),
                      "K": sample.get("K"), "saved": saved, "recomputed": value, "seed_pairs": pairs,
                      "status": "PASS" if good else "FAIL"}
            output.append(result)
            self.finding(result["status"], "stability_reproduction", "Recomputed optimal topic-word Jaccard matching across all ten seed pairs", saved_path,
                         method=sample.get("method"), K=sample.get("K"), saved=saved, recalculated=value)
        write_csv(self.output / "stability_reproduction.csv", output)
        if not output:
            self.finding("INCOMPLETE", "stability_reproduction", "No group had all five audited model descriptors and a uniquely linked stability row")

    def embedding_and_training_records(self):
        if self.np is None:
            self.finding("SKIP", "embedding_arrays", "Numerical dependencies unavailable")
            return
        np = self.np
        results = []
        for path in self.paths:
            if path.name.startswith("glove_6b_") and path.suffix == ".npz":
                try:
                    with np.load(path, allow_pickle=False) as values:
                        words, vectors, found = values["words"], values["vectors"], values["found"]
                        good = vectors.ndim == 2 and len(words) == len(vectors) == len(found) and np.isfinite(vectors).all()
                        if not good:
                            self.finding("FAIL", "english_vector_lookup", "Lookup dimensions or finite values invalid", path)
                        else:
                            lookup = {str(w): i for i, w in enumerate(words)}
                            for c in self.collections:
                                if c["corpus"] != "transport_grievance":
                                    continue
                                ids = [lookup.get(str(w)) for w in c["vocabulary"]]
                                missing = sum(i is None for i in ids)
                                hits = sum(bool(found[i]) for i in ids if i is not None)
                                results.append({"path": self.rel(path), "collection": self.rel(c["directory"]), "dimension": vectors.shape[1],
                                                "vocabulary": len(ids), "lookup_missing": missing, "pretrained_hits": hits,
                                                "oov": len(ids) - missing - hits})
                                self.finding("FAIL" if missing else "PASS", "english_vector_lookup",
                                             "Measured actual prepared-vocabulary lookup coverage; OOV initialization reviewed in collected source", path,
                                             vocabulary=len(ids), missing=missing, pretrained_hits=hits)
                except Exception as error:
                    self.finding("INCOMPLETE", "english_vector_lookup", str(error), path)
            elif path.name == "vectors.npy" and (path.parent / "vocabulary.npy").is_file() and "chinese" in self.rel(path).lower():
                try:
                    vectors = self.array(path)
                    words = self.array(path.parent / "vocabulary.npy")
                    good = vectors.ndim == 2 and len(words) == len(vectors) and np.isfinite(vectors).all() and not (np.linalg.norm(vectors, axis=1) == 0).any()
                    alignment = any(np.array_equal(words, c["vocabulary"]) for c in self.collections if c["corpus"] == "chinese")
                    self.finding("PASS" if good and alignment else "FAIL" if not good else "INCOMPLETE", "chinese_vector_alignment",
                                 "Checked frozen projected vectors and exact prepared-vocabulary order", path, aligned=alignment, shape=list(vectors.shape))
                except Exception as error:
                    self.finding("INCOMPLETE", "chinese_vector_alignment", str(error), path)
            elif path.name == "loss_history.csv":
                try:
                    rows, columns = csv_rows(path)
                    losses = [number(row.get("loss")) for row in rows]
                    epochs = [int_value(row.get("epoch")) for row in rows]
                    good = bool(rows) and None not in losses and None not in epochs and len(set(epochs)) == len(epochs) and epochs == sorted(epochs)
                    self.finding("PASS" if good else "FAIL", "saved_training_history", "Checked finite losses and distinct ordered epochs; monotonic loss is not required", path,
                                 epochs=len(rows), first_loss=losses[0] if rows else None, last_loss=losses[-1] if rows else None)
                    self.queue(path, 15)
                except Exception as error:
                    self.finding("INCOMPLETE", "saved_training_history", str(error), path)
        write_json(self.output / "embedding_coverage.json", results)
        self.finding("INFO", "baseline_fidelity_scope", "Array, coverage and completion checks do not establish faithful reproduction or adequate tuning; collected native source/settings require review")

    def run_core_tests(self):
        if self.options.skip_core_tests:
            self.finding("SKIP", "core_tests", "Disabled by command-line option")
            return
        source = self.root / "src/transport_topics"
        if not source.is_dir():
            self.finding("INCOMPLETE", "core_tests", "src/transport_topics was not found; original source not imported")
            return
        work = self.output / "working_copy"
        destination = work / "src/transport_topics"
        def ignore(folder, names):
            return [name for name in names if name == "__pycache__" or name.endswith((".pyc", ".pyo"))
                    or (Path(folder) / name).is_symlink()]
        shutil.copytree(source, destination, ignore=ignore)
        test_dir = work / "tests"
        test_dir.mkdir(parents=True)
        found = []
        for name in CORE_TESTS:
            path = self.root / "tests/unit" / name
            if path.is_file():
                shutil.copy2(path, test_dir / name)
                found.append(name)
        if not found:
            self.finding("INCOMPLETE", "core_tests", "Selected numerical unit-test files were not found; no broad test discovery executed")
            return
        env = dict(os.environ)
        env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
                    "PYTHONPATH": str(work / "src"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                    "HF_HOME": str(work / "offline_cache"), "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--test-worker", str(work)]
        try:
            process = subprocess.run(command, cwd=work, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, timeout=self.options.test_timeout)
            text = process.stdout.decode("utf-8", errors="replace")
            (self.output / "core_tests_console.txt").write_text(text, encoding="utf-8")
            result_path = work / "core_test_result.json"
            result = read_json(result_path) if result_path.is_file() else {"worker_exit_code": process.returncode}
            write_json(self.output / "core_test_result.json", result)
            complete = len(found) == len(CORE_TESTS) and not result.get("import_errors") and not result.get("skipped")
            failed = bool(result.get("failures") or result.get("errors"))
            state = "FAIL" if failed else "PASS" if process.returncode == 0 and complete else "INCOMPLETE"
            self.finding(state, "core_tests", "Existing numerical tests executed on copied current source; synthetic fixtures only", **result)
            self.log("Core checks: %s tests; %s failures; %s errors; %s import errors." %
                     (result.get("tests_run", "unknown"), result.get("failures", "unknown"),
                      result.get("errors", "unknown"), len(result.get("import_errors", []))))
        except subprocess.TimeoutExpired as error:
            (self.output / "core_tests_console.txt").write_bytes(error.stdout or b"")
            self.finding("INCOMPLETE", "core_tests", "Copied-source tests exceeded timeout; existing experiments were not run")

    def finish(self, interrupted=False):
        self.collect_records()
        budget = self.options.bundle_budget_mb * MIB
        included, omitted = [], []
        for path, priority in sorted(self.bundle_candidates.items(), key=lambda item: (item[1], self.rel(item[0]))):
            size = path.stat().st_size
            if size > self.options.bundle_file_mb * MIB or size > budget:
                omitted.append({"path": self.rel(path), "bytes": size, "reason": "per-file or total package limit"})
                continue
            included.append((path, self.rel(path), size))
            budget -= size
        write_csv(self.output / "package_omissions.csv", omitted, ["path", "bytes", "reason"])
        write_csv(self.output / "package_file_index.csv", [
            {"path": relative, "bytes": size, "sha256": self.digest(path)} for path, relative, size in included])
        if omitted:
            self.finding("INCOMPLETE", "inspection_package_limit", "Some project files were omitted; see package_omissions.csv",
                         omitted_files=len(omitted), included_files=len(included))
        self.finding("INFO", "scientific_scope", "This audit cannot establish superiority, selection robustness, sufficient tuning, unseen-word generalization, journal acceptance or a numerical manuscript rating")
        counts = collections.Counter(r["status"] for r in self.findings)
        summary = {"audit_version": VERSION, "run_id": self.run_id,
                   "audit_execution": "interrupted" if interrupted else "completed",
                   "project_root": str(self.root), "elapsed_seconds": time.monotonic() - self.started,
                   "finding_counts": dict(counts), "experiments_retrained": False,
                   "project_files_replaced": False, "packages_installed_or_updated": False,
                   "writes_confined_to": self.rel(self.output),
                   "new_protocol_topic_counts": [8, 10, 12], "existing_topic_counts_relabelled": False,
                   "scientific_validation_complete": False,
                   "package_project_files": len(included), "package_omitted_files": len(omitted),
                   "all_findings": self.findings}
        write_json(self.output / "audit_report.json", summary)
        write_csv(self.output / "findings.csv", self.findings)
        attention = [r for r in self.findings if r["status"] in {"FAIL", "WARN", "INCOMPLETE"}]
        lines = ["# PEBR step-1 inspection", "", "Execution: " + summary["audit_execution"],
                 "", "This is an implementation and saved-artifact audit. It does not select a winning method or certify the paper.",
                 "", "## Finding counts", "", "| Status | Count |", "|---|---:|"]
        lines += ["| %s | %d |" % (state, counts[state]) for state in ("PASS", "FAIL", "WARN", "INCOMPLETE", "SKIP", "INFO")]
        lines += ["", "PASS applies only to the named check. INCOMPLETE/SKIP means the property has not been established.",
                  "", "## Findings requiring inspection", ""]
        for row in attention[:120]:
            lines.append("- **%s — %s:** %s%s" % (row["status"], row["check"], row["detail"],
                         " (`%s`)" % row["path"] if row.get("path") else ""))
        if len(attention) > 120:
            lines.append("- Further findings are retained in findings.csv and audit_report.json.")
        if not attention:
            lines.append("No automated attention findings. The scientific and baseline fidelity review remains necessary.")
        lines += ["", "## What to return", "", "Return the inspection ZIP printed at the end of the console run. No full project upload is needed.",
                  "", "## Next decision", "", "Review failures first, then missing checks and baseline fidelity. Only then define any corrected implementation.",
                  "Keep new experiments in a separate revision output/cache namespace; preserve old results with their original identities.",
                  "Repeated selection, protected-word replication and simpler-policy comparisons are later experimental steps, not claims established here.",
                  "", "The package contains source/configuration snapshots and small numerical records. It excludes raw post files, pretrained model weights and trained checkpoints.",
                  "Raw-data statistics are computed locally. Original excerpts are not reconstructed from donor words.", ""]
        (self.output / "READ_ME_FIRST.md").write_text("\n".join(lines), encoding="utf-8")
        self.log("AUDIT FINISHED | PASS=%d FAIL=%d WARN=%d INCOMPLETE=%d SKIP=%d" %
                 tuple(counts[state] for state in ("PASS", "FAIL", "WARN", "INCOMPLETE", "SKIP")))
        self.log("Existing experiments were not retrained or replaced. Missing checks are recorded explicitly.")
        zip_path = self.output / (self.run_id + "_step1_inspection.zip")
        # Log final paths before packaging so the included console transcript contains them.
        self.log("REPORT: " + str(self.output / "READ_ME_FIRST.md"))
        self.log("SEND THIS INSPECTION ZIP: " + str(zip_path))
        with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
            for path in sorted(self.output.iterdir()):
                if path.is_file() and path != zip_path:
                    archive.write(path, "audit/" + path.name)
            for path, relative, _ in included:
                archive.write(path, "project_snapshot/" + relative)
            archive.write(Path(__file__).resolve(), "audit/audit_tool.py")
        self.log("Inspection ZIP saved: %.2f MiB" % (zip_path.stat().st_size / MIB))
        return zip_path, counts


def test_worker(work):
    work = Path(work).resolve()
    sys.path.insert(0, str(work / "src"))
    suite = unittest.TestSuite()
    imports = []
    for path in sorted((work / "tests").glob("test_*.py")):
        try:
            spec = importlib.util.spec_from_file_location("audit_" + path.stem, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
        except Exception as error:
            imports.append({"module": path.name, "exception": type(error).__name__ + ": " + str(error)})
            traceback.print_exc()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    payload = {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
               "skipped": len(result.skipped), "import_errors": imports,
               "scope": "synthetic numerical tests on source copied from current project"}
    write_json(work / "core_test_result.json", payload)
    return 0 if result.wasSuccessful() and result.testsRun and not imports else 1


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="Read-only PEBR implementation and saved-results audit; no experiment fitting/downloads.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent,
                        help="Directory containing main.py and src (default: this script's directory)")
    parser.add_argument("--max-models", type=int, default=300)
    parser.add_argument("--max-collections", type=int, default=8)
    parser.add_argument("--max-action-banks", type=int, default=12)
    parser.add_argument("--score-rows", type=int, default=2, help="Saved selected receivers per bank for direct prediction recalculation")
    parser.add_argument("--cv-checks", type=int, default=8, help="Representative saved C_v recalculations; 0 disables")
    parser.add_argument("--hash-budget-mb", type=int, default=512)
    parser.add_argument("--bundle-budget-mb", type=int, default=64)
    parser.add_argument("--bundle-file-mb", type=int, default=8)
    parser.add_argument("--test-timeout", type=int, default=180)
    parser.add_argument("--skip-core-tests", action="store_true")
    parser.add_argument("--test-worker", type=Path, help=argparse.SUPPRESS)
    options = parser.parse_args(argv)
    for key in ("max_models", "max_collections", "max_action_banks", "score_rows", "cv_checks",
                "hash_budget_mb", "bundle_budget_mb", "bundle_file_mb", "test_timeout"):
        if getattr(options, key) < 0:
            parser.error("--" + key.replace("_", "-") + " cannot be negative")
    return options


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    options = parse_arguments(argv)
    if options.test_worker:
        return test_worker(options.test_worker)
    root = options.project_root.resolve()
    if not root.is_dir() or not (root / "main.py").is_file():
        print("Put pebr_step1_audit.py beside your project's main.py, or supply --project-root with that directory.", file=sys.stderr)
        return 1
    # All writes below use the newly created audit directory, never a project pipeline entry point.
    audit = Audit(root, options)
    interrupted = False
    if (root / "outputs/manifests/execution.lock").exists():
        audit.finding("WARN", "concurrent_project_run", "An execution lock exists. Files may be changing; do not delete the lock or rerun training to fix this audit.")
    try:
        for title, function in (
                ("1/12 Inventory", audit.discover),
                ("2/12 Source, configurations and environment", audit.source_snapshot),
                ("3/12 Raw collection statistics", audit.raw_data),
                ("4/12 Saved seed tables and arithmetic", audit.result_tables),
                ("5/12 Existing numerical dependencies", audit.numerical_dependencies),
                ("6/12 Prepared counts, families, masks and donors", audit.prepared_collections),
                ("7/12 Fold-excluded predictive references", audit.predictor_artifacts),
                ("8/12 Borrowing bounds, matched controls and saved word audits", lambda: (audit.borrowing_artifacts(), audit.untouched_word_records())),
                ("9/12 Direct checks of saved selected actions", audit.real_action_checks),
                ("10/12 Model outputs and topic metrics", lambda: (audit.model_artifacts(), audit.stability_reproduction())),
                ("11/12 Representative C_v and baseline records", lambda: (audit.cv_reproduction(), audit.embedding_and_training_records())),
                ("12/12 Core numerical tests on copied source", audit.run_core_tests)):
            audit.phase(title, function)
    except KeyboardInterrupt:
        interrupted = True
        audit.finding("INCOMPLETE", "interrupted", "Audit interrupted by user; partial report retained")
        audit.log("Creating the partial inspection package.")
    try:
        _, counts = audit.finish(interrupted=interrupted)
    except Exception as error:
        audit.log("AUDIT PACKAGING ERROR: " + type(error).__name__ + ": " + str(error))
        traceback.print_exc()
        return 1
    return 1 if interrupted else 2 if counts["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

