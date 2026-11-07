#!/usr/bin/env python3
"""Finish the two-corpus audit using the exact completed-run records.

No training, encoder invocation, installation or replacement of existing files.
The supplied engine supports both nested English metrics and Chinese quality.
"""
from __future__ import annotations
import argparse
import collections
import json
from pathlib import Path
import sys

from pebr_audit_closeout import engine

PRIMARY_RUN = "20261028T170015451848Z_project_run_7add5b34"
CHINESE_RUN = "20261029T084655541657Z_chinese_topics_d8105c07"
METHODS = {"proposed", "nmf_original", "nmf_similarity", "btm", "knntm", "glocom",
           "similarity_matched", "full_pool_loo", "no_downside"}
CONTROL_POLICIES = {"no_borrowing", "similarity_maximum", "similarity_matched", "full_pool_loo", "no_downside", "proposed"}


def source_rows(root):
    sources = [root / "outputs/runs" / PRIMARY_RUN / "results/topic_seed_results.csv",
               root / "outputs/chinese_topics" / CHINESE_RUN / "topic_seed_results.csv"]
    output = []
    for path in sources:
        rows, _ = engine.csv_rows(path)
        for row in rows:
            if row.get("study", "main") in engine.MAIN_STUDIES and engine.row_corpus(row) in {"transport_grievance", "chinese"}:
                output.append((path, row))
    return sources, output


def completed_model_path(root, row):
    condition = row.get("condition") or row.get("collection")
    method = row.get("source_method") or row.get("method")
    identity = row.get("model_identity", "")
    if not condition or not method or len(identity) != 64:
        raise ValueError("Seed record lacks a complete model identity")
    suffix = Path(condition) / method / ("K" + str(row["K"])) / ("seed" + str(row["seed"])) / identity[:16]
    candidates = [root / "outputs/models" / namespace / suffix for namespace in ("topic_models", "chinese_topics")]
    matches = []
    for directory in candidates:
        path = directory / "model.json"
        if path.is_file() and engine.read_json(path).get("identity") == identity:
            matches.append(directory)
    if len(matches) != 1:
        raise ValueError("Expected exactly one saved model for %s K=%s seed=%s; found %d" %
                         (method, row["K"], row["seed"], len(matches)))
    return matches[0]


class Closeout(engine.Audit):
    def result_tables(self):
        super().result_tables()
        # Later checks use only the completed main-comparison rows, not other
        # studies retained in the same historical CSV.
        self.seed_records = list(self.canonical_rows)

    def discover(self):
        roots = set()
        explicit = {self.root / "main.py", Path(__file__).resolve(), Path(engine.__file__).resolve()}
        sources, rows = source_rows(self.root)
        self.canonical_rows = rows
        self.canonical_seed_files = sources
        actual = collections.Counter((engine.row_corpus(r), r.get("method"), int(r["K"]), int(r["seed"])) for _, r in rows)
        expected = {(c, method, k, seed) for c in ("transport_grievance", "chinese")
                    for method in METHODS for k in (8, 12, 16) for seed in range(5)}
        missing = sorted(expected - set(actual))
        duplicate = [list(key) for key, count in actual.items() if count != 1]
        unexpected = sorted(set(actual) - expected)
        self.finding("FAIL" if missing or duplicate or unexpected else "PASS", "canonical_run_coverage",
                     "Exact completed-run records; interrupted runs, copies and sensitivity settings are not pooled",
                     records=len(rows), expected_records=270, missing=missing, duplicate=duplicate, unexpected=unexpected)
        status_paths = [self.root / "outputs/runs" / run / "status.json" for run in (PRIMARY_RUN, CHINESE_RUN)]
        for path in status_paths:
            state = engine.read_json(path).get("state") if path.is_file() else None
            self.finding("PASS" if state == "completed" else "FAIL", "canonical_run_status",
                         "Recorded completion state of source run", path, state=state)
            explicit.add(path)
        for name in ("src", "configs", "requirements", "tests/unit", "third_party/published"):
            p = self.root / name
            if p.is_dir(): roots.add(p)
        roots.update({self.root / "data/raw", self.root / "assets/chinese_study"})
        explicit.update(self.root.glob("*.py"))
        explicit.update(self.root.glob("requirements*.txt"))
        explicit.update(self.root.glob("pyproject.toml"))
        self.model_directories = set()
        collection_ids = {}
        selected_control_ids = set()
        for path, row in rows:
            directory = completed_model_path(self.root, row)
            self.model_directories.add(directory)
            roots.add(directory)
            info = engine.read_json(directory / "model.json")
            collection_ids[info["condition"]] = info["condition_identity"]
            identity_path = directory / "identity.json"
            if identity_path.is_file():
                identity = engine.read_json(identity_path)
                if identity.get("control_identity"):
                    selected_control_ids.add(identity["control_identity"])
            if row.get("evaluation_path"):
                roots.add(self.root / row["evaluation_path"])
            checkpoint = info.get("audit", {}).get("checkpoint")
            if checkpoint:
                directory = (self.root / checkpoint).parent
                for name in ("loss_history.csv", "latest.json"):
                    if (directory / name).is_file(): explicit.add(directory / name)
        self.focus_collection_directories = set()
        for condition, identity in collection_ids.items():
            part = "chinese_transport" if condition.startswith("chinese") else "indian_transport"
            directory = self.root / "data/processed" / part / (condition + "_" + identity[:16])
            if not directory.is_dir():
                raise FileNotFoundError("Exact prepared collection not found: " + str(directory))
            self.focus_collection_directories.add(directory)
            roots.add(directory)
            for name in ("condition.json", "identity.json", "complete.json"):
                if (directory / name).is_file(): self.queue(directory / name, 10)
        for path in sources:
            roots.add(path.parent)
        primary_index = self.root / "outputs/runs" / PRIMARY_RUN / "sharing_controls_index.json"
        chinese_index = self.root / "outputs/chinese_topics" / CHINESE_RUN / "borrowing_controls_index.json"
        control_records = []
        for path in (primary_index, chinese_index):
            values = engine.read_json(path)
            values = values.get("banks", []) if isinstance(values, dict) else values
            explicit.add(path)
            for record in values:
                condition = record.get("condition") or record.get("collection")
                scope = record.get("scope", "topic_learning")
                include = record.get("identity") in selected_control_ids or (
                    condition == "primary_full" and scope == "outer_audit" and set(record.get("variants", [])) == CONTROL_POLICIES)
                if condition in collection_ids and include:
                    control_records.append(record)
        self.focus_controls = []
        self.focus_action_directories = set()
        for record in control_records:
            relative = record.get("path") or record.get("directory")
            directory = self.root / relative
            roots.add(directory)
            self.focus_controls.append(directory)
            identity = engine.read_json(directory / "identity.json")
            if set(identity.get("variants", [])) != CONTROL_POLICIES:
                self.finding("FAIL", "canonical_control_variants", "Declared main control bank differs from expected policies", directory)
            anchor = record.get("anchor_path")
            if not anchor:
                raise ValueError("Main control record lacks its PEBR anchor path")
            action_directory = self.root / anchor
            roots.add(action_directory)
            self.focus_action_directories.add(action_directory)
            action_id = engine.read_json(action_directory / "identity.json")
            condition = record.get("condition") or record.get("collection")
            for key in action_id.get("predictor_keys", []):
                roots.add(self.root / "cache/predictors" / (condition + "_" + key[:16]))
        self.finding("PASS" if len(self.focus_controls) == 9 and len(self.focus_action_directories) == 9 else "INCOMPLETE",
                     "canonical_bank_coverage", "Expected six fitting banks plus three existing primary outer-audit banks",
                     control_banks=len(self.focus_controls), action_banks=len(self.focus_action_directories))
        chinese_run = self.root / "outputs/runs" / CHINESE_RUN
        for name in ("chinese_collections.json", "chinese_encoder_audit.json", "chinese_raw_audit.json", "resolved_protocol.json", "config.json"):
            if (chinese_run / name).is_file(): explicit.add(chinese_run / name)
        roots.update({self.root / "assets/pretrained/word_embeddings", self.root / "assets/pretrained/chinese_word_embeddings"})
        self.paths = []
        for directory in sorted(roots):
            if not engine.inside(directory, self.root) or not directory.is_dir():
                continue
            for path in directory.rglob("*"):
                if path.is_file() and not path.is_symlink() and not any(part in engine.SKIP_DIRS for part in path.relative_to(self.root).parts):
                    explicit.add(path)
        for path in sorted(explicit):
            if path.is_file() and engine.inside(path, self.root):
                self.paths.append(path)
        # Select only the two canonical seed files. Reporting copies are not independent runs.
        self.paths = [p for p in self.paths if p.name != "topic_seed_results.csv" or p in sources]
        self.paths = [p for p in self.paths if p.name != "model.json" or p.parent in self.model_directories]
        self.paths = [p for p in self.paths if p.name != "counts.npz" or p.parent in self.focus_collection_directories]
        for path in self.paths:
            stat = path.stat()
            self.inventory.append({"path": self.rel(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        engine.write_csv(self.output / "file_inventory.csv", self.inventory)
        engine.write_json(self.output / "canonical_sources.json", {
            "primary_run": PRIMARY_RUN, "chinese_run": CHINESE_RUN, "seed_files": [self.rel(p) for p in sources],
            "model_directories": sorted(self.rel(p) for p in self.model_directories),
            "collection_directories": sorted(self.rel(p) for p in self.focus_collection_directories),
            "control_directories": sorted(self.rel(p) for p in self.focus_controls),
            "action_directories": sorted(self.rel(p) for p in self.focus_action_directories),
            "historical_topic_counts": [8, 12, 16], "new_experiments_started": False})
        self.log("Focused inventory: %d files, %d main models, %d action banks." %
                 (len(self.paths), len(self.model_directories), len(self.focus_action_directories)))
        self.finding("INFO", "persistent_lock_marker", "This project's lock file is deliberately retained after release; file existence alone is not evidence of a concurrent run")

    def chinese_query_cache(self):
        try:
            import jieba
            import numpy as np
        except Exception as error:
            self.finding("INCOMPLETE", "chinese_masked_query_cache", "Existing tokenizer could not import: " + str(error))
            return
        c = next((c for c in self.collections if c["corpus"] == "chinese"), None)
        if c is None:
            self.finding("INCOMPLETE", "chinese_masked_query_cache", "Chinese prepared collection was not audited")
            return
        directory = c["directory"] / "topic_learning"
        tokenizer = jieba.Tokenizer()
        tokenizer_dir = self.output / "tokenizer_cache"
        tokenizer_dir.mkdir(exist_ok=True)
        tokenizer.tmp_dir = str(tokenizer_dir)
        tokenizer.cache_file = "closeout_jieba.cache"
        tokenizer.initialize()
        posts = self.get_json(c["directory"] / "posts.json")
        targets = self.matrix(directory / "targets.npz")
        candidates = self.array(directory / "candidates.npy")
        eligible = self.array(directory / "eligible.npy")
        views = candidates.shape[1]
        queries = []
        for i, post in enumerate(posts):
            if not eligible[i]: continue
            text = post.get("original_text")
            if not isinstance(text, str):
                raise ValueError("Original normalized Chinese post text was not saved")
            segments = list(tokenizer.cut(text, HMM=True))
            if "".join(segments) != text:
                raise ValueError("Tokenizer changed character order")
            for m in range(views):
                blocked = set(map(str, c["vocabulary"][targets[i * views + m].indices]))
                queries.append("".join(" " if segment in blocked else segment for segment in segments))
        encoded = self.get_json(directory / "encoder_identity.json")["encoder"]
        identity = engine.canonical_hash({"encoder": encoded, "ordered_texts": queries})
        cache_dir = self.root / "cache/embeddings/chinese" / identity[:24]
        path = cache_dir / "identity.json"
        if not path.is_file():
            self.finding("INCOMPLETE", "chinese_masked_query_cache", "Rebuilt queries did not identify an existing cache; inspect tokenizer/version, do not re-encode automatically", directory,
                         expected_cache=self.rel(cache_dir), rebuilt_queries=len(queries))
            return
        recorded = self.get_json(path)
        vectors = self.array(cache_dir / "vectors.npy")
        ok = recorded.get("identity") == identity and recorded.get("encoder") == encoded and recorded.get("texts") == len(queries) and vectors.shape == (len(queries), 384)
        self.finding("PASS" if ok else "FAIL", "chinese_masked_query_cache",
                     "Rebuilt every masked Chinese query and matched the ordered-text/encoder cache identity; no encoder calls", path,
                     queries=len(queries), cache_identity=identity)
        self.queue(path, 10)
        self.queue(directory / "encoder_identity.json", 10)
        self.receipt(cache_dir)
        if ok:
            # Close the older per-query-hash finding using an exact whole-query-bank identity check.
            for finding in self.findings:
                if finding["check"] == "saved_masked_query_hashes" and finding.get("path") == self.rel(directory):
                    finding["status"] = "INFO"
                    finding["detail"] = "Per-query hashes were not saved; ordered masked-query cache identity was independently checked during closeout"

    def review_summary(self):
        engine.write_json(self.output / "next_protocol.json", {
            "status": "design_for_next_stage_not_an_experiment_result",
            "runner_implementation_pending": True,
            "corpora": ["transport_grievance", "chinese_pooled"],
            "core_experiment_scope_equal_for_both": True,
            "topic_counts": [8, 10, 12], "final_model_seeds": [0, 1, 2, 3, 4],
            "selection_mask_bases": [2711, 3711, 4711, 5711, 6711],
            "outer_reservation_bases": [2712, 3712, 4712, 5712, 6712],
            "default_fold_seed": 2710, "default_reference_seed": 0,
            "targeted_fold_sensitivity_K12": [2710, 3710, 4710],
            "targeted_reference_initializations_K12": [0, 1, 2],
            "views": 20, "candidates_per_view": 20, "max_donors": 3,
            "cap": 1/3, "downside_weight": 1, "positive_budget_cap_fractions": [.25, .5, .75, 1.],
            "core_policies": ["nmf_original", "nmf_similarity", "similarity_matched", "full_pool_loo", "proposed",
                              "fixed_cap_pebr", "predictively_scored_similarity_prefixes", "no_downside",
                              "individual_donor", "single_view"],
            "prefix_order": "mean reciprocal rank over all visible-view lists, absent=0, source-order ties; prefixes 1..3",
            "fixed_cap_definition": "same complete-group search and no-borrowing option, only positive offered budget=cap",
            "metrics": ["npmi", "cv", "diversity", "seed_pair_stability"],
            "single_display_run": {"selection_mask_base": 2711, "final_model_seed": 0},
            "aggregation": "paired seed differences within each selection bank, then across five bank means; final-fit and selection SD kept distinct",
            "preprocessing_changes_from_closeout": False,
            "old_K16_records_preserved": True,
            "guaranteed_performance_improvement": False,
            "disclosure": "Existing corpora and K=8/12 outcomes have already been examined; this is not a blind confirmatory study",
            "interpretation": "Freeze the executable runner and any corrections before new performance outcomes; all declared settings are retained"})
        self.queue(Path(__file__).resolve(), 0)
        self.queue(Path(engine.__file__).resolve(), 0)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"): stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    if not (args.project_root / "main.py").is_file():
        parser.error("Put the new file and pebr_audit_closeout folder beside main.py")
    options = engine.parse_arguments(["--project-root", str(args.project_root), "--max-collections", "2",
                                     "--max-models", "270", "--max-action-banks", "9", "--cv-checks", "8"])
    audit = Closeout(args.project_root, options)
    phases = (
            ("1/11 Resolve exact completed runs and artifacts", audit.discover),
            ("2/11 Source and environment", audit.source_snapshot),
            ("3/11 Canonical seed summaries", audit.result_tables),
            ("4/11 Existing numerical dependencies", audit.numerical_dependencies),
            ("5/11 Both complete prepared corpora", audit.prepared_collections),
            ("6/11 Predictive references and borrowing", lambda: (audit.predictor_artifacts(), audit.borrowing_artifacts(), audit.untouched_word_records())),
            ("7/11 Exact main PEBR action banks", audit.real_action_checks),
            ("8/11 All 270 main models, topic metrics and stability", lambda: (audit.model_artifacts(), audit.stability_reproduction())),
            ("9/11 Balanced C_v checks and baseline records", lambda: (audit.cv_reproduction(), audit.embedding_and_training_records())),
            ("10/11 Rebuild Chinese masked query cache identity", audit.chinese_query_cache),
            ("11/11 Core tests and next-stage design", lambda: (audit.run_core_tests(), audit.review_summary())))
    interrupted = False
    try:
        for title, function in phases: audit.phase(title, function)
    except KeyboardInterrupt:
        interrupted = True
        audit.finding("INCOMPLETE", "interrupted", "Closeout interrupted; partial evidence retained")
    _, counts = audit.finish(interrupted=interrupted)
    return 1 if interrupted else 2 if counts["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
