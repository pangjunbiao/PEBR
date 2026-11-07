"""Run the pinned published topic-model benchmarks on transport grievances."""
from __future__ import annotations

import argparse
import csv
from importlib import metadata
import logging
import os
from pathlib import Path
import platform
import random
import statistics
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np

from inputs import (digest, export_input_bundle, file_digest, load_benchmark_inputs,
                    read, save_json, write_csv)
from stability import seed_stability
from topic_quality import evaluate_topics
from upstream import load_upstream

HERE = Path(__file__).resolve().parent
K = 12
TOP_WORDS = 10
METRICS = ("npmi", "cv", "diversity")
PEBR = "preserve_forward_fixed_cap"
LABEL = {"btm": "BTM", "nmf_original": "NMF", "nmf_similarity": "Similarity sharing",
         "glocom": "GloCOM", "knntm": "kNNTM", PEBR: "PEBR",
         "encot": "EnCOT", "fastopic": "FASTopic", "neuromax": "NeuroMax"}
SETTINGS = {
    "encot": {"epochs": 200, "batch_size": 200, "lr": 0.002,
              "external_cluster_count": 30, "external_cluster_seed": 0,
              "kwargs": {"num_clusters": 30, "en_units": 200, "dropout": 0.0,
                         "beta_temp": 0.2, "weight_loss_ECR": 30.0, "sinkhorn_alpha": 20.0,
                         "sinkhorn_max_iter": 100, "alpha_noise": 0.001,
                         "alpha_augment": 0.01, "weight_ot_doc_cluster": 1.0,
                         "weight_ot_topic_cluster": 1.0}},
    "fastopic": {"epochs": 200, "batch_size": 619, "lr": 0.002,
                 "kwargs": {"DT_alpha": 3.0, "TW_alpha": 2.0, "theta_temp": 1.0}},
    "neuromax": {"epochs": 500, "batch_size": 200, "lr": 0.002, "lr_step": 125,
                  "kwargs": {"num_groups": 5, "en_units": 200, "dropout": 0.2,
                             "beta_temp": 0.2, "weight_loss_ECR": 40.0, "weight_loss_GR": 1.0,
                             "alpha_ECR": 20.0, "alpha_GR": 5.0,
                             "weight_loss_InfoNCE": 50.0, "sinkhorn_max_iter": 1000}}
}


def configure_logger(directory):
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("published_baseline_benchmark")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(directory / "console.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def prepare_model_inputs(data, out):
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits

    params = SETTINGS["encot"]
    with threadpool_limits(limits=1):
        groups = KMeans(n_clusters=params["external_cluster_count"], n_init=10,
                        random_state=params["external_cluster_seed"],
                        algorithm="lloyd").fit_predict(data.embeddings)
    if len(np.unique(groups)) != params["external_cluster_count"]:
        raise ValueError("EnCOT input did not produce the fixed number of global groups")
    global_counts = np.zeros((params["external_cluster_count"], data.counts.shape[1]), dtype=np.float32)
    np.add.at(global_counts, groups, data.counts)
    np.savez_compressed(out / "encot_global_context.npz", group_ids=groups,
                        global_counts=global_counts, source_rows=data.source)
    return np.concatenate((data.counts, global_counts[groups]), axis=1)


def fit(model_name, factory, data, encot_rows, seed, epochs, device, log):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    setting = SETTINGS[model_name]
    kwargs = setting["kwargs"]
    bow = torch.as_tensor(data.counts, device=device, dtype=torch.float32)
    doc_embeddings = torch.as_tensor(data.embeddings, device=device, dtype=torch.float32)
    if model_name == "fastopic":
        model = factory(num_topics=K, **kwargs)
        model.init(bow.shape[1], doc_embeddings.shape[1])
    else:
        model = factory(vocab_size=bow.shape[1], num_topics=K,
                        pretrained_WE=data.word_vectors.copy(), **kwargs)
    model = model.to(device)
    encot_input = (torch.as_tensor(encot_rows, device=device, dtype=torch.float32)
                   if model_name == "encot" else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=setting["lr"])
    scheduler = (torch.optim.lr_scheduler.StepLR(optimizer, step_size=setting["lr_step"], gamma=0.5)
                 if "lr_step" in setting else None)
    batch_rng = torch.Generator().manual_seed(seed)
    start = time.perf_counter()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        indices = (torch.arange(len(bow)) if model_name == "fastopic" else
                   torch.randperm(len(bow), generator=batch_rng))
        batches = list(indices.split(setting["batch_size"]))
        if len(batches[-1]) == 1:
            singleton = batches.pop()
            batches[-1] = torch.cat((batches[-1], singleton))
        totals = {}
        for index in batches:
            index = index.to(device)
            if model_name == "fastopic":
                terms = model(bow[index], doc_embeddings[index])
            elif model_name == "encot":
                terms = model(encot_input[index])
            else:
                terms = model({"data": bow[index], "contextual_embed": doc_embeddings[index]},
                              epoch_id=epoch)
            loss = terms["loss"]
            if not torch.isfinite(loss).item():
                raise ArithmeticError(f"{model_name} seed {seed}: nonfinite loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            for key, term in terms.items():
                value = float(term.detach()) if torch.is_tensor(term) else float(term)
                totals[key] = totals.get(key, 0.0) + value * len(index)
        history.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                        **{k: v / len(bow) for k, v in totals.items()}})
        if scheduler is not None:
            scheduler.step()
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            log.info("%s seed %d | epoch %d/%d | loss %.5f", LABEL[model_name],
                     seed, epoch, epochs, history[-1]["loss"])
    model.eval()
    with torch.no_grad():
        scores = model.get_beta().cpu().numpy()
        if model_name == "fastopic":
            memberships = model.get_theta(doc_embeddings, doc_embeddings)
        else:
            memberships = model.get_theta(encot_input if model_name == "encot" else bow)
        memberships = memberships.cpu().numpy()
    if (scores.shape != (K, len(data.words)) or not np.isfinite(scores).all()
            or memberships.shape != (len(bow), K) or not np.isfinite(memberships).all()
            or (memberships < 0).any() or (memberships.sum(axis=1) <= 0).any()):
        raise ArithmeticError(f"{model_name} seed {seed}: invalid topic or membership outputs")
    if model_name == "neuromax":
        if epochs >= 11 and (model.group_connection_regularizer is None
                             or history[10]["loss_GR"] == 0.0):
            raise RuntimeError("NeuroMax group regularizer did not activate after epoch 10")
    return scores, memberships, history, time.perf_counter() - start


def run_fit(model, factory, data, encot_rows, seed, epochs, device, out, log):
    destination = out / "fits" / model / f"seed{seed}"
    receipt = destination / "metrics.json"
    if receipt.exists():
        row = read(receipt)
        if (row["policy"], row["seed"], row["epochs"]) != (model, seed, epochs):
            raise ValueError(f"Saved fit identity mismatch: {destination}")
        for name, expected in row["files"].items():
            if file_digest(destination / name) != expected:
                raise ValueError(f"Saved fit file changed: {destination / name}")
        log.info("Reusing %s seed %d", LABEL[model], seed)
    else:
        log.info("Fitting %s seed %d", LABEL[model], seed)
        scores, memberships, history, seconds = fit(
            model, factory, data, encot_rows, seed, epochs, device, log)
        quality = evaluate_topics(scores, data.words, data.reference, TOP_WORDS, True)
        row = {"policy": model, "seed": seed, "epochs": epochs, "fit_seconds": seconds, **quality}
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=destination.parent) as stage:
            work = Path(stage) / "fit"
            work.mkdir()
            np.savez_compressed(work / "topics.npz", scores=scores, native_memberships=memberships,
                                vocabulary=data.words, source_rows=data.source)
            write_csv(work / "losses.csv", history)
            (work / "top_words.txt").write_text(
                "\n".join(" ".join(topic) for topic in quality["top_words"]) + "\n", encoding="utf-8")
            row["files"] = {p.name: file_digest(p) for p in work.iterdir()}
            save_json(work / "metrics.json", row)
            work.rename(destination)
    log.info("%s seed %d | NPMI %.5f | C_v %.5f | TD %.5f", LABEL[model], seed,
             row["npmi"], row["cv"], row["diversity"])
    return row


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(fits, mode, out, log):
    old = read_csv(HERE / "existing_summary.csv")
    old_seeds = read_csv(HERE / "existing_seed_metrics.csv")
    fit_rows = []
    for row in fits:
        fit_rows.append({"policy": row["policy"], "method": LABEL[row["policy"]],
                         "seed": row["seed"], "epochs": row["epochs"],
                         **{m: row[m] for m in METRICS}, "fit_seconds": row["fit_seconds"]})
    write_csv(out / "new_seed_metrics.csv", fit_rows)
    report = [{**row, "method": LABEL[row["policy"]], "source": "existing_manuscript"}
              for row in old] if mode == "full" else []
    for model in SETTINGS:
        rows = [r for r in fits if r["policy"] == model]
        stability = seed_stability({r["seed"]: r["top_words"] for r in rows})
        measures = {}
        for metric in METRICS:
            values = [r[metric] for r in rows]
            measures[metric + "_mean"] = statistics.mean(values)
            measures[metric + "_sd"] = statistics.stdev(values) if len(values) > 1 else ""
        report.append({"corpus": "transport_grievance", "K": K, "policy": model,
                       "final_seeds": len(rows), **measures,
                       "stability": stability["stability"], "method": LABEL[model], "source": mode})
    write_csv(out / "summary.csv", report)
    if mode == "full":
        anchor = next(row for row in report if row["policy"] == PEBR)
        comparisons = []
        columns = {metric: metric + "_mean" for metric in METRICS} | {"stability": "stability"}
        for row in report:
            if row["policy"] == PEBR:
                continue
            comparisons.append({"method": row["method"], **{
                "pebr_minus_" + metric: float(anchor[column]) - float(row[column])
                for metric, column in columns.items()}})
        write_csv(out / "pebr_vs_baselines.csv", comparisons)
        # Preserve the original fit identities and provenance alongside the new runs.
        write_csv(out / "existing_seed_metrics.csv", old_seeds)
    log.info("%-22s %12s %12s %12s %12s", "Method", "NPMI", "C_v", "TD", "Stability")
    for row in report:
        cells = [f"{float(row[m + '_mean']):.4f}" for m in METRICS]
        stability = "n/a" if row["stability"] is None else f"{float(row['stability']):.4f}"
        log.info("%-22s %12s %12s %12s %12s", row["method"], *cells, stability)
    if mode == "full":
        for key in (*[m + "_mean" for m in METRICS], "stability"):
            best = max(float(r[key]) for r in report)
            winners = [r["method"] for r in report if abs(float(r[key]) - best) <= 1e-12]
            log.info("Highest %s: %s (%.5f)", key, ", ".join(winners), best)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=HERE.parent)
    parser.add_argument("--data-root", type=Path, help="Verified prepared corpus or data/processed directory")
    parser.add_argument("--glove-file", type=Path, help="Verified project GloVe lookup if outside its default path")
    parser.add_argument("--source-root", type=Path, help="Parent of pinned local encot/fastopic/neuromax repositories")
    parser.add_argument("--mode", choices=("preflight", "smoke", "full", "export-inputs"),
                        default="smoke")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    root = args.project_root.resolve()
    pinned = read(HERE / "protocol_inputs.json")
    if args.mode == "export-inputs":
        data = load_benchmark_inputs(root, args.data_root, args.glove_file, pinned)
        destination = export_input_bundle(root, data, pinned)
        print(f"Verified reproduction inputs: {destination}", flush=True)
        return

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    packages = {name: metadata.version(name) for name in (
        "numpy", "scipy", "scikit-learn", "gensim", "torch", "geomloss",
        "torch-kmeans", "joblib", "smart-open", "threadpoolctl")}
    if packages["gensim"] != "4.4.0":
        raise ValueError("Use Gensim 4.4.0 to match the existing C_v evaluator")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("This PyTorch interpreter has no CUDA device. Use --device cpu.")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seeds = list(range(5)) if args.mode == "full" else [0]
    budgets = {model: cfg["epochs"] if args.mode == "full" else
               (12 if model == "neuromax" else 2) for model, cfg in SETTINGS.items()}
    spec = {"mode": args.mode, "corpus": "transport_grievance", "topics": K, "top_words": TOP_WORDS,
            "seeds": seeds, "epochs": budgets, "settings": SETTINGS, "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.machine(),
            "platform": platform.platform(), "cpu_threads": torch.get_num_threads(),
            "cuda_runtime": torch.version.cuda, "packages": packages, "python": sys.version,
            "input_fingerprints": pinned, "source_patches": {
                "neuromax": "device-safe group tensor and remove unused sentence_transformers import"},
            "code_and_reference_files": {p.name: file_digest(p) for p in sorted(HERE.iterdir())
                                         if p.suffix in {".py", ".csv", ".json"}}}
    base = root / "outputs" / "pebr_extensions" / "published_baseline_benchmark"
    out = base / f"{args.mode}_{digest(spec)[:12]}"
    log = configure_logger(out)
    log.info("Output folder: %s", out)
    log.info("Mode %s | K=%d | seeds=%s | device=%s", args.mode, K, seeds, device)
    data = load_benchmark_inputs(root, args.data_root, args.glove_file, pinned)
    save_json(out / "run_spec.json", spec | {"input_audit": data.audit})
    log.info("Verified 619 original posts, 1296 words, and aligned MiniLM/GloVe inputs")
    factories = {}
    for model in SETTINGS:
        log.info("Loading %s at upstream commit %s", LABEL[model], pinned["upstream"][model]["commit"][:12])
        factories[model] = load_upstream(base / "upstream_cache", model, pinned["upstream"][model], args.source_root)
    encot_rows = prepare_model_inputs(data, out)
    if args.mode == "preflight":
        log.info("Preflight passed")
        return
    fits = [run_fit(model, factories[model], data, encot_rows, seed,
                    budgets[model], device, out, log)
            for model in SETTINGS for seed in seeds]
    summarize(fits, args.mode, out, log)
    save_json(out / "complete.json", {"mode": args.mode, "new_fits": len(fits),
                                     "run_spec_sha256": file_digest(out / "run_spec.json"),
                                     "summary_sha256": file_digest(out / "summary.csv")})
    log.info("Completed %d new fits. Summary: %s", len(fits), out / "summary.csv")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, ArithmeticError, ImportError, RuntimeError,
            metadata.PackageNotFoundError) as error:
        raise SystemExit(f"Benchmark stopped: {error}") from error
