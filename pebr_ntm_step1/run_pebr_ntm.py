"""Evaluate saved PEBR evidence with a ProdLDA neural topic learner."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from stability import seed_stability
from topic_quality import evaluate_topics

HERE = Path(__file__).resolve().parent
CORPORA = {"transport_grievance": ("primary_full", (619, 1296)),
           "chinese_pooled": ("chinese_pooled", (2053, 5002))}
POLICIES = {"matched_sharing": "matched_preserve_fixed",
            "pebr": "preserve_forward_fixed_cap"}
CONDITIONS = ("original", *POLICIES)
ACTION_FIELDS = ("document_index", "source_row", "fold", "donors",
                 "available_donors", "offered_beta", "actual_beta")
METRICS = ("npmi", "cv", "diversity")
DESIGN = {"topics": 12, "mask_seed": 3711, "reference_seed": 0,
          "fold_seed": 2710, "max_donors": 3, "cap": 1 / 3,
          "hidden": 100, "dropout": 0.2, "batch_size": 128,
          "adam_lr": 0.002, "adam_betas": [0.9, 0.999],
          "dirichlet_alpha": 1.0, "top_words": 10}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                             separators=(",", ":")).encode()).hexdigest()


def array_digest(value):
    value = np.asarray(value, dtype="<f8", order="C")
    return digest([value.shape, sha256(value.tobytes()).hexdigest()])


def file_digest(path):
    with path.open("rb") as stream:
        checksum = sha256()
        for block in iter(lambda: stream.read(1 << 20), b""):
            checksum.update(block)
        return checksum.hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as tmp:
        staged = Path(tmp) / path.name
        staged.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")
        staged.replace(path)


def write_csv(path, rows):
    with TemporaryDirectory(dir=path.parent) as tmp:
        staged = Path(tmp) / path.name
        with staged.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        staged.replace(path)


def load_original(folder, corpus):
    counts = sparse.load_npz(folder / "counts.npz").toarray().astype(np.float64)
    words = np.load(folder / "vocabulary.npy", allow_pickle=False)
    source, folds, families = [np.load(folder / f"{field}.npy", allow_pickle=False)
                               for field in ("source_rows", "folds", "families")]
    n, v = CORPORA[corpus][1]
    if (counts.shape != (n, v) or words.shape != (v,) or
            any(a.shape != (n,) for a in (source, folds, families))):
        raise ValueError("Prepared collection shape differs from the accepted corpus")
    if (not np.isfinite(counts).all() or (counts < 0).any() or
            (counts != np.floor(counts)).any() or (counts.sum(axis=1) == 0).any()):
        raise ValueError("Invalid original token counts")
    if (len(set(source.tolist())) != n or len(set(words.tolist())) != v or
            set(folds.tolist()) != set(range(5))):
        raise ValueError("Invalid source identities, vocabulary or fold assignment")
    if (folder / "tokens.json").exists():
        tokens = read(folder / "tokens.json")
    else:
        posts = read(folder / "posts.json")
        if corpus == "chinese_pooled":
            tokens = [post["tokens"] for post in posts]
        else:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            stop = set(ENGLISH_STOP_WORDS) - {"no", "not", "nor", "never"}
            pattern = re.compile(r"(?u)\b[a-z][a-z]+\b")
            tokens = [[w for w in pattern.findall(post["cleaned_text"]) if w not in stop]
                      for post in posts]
    if len(tokens) != n:
        raise ValueError("Original evaluation token rows differ")
    index = {word: j for j, word in enumerate(words)}
    for i, terms in enumerate(tokens):
        observed = {index[word]: count for word, count in Counter(terms).items()
                    if word in index}
        if observed != {j: int(counts[i, j]) for j in np.flatnonzero(counts[i])}:
            raise ValueError(f"Evaluation tokens do not reproduce original count row {i}")
    return SimpleNamespace(counts=counts, words=words, source=source, folds=folds,
                           families=families, tokens=tokens)


def original_fingerprints(data):
    return {key: array_digest(getattr(data, key))
            for key in ("counts", "source", "folds", "families")} | {
                "words": digest(data.words.tolist()), "tokens": digest(data.tokens)}


def action_fingerprint(records):
    return digest([[row[field] for field in ACTION_FIELDS] for row in records])


def restore_target(data, records, condition):
    counts = data.counts
    lengths = counts.sum(axis=1, keepdims=True)
    original = counts / lengths
    mixed = original.copy()
    masses = np.zeros(len(counts))
    donor_counts = np.zeros(len(counts), dtype=int)
    if len(records) != len(counts):
        raise ValueError("Action bank lacks complete receiver coverage")
    for i, row in enumerate(records):
        if (row["document_index"], row["source_row"], row["fold"]) != (
                i, data.source[i], data.folds[i]):
            raise ValueError(f"Action source row or fold mismatch at receiver {i}")
        selected, available = row["donors"], row["available_donors"]
        if (len(set(selected)) != len(selected) or len(selected) > DESIGN["max_donors"]
                or len(set(available)) != len(available)
                or not set(available).issubset(selected)):
            raise ValueError(f"Invalid recorded donor group at receiver {i}")
        if any(j < 0 or j >= len(counts) or data.folds[j] == data.folds[i]
               or data.families[j] == data.families[i] for j in selected):
            raise ValueError(f"Recorded donor violates fold/family eligibility at receiver {i}")
        offered = row["offered_beta"]
        if not 0 <= offered <= DESIGN["cap"] or (not selected and offered != 0):
            raise ValueError(f"Invalid offered weight at receiver {i}")
        if condition == "pebr" and selected and offered != DESIGN["cap"]:
            raise ValueError(f"PEBR action differs from the fixed cap at receiver {i}")
        share = offered / len(selected) if selected else 0.0
        alpha = share * len(available)
        if abs(alpha - row["actual_beta"]) > 1e-12:
            raise ValueError(f"Recorded and reconstructed transfer mass differ at receiver {i}")
        masses[i], donor_counts[i] = alpha, len(available)
        if available:
            mixed[i] = (1 - alpha) * original[i] + share * original[available].sum(axis=0)
    if ((mixed < 0).any() or not np.allclose(mixed.sum(axis=1), 1, atol=1e-12, rtol=0)
            or (0.5 * np.abs(mixed - original).sum(axis=1) > masses + 1e-12).any()):
        raise ValueError("Reconstructed targets fail normalization or borrowing bounds")
    target = mixed * lengths
    if not np.allclose(target.sum(axis=1), lengths[:, 0], atol=1e-10, rtol=0):
        raise ValueError("Training target changed a receiver's total token weight")
    return target, masses, donor_counts


def resolve_corpus(project, data_root, replay_root, corpus, pinned):
    alias = CORPORA[corpus][0]
    if data_root:
        parent_folders = [data_root]
    else:
        parent_folders = sorted(project.glob("outputs/revision_study/step2/*/data/processed"))
        parent_folders += [project / "data" / "processed"]
    names = dict.fromkeys((pinned["prepared_subdirectory"], alias, corpus))
    folders = [root / name for root in parent_folders for name in names]
    problems = []
    for folder in folders:
        if not (folder / "counts.npz").exists():
            continue
        try:
            data = load_original(folder, corpus)
            if original_fingerprints(data) != pinned["original"]:
                raise ValueError("Counts, tokens, vocabulary or row identities differ")
            break
        except (OSError, ValueError, KeyError) as error:
            problems.append(f"{folder} ({error})")
    else:
        raise ValueError(f"Missing verified {corpus} inputs. Use --data-root to point "
                         f"to Step 2 data/processed. Checked {problems or folders}")
    banks = ([replay_root] if replay_root else sorted(
        project.glob("outputs/revision_rule_lab/robustness8/s8_*/reproduction")))
    for bank in banks:
        entries = {arm: bank / corpus / "K12" / name for arm, name in POLICIES.items()}
        if not all((p / "actions.json").is_file() for p in entries.values()):
            continue
        records = {arm: read(p / "actions.json") for arm, p in entries.items()}
        if {arm: action_fingerprint(rows) for arm, rows in records.items()} == pinned["actions"]:
            break
    else:
        raise ValueError(f"Missing verified default K12 {corpus} actions. Use "
                         f"--replay-root to point to the completed Step 8 reproduction folder. "
                         f"Checked {banks}")
    targets = {"original": data.counts}
    weights, donors = {}, {}
    for arm, rows in records.items():
        targets[arm], weights[arm], donors[arm] = restore_target(data, rows, arm)
        matrix = entries[arm] / "transferred.npz"
        if matrix.exists():
            saved = sparse.load_npz(matrix).toarray()
            restored = targets[arm] / data.counts.sum(axis=1, keepdims=True)
            if saved.shape != restored.shape or not np.allclose(saved, restored, atol=2e-12, rtol=0):
                raise ValueError(f"Replay differs from {matrix}")
    if (not np.allclose(weights["pebr"], weights["matched_sharing"], atol=1e-12, rtol=0)
            or not np.array_equal(donors["pebr"], donors["matched_sharing"])):
        raise ValueError("Matched control fails per-receiver transfer weight or donor count")
    binary = sparse.csr_matrix(data.counts > 0, dtype=np.int64)
    from gensim.corpora import Dictionary
    dictionary = Dictionary(data.tokens)
    reference = SimpleNamespace(vocabulary=data.words, documents=len(data.counts),
                                df=np.asarray(binary.sum(axis=0)).ravel(),
                                cooccurrence=(binary.T @ binary).tocsr(),
                                tokens=data.tokens, identity=digest(pinned["original"]),
                                dictionary=lambda: dictionary)
    audit = {"original_directory": str(folder), "replay_directory": str(bank),
             "posts": len(data.counts), "vocabulary": len(data.words),
             "mean_realized_weight": float(weights["pebr"].mean()),
             "positive_transfer_posts": int((weights["pebr"] > 0).sum()),
             "per_receiver_transfer_weight_and_count_match": True}
    print(f"Verified {corpus}: {audit['posts']} posts, {audit['vocabulary']} words", flush=True)
    return data, targets, reference, audit


def fit(data, target, seed, epochs, device):
    import torch
    from torch import nn
    from torch.nn import functional as F

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    k, v = DESIGN["topics"], data.counts.shape[1]
    hidden, dropout = DESIGN["hidden"], DESIGN["dropout"]

    class ProdLDA(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(v, hidden), nn.Softplus(),
                                         nn.Linear(hidden, hidden), nn.Softplus(),
                                         nn.Dropout(dropout))
            self.mean = nn.Sequential(nn.Linear(hidden, k), nn.BatchNorm1d(k, affine=False))
            self.logvar = nn.Sequential(nn.Linear(hidden, k), nn.BatchNorm1d(k, affine=False))
            self.topic_dropout = nn.Dropout(dropout)
            self.decoder = nn.Linear(k, v, bias=False)
            self.decoder_norm = nn.BatchNorm1d(v, affine=False)
            nn.init.xavier_uniform_(self.decoder.weight)

        def posterior(self, counts):
            hidden_counts = self.encoder(counts)
            return self.mean(hidden_counts), self.logvar(hidden_counts)

        def word_logits(self, theta):
            return self.decoder_norm(self.decoder(theta))

    model = ProdLDA().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=DESIGN["adam_lr"],
                                 betas=tuple(DESIGN["adam_betas"]))
    x = torch.as_tensor(data.counts, dtype=torch.float32, device=device)
    y = torch.as_tensor(target, dtype=torch.float32, device=device)
    if not torch.allclose(x.sum(1), y.sum(1), rtol=1e-6, atol=1e-6):
        raise ValueError("Float32 targets changed the per-post reconstruction mass")
    prior_variance = (1.0 - 1.0 / k) / DESIGN["dirichlet_alpha"]
    prior_logvar = float(np.log(prior_variance))
    batch_rng = torch.Generator().manual_seed(seed)
    losses = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        batches = list(torch.randperm(len(x), generator=batch_rng).split(DESIGN["batch_size"]))
        if len(batches[-1]) == 1:
            singleton = batches.pop()
            batches[-1] = torch.cat((batches[-1], singleton))
        sums = torch.zeros(2, device=device)
        for index in batches:
            index = index.to(device)
            mean, logvar = model.posterior(x[index])
            latent = mean + torch.randn_like(mean) * (0.5 * logvar).exp()
            theta = model.topic_dropout(F.softmax(latent, dim=1))
            log_word_probs = F.log_softmax(model.word_logits(theta), dim=1)
            nll = -(y[index] * log_word_probs).sum(dim=1).mean()
            kl = 0.5 * ((logvar.exp() + mean.square()) / prior_variance - 1
                        + prior_logvar - logvar).sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            (nll + kl).backward()
            optimizer.step()
            sums += torch.stack((nll.detach(), kl.detach())) * len(index)
        nll_value, kl_value = (sums / len(x)).cpu().tolist()
        if not np.isfinite([nll_value, kl_value]).all():
            raise ArithmeticError(f"Nonfinite ProdLDA loss at epoch {epoch}")
        losses.append({"epoch": epoch, "reconstruction": nll_value, "kl": kl_value,
                       "objective": nll_value + kl_value})
    model.eval()
    with torch.no_grad():
        memberships = F.softmax(model.posterior(x)[0], dim=1).cpu().numpy()
        logits = model.word_logits(torch.eye(k, device=device))
        scores = F.log_softmax(logits, dim=1).cpu().numpy()
    if (memberships.shape != (len(x), k) or not np.isfinite(memberships).all()
            or not np.allclose(memberships.sum(1), 1, atol=1e-6, rtol=0)
            or scores.shape != (k, v) or not np.isfinite(scores).all()):
        raise ArithmeticError("Invalid original-post memberships or topic-word scores")
    return model, memberships, scores, losses, time.perf_counter() - start


def run_fit(work, corpus, arm, seed, epochs, device, data, target, reference):
    import torch

    destination = work / "fits" / corpus / arm / f"seed{seed}"
    receipt = destination / "result.json"
    if receipt.is_file():
        result = read(receipt)
        if (result["corpus"], result["condition"], result["seed"]) != (corpus, arm, seed):
            raise ValueError(f"Fit receipt has unexpected identity in {destination}")
        for name, expected in result["files"].items():
            if file_digest(destination / name) != expected:
                raise ValueError(f"Saved fit was modified in {destination}")
        return result
    print(f"Fitting {corpus} | {arm} | seed {seed} | {epochs} epochs", flush=True)
    model, memberships, scores, losses, elapsed = fit(data, target, seed, epochs, device)
    quality = evaluate_topics(scores, data.words, reference, DESIGN["top_words"], True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as tmp:
        stage = Path(tmp) / "fit"
        stage.mkdir()
        torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()},
                   stage / "model.pt")
        np.savez_compressed(stage / "outputs.npz", topic_scores=scores,
                            original_post_memberships=memberships, vocabulary=data.words,
                            source_rows=data.source)
        write_csv(stage / "losses.csv", losses)
        result = {"corpus": corpus, "condition": arm, "seed": seed, "epochs": epochs,
                  "fit_seconds": elapsed, "final_objective": losses[-1]["objective"],
                  **quality, "files": {p.name: file_digest(p) for p in stage.iterdir()}}
        save_json(stage / "result.json", result)
        stage.rename(destination)
    return result


def summarize(work, fits):
    summary, paired = [], []
    for corpus in CORPORA:
        base = {}
        for arm in CONDITIONS:
            rows = [r for r in fits if r["corpus"] == corpus and r["condition"] == arm]
            stability = seed_stability({r["seed"]: r["top_words"] for r in rows})
            stats = {"corpus": corpus, "condition": arm, "fits": len(rows)}
            for key in METRICS:
                values = [r[key] for r in rows]
                stats[key + "_mean"] = float(np.mean(values))
                stats[key + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            stats["stability"] = stability["stability"]
            stats["seed_pairs"] = stability["seed_pairs"]
            stats["fit_seconds_mean"] = float(np.mean([r["fit_seconds"] for r in rows]))
            summary.append(stats)
            base[arm] = {r["seed"]: r for r in rows}
        for arm in ("original", "matched_sharing"):
            for seed, treatment in base["pebr"].items():
                control = base[arm][seed]
                paired.append({"corpus": corpus, "seed": seed, "control": arm,
                               **{key + "_delta": treatment[key] - control[key] for key in METRICS}})
    write_csv(work / "summary.csv", summary)
    write_csv(work / "paired_deltas.csv", paired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=HERE.parent)
    parser.add_argument("--data-root", type=Path, help="Prepared Step 2 data/processed directory")
    parser.add_argument("--replay-root", type=Path, help="Completed Step 8 reproduction directory")
    parser.add_argument("--mode", choices=("preflight", "smoke", "full"), default="smoke")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    pinned = read(HERE / "accepted_inputs.json")
    packages = {key: metadata.version(key) for key in ("numpy", "scipy", "gensim", "torch")}
    if packages["gensim"] != "4.4.0":
        raise ValueError("Gensim 4.4.0 is required to match the manuscript's C_v implementation")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable with this interpreter. Select --device cpu.")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    epochs, seeds = (400, list(range(5))) if args.mode == "full" else (2, [0])
    spec = {"mode": args.mode, "epochs": epochs, "seeds": seeds, "design": DESIGN,
            "packages": packages, "python": sys.version, "device": args.device,
            "cuda_runtime": torch.version.cuda, "torch_cpu_threads": torch.get_num_threads(),
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "accepted_inputs": pinned, "inputs": "original counts in every encoder",
            "targets": "original token mass times the policy's normalized distribution",
            "topic_readout": "log word probabilities from one-hot topic inputs in eval mode",
            "code": {name: file_digest(HERE / name) for name in
                     ("run_pebr_ntm.py", "topic_quality.py", "stability.py")}}
    work = (args.project_root.resolve() / "outputs" / "pebr_extensions" / "ntm_step1"
            / f"{args.mode}_{digest(spec)[:12]}")
    inputs = {corpus: resolve_corpus(args.project_root.resolve(), args.data_root,
                                    args.replay_root, corpus, pinned[corpus]) for corpus in CORPORA}
    save_json(work / "run_spec.json", spec | {"paths_and_checks": {c: data[3] for c, data in inputs.items()}})
    print(f"Output folder: {work}", flush=True)
    if args.mode == "preflight":
        print("PREFLIGHT PASSED", flush=True)
        return
    fits = [run_fit(work, corpus, arm, seed, epochs, device, data, targets[arm], reference)
            for corpus, (data, targets, reference, _) in inputs.items()
            for seed in seeds for arm in CONDITIONS]
    summarize(work, fits)
    save_json(work / "complete.json", {"mode": args.mode, "completed_fits": len(fits),
                                      "technical_smoke_only": args.mode == "smoke",
                                      "run_spec_sha256": file_digest(work / "run_spec.json")})
    print(f"Completed {len(fits)} fits. See {work / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, ArithmeticError, ImportError,
            metadata.PackageNotFoundError) as error:
        raise SystemExit(f"Step 1 stopped: {error}") from error
