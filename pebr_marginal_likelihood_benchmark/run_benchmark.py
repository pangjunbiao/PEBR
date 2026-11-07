"""Run the cross-fitted marginal-likelihood retrieval benchmark."""
import argparse
import csv
from datetime import datetime, timezone
from importlib import metadata
import os
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from inputs import CAP, CORPUS, HERE, LIMIT, MASK_SEED, N_VIEWS, TOPICS, file_digest, resolve, save_json
from pebr_core.numerics.nmf import fit_nmf
from pebr_core.proposed.actions import BorrowingAction
from pebr_core.proposed.transfer import check_transfer, normalized_counts, transfer_action
from pebr_core.representations.tfidf import idf_weights, transform
from selection import DOWNSIDE, MARGIN, REGULARIZATION, STARTS, run_selection
from topic_quality import evaluate_topics

POLICIES = {
    "nmf_original": "NMF without borrowing",
    "realm_inspired": "Marginal-likelihood retrieval (REALM-inspired)",
    "pebr_one_donor": "PEBR (one donor)",
    "pebr": "PEBR",
}
METRICS = ("npmi", "cv", "diversity")
TOP_WORDS = 10


def write_csv(path, rows):
    with TemporaryDirectory(dir=path.parent) as tmp:
        staged = Path(tmp) / path.name
        with staged.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        staged.replace(path)


def final_inputs(data, bank, selected, work):
    original = normalized_counts(data.counts)
    matrices, rates = {}, []
    for policy in POLICIES:
        actions, rows = [], []
        for i, donor_ids in enumerate(selected[policy]):
            action = BorrowingAction(donor_ids, CAP if donor_ids else 0.0)
            row, weight, available = transfer_action(original[i], action, bank.full[i], original)
            distance = check_transfer(original[i], row, weight, CAP)
            actions.append({"receiver": i, "source_row": int(data.source[i]),
                            "fold": int(data.folds[i]), "donors": list(action.donors),
                            "contributing_donors": list(available), "offered_weight": action.beta,
                            "realized_weight": weight, "total_variation": distance})
            rows.append(row)
        directory = work / "transfer" / policy
        save_json(directory / "actions.json", actions)
        matrices[policy] = sparse.csr_matrix(rows)
        sparse.save_npz(directory / "normalized_counts.npz", matrices[policy])
        rates.append({"policy": policy, "label": POLICIES[policy],
                      "selected_receivers": sum(bool(a["donors"]) for a in actions),
                      "transferred_receivers": sum(bool(a["contributing_donors"]) for a in actions),
                      "total_contributing_donors": sum(len(a["contributing_donors"]) for a in actions),
                      "mean_realized_weight": float(np.mean([a["realized_weight"] for a in actions]))})
        print(f"{POLICIES[policy]} | selected {rates[-1]['selected_receivers']}/{len(actions)}"
              f" | transferred {rates[-1]['transferred_receivers']}/{len(actions)}"
              f" | mean realized weight {rates[-1]['mean_realized_weight']:.4f}", flush=True)
    write_csv(work / "transfer_summary.csv", rates)
    return matrices


def fit_and_score(data, matrices, seeds, work):
    from gensim.corpora import Dictionary
    original = sparse.csr_matrix(data.counts)
    binary = original.astype(bool).astype(np.int64)
    dictionary = Dictionary(data.tokens)
    ref = SimpleNamespace(vocabulary=data.words, documents=original.shape[0],
                          df=np.asarray(binary.sum(axis=0)).ravel(), cooccurrence=(binary.T @ binary).tocsr(),
                          tokens=data.tokens, identity="verified_original_transport_grievance",
                          dictionary=lambda: dictionary)
    idf = idf_weights(original)
    fitting = {p: transform(original if p == "nmf_original" else matrix, idf)
               for p, matrix in matrices.items()}
    records = []
    for seed in seeds:
        for policy, matrix in fitting.items():
            print(f"Fit | {POLICIES[policy]} | seed {seed}", flush=True)
            _, topics, audit = fit_nmf(matrix, TOPICS, seed)
            directory = work / "fits" / policy / f"seed{seed}"
            save_json(directory / "fit_audit.json", audit)
            if not audit["converged"]:
                raise RuntimeError(f"NMF failed to converge. See {directory}")
            quality = evaluate_topics(topics, data.words, ref, TOP_WORDS)
            save_json(directory / "quality.json", quality)
            np.save(directory / "topic_word_weights.npy", topics, allow_pickle=False)
            records.append({"corpus": CORPUS, "K": TOPICS, "mask_seed": MASK_SEED,
                            "policy": policy, "label": POLICIES[policy], "seed": seed,
                            **{key: quality[key] for key in METRICS}})
            write_csv(work / "per_seed.csv", records)
            print(f"      NPMI {quality['npmi']:.4f} | C_v {quality['cv']:.4f}"
                  f" | TD {quality['diversity']:.4f}", flush=True)
    return records


def summarize(records, work):
    summaries = []
    for policy, label in POLICIES.items():
        rows = [r for r in records if r["policy"] == policy]
        summary = {"policy": policy, "label": label, "n_seeds": len(rows)}
        fields = []
        for key, title in zip(METRICS, ("NPMI", "C_v", "TD")):
            values = [r[key] for r in rows]
            average = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else None
            summary.update({f"{key}_mean": average, f"{key}_sd": sd})
            fields.append(f"{title} {average:.4f}" + (f" +/- {sd:.4f}" if sd is not None else " (one seed)"))
        summaries.append(summary)
        print(f"{label} | " + " | ".join(fields), flush=True)
    write_csv(work / "summary.csv", summaries)
    by_key = {(r["policy"], r["seed"]): r for r in records}
    contrasts = [("pebr_one_donor", "realm_inspired"),
                 ("pebr", "realm_inspired"), ("pebr", "pebr_one_donor"),
                 ("realm_inspired", "nmf_original"), ("pebr_one_donor", "nmf_original")]
    deltas = []
    for policy, comparison in contrasts:
        for seed in sorted({r["seed"] for r in records}):
            left, right = by_key[policy, seed], by_key[comparison, seed]
            deltas.append({"policy": policy, "comparison": comparison, "seed": seed,
                           **{f"{m}_delta": left[m] - right[m] for m in METRICS}})
    write_csv(work / "paired_deltas.csv", deltas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=HERE.parent)
    parser.add_argument("--mode", choices=("preflight", "smoke", "full"), default="preflight")
    parser.add_argument("--data-root", type=Path, help="Prepared data/processed root or verified corpus folder")
    parser.add_argument("--mask-bank", type=Path, help="Completed transport_grievance/mask3711 directory")
    parser.add_argument("--query-cache", type=Path, help="Directory containing the verified vectors_*.npy and .json")
    parser.add_argument("--reference-cache", type=Path, help="Completed cache/predictors directory")
    parser.add_argument("--pebr-actions", type=Path, help="Reviewed default K12 PEBR actions.json")
    args = parser.parse_args()
    args.project_root = args.project_root.resolve()
    for name in ("data_root", "mask_bank", "query_cache", "reference_cache", "pebr_actions"):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, (args.project_root / path).resolve())

    seeds = list(range(5)) if args.mode == "full" else [0] if args.mode == "smoke" else []
    project = args.project_root
    if not project.is_dir():
        parser.error(f"Project directory does not exist: {project}")
    if metadata.version("gensim") != "4.4.0":
        parser.error("Gensim 4.4.0 is required for the reviewed C_v implementation.")
    with threadpool_limits(limits=1):
        print("Verifying corpus, mask3711, cached query vectors, predictors and default PEBR replay...", flush=True)
        data, bank, frozen, default, inputs_audit = resolve(project, args)
        versions = {name: metadata.version(name) for name in
                    ("numpy", "scipy", "scikit-learn", "gensim", "threadpoolctl")}
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        work = project / "outputs" / "pebr_extensions" / "marginal_likelihood_retrieval" / f"{args.mode}_{stamp}"
        work.mkdir(parents=True, exist_ok=False)
        print(f"Verified {CORPUS}: {len(data.counts)} posts, {len(data.words)} words"
              f" | {bank.eligible.sum()} eligible receivers\nOutput folder: {work}", flush=True)
        manifest = {
            "status": "running", "mode": args.mode, "reportable_run": False,
            "project": str(project), "python": sys.version.split()[0], "platform": platform.platform(),
            "versions": versions, "pid": os.getpid(), "inputs": inputs_audit,
            "design": {"corpus": CORPUS, "topics": TOPICS, "mask_seed": MASK_SEED, "mask_views": N_VIEWS,
                       "candidate_limit": LIMIT, "fold_seed": 2710, "reference_seed": 0,
                       "transfer_cap": CAP, "pebr_margin": MARGIN, "pebr_downside_weight": DOWNSIDE,
                       "retriever_l2": REGULARIZATION, "retriever_optimizer_starts": STARTS,
                       "nmf_final_seeds": seeds, "nmf_init": "random", "nmf_loss": "frobenius",
                       "nmf_solver": "cd", "nmf_tolerance": 1e-4, "nmf_max_iterations": 2000,
                       "top_words": TOP_WORDS, "threads": 1,
                       "smoke_train_receivers_per_inner_fold": 8 if args.mode == "smoke" else None,
                       "smoke_eval_receivers_per_outer_fold": 4 if args.mode == "smoke" else None},
            "source_sha256": {p.relative_to(HERE).as_posix(): file_digest(p)
                              for p in sorted(HERE.rglob("*.py"))},
        }
        save_json(work / "run_manifest.json", manifest)
        if args.mode == "preflight":
            manifest["status"] = "preflight_complete"
            save_json(work / "run_manifest.json", manifest)
            print("Preflight complete. The verified inputs are ready for a smoke or full run.")
            return

        try:
            choices, active, decisions = run_selection(
                data, bank, frozen, args.mode == "smoke",
                lambda name, rows: save_json(work / "retrieval_fit" / f"{name}.json", rows))
            choices["nmf_original"] = [()] * len(data.counts)
            choices["pebr"] = [donors if i in active else () for i, donors in enumerate(default)]
            write_csv(work / "receiver_decisions.csv", decisions)
            matrices = final_inputs(data, bank, choices, work)
            records = fit_and_score(data, matrices, seeds, work)
            print("\nTopic quality across final-model seeds", flush=True)
            summarize(records, work)
        except Exception as error:
            manifest.update(status="failed", failure=str(error))
            save_json(work / "run_manifest.json", manifest)
            raise
        manifest.update(status="complete", reportable_run=(args.mode == "full"),
                        fits=len(records), evaluated_receivers=len(active))
        save_json(work / "run_manifest.json", manifest)
        print(f"\nCompleted {len(records)} fits. Summary: {work / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
