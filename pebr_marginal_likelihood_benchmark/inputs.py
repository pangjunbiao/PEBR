"""Find and verify the fixed transport-grievance experiment inputs."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from pebr_core.data.preprocessing import WORD
from pebr_core.proposed.candidates import retrieve
from pebr_core.proposed.masks import build_views
from pebr_core.proposed.reference import PredictiveReference

HERE = Path(__file__).resolve().parent
CORPUS = "transport_grievance"
SHAPE = (619, 1296)
N_VIEWS = 20
LIMIT = 20
MASK_SEED = 3711
TOPICS = 12
CAP = 1 / 3
ACTION_FIELDS = ("document_index", "source_row", "fold", "donors",
                 "available_donors", "offered_beta", "actual_beta")


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


def load_original(folder):
    counts = sparse.load_npz(folder / "counts.npz").toarray().astype(np.float64)
    words = np.load(folder / "vocabulary.npy", allow_pickle=False)
    source, folds, families = [np.load(folder / f"{field}.npy", allow_pickle=False)
                               for field in ("source_rows", "folds", "families")]
    n, v = SHAPE
    if (counts.shape != (n, v) or words.shape != (v,) or
            any(a.shape != (n,) for a in (source, folds, families))):
        raise ValueError("Prepared collection shape differs from the accepted corpus")
    if (not np.isfinite(counts).all() or (counts < 0).any() or
            (counts != np.floor(counts)).any() or (counts.sum(axis=1) == 0).any()):
        raise ValueError("Invalid original token counts")
    if (len(set(source.tolist())) != n or len(set(words.tolist())) != v or
            set(folds.tolist()) != set(range(5))):
        raise ValueError("Invalid source identities, vocabulary or fold assignment")
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    posts = read(folder / "posts.json")
    texts = [post["cleaned_text"] for post in posts]
    stop = set(ENGLISH_STOP_WORDS) - {"no", "not", "nor", "never"}
    tokens = [[w for w in WORD.findall(text) if w not in stop] for text in texts]
    if len(tokens) != n:
        raise ValueError("Original evaluation token rows differ")
    index = {word: j for j, word in enumerate(words)}
    for i, terms in enumerate(tokens):
        observed = {index[word]: count for word, count in Counter(terms).items()
                    if word in index}
        if observed != {j: int(counts[i, j]) for j in np.flatnonzero(counts[i])}:
            raise ValueError(f"Evaluation tokens do not reproduce original count row {i}")
    return SimpleNamespace(counts=counts, words=words, source=source, folds=folds,
                           families=families, tokens=tokens, texts=texts)


def original_fingerprints(data):
    return {key: array_digest(getattr(data, key))
            for key in ("counts", "source", "folds", "families")} | {
                "words": digest(data.words.tolist()), "tokens": digest(data.tokens)}


def action_fingerprint(records):
    return digest([[row[field] for field in ACTION_FIELDS] for row in records])


def cache_key(value):
    """Match the frozen encoder's Unicode-aware cache key."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return sha256(encoded).hexdigest()


def same_matrix(left, right):
    difference = (left - right).tocsr()
    return not difference.nnz or np.max(np.abs(difference.data)) <= 2e-12


def first_verified(paths, loader, hint):
    failures = []
    for path in dict.fromkeys(paths):
        try:
            return loader(path), path
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(f"{path} ({error})")
    detail = "\n  ".join(failures[:8])
    raise ValueError(f"{hint}" + (f"\n  {detail}" if detail else ""))


def load_mask_bank(directory, cache, data, pins):
    expected_encoder = pins["encoder_identity"]
    files = ("base.npz", "visible.npz", "targets.npz", "eligible.npy", "candidates.npy",
             "full_candidates.npy", "full_receiver_embeddings.npy",
             "query_positions.npy", "query_hashes.npy", "summary.json")
    receipt = read(directory / "complete.json")["files"]
    for name in files:
        if file_digest(directory / name) != receipt.get(name):
            raise ValueError(f"Saved masking-bank checksum mismatch for {name}")
    summary = read(directory / "summary.json")
    if (summary["corpus"], summary["mask_base"], summary["encoder_identity"]) != (
            CORPUS, MASK_SEED, expected_encoder):
        raise ValueError("Mask, corpus or frozen encoder differs from the reviewed run")
    config = {"method": {"views": {"count": N_VIEWS, "min_word_types": 4, "hide_fraction": 0.5},
                         "predictor": {"mask_seed": MASK_SEED}}}
    view = build_views(sparse.csr_matrix(data.counts), data.words, data.texts, data.source, config)
    for name in ("base", "visible", "targets"):
        saved = sparse.load_npz(directory / f"{name}.npz")
        if saved.shape != view[name].shape or not same_matrix(saved, view[name]):
            raise ValueError(f"Mask {MASK_SEED} reconstruction differs for {name}")
    for filename, expected in (("query_positions", view["positions"]), ("eligible", view["eligible"]),
                               ("query_hashes", [cache_key(q) for q in view["queries"]])):
        if not np.array_equal(np.load(directory / f"{filename}.npy", allow_pickle=False), expected):
            raise ValueError(f"Mask {MASK_SEED} reconstruction differs for {filename}")
    if len(view["queries"]) != 12320:
        raise ValueError("Unexpected number of masked queries")

    key = cache_key({"encoder": expected_encoder, "texts": view["queries"]})
    cache_dir = cache if cache is not None else directory.parents[2] / "cache" / "embeddings"
    path = cache_dir / f"vectors_{key}.npy"
    vector_receipt = read(path.with_suffix(".json"))
    if (vector_receipt["identity"] != key
            or pins["masked_query_cache"] != {"identity": key, "sha256": file_digest(path)}
            or vector_receipt["sha256"] != pins["masked_query_cache"]["sha256"]):
        raise ValueError("Cached masked-query embedding checksum or identity mismatch")
    vectors = np.load(path, allow_pickle=False)
    if (vectors.shape != (len(view["positions"]), data.embeddings.shape[1])
            or vectors.dtype != data.embeddings.dtype or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-6, rtol=0)):
        raise ValueError("Invalid frozen masked-query embeddings")

    n = len(data.counts)
    positive = data.counts.sum(axis=1) > 0
    saved_lists = np.load(directory / "candidates.npy", allow_pickle=False)
    saved_full = np.load(directory / "full_candidates.npy", allow_pickle=False)
    full_vectors = np.load(directory / "full_receiver_embeddings.npy", allow_pickle=False)
    lists = np.full((n * N_VIEWS, LIMIT), -1, dtype=np.int32)
    lists[view["positions"]] = retrieve(vectors, view["positions"] // N_VIEWS, data.embeddings,
                                        data.families, data.folds, positive, LIMIT)
    full = retrieve(full_vectors, np.arange(n), data.embeddings, data.families, data.folds,
                    positive, LIMIT)
    if (not np.array_equal(lists.reshape(n, N_VIEWS, LIMIT), saved_lists)
            or not np.array_equal(full, saved_full)):
        raise ValueError("Cached embeddings failed to replay the saved candidate lists exactly")
    return SimpleNamespace(visible=view["visible"], hidden=view["targets"],
                           eligible=view["eligible"], positions=view["positions"],
                           candidates=saved_lists, full=saved_full, vectors=vectors,
                           vector_path=path, vector_sha256=vector_receipt["sha256"],
                           bank_receipt_sha256=file_digest(directory / "complete.json"))


def read_frozen_references(cache, data, pins):
    settings = {"folds": 5, "seed": 0, "tolerance": 1e-4, "max_iterations": 2000,
                "smoothing": 0.1, "unigram_smoothing": 0.1}
    refs, origins = {}, {}
    for path in sorted(cache.glob("*/identity.json")):
        identity = read(path)
        fold = identity.get("excluded_fold")
        if (identity.get("condition") != pins["condition"] or identity.get("topics") != TOPICS
                or identity.get("settings") != settings or fold not in range(5) or fold in refs):
            continue
        arrays = {key: np.load(path.with_name(f"{key}.npy"), allow_pickle=False)
                  for key in pins["references"][str(fold)]}
        if {key: array_digest(value) for key, value in arrays.items()} != pins["references"][str(fold)]:
            continue
        audit = read(path.with_name("fit.json"))
        if (not audit["converged"] or not audit["receiver_fold_excluded"]
                or not np.array_equal(arrays["training_indices"], np.flatnonzero(data.folds != fold))):
            raise ValueError(f"Invalid reference fit for receiver fold {fold}")
        refs[fold] = PredictiveReference(arrays["dictionary"], arrays["idf"], arrays["prior"])
        origins[str(fold)] = str(path.parent)
    if set(refs) != set(range(5)):
        raise ValueError("Expected five verified default K12 reference predictors")
    return refs, origins


def validate_actions(rows, data, bank, expected_fingerprint):
    if len(rows) != len(data.counts) or action_fingerprint(rows) != expected_fingerprint:
        raise ValueError("Default PEBR action records differ from the reviewed replay")
    for i, row in enumerate(rows):
        if (row["document_index"], row["source_row"], row["fold"]) != (
                i, data.source[i], data.folds[i]):
            raise ValueError(f"Default PEBR action identities differ at receiver {i}")
        selected = tuple(row["donors"])
        available = sorted(set(selected).intersection(bank.full[i]))
        offered = CAP if selected else 0.0
        realized = offered * len(available) / len(selected) if selected else 0.0
        if (len(set(selected)) != len(selected) or len(selected) > 3
                or not set(selected).issubset(bank.candidates[i].ravel())
                or (selected and not bank.eligible[i])
                or available != sorted(row["available_donors"])
                or abs(offered - row["offered_beta"]) > 1e-12
                or abs(realized - row["actual_beta"]) > 1e-12):
            raise ValueError(f"Default PEBR action/availability differs at receiver {i}")
    return [tuple(row["donors"]) for row in rows]


def resolve(project, args):
    pins = read(HERE / "accepted_inputs.json")
    if args.data_root:
        roots = [args.data_root]
    else:
        roots = sorted(project.glob("outputs/revision_study/step2/*/data/processed"), reverse=True)
        roots += [project / "data" / "processed"]
    folders = [path for root in roots for path in (root / pins["prepared_subdirectory"], root)
               if (path / "counts.npz").is_file()]

    def prepared(path):
        data = load_original(path)
        if original_fingerprints(data) != pins["original"]:
            raise ValueError("Counts, folds, tokens, vocabulary or source IDs differ")
        data.embeddings = np.load(path / "embeddings.npy", allow_pickle=False)
        if array_digest(data.embeddings) != pins["embeddings"]:
            raise ValueError("Full donor embeddings differ")
        return data

    data, data_path = first_verified(folders, prepared,
                                     "Verified corpus missing. Use --data-root for the prepared data folder.")
    runs = sorted(project.glob("outputs/revision_rule_lab/fresh_selection/*"), reverse=True)
    banks = [args.mask_bank] if args.mask_bank else [
        p / "mask_banks" / CORPUS / f"mask{MASK_SEED}" for p in runs
        if (p / "mask_banks" / CORPUS / f"mask{MASK_SEED}" / "complete.json").is_file()]
    bank, bank_path = first_verified(
        banks, lambda p: load_mask_bank(p, args.query_cache, data, pins),
        "Verified mask3711 bank or query cache missing. See --mask-bank and --query-cache in README.")

    reference_caches = [args.reference_cache] if args.reference_cache else sorted(
        project.glob("outputs/revision_study/step3/*/cache/predictors"), reverse=True)
    reference_caches += [] if args.reference_cache else sorted(
        project.glob("outputs/revision_rule_lab/robustness8/s8_*/cache/predictors"), reverse=True)
    (references, ref_origins), _ = first_verified(
        reference_caches, lambda p: read_frozen_references(p, data, pins),
        "Verified reference predictors missing. Use --reference-cache for cache/predictors.")
    paths = [args.pebr_actions] if args.pebr_actions else sorted(
        project.glob(f"outputs/revision_rule_lab/robustness8/s8_*/reproduction/{CORPUS}/K12/"
                     "preserve_forward_fixed_cap/actions.json"), reverse=True)
    paths += [] if args.pebr_actions else [
        p / "selection" / CORPUS / f"mask{MASK_SEED}" / "K12" / "preserve_forward_fixed_cap" / "actions.json"
        for p in runs if (p / "selection" / CORPUS / f"mask{MASK_SEED}" / "K12" /
                          "preserve_forward_fixed_cap" / "actions.json").is_file()]
    actions, action_path = first_verified(
        paths, lambda p: validate_actions(read(p), data, bank, pins["actions"]["pebr"]),
        "Verified default PEBR actions missing. Use --pebr-actions for the audited actions.json.")
    audit = {"prepared_corpus": str(data_path), "shape": list(data.counts.shape),
             "eligible_receivers": int(bank.eligible.sum()), "mask_bank": str(bank_path),
             "mask_bank_receipt_sha256": bank.bank_receipt_sha256, "candidate_lists_replayed": True,
             "visible_query_cache": str(bank.vector_path), "visible_query_sha256": bank.vector_sha256,
             "default_pebr_actions": str(action_path), "reference_fits": ref_origins,
             "expected_fingerprints": pins}
    return data, bank, references, actions, audit
