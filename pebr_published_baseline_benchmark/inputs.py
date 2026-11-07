"""Load the verified corpus and build the original-text evaluation reference."""
from __future__ import annotations

from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from scipy import sparse


CORPORA = {"transport_grievance": ("primary_full", (619, 1296))}
INPUT_BUNDLE = "pebr_benchmark_inputs"
GLOVE_NAME = "glove_6b_200d_transport_lookup.npz"

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


def load_benchmark_inputs(project, data_root, glove_file, pinned):
    from gensim.corpora import Dictionary

    bundle = project / INPUT_BUNDLE
    parents = ([data_root] if data_root else
               sorted(project.glob("outputs/revision_study/step2/*/data/processed")) +
               [project / "data/processed"])
    names = (pinned["prepared_subdirectory"], "primary_full", "transport_grievance")
    folders = [parent / name for parent in parents for name in names]
    if data_root:
        folders.insert(0, data_root)
    else:
        folders.insert(0, bundle / "corpus")
    errors = []
    for folder in folders:
        if not (folder / "counts.npz").is_file():
            continue
        try:
            data = load_original(folder, "transport_grievance")
            if original_fingerprints(data) != pinned["original"]:
                raise ValueError("original counts, tokens, vocabulary, or row IDs differ")
            if file_digest(folder / "embeddings.npy") != pinned["embeddings_sha256"]:
                raise ValueError("original document embeddings differ")
            data.embeddings = np.load(folder / "embeddings.npy", allow_pickle=False)
            break
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"{folder}: {exc}")
    else:
        raise ValueError(f"Verified corpus missing. Place {INPUT_BUNDLE} beside the code "
                         "folder, or set --data-root to the prepared corpus directory. "
                         + " | ".join(errors))
    if data.embeddings.shape != (619, 384) or not np.isfinite(data.embeddings).all():
        raise ValueError("Expected 619 aligned MiniLM vectors with 384 finite coordinates")
    if glove_file is None:
        bundled_glove = bundle / GLOVE_NAME
        glove_file = (bundled_glove if bundled_glove.is_file()
                      else project / "assets/pretrained/word_embeddings" / GLOVE_NAME)
    if not glove_file.is_file() or file_digest(glove_file) != pinned["glove_sha256"]:
        raise ValueError("Verified GloVe lookup missing. Set --glove-file to the original "
                         f"200-dimensional project lookup. Checked {glove_file}")
    with np.load(glove_file, allow_pickle=False) as glove:
        lookup = {str(word): i for i, word in enumerate(glove["words"])}
        indices = [lookup[str(word)] for word in data.words]
        vectors = glove["vectors"][indices].astype(np.float32, copy=True)
        found = glove["found"][indices]
    # Same out-of-vocabulary stream as the existing project word-vector loader.
    for i in np.flatnonzero(~found):
        payload = json.dumps([0, "glove_oov", [str(data.words[i]), 200]],
                             ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        seed = int.from_bytes(sha256(payload).digest()[:4], "big")
        vectors[i] = np.random.default_rng(seed).normal(0, 0.6, 200)
    if vectors.shape != (1296, 200) or not np.isfinite(vectors).all() or (np.linalg.norm(vectors, axis=1) == 0).any():
        raise ValueError("Invalid vocabulary-aligned GloVe vectors")
    data.word_vectors = vectors
    data.audit = {"data_directory": str(folder.resolve()), "glove_file": str(glove_file.resolve()),
                  "posts": 619, "vocabulary": 1296, "glove_found": int(found.sum()),
                  "document_encoder": "all-MiniLM-L6-v2",
                  "embedding_normalization": "fixed unit-normalized project vectors",
                  "aligned_glove_sha256": array_digest(vectors)}
    binary = sparse.csr_matrix(data.counts > 0, dtype=np.int64)
    dictionary = Dictionary(data.tokens)
    data.reference = SimpleNamespace(vocabulary=data.words, documents=len(data.counts),
        df=np.asarray(binary.sum(axis=0)).ravel(), cooccurrence=(binary.T @ binary).tocsr(),
        tokens=data.tokens, identity=digest(pinned["original"]), dictionary=lambda: dictionary)
    return data


def export_input_bundle(project, data, pinned):
    destination = project / INPUT_BUNDLE
    if destination.exists():
        load_benchmark_inputs(project, destination / "corpus", destination / GLOVE_NAME, pinned)
        return destination

    original = Path(data.audit["data_directory"])
    with TemporaryDirectory(dir=project) as tmp:
        staged = Path(tmp) / INPUT_BUNDLE
        corpus = staged / "corpus"
        corpus.mkdir(parents=True)
        for name in ("counts.npz", "vocabulary.npy", "source_rows.npy",
                     "folds.npy", "families.npy", "embeddings.npy"):
            shutil.copyfile(original / name, corpus / name)
        save_json(corpus / "tokens.json", data.tokens)
        shutil.copyfile(data.audit["glove_file"], staged / GLOVE_NAME)
        load_benchmark_inputs(staged, corpus, staged / GLOVE_NAME, pinned)
        staged.rename(destination)
    return destination
