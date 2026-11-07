"""Load the verified corpus and build the original-text evaluation reference."""
from __future__ import annotations

from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from scipy import sparse


CORPUS = "chinese_pooled"
SHAPE = (2053, 5002)

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
    if (folder / "tokens.json").exists():
        tokens = read(folder / "tokens.json")
    else:
        posts = read(folder / "posts.json")
        tokens = [post["tokens"] for post in posts]
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


def load_word_vectors(path, vocabulary, pinned):
    """Read the Chinese 200-dimensional initialization used in the original study."""
    if path.is_dir():
        vectors = np.load(path / "vectors.npy", allow_pickle=False)
        saved_words = np.load(path / "vocabulary.npy", allow_pickle=False)
    elif path.is_file() and path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as exported:
            vectors, saved_words = exported["vectors"], exported["vocabulary"]
    else:
        raise ValueError(f"Chinese word vectors missing at {path}. Use --word-vectors "
                         "with the project's 200D cache directory or audited Chinese NPZ export.")
    if (not np.array_equal(saved_words, vocabulary) or vectors.shape != (SHAPE[1], 200)
            or vectors.dtype != np.float32 or not np.isfinite(vectors).all()
            or np.any(np.linalg.norm(vectors, axis=1) <= 0)):
        raise ValueError("Chinese word-vector alignment or values differ")
    fingerprint = sha256(np.asarray(vectors, dtype="<f4", order="C").tobytes()).hexdigest()
    if fingerprint != pinned["word_vectors_values_sha256"]:
        raise ValueError("Chinese word vectors differ from the audited 200D initialization")
    return vectors, fingerprint


def load_benchmark_inputs(project, data_root, word_vectors, pinned):
    from gensim.corpora import Dictionary

    parents = ([data_root] if data_root else
               sorted(project.glob("outputs/revision_study/step2/*/data/processed"), reverse=True) +
               [project / "data/processed"])
    names = (pinned["prepared_subdirectory"], CORPUS)
    folders = [parent / name for parent in parents for name in names]
    if data_root:
        folders.insert(0, data_root)
    errors = []
    for folder in dict.fromkeys(folders):
        if not (folder / "counts.npz").is_file():
            continue
        try:
            data = load_original(folder)
            if original_fingerprints(data) != pinned["original"]:
                raise ValueError("original counts, tokens, vocabulary, or row IDs differ")
            if file_digest(folder / "embeddings.npy") != pinned["embeddings_sha256"]:
                raise ValueError("original document embeddings differ")
            data.embeddings = np.load(folder / "embeddings.npy", allow_pickle=False)
            break
        except (OSError, ValueError, KeyError) as error:
            errors.append(f"{folder}: {error}")
    else:
        detail = "\n".join(errors)
        raise ValueError("Verified Chinese corpus missing. Set --data-root to the completed "
                         f"prepared corpus directory.\n{detail}")
    if (data.embeddings.shape != (SHAPE[0], 384) or not np.isfinite(data.embeddings).all()
            or not np.allclose(np.linalg.norm(data.embeddings, axis=1), 1, atol=2e-6, rtol=0)):
        raise ValueError("Expected 2,053 aligned unit-normalized multilingual MiniLM vectors")

    path = word_vectors if word_vectors else project / pinned["word_vector_directory"]
    if word_vectors is None and not path.is_dir():
        snapshots = sorted(project.glob("outputs/revision_study/step2/*"), reverse=True)
        path = next((p / pinned["word_vector_directory"] for p in snapshots
                     if (p / pinned["word_vector_directory"]).is_dir()), path)
    data.word_vectors, fingerprint = load_word_vectors(path, data.words, pinned)
    data.audit = {
        "data_directory": str(folder.resolve()),
        "word_vectors_path": str(path.resolve()),
        "posts": SHAPE[0],
        "vocabulary": SHAPE[1],
        "document_encoder": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "document_embeddings_sha256": pinned["embeddings_sha256"],
        "word_vector_source": "frozen multilingual encoder vocabulary embeddings, centered unwhitened PCA to 200D",
        "word_vectors_values_sha256": fingerprint,
        "word_vector_dimension": 200,
    }
    binary = sparse.csr_matrix(data.counts > 0, dtype=np.int64)
    dictionary = Dictionary(data.tokens)
    data.reference = SimpleNamespace(
        vocabulary=data.words, documents=len(data.counts),
        df=np.asarray(binary.sum(axis=0)).ravel(),
        cooccurrence=(binary.T @ binary).tocsr(),
        tokens=data.tokens, identity=digest(pinned["original"]), dictionary=lambda: dictionary,
    )
    return data
