"""Read verified preparation artifacts without encoding or loading annotations."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from scipy import sparse

from ..runtime.artifacts import file_hash, safe_path
from .preparation import check_receipt


@dataclass
class PreparedCollection:
    summary: dict
    directory: Path
    counts: sparse.csr_matrix
    folds: np.ndarray
    families: np.ndarray
    source_rows: np.ndarray


@dataclass
class SelectionBank:
    """The selection interface deliberately has no outer-target field."""
    scope: str
    directory: Path
    base: sparse.csr_matrix
    visible: sparse.csr_matrix
    targets: sparse.csr_matrix
    eligible: np.ndarray
    candidates: np.ndarray
    full_candidates: np.ndarray


def preparation_dependency(root, config):
    expected_scope = config['preparation']['scope']
    for run in sorted((root / 'outputs/runs').glob('*'), reverse=True):
        status_path, preparation_path = run / 'status.json', run / 'corpus_preparation.json'
        if not status_path.is_file() or not preparation_path.is_file():
            continue
        status = json.loads(status_path.read_text(encoding='utf-8'))
        if status.get('state') != 'completed' or status.get('stage') != 'corpus_preparation':
            continue
        preparation = json.loads(preparation_path.read_text(encoding='utf-8'))
        if preparation.get('scope') != expected_scope:
            continue
        manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
        for name in ('corpus_preparation.json', 'encoder_audit.json'):
            if file_hash(run / name) != manifest['artifacts'][name]['sha256']:
                raise ValueError('The preparation run metadata changed: ' + name)
        sample = safe_path(root, preparation['sample_manifest'])
        if file_hash(sample) != preparation['sample_manifest_sha256']:
            raise ValueError('The frozen sample manifest changed')
        encoder_audit = json.loads((run / 'encoder_audit.json').read_text(encoding='utf-8'))
        for relative, expected in (
            ('assets/pretrained/encoder/onnx/model.onnx', encoder_audit['model_sha256']),
            ('assets/pretrained/encoder/tokenizer.json', encoder_audit['tokenizer_sha256'])):
            if file_hash(root / relative) != expected:
                raise ValueError('A frozen encoder source file changed')
        return preparation, {'preparation_run_id': run.name,
                             'preparation_sha256': file_hash(preparation_path),
                             'sample_manifest_sha256': preparation['sample_manifest_sha256'],
                             'scope': expected_scope, 'encoder_audit': encoder_audit,
                             'data_and_encoder_reused': True}
    raise FileNotFoundError('No completed matching corpus preparation was found. Run main.py --config configs/experiments/corpus_preparation.yaml first.')


def load_collection(root, summary, config):
    directory = safe_path(root, summary['path'])
    if not check_receipt(directory):
        raise ValueError('The fitting collection has no complete preparation receipt')
    spec = json.loads((directory / 'identity.json').read_text(encoding='utf-8'))
    expected = {'views': config['method']['views'],
                'neighbors': config['method']['candidates']['per_view'],
                'outer_fraction': config['evaluation']['prediction']['outer_fraction']}
    if any(spec[key] != value for key, value in expected.items()):
        raise ValueError('Mask/retrieval settings changed; prepare evidence under those settings before borrowing')
    seeds = {key: config['method']['predictor'][key] for key in ('folds', 'fold_seed', 'mask_seed', 'outer_seed')}
    if spec['predictor_seeds'] != seeds:
        raise ValueError('Family-fold or mask seeds changed after preparation')
    counts = sparse.load_npz(directory / 'counts.npz').tocsr().astype(np.float64)
    folds = np.load(directory / 'folds.npy', allow_pickle=False)
    families = np.load(directory / 'families.npy', allow_pickle=False)
    rows = np.load(directory / 'source_rows.npy', allow_pickle=False)
    n = counts.shape[0]
    if counts.shape != (summary['posts'], summary['vocabulary']) or any(len(x) != n for x in (folds, families, rows)):
        raise ValueError('Prepared collection dimensions differ from its receipt')
    if not np.isfinite(counts.data).all() or (counts.data < 0).any():
        raise ValueError('Prepared counts must be finite and nonnegative')
    if len(rows) > 1 and not np.all(np.diff(rows) > 0):
        raise ValueError('Retained posts must preserve distinct increasing source-row order')
    if (folds < 0).any() or (folds >= seeds['folds']).any():
        raise ValueError('A prepared family fold is outside the declared range')
    for family in np.unique(families):
        if len(np.unique(folds[families == family])) != 1:
            raise ValueError('A text family crosses predictive folds')
    return PreparedCollection(summary, directory, counts, folds, families, rows)


def load_selection_bank(collection, scope, config):
    if scope not in ('topic_learning', 'outer_audit'):
        raise ValueError('Unknown selection-bank scope')
    directory = collection.directory / scope
    if not check_receipt(directory):
        raise ValueError('A prepared selection bank is incomplete')
    matrices = {name: sparse.load_npz(directory / (name + '.npz')).tocsr()
                for name in ('base', 'visible', 'targets')}
    arrays = {name: np.load(directory / (name + '.npy'), allow_pickle=False)
              for name in ('eligible', 'candidates', 'full_candidates')}
    n, vocabulary = collection.counts.shape
    views, neighbors = config['method']['views']['count'], config['method']['candidates']['per_view']
    if (matrices['base'].shape != (n, vocabulary)
            or any(matrices[k].shape != (n * views, vocabulary) for k in ('visible', 'targets'))
            or arrays['eligible'].shape != (n,)
            or arrays['candidates'].shape != (n, views, neighbors)
            or arrays['full_candidates'].shape != (n, neighbors)):
        raise ValueError('Prepared selection-bank dimensions differ from configuration')
    if matrices['visible'].multiply(matrices['targets']).nnz:
        raise ValueError('Inner visible and hidden words overlap')
    positive = np.asarray(collection.counts.sum(axis=1)).ravel() > 0
    for lists, receivers in (
        (arrays['candidates'].reshape(n, -1), np.arange(n)),
        (arrays['full_candidates'], np.arange(n))):
        for i, candidates in zip(receivers, lists):
            ids = candidates[candidates >= 0]
            if len(ids) and (ids.max() >= n or np.any(collection.folds[ids] == collection.folds[i])
                            or np.any(collection.families[ids] == collection.families[i])
                            or np.any(~positive[ids])):
                raise ValueError('A donor violates the family/fold/nonempty boundary')
    return SelectionBank(scope, directory, **matrices, **arrays)
