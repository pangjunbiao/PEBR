"""Exact compact exports from the user's completed full study; no raw texts or model weights."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import zipfile

from transport_topics.runtime.artifacts import file_hash


@contextmanager
def completed_exports():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / 'completed source with spaces'
        with zipfile.ZipFile(Path(__file__).parent / 'fixtures/completed_study_exports.zip') as z:
            z.extractall(root)
        yield root


def refresh_manifest(root, relative):
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    path = root / relative
    manifest['artifacts'][relative] = {'sha256': file_hash(path), 'bytes': path.stat().st_size}
    (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')


def revise_json(root, name, update):
    path = root / name
    record = json.loads(path.read_text(encoding='utf-8'))
    update(record)
    path.write_text(json.dumps(record), encoding='utf-8')
    refresh_manifest(root, name)
