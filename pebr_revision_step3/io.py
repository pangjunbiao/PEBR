"""Atomic checkpoints, immutable-input checks and conservative resumption."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path

PROTOCOL_SHA256 = 'd25536e04233a643862676601e5e441ac4f7c2f4982d98793df495d74b3ea8ec'

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def inside(root, relative):
    root = Path(root).resolve()
    target = (root / relative).resolve()
    try: target.relative_to(root)
    except ValueError: raise ValueError('Path escapes its declared root: ' + str(relative))
    return target

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)

def write_csv(path, rows, fields=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)

def checkpoint(path, identity, compute):
    path = Path(path)
    if path.is_file():
        saved = read_json(path)
        if saved['identity'] != identity or saved['sha256'] != canonical_hash(saved['record']):
            raise ValueError('Changed receiver checkpoint; preserve for inspection: ' + str(path))
        return saved['record'], True
    value = compute()
    write_json(path, {'identity': identity, 'record': value, 'sha256': canonical_hash(value)})
    return value, False

def verify_step2(directory):
    directory = Path(directory).resolve()
    report = read_json(directory/'step2_report.json')
    if not report.get('stage2_preparation_complete') or report.get('status') != 'prepared_for_runner_implementation':
        raise ValueError('The selected Step 2 workspace is not complete')
    checks = report.get('control_checks', {})
    if not checks.get('passed') or checks.get('tests_run') != 24 or checks.get('failures') or checks.get('errors') or checks.get('skipped'):
        raise ValueError('Step 2 control checks did not pass completely')
    protocol = read_json(directory/'configs/revision_protocol.json')
    if canonical_hash(protocol) != PROTOCOL_SHA256 or report.get('protocol_sha256') != PROTOCOL_SHA256:
        raise ValueError('The frozen scientific protocol changed')
    manifest = read_json(directory/'snapshot_manifest.json')
    files = manifest.get('files')
    if not isinstance(files, list) or not files:
        raise ValueError('Missing snapshot file list')
    if manifest.get('protocol_sha256') != PROTOCOL_SHA256 or manifest.get('snapshot_hash') != canonical_hash(files):
        raise ValueError('Snapshot manifest identity changed')
    if len(files) != report.get('frozen_source_and_input_files'):
        raise ValueError('Snapshot coverage differs from its preparation report')
    destinations = [row['destination'] for row in files]
    if len(destinations) != len(set(destinations)):
        raise ValueError('Duplicate snapshot destinations')
    for row in files:
        path = inside(directory, row['destination'])
        if not path.is_file() or path.stat().st_size != row['bytes'] or file_hash(path) != row['sha256']:
            raise ValueError('A frozen input differs from Step 2: ' + row['destination'])
    source_files = [p for p in destinations if p.startswith('src/transport_topics/') and p.endswith('.py')]
    actual = {p.relative_to(directory).as_posix() for p in (directory/'src/transport_topics').rglob('*.py')}
    if set(source_files) != actual or len(source_files) != 96:
        raise ValueError('Scientific source inventory differs from the accepted snapshot')
    collections = read_json(directory/'provenance/collections.json')
    if len(collections) != 2 or {c['corpus'] for c in collections} != {'transport_grievance', 'chinese_pooled'}:
        raise ValueError('Exactly the two declared full corpora are required')
    return report, protocol, collections, manifest

def resolve_step2(root, explicit=None):
    if explicit is not None:
        path = Path(explicit)
        return path.resolve() if path.is_absolute() else (Path(root)/path).resolve()
    candidates = sorted((Path(root)/'outputs/revision_study/step2').glob('*/step2_report.json'), reverse=True)
    for path in candidates:
        data = read_json(path)
        if data.get('stage2_preparation_complete'): return path.parent.resolve()
    raise FileNotFoundError('No completed Step 2 workspace was found')

def save_receipt(directory):
    directory = Path(directory)
    files = {p.relative_to(directory).as_posix(): file_hash(p) for p in sorted(directory.rglob('*'))
             if p.is_file() and p.name != 'complete.json' and '.partial' not in p.name}
    write_json(directory/'complete.json', {'files': files})

def complete(directory):
    directory = Path(directory); p = directory/'complete.json'
    if not p.is_file(): return False
    for relative, digest in read_json(p)['files'].items():
        target = inside(directory, relative)
        if not target.is_file() or file_hash(target) != digest:
            raise ValueError('Changed completed artifact: ' + str(target))
    return True

def acquire_run_lock(path):
    """OS-held lock: released even if Python or the computer stops abruptly."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    stream=path.open('a+b')
    if stream.seek(0,2)==0:stream.write(b'0');stream.flush()
    stream.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError as error:
        stream.close()
        raise RuntimeError('Another process is using this revision workspace; wait for it to finish.') from error
    return stream

def release_run_lock(stream):
    if stream is None:return
    stream.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
    finally:stream.close()
