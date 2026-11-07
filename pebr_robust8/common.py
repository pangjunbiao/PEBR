"""Small I/O helpers; all writes are restricted to the new review workspace."""
import csv
import hashlib
import json
from pathlib import Path
import traceback

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()

def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def inside(root, relative):
    path = (Path(root)/relative).resolve()
    path.relative_to(Path(root).resolve())
    return path

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def table(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

class Findings:
    def __init__(self, work, logger):
        self.work, self.logger, self.rows = work, logger, []

    def add(self, status, check, detail, **extra):
        self.rows.append(dict(status=status, check=check, detail=detail, **extra))
        write(self.work/'findings.json', self.rows)
        if status != 'PASS':
            self.logger.info('%s | %s | %s', status, check, detail)

    def failure(self, check, error, **extra):
        status = 'INCOMPLETE' if isinstance(error, (FileNotFoundError, ModuleNotFoundError)) else 'FAIL'
        self.add(status, check, type(error).__name__+': '+str(error), traceback=traceback.format_exc(), **extra)

    def counts(self):
        return {s: sum(r['status'] == s for r in self.rows) for s in ('PASS', 'WARN', 'INCOMPLETE', 'FAIL')}
