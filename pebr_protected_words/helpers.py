"""Resolve and verify the completed default-bank runner before importing it."""
import hashlib
import json
from pathlib import Path
import sys

ACCEPTED_STEP3='20261101T095053677062Z_d25536e042_b3f523_8f044d9c431e'

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

