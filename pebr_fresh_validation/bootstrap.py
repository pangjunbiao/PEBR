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

def resolve_and_bind(project,explicit=None):
    project=Path(project).resolve()
    p=Path(explicit) if explicit is not None else Path('outputs/revision_study/step3')/ACCEPTED_STEP3
    p=p.resolve() if p.is_absolute() else (project/p).resolve()
    report=read_json(p/'step3_report.json');identity=read_json(p/'identity.json')
    if (report.get('status')!='default_bank_comparison_complete' or report.get('completed_final_fits_available')!=300
        or not report.get('step2_snapshot_reverified_after_run') or not report.get('reproduction',{}).get('passed')):
        raise ValueError('A complete verified Step 3 comparison with 300 fits is required')
    if report['runner_identity']!=canonical_hash(identity):raise ValueError('Step 3 runner identity differs')
    source=p/'runner_source'
    for name,h in identity['runner_source'].items():
        file=source/name if name=='pebr_step3_run.py' else source/'pebr_revision_step3'/name
        if not file.is_file() or digest(file)!=h:raise ValueError('Frozen Step 3 source changed: '+name)
    actual={x.name for x in (source/'pebr_revision_step3').glob('*.py')}
    expected=set(identity['runner_source'])-{'pebr_step3_run.py'}
    if actual!=expected:raise ValueError('Unexpected files in the frozen Step 3 source')
    sys.path.insert(0,str(source))
    from pebr_revision_step3.runner import bind_frozen_source
    step2=Path(report['step2_workspace'])
    if not step2.is_dir():
        step2=project/'outputs/revision_study/step2'/p.name.rsplit('_',1)[0]
    step2=step2.resolve()
    from pebr_revision_step3.io import verify_step2
    accepted,protocol,descriptions,manifest=verify_step2(step2)
    if manifest['snapshot_hash']!=identity['snapshot']:raise ValueError('Step 3 used a different input snapshot')
    bind_frozen_source(step2)
    return p,step2,report,identity,protocol,descriptions,manifest
