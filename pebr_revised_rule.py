"""Inspect or reproduce the focused receiver-preserving PEBR revision.

Extract this file and pebr_revised_rule/ beside the existing main.py.
Existing source, configurations, preparation and experiment outputs are kept.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import subprocess
import sys
import zipfile

sys.dont_write_bytecode=True


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def find_export(project,bundle,explicit):
    candidates=[explicit] if explicit else sorted((project/'outputs/revision_rule_lab/exports').glob('*/PEBR_focused_rule_inputs.zip'),reverse=True)
    expected=(bundle/'EXPECTED_INPUT_MANIFEST_SHA256.txt').read_text().strip()
    for path in candidates:
        path=Path(path)
        if not path.is_absolute():path=project/path
        if not path.is_file():continue
        with zipfile.ZipFile(path) as z:
            if hashlib.sha256(z.read('export_manifest.json')).hexdigest()==expected:return path.resolve()
    raise FileNotFoundError('The accepted PEBR_focused_rule_inputs.zip was not found; provide --inputs with its path. No training started.')


def prepare(project,bundle,archive):
    manifest=read(bundle/'CODE_MANIFEST.json')
    for r in manifest:
        if sha(bundle/r['path'])!=r['sha256']:raise ValueError('Bundled code changed: '+r['path'])
    identity=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:12]
    workspace=project/'outputs/revision_rule_lab/revised_fixed_cap'/identity
    workspace.mkdir(parents=True,exist_ok=True)
    lock=workspace/'RUNNING.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump({'pid':os.getpid(),'python':sys.executable},f)
    try:
        for r in manifest:
            relative=Path(r['path']).relative_to('engine')
            target=workspace/relative;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and sha(target)!=r['sha256']:raise ValueError('Workspace code changed: '+str(target))
            if not target.exists():shutil.copyfile(bundle/r['path'],target)
        with zipfile.ZipFile(archive) as z:
            supplied=json.loads(z.read('export_manifest.json'))
            if len(supplied['files'])!=389:raise ValueError('Unexpected input member count')
            for r in supplied['files']:
                rel=PurePosixPath(r['member'])
                if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe archive member')
                data=z.read(r['member'])
                if len(data)!=r['bytes'] or hashlib.sha256(data).hexdigest()!=r['sha256']:raise ValueError('Input content differs: '+r['member'])
                target=workspace/'inputs'/Path(*rel.parts);target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():
                    if sha(target)!=r['sha256']:raise ValueError('Existing input changed: '+str(target))
                else:target.write_bytes(data)
            (workspace/'inputs/export_manifest.json').write_bytes(z.read('export_manifest.json'))
        return workspace,lock
    except Exception:
        lock.unlink(missing_ok=True);raise


def compare_candidate(workspace,bundle):
    expected=read(bundle/'reports/revised_candidate_seed_results.json')
    for e in expected:
        p=workspace/'candidate_replay/fits'/e['corpus']/('K%d'%e['K'])/'preserve_forward_fixed_cap'/('seed%d'%e['seed'])/'result.json'
        a=read(p)
        if a['top_words']!=e['top_words'] or max(abs(a[m]-e[m]) for m in ('npmi','cv','diversity'))>2e-6:
            raise ArithmeticError('Candidate replay differs: '+str(p))
    for corpus in ('transport_grievance','chinese_pooled'):
        for k in (8,10,12):
            a=read(workspace/'candidate_replay/selection'/corpus/('K%d'%k)/'actions.json')
            b=read(bundle/'reports/expected_actions'/corpus/('K%d.json'%k))
            if len(a)!=len(b):raise ValueError('Candidate action count differs')
            for x,y in zip(a,b):
                if x['donors']!=y['donors'] or x['available_donors']!=y['available_donors'] or max(abs(x[t]-y[t]) for t in ('offered_beta','actual_beta'))>1e-12:
                    raise ArithmeticError('Candidate selected actions differ')
    print('REPLAY VERIFIED | 30 revised fits and every selected action match the supplied results.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path('.'))
    parser.add_argument('--mode',choices=['results','candidate','full'],default='results')
    parser.add_argument('--inputs',type=Path,default=None)
    parser.add_argument('--dependencies',type=Path,default=None,help='Optional isolated numerical packages; never installs packages.')
    args=parser.parse_args();bundle=Path(__file__).resolve().parent/'pebr_revised_rule'
    if args.mode=='results':
        print((bundle/'reports/RESULTS_READ_FIRST.md').read_text(encoding='utf-8'))
        print('\nSUPPLIED MODELS:',bundle/'models');return 0
    project=args.project_root.resolve()
    if not (project/'main.py').is_file():raise ValueError('Run beside the existing main.py')
    archive=find_export(project,bundle,args.inputs)
    workspace,lock=prepare(project,bundle,archive)
    env=dict(os.environ)
    if args.dependencies:
        env['PYTHONPATH']=str(args.dependencies.resolve())+os.pathsep+env.get('PYTHONPATH','')
    def run(script,*arguments):
        subprocess.run([sys.executable,'-B',str(workspace/script),*arguments],cwd=workspace,env=env,check=True)
    print('FOCUSED REVISION WORKSPACE:',workspace,flush=True)
    try:
        run('run_trial.py','--phase','reproduce')
        if args.mode=='candidate':
            run('run_candidate.py');compare_candidate(workspace,bundle)
            run('export_models.py','--source-run','candidate_replay')
        else:
            run('run_trial.py','--phase','trial')
            run('run_forward_controls.py');run('run_fixed_controls.py');run('run_prefix_check.py')
            run('run_candidate.py','--verify-existing');run('write_results.py')
            run('export_models.py')
        print('COMPLETE | Original project source/configurations/experiments were not replaced.',flush=True)
        print('RESULTS DIRECTORY:',workspace,flush=True)
        inspection=workspace/'PEBR_revised_candidate_inspection.zip'
        with zipfile.ZipFile(inspection,'w',zipfile.ZIP_DEFLATED) as z:
            scopes=['candidate_replay','run_v1/baseline_reproduction.json','run_v1/trial_identity.json']
            if args.mode=='full':scopes+=['final_report']
            for scope in scopes:
                path=workspace/scope
                files=[path] if path.is_file() else path.rglob('*')
                for f in files:
                    if f.is_file() and f.suffix in {'.json','.csv','.md'}:z.write(f,f.relative_to(workspace))
        print('SEND THIS INSPECTION ZIP:',inspection,flush=True)
    finally:lock.unlink(missing_ok=True)
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:
        print('REVISION BLOCKED | '+type(error).__name__+': '+str(error),flush=True)
        raise SystemExit(1)
