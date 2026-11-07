"""Export only the saved numerical inputs needed for a focused rule experiment.

No fitting, encoding, package installation or change to the existing method.
The resulting ZIP is for analysis; it is not a new or improved model.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path,PurePosixPath
import sys
import traceback
import uuid
import zipfile

sys.dont_write_bytecode=True
ACCEPTED_STEP3='20261101T095053677062Z_d25536e042_b3f523_8f044d9c431e'
PROTOCOL='d25536e04233a643862676601e5e441ac4f7c2f4982d98793df495d74b3ea8ec'

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(2**20),b''):h.update(data)
    return h.hexdigest()

def checked_member(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name:raise ValueError('Unsafe ZIP member')
    return p.as_posix()

class Export:
    def __init__(self,archive,project):
        self.z=archive;self.project=project;self.records=[];self.names=set()
    def file(self,path,member):
        path=Path(path).resolve();member=checked_member(member)
        if member in self.names:raise ValueError('Duplicate export member: '+member)
        try:source=path.relative_to(self.project).as_posix()
        except ValueError:raise ValueError('Export source escapes the project: '+str(path))
        if not path.is_file() or path.is_symlink():raise FileNotFoundError(str(path))
        h=hashlib.sha256();size=0
        with path.open('rb') as incoming,self.z.open(member,'w',force_zip64=True) as outgoing:
            for data in iter(lambda:incoming.read(2**20),b''):
                outgoing.write(data);h.update(data);size+=len(data)
        self.names.add(member);self.records.append({'member':member,'source':source,'bytes':size,'sha256':h.hexdigest()})
    def value(self,value,member):
        member=checked_member(member)
        if member in self.names:raise ValueError('Duplicate export member')
        data=json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf-8')
        self.z.writestr(member,data);self.names.add(member)
        self.records.append({'member':member,'source':'derived metadata or token-only reference',
                             'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})

def run(project,explicit=None):
    project=Path(project).resolve();folder=None
    if not (project/'main.py').is_file():raise ValueError('Run beside existing main.py')
    step3=Path(explicit) if explicit is not None else Path('outputs/revision_study/step3')/ACCEPTED_STEP3
    if not step3.is_absolute():step3=project/step3
    step3=step3.resolve();report=read_json(step3/'step3_report.json');identity=read_json(step3/'identity.json')
    if report.get('status')!='default_bank_comparison_complete' or report.get('completed_final_fits_available')!=300:
        raise ValueError('The completed 300-fit Step 3 comparison is required')
    if not report.get('reproduction',{}).get('passed') or canonical_hash(identity)!=report['runner_identity']:
        raise ValueError('Step 3 reproduction/identity check failed')
    source=step3/'runner_source'
    for name,digest in identity['runner_source'].items():
        path=source/name if name=='pebr_step3_run.py' else source/'pebr_revision_step3'/name
        if sha(path)!=digest:raise ValueError('Frozen runner source changed: '+name)
    sys.path.insert(0,str(source))
    from pebr_revision_step3.io import verify_step2,complete,inside
    step2=Path(report['step2_workspace'])
    if not step2.is_dir():step2=project/'outputs/revision_study/step2'/step3.name.rsplit('_',1)[0]
    step2=step2.resolve();accepted,protocol,collections,snapshot=verify_step2(step2)
    if protocol is None or canonical_hash(protocol)!=PROTOCOL or snapshot['snapshot_hash']!=identity['snapshot']:
        raise ValueError('The scientific protocol/snapshot differs')
    from pebr_revision_step3.runner import validate_environment,bind_frozen_source,load_collections
    validate_environment(read_json(step2/'environment.json'))
    bind_frozen_source(step2)
    aligned=load_collections(step2,collections,protocol)
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder=project/'outputs/revision_rule_lab/exports'/(stamp+'_'+uuid.uuid4().hex[:6])
    folder.mkdir(parents=True,exist_ok=False)
    archive=folder/'PEBR_focused_rule_inputs.zip';partial=folder/'PEBR_focused_rule_inputs.partial.zip'
    print('PEBR FOCUSED RULE INPUT EXPORT | no fitting, encoding or existing-file replacement',flush=True)
    print('Verified frozen Step 2/3 inputs; exporting selected numerical files only.',flush=True)
    collection_by_identity={c['summary']['identity_sha256']:c for c in collections}
    reference_coverage=set();scope_availability={}
    with zipfile.ZipFile(partial,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        writer=Export(z,project)
        writer.value(protocol,'protocol.json')
        writer.value(collections,'collections.json')
        for name in ('identity.json','step3_report.json','reproduction.json','phase_timings.csv'):
            writer.file(step3/name,'step3/'+name)
        for name in ('topic_seed_results.csv','multi_seed_summary.csv','paired_pebr_contrasts.csv',
                     'stability_pairs.csv','topic_words.json'):
            writer.file(step3/'results'/name,'step3/results/'+name)
        for name in ('environment.json','snapshot_manifest.json'):
            writer.file(step2/name,'provenance/'+Path(name).name)
        for item in collections:
            corpus=item['corpus'];summary=item['summary'];directory=inside(step2,item['relative_directory'])
            print('Collection | %s | %s posts, %s terms'%(corpus,summary['posts'],summary['vocabulary']),flush=True)
            for name in ('counts.npz','vocabulary.npy','folds.npy','families.npy','source_rows.npy','embeddings.npy','identity.json'):
                writer.file(directory/name,'data/'+corpus+'/'+name)
            tokens=aligned[corpus]['reference'].tokens
            if len(tokens)!=summary['posts']:raise ValueError('Prepared token-reference count differs')
            writer.value(tokens,'data/'+corpus+'/tokens.json')
            writer.value(summary,'data/'+corpus+'/summary.json')
            scope_availability[corpus]=[]
            for scope in ('topic_learning','outer_audit'):
                bank=directory/scope
                if not bank.is_dir():continue
                if not complete(bank):raise ValueError('A prepared evidence bank is incomplete')
                for name in ('base.npz','visible.npz','targets.npz','eligible.npy','candidates.npy',
                             'full_candidates.npy','full_receiver_embeddings.npy'):
                    writer.file(bank/name,'data/'+corpus+'/'+scope+'/'+name)
                if scope=='outer_audit':writer.file(bank/'outer.npz','data/'+corpus+'/'+scope+'/outer.npz')
                scope_availability[corpus].append(scope)
            for k in protocol['topic_counts']:
                parent=step3/'selection'/corpus/('K%d'%k)
                for policy in protocol['policies']:
                    if policy=='nmf_original':continue
                    if not complete(parent/policy):raise ValueError('A Step 3 policy bank is incomplete')
                    for name in ('actions.json','summary.json'):
                        writer.file(parent/policy/name,'actions/'+corpus+'/K%d/'%k+policy+'/'+name)
                for policy in ('nmf_original','proposed'):
                    fitted=step3/'fits'/corpus/('K%d'%k)/policy/'seed0'
                    if not complete(fitted):raise ValueError('A seed-0 numerical baseline is incomplete')
                    for name in ('topic_scores.npy','quality.json','audit.json','result.json'):
                        writer.file(fitted/name,'baseline_seed0/'+corpus+'/K%d/'%k+policy+'/'+name)
        desired={'folds':5,'seed':0,'tolerance':.0001,'max_iterations':2000,'smoothing':.1,'unigram_smoothing':.1}
        for directory in sorted((step3/'cache/predictors').iterdir()):
            if not directory.is_dir() or not (directory/'identity.json').is_file():continue
            spec=read_json(directory/'identity.json');condition=spec.get('condition')
            if condition not in collection_by_identity or spec.get('topics') not in protocol['topic_counts'] or spec.get('settings')!=desired:continue
            corpus=collection_by_identity[condition]['corpus'];k=int(spec['topics']);fold=int(spec['excluded_fold'])
            key=(corpus,k,fold)
            if key in reference_coverage or fold not in range(5) or not complete(directory):raise ValueError('Duplicate/incomplete frozen reference')
            reference_coverage.add(key)
            for name in ('dictionary.npy','idf.npy','prior.npy','training_indices.npy','identity.json','fit.json'):
                writer.file(directory/name,'predictors/'+corpus+'/K%d/fold%d/'%(k,fold)+name)
        expected={(c,k,f) for c in protocol['corpora'] for k in protocol['topic_counts'] for f in range(5)}
        if reference_coverage!=expected:raise ValueError('Expected all 30 fixed predictive references')
        manifest={'schema':'pebr_focused_rule_inputs_v1','purpose':'offline development/diagnosis of a candidate rule; not final validation',
            'training_performed':False,'encoder_loaded':False,'project_files_replaced':False,
            'step3_run':step3.name,'protocol_sha256':PROTOCOL,'reference_predictors':30,
            'scope_availability':scope_availability,'token_reference_policy':'token lists only; no original raw-text fields or service labels',
            'large_pretrained_assets_included':False,'source_code_included':False,
            'files':writer.records}
        z.writestr('export_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(partial) as z:
        if z.testzip() is not None:raise IOError('Export ZIP integrity check failed')
    partial.replace(archive)
    verify_step2(step2)
    print('EXPORT COMPLETE | files=%d | %.2f MiB | no model fitting'%(len(writer.records),archive.stat().st_size/2**20),flush=True)
    print('SEND THIS ZIP: '+str(archive),flush=True)
    return archive

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path('.'))
    parser.add_argument('--step3-workspace',type=Path,default=None)
    args=parser.parse_args()
    try:run(args.project_root,args.step3_workspace);return 0
    except Exception as error:
        print('EXPORT BLOCKED | '+type(error).__name__+': '+str(error),flush=True)
        traceback.print_exc();return 1

if __name__=='__main__':raise SystemExit(main())
