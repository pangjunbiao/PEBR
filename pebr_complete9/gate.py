"""Accept the inspected complete Step 8, then bind its frozen predecessors."""
from pathlib import Path
import sys
from .common import read,digest,key
def bind(project,explicit,package):
    accepted=read(package/'s9_accepted_step8.json');errors=[]
    choices=[Path(explicit)] if explicit else sorted((project/'outputs/revision_rule_lab/robustness8').glob('*'))
    for item in choices:
        root=item.resolve() if item.is_absolute() else (project/item).resolve()
        if not (root/'step8_report.json').is_file():continue
        try:
            report,identity=read(root/'step8_report.json'),read(root/'identity.json')
            if report.get('status')!='reference_fold_sensitivity_complete' or not report.get('step8_complete'):continue
            if (key(identity)!=report['runner_identity'] or identity['runner_source']!=accepted['source_manifest'] or
                identity['protocol']!=accepted['protocol'] or not report.get('default_selector_reproduction_passed') or
                not report.get('frozen_inputs_reverified_after_run')):raise ValueError('Step 8 scientific identity or completion differs')
            for name,sha in accepted['tables_sha256'].items():
                if digest(root/name)!=sha:raise ValueError('An inspected Step 8 table changed: '+name)
            source=root/'runner_source'
            for name,sha in accepted['source_manifest'].items():
                if digest(source/name)!=sha:raise ValueError('Frozen Step 8 source changed: '+name)
            sys.path.insert(0,str(source))
            import pebr_robust8
            Path(pebr_robust8.__file__).resolve().relative_to(source)
            from pebr_robust8.gate import bind as previous
            seven=Path(report['step7_workspace'])
            if not seven.is_dir():seven=project/'outputs/revision_rule_lab/review7'/('s7_'+identity['step7_identity'][:12])
            s7,s6,s5,s3,s2,fresh,descriptions,snapshot,checks=previous(project,seven,source)
            for name,sha in accepted.get('step7_inputs_sha256',{}).items():
                if digest(s7/name)!=sha:raise ValueError('Verified baseline/case metadata changed: '+name)
            return root,s7,s6,s5,s3,s2,fresh,descriptions,snapshot,{'step8_identity':report['runner_identity'],'previous':checks}
        except (ValueError,KeyError,OSError) as e:errors.append(str(root)+': '+str(e))
    raise ValueError('No unchanged inspected complete Step 8 found. '+'; '.join(errors))
