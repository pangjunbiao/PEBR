"""Bind the inspected completed Step 7 and its frozen numerical lineage."""
from pathlib import Path
import sys
from .common import read, digest, key

def bind(project, explicit, package):
    accepted = read(package/'s8_accepted_step7.json')
    choices = [Path(explicit)] if explicit else sorted((project/'outputs/revision_rule_lab/review7').glob('*'))
    errors = []
    for value in choices:
        root = value.resolve() if value.is_absolute() else (project/value).resolve()
        if not (root/'step7_report.json').is_file():
            continue
        try:
            report, identity = read(root/'step7_report.json'), read(root/'identity.json')
            if not report.get('step7_execution_complete') or report.get('status') != 'review_export_complete':
                continue
            if not report.get('baseline_fidelity_checks_complete') or report['findings']['FAIL'] or report['findings']['INCOMPLETE']:
                raise ValueError('The accepted baseline/case review is incomplete')
            if identity['runner_source'] != accepted['source_manifest']:
                raise ValueError('Step 7 used another scientific version')
            for relative, sha in accepted['tables_sha256'].items():
                if digest(root/relative) != sha:
                    raise ValueError('An inspected Step 7 numerical table changed: '+relative)
            source = root/'runner_source'
            for relative, sha in accepted['source_manifest'].items():
                if digest(source/relative) != sha:
                    raise ValueError('The frozen Step 7 runner changed: '+relative)
            sys.path.insert(0, str(source))
            import pebr_review7
            Path(pebr_review7.__file__).resolve().relative_to(source)
            from pebr_review7.gate import bind as bind_previous
            six = Path(report['step6_workspace'])
            if not six.is_dir():
                six = project/'outputs/revision_rule_lab/protected_words'/('s6_'+identity['step6_identity'][:12])
            step6, step5, step3, step2, protocol, descriptions, snapshot, previous = bind_previous(project, six, source)
            if identity['snapshot_hash'] != snapshot['snapshot_hash'] or not report.get('frozen_inputs_reverified_after_run'):
                raise ValueError('Step 7 prepared-input lineage differs')
            return root, step6, step5, step3, step2, protocol, descriptions, snapshot, {
                'step7_identity': key(identity), 'step7_tables_verified': len(accepted['tables_sha256']),
                'step7_source_files_verified': len(accepted['source_manifest']), 'previous': previous}
        except (ValueError, KeyError, OSError) as error:
            errors.append(str(root)+': '+str(error))
    raise ValueError('No complete unchanged inspected Step 7 found. '+'; '.join(errors))
