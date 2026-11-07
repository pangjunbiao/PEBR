"""Accept only the inspected, complete Step 6 numerical and scientific version."""
from pathlib import Path
import sys
from .common import read, digest, key

def bind(project, explicit, package):
    accepted = read(package/'s7_accepted_step6.json')
    candidates = [Path(explicit)] if explicit else sorted(
        (project/'outputs/revision_rule_lab/protected_words').glob('*'))
    errors = []
    for item in candidates:
        root = item.resolve() if item.is_absolute() else (project/item).resolve()
        if not (root/'step6_report.json').is_file():
            continue
        try:
            report, identity = read(root/'step6_report.json'), read(root/'identity.json')
            if not report.get('protected_word_study_complete') or report.get('status') != 'protected_word_audit_complete':
                continue
            if (report['runner_identity'] != key(identity) or
                identity['protocol'] != accepted['protocol'] or
                identity['runner_source'] != accepted['source_manifest']):
                raise ValueError('The protected study uses another scientific version')
            if not report.get('frozen_inputs_reverified_after_run') or report['completed_audit_configurations'] != 30:
                raise ValueError('Protected study completion checks are missing')
            for relative, sha in accepted['tables_sha256'].items():
                if digest(root/relative) != sha:
                    raise ValueError('An inspected protected table changed: '+relative)
            source = root/'runner_source'
            for relative, sha in accepted['source_manifest'].items():
                if digest(source/relative) != sha:
                    raise ValueError('The protected study source changed: '+relative)
            sys.path.insert(0, str(source))
            import pebr_protected_words
            Path(pebr_protected_words.__file__).resolve().relative_to(source)
            from pebr_protected_words.gate import bind as bind_step5
            p5 = Path(report['step5_workspace'])
            if not p5.is_dir():
                # Relocated copies can be resolved by their preserved basename.
                name = str(report['step5_workspace']).replace('\\', '/').rstrip('/').split('/')[-1]
                p5 = project/'outputs/revision_rule_lab/fresh_selection'/name
            step5, step3, step2, protocol, descriptions, snapshot, checks = bind_step5(project, p5, source)
            return root, step5, step3, step2, protocol, descriptions, snapshot, {
                'step6_identity': report['runner_identity'], 'step6_tables_verified': len(accepted['tables_sha256']),
                'step6_source_files_verified': len(accepted['source_manifest']), 'step5': checks}
        except (OSError, ValueError, KeyError) as error:
            errors.append(str(root)+': '+str(error))
    raise ValueError('No unchanged accepted complete Step 6 workspace found. '+'; '.join(errors))
