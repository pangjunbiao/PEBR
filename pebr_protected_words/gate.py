"""Bind the verified Step 5 source and its unchanged Step 2/3 numerical inputs."""
import sys
from pathlib import Path
from .helpers import read_json, digest, canonical_hash

def bind(project, explicit, package):
    expected = read_json(package/'accepted_step5_source_manifest.json')
    paths = [Path(explicit)] if explicit else sorted((project/'outputs/revision_rule_lab/fresh_selection').glob('*'))
    failures = []
    for value in paths:
        root = value.resolve() if value.is_absolute() else (project/value).resolve()
        try:
            if not (root/'validation_report.json').is_file():
                continue
            report = read_json(root/'validation_report.json')
            if report.get('status') != 'fresh_selection_validation_complete':
                continue
            identity = read_json(root/'identity.json')
            if identity['protocol'] != read_json(package/'accepted_step5_protocol.json'):
                raise ValueError('The completed selection study uses a different protocol')
            if not report.get('selection_validation_complete') or report.get('total_unique_fit_records') != 1230:
                raise ValueError('Step 5 fitted-seed coverage is incomplete')
            if identity['source_manifest'] != expected or report['runner_identity'] != canonical_hash(identity):
                raise ValueError('The completed selection study uses a different scientific version')
            if not report.get('frozen_inputs_reverified_after_run') or report.get('development_bank_in_validation'):
                raise ValueError('Step 5 frozen-input/fresh-mask checks did not pass')
            if read_json(root/'results/coverage.json') != {'unique_fit_records': 1230, 'expected': 1230, 'complete': True}:
                raise ValueError('Step 5 numerical coverage changed')
            for relative, wanted_hash in read_json(package/'accepted_step5_tables_sha256.json').items():
                if digest(root/relative) != wanted_hash:
                    raise ValueError('An inspected Step 5 numerical table changed: '+relative)
            source = root/'runner_source'
            for relative, wanted in expected.items():
                if digest(source/relative) != wanted:
                    raise ValueError('Frozen Step 5 source changed: '+relative)
            rows = read_json(root/'results/topic_words.json')
            protocol = identity['protocol']
            keys = {(r['corpus'], int(r['K']), r['policy'], str(r['mask_base']), int(r['seed'])) for r in rows}
            wanted = {(c, k, p, 'not_applicable' if p == 'nmf_original' else str(m), s)
                      for c in protocol['corpora'] for k in protocol['topic_counts'] for p in protocol['policies']
                      for m in (['not_applicable'] if p == 'nmf_original' else protocol['selection_mask_bases'])
                      for s in protocol['final_model_seeds']}
            if len(rows) != 1230 or keys != wanted or not all(r['fit_converged'] for r in rows):
                raise ValueError('Step 5 seed records are incomplete or invalid')
            # Import from its frozen copy, never a mutable project-root copy.
            sys.path.insert(0, str(source))
            import pebr_fresh_validation
            Path(pebr_fresh_validation.__file__).resolve().relative_to(source.resolve())
            from pebr_fresh_validation.bootstrap import resolve_and_bind
            step3_path = Path(report['step3_workspace'])
            if not step3_path.is_dir():
                step3_path = project/'outputs/revision_study/step3'/root.name.rsplit('_', 1)[0]
            step3, step2, prior_report, prior_identity, old_protocol, descriptions, manifest = resolve_and_bind(project, step3_path)
            if identity['step2_snapshot'] != manifest['snapshot_hash']:
                raise ValueError('Step 5 and the protected audit use different prepared inputs')
            return root, step3, step2, protocol, descriptions, manifest, {
                'passed': True, 'step5_unique_fit_records': 1230,
                'source_files_verified': len(expected), 'step5_identity': report['runner_identity'],
                'model_arrays_reopened': False,
                'scope': 'Stage completion, source version, fitted-seed records and frozen scientific inputs'}
        except (ValueError, KeyError, OSError) as error:
            failures.append(str(root)+': '+str(error))
    raise ValueError('No complete unchanged accepted Step 5 workspace found. '+ '; '.join(failures))

def audit_protocol(fresh):
    import copy
    p = copy.deepcopy(fresh)
    p.update(stage='repeated_protected_word_audit',
             outer_target_bases=[2712, 3712, 4712, 5712, 6712],
             known_outer_target_base=2712,
             fresh_outer_target_bases=[3712, 4712, 5712, 6712],
             inner_mask_base=2711, selection_mask_bases=[2711],
             outer_type_fraction=0.2, outer_minimum_types=1,
             loss_threshold=1e-10, new_final_fits=0,
             audit_configurations=30, policy_configuration_summaries=330,
             targets_opened_after_configuration_actions_saved=True,
             tuning_from_audit_outcomes=False,
             zero_transfer_queries_retained_in_common_cohort=True,
             inferential_scope='Prediction on protected receiver word types with fixed fold-excluded predictors; not final-model or new-corpus performance.')
    p.pop('shared_no_borrowing_fits', None)
    p.pop('reused_no_borrowing_fits', None)
    p.pop('fresh_mask_results_not_known_at_freeze', None)
    p.pop('final_model_seeds', None)
    p['aggregation'] = {'unit': 'Mean per-post average log-probability gain within each protected bank',
                        'outer_repeats': 'PRIMARY: mean/SD across four fresh protected banks; SECONDARY: all five including the previously used bank 2712',
                        'fitting_seeds': 'No final topic models are fitted; final-model seeds are not audit replicates',
                        'paired': 'Same target words, query cohort, folds, predictor and permitted donor catalogue for every policy',
                        'uncertainty': 'Overlapping word holdouts on the same corpus; SD is not a confidence interval'}
    return p
