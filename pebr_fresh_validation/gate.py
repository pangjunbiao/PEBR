"""Verify the already completed revised-candidate replay without repeating fitting."""
from pathlib import Path
from .bootstrap import read_json, digest

def verify_candidate(project, explicit, accepted, descriptions, step2):
    candidates = [Path(explicit)] if explicit else sorted((project/'outputs/revision_rule_lab/revised_fixed_cap').glob('*'))
    expected = read_json(accepted/'revised_candidate_seed_results.json')
    expected = {(r['corpus'], r['K'], r['seed']): r for r in expected}
    wanted_identity = read_json(accepted/'trial_identity.json')
    failures = []
    for candidate in candidates:
        root = candidate.resolve() if candidate.is_absolute() else (project/candidate).resolve()
        try:
            if not (root/'candidate_replay/CANDIDATE_COMPLETE.json').is_file():
                continue
            marker = read_json(root/'candidate_replay/CANDIDATE_COMPLETE.json')
            if marker['kernel_sha256'] != 'ebc4ad3cacb5a0950bbd2e524ddab73de1f49eb085216b41a3a15ca3ed4df381':
                raise ValueError('Candidate selection kernel changed')
            if read_json(root/'candidate_replay/trial_identity.json') != wanted_identity:
                raise ValueError('Candidate input/source/environment identity changed')
            baseline = read_json(root/'run_v1/baseline_reproduction.json')
            if not baseline.get('passed') or len(baseline.get('checks', [])) != 28 or not all(x['passed'] for x in baseline['checks']):
                raise ValueError('The 28-check baseline replay did not pass')
            for rec in read_json(accepted/'candidate_code_manifest.json'):
                path = root/Path(rec['path']).relative_to('engine')
                if digest(path) != rec['sha256']:
                    raise ValueError('Candidate scientific code changed: '+str(path))
            if digest(root/'inputs/export_manifest.json') != (accepted/'EXPECTED_INPUT_MANIFEST_SHA256.txt').read_text().strip():
                raise ValueError('The accepted numerical export changed')
            manifest = read_json(root/'inputs/export_manifest.json')
            files = manifest['files']
            for rec in files:
                # Exporter uses destination paths relative to the export root.
                relative = rec.get('member', rec.get('path', rec.get('destination')))
                if relative is None:
                    raise ValueError('Unsupported numerical-export manifest entry')
                target = (root/'inputs'/relative).resolve()
                target.relative_to((root/'inputs').resolve())
                if digest(target) != rec['sha256']:
                    raise ValueError('A candidate numerical input changed: '+relative)
            maximum = 0.
            keys = set()
            for path in (root/'candidate_replay/fits').glob('*/*/preserve_forward_fixed_cap/seed*/result.json'):
                row = read_json(path)
                key = (row['corpus'], int(row['K']), int(row['seed']))
                if key in keys or key not in expected:
                    raise ValueError('Unexpected or duplicated candidate fit')
                keys.add(key)
                quality, audit = read_json(path.parent/'quality.json'), read_json(path.parent/'audit.json')
                if not row['fit_converged'] or not audit['fit']['converged']:
                    raise ValueError('Candidate fit is not converged')
                if row['top_words'] != expected[key]['top_words'] or row['top_words'] != quality['top_words']:
                    raise ValueError('Candidate top-word extraction differs')
                for metric in ('npmi', 'cv', 'diversity'):
                    error = abs(row[metric]-expected[key][metric])
                    maximum = max(maximum, error)
                    if error > 2e-8 or abs(row[metric]-quality[metric]) > 1e-12:
                        raise ValueError('Candidate topic metrics differ')
            if keys != set(expected):
                raise ValueError('Expected all 30 completed revised fits')
            posts = 0
            for corpus in wanted_identity['corpora']:
                desc = next(d for d in descriptions if d['corpus'] == corpus)
                prepared = step2/desc['relative_directory']
                for name in ('counts.npz', 'vocabulary.npy', 'folds.npy', 'families.npy', 'source_rows.npy', 'embeddings.npy'):
                    if digest(root/'inputs/data'/corpus/name) != digest(prepared/name):
                        raise ValueError('Candidate and Step 2 original inputs differ: '+corpus+'/'+name)
                for k in wanted_identity['topics']:
                    path = root/'candidate_replay/selection'/corpus/('K%d'%k)/'actions.json'
                    actions = read_json(path)
                    reference = read_json(accepted/'expected_actions'/corpus/('K%d.json'%k))
                    if len(actions) != len(reference):
                        raise ValueError('Candidate action-bank size differs')
                    for a, b in zip(actions, reference):
                        for field in ('donors', 'offered_beta', 'actual_beta', 'available_donors'):
                            if a[field] != b[field]:
                                raise ValueError('Candidate selected action changed')
                    posts += len(actions)
            return root, {'passed': True, 'fit_records': 30, 'baseline_checks': 28,
                          'selected_actions_checked': posts, 'maximum_metric_error': maximum,
                          'candidate_input_files_verified': len(files),
                          'dictionary_membership_arrays_inspected': False,
                          'independent_selection_repeats_in_this_replay': 0}
        except (ValueError, KeyError, OSError) as error:
            failures.append(str(root)+': '+str(error))
    raise ValueError('No unchanged accepted revised-candidate replay was found. '+ '; '.join(failures))
