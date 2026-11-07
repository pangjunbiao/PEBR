"""Gate publication on a complete, checksummed study and independently reconcile exports."""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path

from ..runtime.artifacts import file_hash, safe_path
from ..runtime.checkpointing import validate_run_id
from .common import ALL_TOPIC_METHODS, RESULT_NAMES, avg_sd, numeric, one, read_csv

REQUIRED_JSON = ('status.json', 'evaluation_summary.json', 'model_integration.json', 'study_plan.json',
    'dataset_evaluation_profile.json', 'models_index.json', 'study_models_index.json',
    'study_evaluations_index.json', 'study_controls_index.json', 'source_fingerprints.json',
    'verification.json', 'environment.json', 'timing.json', 'coherence_references.json',
    'preflight_plan.json', 'preparation_dependency.json')


def resolve_study(root, run_id=None):
    root = Path(root)
    if run_id:
        path = safe_path(root, 'outputs/runs/' + validate_run_id(run_id))
        if not path.is_dir():
            raise FileNotFoundError('Requested source run does not exist: ' + run_id)
        return path
    for path in sorted((root / 'outputs/runs').glob('*'), reverse=True):
        status = path / 'status.json'
        if not status.is_file():
            continue
        record = json.loads(status.read_text(encoding='utf-8'))
        if record.get('stage') == 'full_study' and record.get('state') == 'completed' and record.get('requested_evaluation_complete'):
            return path
    raise FileNotFoundError('No completed full study is available for publication. Complete phase five before running reports.')


def _same(actual, expected, description):
    if expected is None:
        if actual not in (None, ''):
            raise ValueError('Unexpected uncertainty value: ' + description)
    elif not math.isclose(numeric(actual), numeric(expected), rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError('Saved aggregation differs from its constituent results: ' + description)


def validate_study(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    required = list(REQUIRED_JSON) + ['config.yaml'] + ['results/' + name for name in RESULT_NAMES]
    for name in required:
        path = safe_path(directory, name)
        expected = manifest['artifacts'].get(name)
        if not expected or not path.is_file() or file_hash(path) != expected['sha256'] or path.stat().st_size != expected['bytes']:
            raise ValueError('Source study artifact is missing or changed: ' + name)
    data = {name[:-5]: json.loads((directory / name).read_text(encoding='utf-8')) for name in REQUIRED_JSON}
    import yaml
    config = yaml.safe_load((directory / 'config.yaml').read_text(encoding='utf-8'))
    status, summary = data['status'], data['evaluation_summary']
    if status.get('state') != 'completed' or status.get('stage') != 'full_study' or not status.get('requested_evaluation_complete'):
        raise ValueError('Publication requires a successfully completed full-study source run')
    if not summary.get('complete_registered_study') or summary.get('development_only') or status.get('development_only'):
        raise ValueError('A smoke or partial study cannot be presented as the registered full experiment')
    verification = data['verification']
    if not verification.get('passed') or any(verification.get(key) for key in ('errors', 'failures', 'skipped')):
        raise ValueError('The source study did not pass all recorded implementation checks')
    if summary.get('settings_selected_from_results') or summary.get('proposed_rule_changed'):
        raise ValueError('The registered reporting route does not accept a result-selected protocol')
    tables = {name: read_csv(directory / 'results' / name) for name in RESULT_NAMES}
    for collection in data['dataset_evaluation_profile']['preparation']['conditions']:
        for bank in collection['banks'].values():
            if any(bank.get(k, 0) for k in ('inner_query_leaks', 'outer_query_leaks', 'donor_boundary_violations')):
                raise ValueError('A recorded information-boundary invariant failed')
    for name, rows in tables.items():
        if len(rows) != summary['result_files'][name]['rows']:
            raise ValueError('Exported CSV row count differs: ' + name)
    models = data['study_models_index']['models']
    if len({m['identity'] for m in models}) != summary['models_unique'] or len(models) != summary['models_unique']:
        raise ValueError('Unique model counts do not reconcile')
    if len(data['study_evaluations_index']['evaluations']) != summary['logical_evaluations']:
        raise ValueError('Logical evaluation counts do not reconcile')
    if data['dataset_evaluation_profile'].get('query_catalogue_depends_on_method') is not False:
        raise ValueError('Publication requires a fixed query catalogue shared by all methods')
    integration = data['model_integration']
    if len(data['models_index']['models']) != integration['planned_model_outputs'] or not integration.get('method_seed_condition_coverage_verified'):
        raise ValueError('The registered main model integration is incomplete')
    runtime = tables['runtime_and_fits.csv']
    if {r['model_identity'] for r in runtime} != {m['identity'] for m in models} or len(runtime) != len(models):
        raise ValueError('Runtime export does not cover each distinct model exactly once')
    nmf = [r for r in runtime if r['converged_nmf']]
    if any(r['converged_nmf'] != 'True' for r in nmf):
        raise ValueError('A recorded NMF fit did not converge')
    for r in runtime:
        if r['method'] in ('knntm', 'glocom') and int(r['native_epochs']) != config['model_settings'][r['method']]['epochs']:
            raise ValueError('A native baseline did not complete its fixed training budget')
    controls = data['study_controls_index']['banks']
    if any(c.get(k, 0) for c in controls for k in ('bounds_violations', 'matched_amount_count_violations', 'proposed_representation_changes')):
        raise ValueError('A saved transfer-control invariant failed')
    if any(int(r[k]) for r in tables['borrowing_diagnostics.csv'] for k in ('bounds_violations', 'matched_mass_count_violations')):
        raise ValueError('A borrowing diagnostic invariant failed')

    topics = tables['topic_seed_results.csv']
    for row in topics:
        if row['development_only'] != 'False' or int(row['seed']) not in config['seeds']:
            raise ValueError('Development results or an undeclared seed entered the topic export')
        for metric, low in (('npmi', -1), ('diversity', 0), ('cv', 0)):
            if not low - 1e-12 <= numeric(row[metric]) <= 1 + 1e-12:
                raise ValueError('Topic metric is outside its range: ' + metric)
    expected = {(condition, method, str(k), str(seed)) for condition in data['study_plan']['base_collections']
                for method in ALL_TOPIC_METHODS for k in config['topics']['counts'] for seed in config['seeds']}
    observed = [(r['condition'], r['method'], r['K'], r['seed']) for r in topics if r['study'] == 'main']
    if set(observed) != expected or len(observed) != len(expected):
        raise ValueError('The main topic benchmark is incomplete or has duplicate conditions')
    groups = defaultdict(list)
    for row in topics:
        if row['study'] == 'main':
            groups[(row['dataset'], row['method'], row['K'])].append(row)
    for (dataset, method, k), group in groups.items():
        saved = one(tables['topic_quality_summary.csv'], dataset=dataset, method=method, K=k)
        collections = defaultdict(list)
        for row in group:
            collections[row['condition']].append(row)
        for metric in ('npmi', 'diversity', 'cv'):
            sample_means = [avg_sd(r[metric] for r in sample)[0] for sample in collections.values()]
            _same(saved[metric + '_mean'], avg_sd(sample_means)[0], str((dataset, method, k, metric)))
            if dataset == 'primary':
                _same(saved[metric + '_seed_sd'], avg_sd(r[metric] for r in group)[1], 'primary seed SD')
            else:
                _same(saved[metric + '_sample_sd'], avg_sd(sample_means)[1], 'external sample SD')
                within = [avg_sd(r[metric] for r in sample)[1] for sample in collections.values()]
                _same(saved[metric + '_mean_within_sample_seed_sd'], avg_sd(within)[0], 'external within-sample seed SD')

    retrieval = tables['retrieval_seed_results.csv']
    categories = tables['retrieval_category_results.csv']
    by_model = defaultdict(list)
    for row in categories:
        by_model[(row['study'], row['job_id'], row['method'], row['K'], row['seed'], row['B'])].append(row)
    cohort = data['dataset_evaluation_profile']['query_categories']
    for row in retrieval:
        cat = by_model[(row['study'], row['job_id'], row['method'], row['K'], row['seed'], row['B'])]
        if {r['category']: int(r['category_queries']) for r in cat} != cohort or len(cat) != len(cohort):
            raise ValueError('Retrieval categories or query cohort differ across methods')
        if int(row['queries']) != sum(cohort.values()):
            raise ValueError('A retrieval output dropped queries')
        for short, column in (('ndcg', 'macro_ndcg'), ('recall', 'macro_recall')):
            if any(not 0 <= numeric(r[short]) <= 1 for r in cat):
                raise ValueError('Retrieval metric is outside [0,1]')
            _same(row[column], avg_sd(r[short] for r in cat)[0], 'category-macro retrieval')
    for row in tables['retrieval_summary.csv']:
        raw = [r for r in retrieval if r['study'] == 'main' and (r['method'], r['K'], r['B']) == (row['method'], row['K'], row['B'])]
        if len(raw) != int(row['model_outputs']):
            raise ValueError('Retrieval seed coverage differs from summary')
        for metric in ('macro_ndcg', 'macro_recall'):
            value, sd = avg_sd(r[metric] for r in raw)
            _same(row[metric + '_mean'], value, metric)
            _same(row[metric + '_seed_sd'], sd, metric + ' SD')
    main_k, main_b = str(config['topics']['main_count']), str(config['retrieval']['main_budget'])
    for row in tables['retrieval_contrasts.csv']:
        for key in ('difference', 'ci_low', 'ci_high'):
            numeric(row[key])
        if numeric(row['ci_low']) > numeric(row['ci_high']):
            raise ValueError('A saved uncertainty interval is reversed')
        source = tables['retrieval_summary.csv']
        prop = one(source, method='proposed', K=row['K'], B=row['B'])
        other_k = '' if row['comparator'] == 'minilm_retrieval' else row['K']
        other = one(source, method=row['comparator'], K=other_k, B=row['B'])
        metric = 'macro_' + row['metric'] + '_mean'
        _same(row['difference'], numeric(prop[metric]) - numeric(other[metric]), 'paired contrast point estimate')
    for row in tables['sensitivity_summary.csv']:
        raw = [r for r in tables['sensitivity_results.csv'] if (r['factor'], r['value']) == (row['factor'], row['value'])]
        for metric in ('npmi', 'diversity', 'cv', 'macro_ndcg', 'macro_recall'):
            value, sd = avg_sd(r[metric] for r in raw)
            _same(row[metric + '_mean'], value, 'sensitivity mean')
            _same(row[metric + '_seed_sd'], sd, 'sensitivity SD')
    size_original = {(r['job_id'], r['seed']): numeric(r['npmi']) for r in tables['corpus_size_results.csv'] if r['method'] == 'nmf_original'}
    for row in tables['corpus_size_deltas.csv']:
        _same(row['npmi_change_from_no_borrowing'], numeric(row['npmi']) - size_original[(row['job_id'], row['seed'])], 'paired corpus-size difference')
    for row in tables['corpus_size_summary.csv']:
        raw = [r for r in tables['corpus_size_deltas.csv'] if (r['dataset'], r['target_size'], r['method']) == (row['dataset'], row['target_size'], row['method'])]
        ladders = defaultdict(list)
        for r in raw:
            ladders[r['sample_index']].append(r['npmi_change_from_no_borrowing'])
        value, sd = avg_sd(avg_sd(v)[0] for v in ladders.values())
        _same(row['mean_npmi_change'], value, 'corpus-size mean')
        _same(row['ladder_mean_sd'], sd, 'corpus-size ladder SD')
    pairs = defaultdict(list)
    for row in tables['seed_stability_pairs.csv']:
        pairs[(row['study'], row['job_id'], row['method'], row['K'])].append(row['matched_jaccard'])
    for row in tables['seed_stability.csv']:
        values = pairs[(row['study'], row['job_id'], row['method'], row['K'])]
        if len(values) != int(row['seed_pairs']):
            raise ValueError('Seed-stability pair counts differ')
        _same(row['stability'], avg_sd(values)[0], 'seed stability')
    audit = {'passed': True, 'source_run_id': manifest['run_id'], 'source_manifest_sha256': file_hash(directory / 'manifest.json'),
             'checksummed_input_artifacts': len(required), 'verified_implementation_tests': verification['tests_run'],
             'unique_models': len(models), 'logical_evaluations': summary['logical_evaluations'],
             'converged_final_nmf_fits': len(nmf), 'cuda_neural_fits': sum(r['device'].startswith('cuda') for r in runtime),
             'control_banks': len(controls), 'borrowing_invariant_violations': 0,
             'recorded_information_boundary_violations': 0,
             'independent_aggregate_reconciliation': True, 'source_evaluations_recomputed': False,
             'main_K': int(main_k), 'main_B': int(main_b),
             'scientific_superiority_is_not_a_completion_check': True}
    return {'directory': directory, 'data': data, 'tables': tables, 'config': config, 'audit': audit}
