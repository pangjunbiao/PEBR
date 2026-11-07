"""Inspect existing BTM/kNNTM/GloCOM outputs without training or cache creation."""
from collections import defaultdict
from pathlib import Path
import hashlib
import numpy as np
from .common import read, digest, key, inside, write, table, csv_rows
from .native_math import beta as replay_beta, theta as replay_theta

METHODS = ('btm', 'knntm', 'glocom')
COUNTS = (8, 12)  # Existing historical runs; K=10 is inventoried, never substituted.

def check(condition, message):
    if not condition:
        raise ValueError(message)

def compare(actual, expected, tolerance, description):
    check(actual.shape == expected.shape, description+' shape differs')
    error = float(np.max(np.abs(actual-expected)))
    check(np.isfinite(actual).all() and np.isfinite(expected).all() and error <= tolerance,
          description+' max absolute error '+str(error)+' exceeds '+str(tolerance))
    return error

def verify_receipt(directory):
    from transport_topics.data.preparation import check_receipt
    check(check_receipt(directory), 'Missing or changed model/cache receipt: '+str(directory))

def distance_checks(project, c, audit, state, vectors):
    """Read the completed OT cache; replay a small fixed source-order pair sample."""
    spec = audit['input_preparation']['transport']
    directory = project/'cache/baseline_inputs/knntm'/spec['identity'][:20]
    verify_receipt(directory)
    check(key(read(directory/'identity.json')) == spec['identity'], 'OT cache identity differs')
    n = c['collection'].counts.shape[0]
    matrices = []
    for file, tensor in (('M_cos', 'M_cos_dist'), ('M_coo', 'M_coo_dist')):
        with np.load(directory/(file+'.npz'), allow_pickle=False) as data:
            matrix = data['arr_0'].copy()
        check(matrix.shape == (n, n) and np.isfinite(matrix).all(), 'Invalid '+file+' matrix')
        check(np.max(np.abs(matrix-matrix.T)) < 2e-6 and np.max(np.abs(np.diag(matrix))) < 2e-6,
              'Asymmetric or nonzero-diagonal '+file)
        check(matrix.min() >= -2e-6 and matrix.max() <= 1.000002, 'Out-of-range transport distance')
        compare(matrix, state[tensor], 1e-7, 'Checkpoint/cache '+file)
        matrices.append(matrix)
    import ot
    counts = c['collection'].counts
    unit = vectors/np.linalg.norm(vectors, axis=1, keepdims=True)
    pairs = sorted({(0, 1), (n//3, n//2), (n//2, n-1), (0, n-1)})
    errors = []
    for i, j in pairs:
        a, b = counts[i], counts[j]
        aa, bb = a.indices, b.indices
        co = c['reference'].cooccurrence[aa][:, bb].toarray().astype(float)
        df = c['reference'].df
        cosine = np.clip((1-unit[aa].astype(float)@unit[bb].astype(float).T)/2, 0, 1)
        occurrence = np.clip(1-co*(1/df[aa, None]+1/df[None, bb])/2, 0, 1)
        cosine[aa[:, None] == bb[None, :]] = 0
        occurrence[aa[:, None] == bb[None, :]] = 0
        for index, cost in enumerate((cosine, occurrence)):
            value = ot.emd2(a.data/a.data.sum(), b.data/b.data.sum(), cost,
                            numItermax=100000, numThreads=1, check_marginals=True)
            errors.append(abs(float(value)-float(matrices[index][i, j])))
    check(max(errors) < 2e-6, 'Fixed-pair exact OT reconstruction differs')
    return {'checked_pairs': pairs, 'max_ot_error': max(errors), 'ot_version': ot.__version__,
            'all_cached_matrix_entries_checked_against_checkpoint': True}

class Audit:
    def __init__(self, project, step2, work, corpora, findings, logger):
        self.project, self.step2, self.work = project, step2, work
        self.corpora, self.findings, self.logger = corpora, findings, logger
        self.vector_cache, self.context_cache, self.distance_cache = {}, {}, {}
        self.metrics, self.records, self.losses, self.topics, self.diagnostics = [], [], [], [], []
        self.ledger = []
        self.historical = {}
        for path in sorted((step2/'provenance/historical_results').glob('*_seed_results.csv')):
            for row in csv_rows(path):
                condition = row.get('collection') or row.get('condition')
                if condition not in ('primary_full', 'chinese_pooled') or row.get('method') not in METHODS:
                    continue
                if row.get('study') not in ('main', 'chinese_full'):
                    continue
                case = (condition, int(row['K']), row['method'], int(row['seed']))
                check(case not in self.historical, 'Duplicate historical baseline metric record')
                self.historical[case] = row

    def vectors(self, corpus, method, info):
        dimension = 300 if method == 'knntm' else 200
        cache_key = (corpus, dimension)
        if cache_key not in self.vector_cache:
            if corpus == 'chinese_pooled':
                from transport_topics.chinese.embeddings import load_word_vectors
            else:
                from transport_topics.representations.word_embeddings import load_word_vectors
            values, meta = load_word_vectors(self.project, self.corpora[corpus]['vocabulary'], dimension)
            check(values.shape == (len(self.corpora[corpus]['vocabulary']), dimension) and
                  np.isfinite(values).all() and np.all(np.linalg.norm(values, axis=1) > 0), 'Invalid aligned embeddings')
            self.vector_cache[cache_key] = values, meta
            out = self.work/'baseline_inputs'/(corpus+'_'+str(dimension)+'.npz')
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, vocabulary=self.corpora[corpus]['vocabulary'], vectors=values)
            write(out.with_suffix('.json'), meta)
        values, meta = self.vector_cache[cache_key]
        saved = info['audit']['input_preparation']['word_vectors']
        check(meta['aligned_values_sha256'] == saved['aligned_values_sha256'], 'Aligned baseline word vectors differ from training')
        check(hashlib.sha256(values.tobytes()).hexdigest() == saved['aligned_values_sha256'], 'Aligned embedding bytes differ')
        return values, meta

    def checkpoint(self, corpus, method, info, h, theta):
        audit, c = info['audit'], self.corpora[corpus]
        for receipt in audit['implementation']['files']:
            path = inside(self.project, receipt['path'])
            check(digest(path) == receipt['sha256'], 'Pinned native source changed: '+receipt['path'])
        check(audit['implementation']['commit'] == '4094055b9e2d0169b0aa75d5aed7220e9509f0de', 'Unreviewed source revision')
        path = inside(self.project, audit['checkpoint'])
        check(digest(path) == audit['checkpoint_sha256'], 'Neural checkpoint SHA differs')
        import torch
        # Never load an arbitrary pickle with weights_only=False as a fallback.
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        check(checkpoint['identity'] == info['identity'], 'Checkpoint/model identity differs')
        check(int(checkpoint['epoch']) == audit['completed_epochs'] == audit['epochs'] == 200,
              'Neural epoch coverage differs')
        state = {name: value.detach().cpu().numpy() for name, value in checkpoint['model'].items()}
        check(all(np.isfinite(value).all() for value in state.values()), 'Nonfinite checkpoint parameter/buffer')
        losses = checkpoint['losses']
        check([r['epoch'] for r in losses] == list(range(1, 201)) and
              all(np.isfinite(r['loss']) for r in losses), 'Incomplete/nonfinite training-loss history')
        loss_file = path.parent/'loss_history.csv'
        if loss_file.is_file():
            saved = csv_rows(loss_file)
            check(len(saved) == 200 and all(int(a['epoch']) == b['epoch'] and abs(float(a['loss'])-b['loss']) < 1e-7
                  for a, b in zip(saved, losses)), 'Checkpoint/loss CSV histories disagree')
        settings = audit['settings']
        vectors, meta = self.vectors(corpus, method, info)
        contexts = None
        extra = {}
        if method == 'glocom':
            from transport_topics.baselines.inputs import global_context
            context_key = (corpus, settings['clusters'], settings['cluster_seed'])
            if context_key not in self.context_cache:
                embedding = np.load(c['collection'].directory/'embeddings.npy', allow_pickle=False)
                context, groups = global_context(c['collection'].counts, embedding, settings['clusters'], settings['cluster_seed'])
                self.context_cache[context_key] = context
                extra['reconstructed_global_cluster_counts'] = np.bincount(groups).tolist()
            contexts = self.context_cache[context_key]
        else:
            cache_key = (corpus, audit['input_preparation']['transport']['identity'])
            if cache_key not in self.distance_cache:
                self.distance_cache[cache_key] = distance_checks(self.project, c, audit, state, vectors)
            else:
                # Each seed's checkpoint still must contain the same prescribed inputs.
                directory = self.project/'cache/baseline_inputs/knntm'/cache_key[1][:20]
                for f, s in (('M_cos', 'M_cos_dist'), ('M_coo', 'M_coo_dist')):
                    with np.load(directory/(f+'.npz'), allow_pickle=False) as z:
                        compare(z['arr_0'], state[s], 1e-7, 'Checkpoint/cache '+f)
            extra['exact_ot_diagnostic'] = self.distance_cache[cache_key]
        native_h = replay_beta(state, method, settings.get('beta_temperature', .2))
        native_theta = replay_theta(c['collection'].counts, state, method, contexts)
        # GPU-produced snapshots and CPU replay can differ by float32 rounding.
        extra['decoder_max_abs_error'] = compare(native_h, h, 3e-5, 'Native topic scores')
        extra['membership_max_abs_error'] = compare(native_theta, theta, 3e-5, 'Native original-post memberships')
        left = np.argsort(-h, axis=1, kind='stable')[:, :10]
        right = np.argsort(-native_h, axis=1, kind='stable')[:, :10]
        extra['cpu_replay_top10_set_changes'] = sum(set(a) != set(b) for a, b in zip(left, right))
        if extra['cpu_replay_top10_set_changes']:
            self.findings.add('WARN', 'float32_top_word_boundary', 'CPU replay changes a top-ten boundary; saved scores retained for metrics.',
                              corpus=corpus, method=method, K=info['topics'], seed=info['seed'])
        for row in losses:
            self.losses.append(dict(corpus=corpus, method=method, K=info['topics'], seed=info['seed'], **row))
        self.diagnostics.append(dict(corpus=corpus, method=method, K=info['topics'], seed=info['seed'],
            first20_mean_loss=float(np.mean([r['loss'] for r in losses[:20]])),
            last20_mean_loss=float(np.mean([r['loss'] for r in losses[-20:]])),
            observed_epochs=200, **extra))
        if info['seed'] == 0 and info['topics'] == 12:
            names = ('fcd1.weight',) if method == 'knntm' else ('topic_embeddings', 'word_embeddings')
            target = self.work/'decoder_samples'/(corpus+'_'+method+'.npz')
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **{n: state[n] for n in names})
        return extra

    def one(self, corpus, directory, info):
        from transport_topics.evaluation.topic_quality import evaluate_topics
        c, k = self.corpora[corpus], int(info['topics'])
        verify_receipt(directory)
        check(key(read(directory/'identity.json')) == info['identity'], 'Model identity checksum differs')
        check(info['condition_identity'] == c['collection'].summary['identity_sha256'], 'Baseline used different prepared corpus')
        check(np.array_equal(np.load(directory/'vocabulary.npy', allow_pickle=False), c['vocabulary']), 'Baseline vocabulary order differs')
        check(np.array_equal(np.load(directory/'source_rows.npy', allow_pickle=False), c['collection'].source_rows), 'Baseline document identities/order differ')
        h, theta, features = [np.load(directory/(name+'.npy'), allow_pickle=False)
                              for name in ('topic_scores', 'doc_topics', 'retrieval_features')]
        n, v = c['collection'].counts.shape
        check(h.shape == (k, v) and theta.shape == (n, k) and features.shape == (n, k), 'Baseline array shape differs')
        check(all(np.isfinite(x).all() for x in (h, theta, features)), 'Nonfinite baseline output')
        check(np.min(theta) >= -1e-8 and np.max(np.abs(theta.sum(axis=1)-1)) < 2e-5, 'Invalid baseline topic-membership probabilities')
        compare(theta, features, 1e-8, 'Baseline retrieval features/native memberships')
        if info['method'] == 'btm':
            check(h.min() >= 0 and np.max(np.abs(h.sum(axis=1)-1)) < 2e-5, 'Invalid BTM word distributions')
            audit = info['audit']
            check(audit['iterations'] == 1000 and audit['window'] == 15 and audit['version'] == '1.0.0', 'Unexpected BTM settings')
            check(abs(audit.get('alpha', 50/k)-50/k) < 1e-10 and abs(audit.get('beta', .01)-.01) < 1e-12,
                  'Unexpected BTM priors')
        else:
            self.checkpoint(corpus, info['method'], info, h, theta)
        quality = evaluate_topics(h, c['vocabulary'], c['reference'], 10, True)
        record = dict(corpus=corpus, method=info['method'], K=k, seed=info['seed'],
                      npmi=quality['npmi'], cv=quality['cv'], diversity=quality['diversity'])
        old = self.historical.get((info['condition'], k, info['method'], int(info['seed'])))
        if old is not None:
            check(old.get('model_identity', info['identity']) == info['identity'], 'Historical baseline table refers to a different model')
            for field in ('npmi', 'cv', 'diversity'):
                check(abs(float(old[field])-record[field]) < 2e-8,
                      'Recomputed baseline '+field+' differs from its historical seed table')
        self.metrics.append(record)
        self.topics.append(dict(**record, top_words=quality['top_words']))
        self.records.append(info)
        self.diagnostics.append(dict(corpus=corpus, method=info['method'], K=k, seed=info['seed'],
            constant_score_topics=int(np.sum(np.ptp(h, axis=1) == 0)),
            dominant_topic_counts=np.bincount(theta.argmax(axis=1), minlength=k).tolist(),
            mean_membership_entropy=float(np.mean(-np.sum(theta*np.log(np.maximum(theta, 1e-30)), axis=1))),
            signed_topic_scores_preserved=bool(h.min() < 0)))
        target = self.work/'baseline_arrays'/corpus/('K%d'%k)/info['method']/('s%d.npz'%info['seed'])
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, topic_scores=h, doc_topics=theta, vocabulary=c['vocabulary'],
                            source_rows=c['collection'].source_rows)
        return record

    def run(self):
        canonical = read(self.step2/'provenance/step1_closeout/canonical_sources.json')
        cases, inventory = defaultdict(list), []
        mapping = {'primary_full': 'transport_grievance', 'chinese_pooled': 'chinese_pooled'}
        for relative in canonical['model_directories']:
            p = inside(self.project, relative)
            info = read(p/'model.json')
            if info.get('method') not in METHODS or info.get('condition') not in mapping:
                continue
            corpus = mapping[info['condition']]
            inventory.append(dict(corpus=corpus, method=info['method'], K=int(info['topics']), seed=int(info['seed']), path=relative))
            cases[(corpus, int(info['topics']), info['method'], int(info['seed']))].append((p, info))
        table(self.work/'baseline_inventory.csv', inventory)
        completed = 0
        for corpus in self.corpora:
            for k in COUNTS:
                for method in METHODS:
                    for seed in range(5):
                        case = (corpus, k, method, seed)
                        label = '%s K%d %s seed%d'%case
                        try:
                            matches = cases.get(case, [])
                            if not matches:
                                raise FileNotFoundError('Missing expected saved baseline '+label)
                            check(len(matches) == 1, 'Multiple canonical baseline runs; no score-based choice: '+label)
                            self.logger.info('Baseline %d/60 | %s', completed+1, label)
                            directory, info = matches[0]
                            row = self.one(corpus, directory, info)
                            self.ledger.append(dict(case=label, model_path=str(directory), model_identity=info['identity'],
                                                   model_receipt_sha256=digest(directory/'complete.json'),
                                                   checkpoint=info['audit'].get('checkpoint', ''),
                                                   checkpoint_sha256=info['audit'].get('checkpoint_sha256', ''), **row))
                            self.findings.add('PASS', 'saved_baseline_replay', label)
                            completed += 1
                        except Exception as error:
                            self.findings.failure('saved_baseline_replay', error, case=label)
                        self.flush()
        # Historical K16 remains historical. No claim that the revised K10 benchmark exists.
        missing10 = [dict(corpus=c, K=10, method=m, seed=s) for c in self.corpora for m in METHODS for s in range(5)
                     if len(cases.get((c, 10, m, s), [])) != 1]
        if missing10:
            self.findings.add('WARN', 'K10_benchmark_coverage',
                '%d historical baseline fits are missing at K10. PEBR policy comparisons at K10 remain valid; a complete all-method K10 benchmark cannot be claimed.'%len(missing10))
        self.findings.add('WARN', 'baseline_configuration_adequacy',
            'Faithful saved-output replay and finite 200-epoch training do not establish adequate tuning or convergence. Inspect training diagnostics and settings before claiming general neural-baseline inferiority.')
        summaries = []
        from transport_topics.evaluation.stability import seed_stability
        for c in self.corpora:
            for k in COUNTS:
                for m in METHODS:
                    rows = [r for r in self.topics if (r['corpus'], r['K'], r['method']) == (c, k, m)]
                    if len(rows) != 5:
                        continue
                    item = dict(corpus=c, K=k, method=m, seeds=5,
                                stability=seed_stability({r['seed']: r['top_words'] for r in rows})['stability'])
                    for field in ('npmi', 'cv', 'diversity'):
                        item[field+'_mean'] = float(np.mean([r[field] for r in rows]))
                        item[field+'_sample_sd'] = float(np.std([r[field] for r in rows], ddof=1))
                    summaries.append(item)
        table(self.work/'baseline_summaries.csv', summaries)
        write(self.work/'baseline_coverage.json', dict(expected_saved_fits=60, verified_saved_fits=completed,
              fidelity_checks_complete=completed == 60, native_training_repeated=False,
              tuning_adequacy_established=False, K10_missing=missing10))
        return completed == 60

    def flush(self):
        table(self.work/'baseline_seed_metrics.csv', self.metrics)
        table(self.work/'baseline_run_ledger.csv', self.ledger)
        table(self.work/'training_losses.csv', self.losses)
        write(self.work/'baseline_diagnostics.json', self.diagnostics)
        write(self.work/'baseline_top_words.json', self.topics)
        write(self.work/'baseline_metadata.json', self.records)
