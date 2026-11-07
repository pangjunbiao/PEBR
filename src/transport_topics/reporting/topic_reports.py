"""Systematic topic reports from original posts; no automatic semantic validation claims."""
from __future__ import annotations

import html
import json
from pathlib import Path
import numpy as np

from ..runtime.artifacts import file_hash, safe_path, write_json
from .common import VARIANTS, write_csv, write_text


def verify_receipt(directory):
    directory = Path(directory)
    receipt = json.loads((directory / 'complete.json').read_text(encoding='utf-8'))
    for name, expected in receipt['files'].items():
        path = safe_path(directory, name)
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError('Saved model/evidence receipt failed: ' + str(path))


def make_topic_report(model, scores, features, vocabulary, posts, actions=None, top_words=10, supports=5, families=None):
    scores, features = np.asarray(scores), np.asarray(features)
    if scores.shape != (model['topics'], len(vocabulary)) or features.shape != (len(posts), model['topics']):
        raise ValueError('Topic-report coordinates do not align with original posts')
    if not np.isfinite(scores).all() or not np.isfinite(features).all() or np.any(features < 0):
        raise ValueError('Topic reports require finite scores and nonnegative original-post features')
    source_rows = [int(p['source_row']) for p in posts]
    if len(set(source_rows)) != len(posts) or source_rows != sorted(source_rows):
        raise ValueError('Supporting original posts must preserve distinct source-row order')
    report = {'model_identity': model['identity'], 'condition': model['condition'], 'method': model['method'],
        'K': model['topics'], 'seed': model['seed'], 'topics': [],
        'selection_policy': 'All topics, saved topic order, top original-post feature values; source-order ties',
        'semantic_validation_performed': False,
        'support_definition': 'Original-post topic association; neither a confirmed complaint count nor semantic verification'}
    for k in range(model['topics']):
        indices = np.argsort(-scores[k], kind='stable')[:min(top_words, len(vocabulary))]
        words = [str(vocabulary[v]) for v in indices]
        order = [int(i) for i in np.argsort(-features[:, k], kind='stable') if features[i, k] > 0]
        examples = []
        for i in order[:supports]:
            original = posts[i]
            item = {'source_row': int(original['source_row']), 'source_id': original['source_id'],
                    'original_text': original['original_text'], 'original_post_topic_weight': float(features[i, k]),
                    'top_words_present_in_original_cleaned_text': [w for w in words if w in set(original['cleaned_text'].split())]}
            if actions is not None:
                item['borrowing_provenance'] = actions[int(original['source_row'])]
            examples.append(item)
        record = {'topic_index': k + 1, 'top_words': words, 'ranked_original_posts': examples,
                  'original_posts_with_positive_association': len(order)}
        if families is not None:
            record['text_families_with_positive_association'] = len(set(int(families[i]) for i in order))
        report['topics'].append(record)
    return report


def _html_report(report):
    esc = lambda s: html.escape(str(s), quote=True)
    pieces = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Original topic evidence</title>',
        '<style>body{font:16px/1.55 system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1.3rem;color:#182331}'
        'article{border-top:1px solid #cbd5df;margin-top:2rem}blockquote{margin:1rem 0;padding:1rem;background:#f2f5f8}'
        'pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}small{color:#495565}</style>',
        '<h1>Topic evidence: %s</h1>' % esc(report['method']),
        '<p>%s; K=%s; seed=%s. All topics are shown in saved order.</p>' % (esc(report['condition']), report['K'], report['seed']),
        '<p>These are model associations. Original complaints remain verbatim; topic words and donor records are separate. '
        'No human or model judge has certified that the posts describe the same incident.</p>']
    for topic in report['topics']:
        pieces += ['<article><h2>Topic %s</h2><p><b>Topic words:</b> %s</p>' % (topic['topic_index'], esc(', '.join(topic['top_words'])))]
        for post in topic['ranked_original_posts']:
            pieces += ['<blockquote><small>%s; original-post weight %.5f</small><p>%s</p>' %
                       (esc(post['source_id']), post['original_post_topic_weight'], esc(post['original_text'])),
                       '<small>Listed words present in original text: %s</small>' % esc(', '.join(post['top_words_present_in_original_cleaned_text']))]
            if 'borrowing_provenance' in post:
                pieces += ['<details><summary>Borrowing record</summary><pre>%s</pre></details>' % esc(json.dumps(post['borrowing_provenance'], indent=2))]
            pieces += ['</blockquote>']
        pieces += ['</article>']
    return '\n'.join(pieces) + '</html>\n'


def export_topic_reports(root, bundle, destination, logger):
    """Read stored main K=12 models only; selection does not depend on their scores."""
    root, destination = Path(root), Path(destination)
    profile = bundle['data']['dataset_evaluation_profile']['preparation']['conditions']
    k = bundle['config']['topics']['main_count']
    models = [m for m in bundle['data']['models_index']['models'] if m['topic_model'] and m['topics'] == k]
    collections, provenance, summary, index = {}, {}, [], []
    controls = bundle['data']['study_controls_index']['banks']
    for condition in profile:
        directory = safe_path(root, condition['path'])
        logger.info('TOPIC EVIDENCE | verifying original collection %s.', condition['name'])
        verify_receipt(directory)
        posts = json.loads((directory / 'posts.json').read_text(encoding='utf-8'))
        rows = np.load(directory / 'source_rows.npy', allow_pickle=False)
        if list(map(int, rows)) != [int(p['source_row']) for p in posts]:
            raise ValueError('Original topic-evidence texts have a different source order')
        families = np.load(directory / 'families.npy', allow_pickle=False)
        collections[condition['name']] = (posts, families)
        bank = [c for c in controls if c['job']['study'] == 'main' and c['condition'] == condition['name'] and c['topics'] == k and c['scope'] == 'topic_learning']
        if len(bank) != 1:
            raise ValueError('Missing unique full-view borrowing provenance bank')
        control_directory = safe_path(root, bank[0]['path'])
        verify_receipt(control_directory)
        actions = json.loads((control_directory / 'actions.json').read_text(encoding='utf-8'))
        if [int(a['source_row']) for a in actions] != list(map(int, rows)):
            raise ValueError('Borrowing provenance and original posts are misaligned')
        lookup = {int(p['source_row']): p['source_id'] for p in posts}
        for action in actions:
            for variant, evidence in action['variants'].items():
                evidence['selected_source_ids'] = [lookup[int(i)] for i in evidence['selected_source_rows']]
                evidence['available_source_ids'] = [lookup[int(i)] for i in evidence['available_source_rows']]
        provenance[condition['name']] = {int(a['source_row']): a['variants'] for a in actions}
        write_json(destination / ('borrowing_' + condition['name'] + '.json'), actions, overwrite=True)
    for count, model in enumerate(models, 1):
        directory = safe_path(root, model['path'])
        verify_receipt(directory)
        posts, families = collections[model['condition']]
        rows = np.load(directory / 'source_rows.npy', allow_pickle=False)
        if list(map(int, rows)) != [int(p['source_row']) for p in posts]:
            raise ValueError('Saved original-post memberships and texts are misaligned')
        variant = VARIANTS.get(model['method'])
        action = None if variant is None else {row: variants[variant] for row, variants in provenance[model['condition']].items()}
        report = make_topic_report(model, np.load(directory / 'topic_scores.npy', allow_pickle=False),
             np.load(directory / 'retrieval_features.npy', allow_pickle=False),
             np.load(directory / 'vocabulary.npy', allow_pickle=False), posts, action,
             bundle['config']['topics']['top_words'], families=families)
        name = '%s_%s_K%d_seed%d' % (model['condition'], model['method'], model['topics'], model['seed'])
        write_json(destination / (name + '.json'), report, overwrite=True)
        write_text(destination / (name + '.html'), _html_report(report))
        index.append({'condition': model['condition'], 'method': model['method'], 'K': k, 'seed': model['seed'],
                      'model_identity': model['identity'], 'json': name + '.json', 'html': name + '.html'})
        for topic in report['topics']:
            summary.append({'condition': model['condition'], 'method': model['method'], 'K': k, 'seed': model['seed'],
                'topic_index': topic['topic_index'], 'top_words': ', '.join(topic['top_words']),
                'representative_original_source_ids': '; '.join(p['source_id'] for p in topic['ranked_original_posts']),
                'semantic_validation_performed': False})
        if count == 1 or count % 25 == 0 or count == len(models):
            logger.info('TOPIC EVIDENCE | %d/%d model reports exported.', count, len(models))
    write_csv(destination / 'topic_index.csv', index)
    write_csv(destination / 'topic_words_and_source_ids.csv', summary)
    write_text(destination / 'index.html', '<!doctype html><html lang="en"><meta charset="utf-8"><title>Topic reports</title><h1>All topic reports</h1><ul>' +
        ''.join('<li><a href="%s">%s</a></li>' % (html.escape(r['html'], quote=True), html.escape(r['html'])) for r in index) + '</ul></html>')
    return {'model_reports': len(models), 'topic_reports': len(summary), 'provenance_collections': len(collections),
            'original_supports_only': True, 'raw_text_in_compact_inspection': False,
            'systematic_display_rule': 'Every main K=12 topic model, every seed, every topic; top five original-post associations'}
