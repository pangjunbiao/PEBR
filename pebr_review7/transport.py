"""Two distinct, traceable illustrations; no semantic labels inferred from donors."""
from collections import Counter
from pathlib import Path
import re
import numpy as np
from .common import read, write, table, csv_rows
from .native_math import normalized_contributions

ANCHORS = (
    (271, 'Cab fares and payment demands', 'driver asked for more fare'),
    (357, 'Train delays', 'running 2 hours 15 mins late'),
    (264, 'Coach facilities and condition', 'no water no Beddings'),
    (14, 'Interstate bus-service requests', 'approve interstate Agreement KSRTC and Apsrtc'))
LABELS = {'nmf_original': 'NMF', 'nmf_similarity': 'Similarity sharing',
          'preserve_fixed_prefix': 'Same-score similarity prefix',
          'preserve_forward_fixed_cap': 'PEBR',
          'matched_preserve_fixed': 'Matched-amount sharing',
          'weighted_hidden_fixed': 'Without visible-loss charge',
          'preserve_fixed_no_downside': 'Without downside penalty'}
DISPLAY = ('nmf_original', 'nmf_similarity', 'preserve_fixed_prefix', 'preserve_forward_fixed_cap')

def redact(text):
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', str(text), flags=re.I)
    text = re.sub(r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b', '[EMAIL]', text)
    text = re.sub(r'(?<!\w)@[A-Za-z0-9_]+', '[USER]', text)
    text = re.sub(r'\b\d{7,}\b', '[NUMBER]', text)
    text = re.sub(r'(?i)\b(pnr|booking\s*(?:id|number|no\.?))\s*[:#-]?\s*[A-Za-z0-9-]{5,}', r'\1 [ID]', text)
    return ' '.join(text.split())

def excerpt(text, phrase=None, words=(), budget=240):
    text = redact(text)
    if phrase and phrase in text:
        return '[...] '+phrase+' [...]', 'verified supplied excerpt'
    start = 0
    for word in words:
        match = re.search(r'(?<!\w)'+re.escape(word)+r'(?!\w)', text, re.I)
        if match:
            start = max(0, match.start()-60)
            break
    if len(text) <= budget:
        return text, 'complete redacted text'
    if start:
        boundary = text.find(' ', start)
        start = boundary+1 if boundary >= 0 else start
    end = min(len(text), start+budget)
    if end < len(text):
        boundary = text.rfind(' ', start, end)
        end = boundary if boundary > start else end
    return ('[...] ' if start else '')+text[start:end]+(' [...]' if end < len(text) else ''), 'fixed-length redacted excerpt'

def latex(text):
    return ''.join({'\\':r'\textbackslash{}', '&':r'\&', '%':r'\%', '$':r'\$', '#':r'\#',
                    '_':r'\_', '{':r'\{', '}':r'\}', '~':r'\textasciitilde{}', '^':r'\textasciicircum{}'}.get(c, c) for c in str(text))

def source_identity(record, source_row):
    return str(record.get('source_id') or record.get('id') or ('grievance.csv:data_row:'+str(source_row)))

def fit_path(row, step5, step3):
    if row['policy'] == 'nmf_original':
        path = step3/'fits'/row['corpus']/('K%d'%int(row['K']))/'nmf_original'/('seed%d'%int(row['seed']))
    else:
        path = step5/'fits'/row['corpus']/('mask%d'%int(row['mask_base']))/('K%d'%int(row['K']))/row['policy']/('seed%d'%int(row['seed']))
    return path

def selected_examples(outcomes, source_rows):
    """Outcome-based illustrations are separate from four prespecified anchors."""
    full, other = 'preserve_forward_fixed_cap', 'weighted_hidden_fixed'
    indexed = {(r['policy'], int(r['document_index'])): r for r in outcomes}
    indices = sorted({int(r['document_index']) for r in outcomes}, key=lambda i: int(source_rows[i]))
    contrast, failure = None, None
    for i in indices:
        f, o = indexed.get((full, i)), indexed.get((other, i))
        if f is None or o is None:
            continue
        a, b = float(f['gain_nats_per_token']), float(o['gain_nats_per_token'])
        if contrast is None and a > 0 and b < 0 and a-b >= .02:
            contrast = i
        if failure is None and a < -.01:
            failure = i
    return contrast, failure

def protected_cases(step6, step3, work, c, protocol, findings, logger):
    from scipy import sparse
    from pebr_revision_step3.io import complete
    from pebr_fresh_validation.references import read_references
    from pebr_protected_words.banks import load_bank
    from transport_topics.proposed.transfer import normalized_counts, transfer_action, check_transfer
    from transport_topics.proposed.actions import BorrowingAction
    corpus, k, outer_seed = 'transport_grievance', 12, 3712
    collection, vocabulary = c['collection'], c['vocabulary']
    p6 = read(step6/'protocol.json')
    bank_dir = step6/'protected_banks'/corpus/('outer%d'%outer_seed)
    bank = load_bank(bank_dir, collection, p6)
    targets = sparse.load_npz(bank_dir/'outer.npz').tocsr()
    directory = step6/'audits'/corpus/('outer%d'%outer_seed)/'K12'
    if not complete(directory):
        raise ValueError('Protected-case audit receipt failed')
    outcomes = csv_rows(directory/'post_outcomes.csv')
    scores = {(r['policy'], int(r['document_index'])): r for r in outcomes}
    policies = tuple(LABELS)
    actions = {}
    selection_root = step6/'selection'/corpus/('outer%d'%outer_seed)/'K12'
    if not complete(selection_root):
        raise ValueError('Protected-case selected-action receipt failed')
    for policy in policies:
        if policy == 'nmf_original':
            continue
        path = selection_root/policy
        actions[policy] = read(path/'actions.json')
    posts = read(collection.directory/'posts.json')
    row_map = {int(s): i for i, s in enumerate(collection.source_rows)}
    case_specs = []
    for source, title, phrase in ANCHORS:
        if source not in row_map:
            findings.add('INCOMPLETE', 'fixed_transport_anchor', 'Source row is absent from the prepared primary collection', source_row=source)
            continue
        i = row_map[source]
        case_specs.append((i, title, phrase, 'prespecified transport anchor'))
    contrast, failure = selected_examples(outcomes, collection.source_rows)
    used = {s[0] for s in case_specs}
    for i, title, reason in ((contrast, 'Predictive contrast illustration', 'first source row: PEBR gain > 0, control gain < 0, gap >= 0.02'),
                             (failure, 'Unsuccessful protected-word example', 'first source row: PEBR gain < -0.01')):
        if i is not None and i not in used:
            case_specs.append((i, title, None, reason))
            used.add(i)
    refs = read_references(step3, collection, k, protocol)
    original = normalized_counts(collection.counts)
    base = normalized_counts(bank.base)
    cases, gain_rows, token_rows, donor_rows = [], [], [], []
    for i, title, phrase, reason in case_specs:
        source = int(collection.source_rows[i])
        selected_record = dict(source_row=source, document_index=i, title=title, selection_reason=reason,
            source_id=source_identity(posts[i], source), outer_seed=outer_seed, K=k)
        target = targets[i].toarray().ravel()
        ids = np.flatnonzero(target)
        words = [str(x) for x in vocabulary[ids]]
        quote, quote_rule = excerpt(posts[i]['original_text'], phrase, words)
        selected_record.update(receiver_excerpt=quote, excerpt_rule=quote_rule, protected_words=words,
                               protected_token_counts=target[ids].astype(int).tolist())
        if ('nmf_original', i) not in scores:
            selected_record.update(eligible_protected_query=False, explanation='Fixed anchor retained; it is not in this protected audit cohort.')
            cases.append(selected_record)
            continue
        selected_record['eligible_protected_query'] = True
        distributions, records = [], []
        for policy in policies:
            rec = {'donors': [], 'offered_beta': 0., 'actual_beta': 0., 'available_donors': [], 'selection_utility': 0.}
            if policy != 'nmf_original':
                rec = actions[policy][i]
                if int(rec['document_index']) != i:
                    raise ValueError('Protected action source order changed')
            action = BorrowingAction(tuple(rec['donors']), float(rec['offered_beta']))
            q, mass, available = transfer_action(base[i], action, bank.full_candidates[i], original)
            check_transfer(base[i], q, mass, p6['cap'])
            if abs(mass-float(rec['actual_beta'])) > 1e-12 or list(available) != rec['available_donors']:
                raise ValueError('Protected case does not replay the saved transfer')
            distributions.append(q)
            records.append(rec)
        probabilities = refs[int(collection.folds[i])].predict(np.asarray(distributions), ids)
        log_score = np.log(probabilities)@target[ids]/target[ids].sum()
        gains = log_score-log_score[0]
        for index, (policy, rec) in enumerate(zip(policies, records)):
            saved_gain = float(scores[(policy, i)]['gain_nats_per_token'])
            if abs(saved_gain-float(gains[index])) > 2e-8:
                raise ValueError('Protected case gain differs from completed Step 6')
            available = set(rec['available_donors'])
            gain_rows.append(dict(**selected_record, policy=policy, method=LABELS[policy],
                gain_nats_per_token=saved_gain, offered_beta=rec['offered_beta'], realized_mass=rec['actual_beta'],
                selected_donors=len(rec['donors']), contributing_donors=len(available),
                selection_utility=rec.get('selection_utility'), selected_source_rows=';'.join(str(int(collection.source_rows[j])) for j in rec['donors'])))
            for position, word in enumerate(words):
                token_rows.append(dict(source_row=source, word=word, original_target_count=int(target[ids[position]]),
                    policy=policy, no_borrowing_probability=float(probabilities[0, position]),
                    policy_probability=float(probabilities[index, position]),
                    log_probability_gain=float(np.log(probabilities[index, position]/probabilities[0, position]))))
            for j in rec['donors']:
                j = int(j)
                donor_source = int(collection.source_rows[j])
                overlap = [word for word, col in zip(words, ids) if collection.counts[j, col] > 0]
                donor_quote, donor_excerpt_rule = excerpt(posts[j]['original_text'], words=overlap)
                donor_rows.append(dict(receiver_source_row=source, policy=policy, donor_source_row=donor_source,
                    donor_source_id=source_identity(posts[j], donor_source), contributes_at_transfer=j in available,
                    offered_share=float(rec['offered_beta'])/len(rec['donors']),
                    realized_share=float(rec['offered_beta'])/len(rec['donors']) if j in available else 0.,
                    protected_words_also_present_in_donor=';'.join(overlap), donor_excerpt=donor_quote,
                    excerpt_rule=donor_excerpt_rule))
        cases.append(selected_record)
        logger.info('Transportation protected case | source row %d | %s', source, reason)
    dest = work/'transportation'
    write(dest/'protected_cases.json', cases)
    table(dest/'protected_case_policy_gains.csv', gain_rows)
    table(dest/'protected_case_word_probabilities.csv', token_rows)
    table(dest/'protected_case_donors.csv', donor_rows)
    write(dest/'protected_case_selection.json', dict(outer_seed=outer_seed, fixed_source_rows=[a[0] for a in ANCHORS],
        predictive_contrast_source_row=None if contrast is None else int(collection.source_rows[contrast]),
        failure_source_row=None if failure is None else int(collection.source_rows[failure]),
        contrast_and_failure_selected_after_observing_saved_outcomes=True,
        illustration_not_independent_validation=True, factual_completion_claim=False))
    repeats = []
    for outer in p6['fresh_outer_target_bases']:
        repeat_path = step6/'audits'/corpus/('outer%d'%outer)/'K12'
        if not complete(repeat_path):
            raise ValueError('A fixed-anchor protected-repeat audit is incomplete')
        for row in csv_rows(repeat_path/'post_outcomes.csv'):
            if int(row['source_row']) in {a[0] for a in ANCHORS} and row['policy'] in policies:
                repeats.append(row)
    table(dest/'fixed_anchor_protected_repeats.csv', repeats)
    write(dest/'fixed_anchor_repeat_scope.json', {
        'outer_banks': p6['fresh_outer_target_bases'], 'known_bank_2712_excluded': True,
        'targets_change_across_banks': True, 'overlapping_holdouts_not_independent_incidents': True,
        'first_fresh_bank_preselected_for_detailed_display': 3712})
    lines = ['# Protected-word transportation cases', '',
        'Fixed audit: primary corpus, K=12, outer target bank 3712. These are saved-action replays, not new fits.', '',
        'The first four anchors were fixed in the manuscript. Any added contrast/failure examples are explicitly outcome-selected illustrations.', '',
        'Prediction concerns words actually present in the receiver. A donor does not establish the cause of a service incident.', '']
    for case in cases:
        source = case['source_row']
        lines += ['## '+case['title']+' — source row '+str(source), '',
                  'Selection: '+case['selection_reason']+'.', '', '> '+case['receiver_excerpt'], '',
                  'Protected words: '+', '.join(case['protected_words'])+'.', '',
                  '| Policy | Gain (nats/token) | Realized mass | Contributing donors |',
                  '| --- | ---: | ---: | ---: |']
        for row in gain_rows:
            if row['source_row'] == source:
                lines.append('| %s | %.6f | %.4f | %d |'%(row['method'], row['gain_nats_per_token'], row['realized_mass'], row['contributing_donors']))
        if not case['eligible_protected_query']:
            lines += ['', case['explanation']]
        lines += ['', 'See the donor ledger and per-word probability table for the original evidence and exact score decomposition.', '']
    (dest/'PROTECTED_CASES.md').write_text('\n'.join(lines), encoding='utf-8')
    return len(cases)

def topic_cases(step5, step3, work, c, findings, logger):
    from pebr_revision_step3.io import complete
    from transport_topics.evaluation.topic_quality import top_words
    corpus, k, mask = 'transport_grievance', 12, 3711
    collection, vocabulary = c['collection'], c['vocabulary']
    posts = read(collection.directory/'posts.json')
    source_map = {int(s): i for i, s in enumerate(collection.source_rows)}
    rows = [r for r in read(step5/'results/topic_words.json') if r['corpus'] == corpus and int(r['K']) == k
            and r['policy'] in DISPLAY and (r['policy'] == 'nmf_original' or int(r['mask_base']) == mask)]
    if len(rows) != 20:
        raise ValueError('Expected 20 saved fits for the fixed final-topic case display')
    output, counters = [], {}
    for row in rows:
        path = fit_path(row, step5, step3)
        if not complete(path):
            raise ValueError('Topic-case model receipt failed: '+str(path))
        if not np.array_equal(np.load(path/'source_rows.npy', allow_pickle=False), collection.source_rows):
            raise ValueError('Topic-case document identities/order differ')
        if not np.array_equal(np.load(path/'vocabulary.npy', allow_pickle=False), vocabulary):
            raise ValueError('Topic-case vocabulary identities/order differ')
        h, w, saved = [np.load(path/(name+'.npy'), allow_pickle=False)
                       for name in ('topic_scores', 'doc_topics', 'retrieval_features')]
        theta = normalized_contributions(w, h)
        if not np.allclose(theta, saved, atol=2e-10, rtol=2e-10):
            raise ValueError('Original-post normalized topic contributions do not replay')
        words = top_words(h, vocabulary, 10)
        if words.tolist() != row['top_words']:
            raise ValueError('Saved model/top-word record differs')
        for source, title, phrase in ANCHORS:
            if source not in source_map:
                continue
            i = source_map[source]
            if not np.any(theta[i]):
                findings.add('INCOMPLETE', 'anchor_topic_membership', 'Fixed anchor has zero original-post contribution', source_row=source)
                continue
            topic = int(np.argmax(theta[i]))
            quote, rule = excerpt(posts[i]['original_text'], phrase)
            item = dict(source_row=source, title=title, excerpt=quote, excerpt_rule=rule,
                        method=LABELS[row['policy']], policy=row['policy'], K=k, mask_base='not_applicable' if row['policy']=='nmf_original' else mask,
                        final_seed=int(row['seed']), selected_topic_zero_based=topic,
                        dominant_contribution=float(theta[i, topic]), top_words=words[topic].tolist())
            output.append(item)
            counters.setdefault((source, row['policy']), Counter()).update(item['top_words'])
    dest = work/'transportation'
    write(dest/'topic_cases_five_seeds.json', output)
    table(dest/'topic_cases_five_seeds.csv', [dict(r, top_words=', '.join(r['top_words'])) for r in output])
    recurrence = []
    for source, title, phrase in ANCHORS:
        shown = sorted({word for policy in DISPLAY for word, count in counters.get((source, policy), {}).items() if count >= 3})
        for word in shown:
            for policy in DISPLAY:
                recurrence.append(dict(source_row=source, title=title, word=word, policy=policy,
                                       count=counters.get((source, policy), {}).get(word, 0), fits=5))
    table(dest/'topic_word_recurrence.csv', recurrence)
    selected = [r for r in output if r['final_seed'] == 0]
    table(dest/'topic_cases_seed0.csv', [dict(r, top_words=', '.join(r['top_words'])) for r in selected])
    text = [r'\begin{table*}[!t]', r'\centering\footnotesize',
        r'\caption{Illustrative transport-service topics at $K=12$, using the first fresh selection bank and final seed 0.}',
        r'\label{tab:transport_cases_revised}', r'\setlength{\tabcolsep}{3pt}',
        r'\begin{tabularx}{\textwidth}{@{}p{0.20\textwidth}*{4}{>{\raggedright\arraybackslash}X}@{}}',
        r'\toprule', r'Anchor and original excerpt & NMF & Similarity sharing & Same-score prefix & PEBR \\', r'\midrule']
    for source, title, phrase in ANCHORS:
        case = {r['policy']: r for r in selected if r['source_row'] == source}
        if set(case) != set(DISPLAY):
            continue
        first = case[DISPLAY[0]]
        cells = [r'\textbf{'+latex(title)+r'}\par Source row '+str(source)+r'\par\textit{'+latex(first['excerpt'])+'}']
        cells += [latex(', '.join(case[p]['top_words'])) for p in DISPLAY]
        text += [' & '.join(cells)+r' \\', r'\addlinespace[3pt]']
    text += [r'\bottomrule', r'\end{tabularx}', r'\par\vspace{2pt}',
        r'\parbox{\textwidth}{\scriptsize Each topic is selected by the anchor\textquotesingle s largest normalized contribution inferred from its original words. No descriptor words are edited. Source rows are zero-based data-row indices. Excerpts retain wording; redactions and omissions are marked. These fixed examples illustrate interpretation, not a separate performance evaluation.}',
        r'\end{table*}']
    (dest/'transport_cases_revised.tex').write_text('\n'.join(text)+'\n', encoding='utf-8')
    make_heatmaps(dest, recurrence, findings)
    logger.info('Final transportation topics | 4 fixed anchors, 4 methods, 5 final seeds within bank 3711')
    return len(output)

def make_heatmaps(dest, recurrence, findings):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        findings.failure('transportation_heatmaps', error)
        return
    for number, anchors in enumerate((ANCHORS[:2], ANCHORS[2:]), 1):
        panels = []
        for source, title, _ in anchors:
            rows = [r for r in recurrence if r['source_row'] == source]
            words = sorted({r['word'] for r in rows})
            lookup = {(r['word'], r['policy']): r['count'] for r in rows}
            values = np.asarray([[lookup.get((w, p), 0) for p in DISPLAY] for w in words])
            panels.append((title, words, values))
        if any(not p[1] for p in panels):
            findings.add('INCOMPLETE', 'transportation_heatmap', 'A fixed-anchor panel has no available recurrence words')
            continue
        height = max(3.5, 1.8+.18*max(len(p[1]) for p in panels))
        fig, axes = plt.subplots(1, 2, figsize=(7.25, height), layout='constrained')
        for ax, (title, words, values) in zip(axes, panels):
            im = ax.imshow(values, aspect='auto', vmin=0, vmax=5, cmap='Blues')
            ax.set_title(title, fontsize=9)
            ax.set_yticks(range(len(words)), words, fontsize=7)
            ax.set_xticks(range(4), ['NMF', 'Similarity', 'Prefix', 'PEBR'], rotation=35, ha='right', fontsize=8)
            for i in range(len(words)):
                for j in range(4):
                    ax.text(j, i, str(values[i,j]), ha='center', va='center', fontsize=6.5,
                            color='white' if values[i,j] >= 4 else 'black')
        bar = fig.colorbar(im, ax=axes, shrink=.55, ticks=range(6), pad=.02)
        bar.set_label('Model fits (out of 5)', fontsize=8)
        base = dest/('service_word_recurrence_revised_%d'%number)
        fig.savefig(base.with_suffix('.pdf'))
        fig.savefig(base.with_suffix('.png'), dpi=150)
        plt.close(fig)
    findings.add('PASS', 'transportation_heatmaps', 'Recurrence counts reflect five fits within one fixed bank; no counts represent passengers or incidents.')

def run(step6, step5, step3, work, c, protocol, findings, logger):
    completed = {}
    for name, function, args in (
        ('protected_cases', protected_cases, (step6, step3, work, c, protocol, findings, logger)),
        ('final_topic_cases', topic_cases, (step5, step3, work, c, findings, logger))):
        try:
            completed[name] = function(*args)
            findings.add('PASS', name, 'Saved numerical inputs and original-source mapping replayed')
        except Exception as error:
            findings.failure(name, error)
    write(work/'transportation/interpretation_scope.json', {
        'protected_cases': 'Outer bank 3712; fold-excluded prediction of genuinely observed receiver words. No final topic fits.',
        'final_topic_cases': 'Fresh selection bank 3711; K12; final seed0 display and five-seed recurrence. Original-post memberships.',
        'same_selected_actions_used_for_both_panels': False,
        'case_illustrations_are_independent_validation_studies': False,
        'passenger_service_improvement_or_incident_cause_established': False,
        'supporting_claim': 'Inspectable evidence admission and readable topic descriptions linked to unchanged source posts.',
        'completed': completed})
    return set(completed) == {'protected_cases', 'final_topic_cases'}
