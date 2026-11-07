"""Distinguish explicit admission refusal from zero transfer after fallback."""
from .common import read,write,table,csv_rows

def decision_kind(record):
    if not record['donors']:
        if float(record['offered_beta'])!=0 or float(record['actual_beta'])!=0:
            raise ValueError('Empty action has nonzero borrowing')
        return 'explicit_no_borrowing'
    if float(record['actual_beta'])==0:
        return 'selected_donors_unavailable_zero_transfer'
    if len(record['available_donors'])<len(record['donors']):
        return 'partial_transfer_with_unused_shares_returned'
    return 'selected_donors_all_available'

def first_explicit(actions,similarity):
    for rec in sorted(actions,key=lambda r:int(r['source_row'])):
        i=int(rec['document_index'])
        if rec['eligible'] and not rec['donors'] and float(similarity[i]['actual_beta'])>0:
            return i
    return None

def export(step6,work,collection):
    from pebr_revision_step3.io import complete
    from pebr_review7.transport import excerpt,LABELS
    base=step6/'selection/transport_grievance/outer3712/K12'
    if not complete(base):raise ValueError('Protected transportation actions changed')
    actions={p:read(base/p/'actions.json') for p in LABELS if p!='nmf_original'}
    anchor='preserve_forward_fixed_cap'
    selected=first_explicit(actions[anchor],actions['nmf_similarity'])
    indices=[i for i,r in enumerate(actions[anchor]) if int(r['source_row']) in (271,357,264,14,5)]
    if selected is not None and selected not in indices:indices.append(selected)
    records=read(collection.directory/'posts.json')
    outcomes=csv_rows(step6/'audits/transport_grievance/outer3712/K12/post_outcomes.csv')
    scores={(r['policy'],int(r['document_index'])):r for r in outcomes}
    counts=[];contrasts=[]
    for i in indices:
        rec=actions[anchor][i]
        if int(rec['document_index'])!=i:raise ValueError('Transportation action/document ordering differs')
        source=int(collection.source_rows[i]);quote,rule=excerpt(records[i]['original_text'])
        counts.append({'source_row':source,'document_index':i,'decision':decision_kind(rec),
            'selected_donor_source_rows':';'.join(str(int(collection.source_rows[j])) for j in rec['donors']),
            'contributing_donor_source_rows':';'.join(str(int(collection.source_rows[j])) for j in rec['available_donors']),
            'offered_beta':rec['offered_beta'],'realized_mass':rec['actual_beta'],
            'selected_donor_count':len(rec['donors']),'contributing_donor_count':len(rec['available_donors']),
            'receiver_excerpt':quote,'excerpt_rule':rule,
            'selection_basis':'first eligible source-order explicit empty action; not selected using held-out gain' if i==selected else 'existing fixed anchor or existing failure example'})
        for p in LABELS:
            saved=scores.get((p,i))
            if saved is not None:
                contrasts.append(dict(saved,method=LABELS[p],case_decision=decision_kind(rec),
                                      explicit_no_borrowing_illustration=i==selected))
    dest=work/'transportation_clarification'
    table(dest/'decision_types.csv',counts);table(dest/'case_policy_gains.csv',contrasts)
    write(dest/'selection_scope.json',{'K':12,'outer_seed':3712,
        'explicit_no_borrowing_source_row':None if selected is None else int(collection.source_rows[selected]),
        'selected_before_reading_prediction_outcomes':True,
        'criterion':'First source-order eligible receiver with empty PEBR donor group and positive ordinary-sharing mass.',
        'held_out_gain_used_to_choose_case':False,'case_is_illustration_not_new_quantitative_study':True})
    bus=next((r for r in counts if r['source_row']==14),None)
    bus_fallback=bool(bus and bus['decision']=='selected_donors_unavailable_zero_transfer')
    bus_note=('The original bus anchor (source row 14) has a selected donor that is unavailable at final transfer. Zero realized mass is therefore fallback, not an explicit no-borrowing admission decision.'
        if bus_fallback else 'Classify the bus anchor by its saved action metadata; zero realized transfer alone does not establish an explicit no-borrowing choice.')
    lines=['# Transportation interpretation correction','',bus_note,
        'For the train anchor (row 357), show selected and contributing donors separately: only available donors supply the realized transfer. Do not describe unavailable donors as transferred evidence.','',
        '| Source row | Decision | Selected donors | Contributing donors | Offered weight | Realized mass |',
        '| ---: | --- | ---: | ---: | ---: | ---: |']
    for r in counts:
        lines.append('| %d | %s | %d | %d | %.4f | %.4f |'%(r['source_row'],r['decision'],r['selected_donor_count'],r['contributing_donor_count'],r['offered_beta'],r['realized_mass']))
    lines+=['','An explicit no-borrowing example is chosen from action metadata in source order, without selecting for a favorable protected outcome. All competing policy gains are exported.',
        'These examples explain actual evidence decisions. They do not prove semantic correctness for all complaints, establish incident causes or measure passenger-service improvements.']
    (dest/'READ_ME.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {'case_rows':len(counts),'explicit_no_borrowing_source_row':None if selected is None else int(collection.source_rows[selected]),
            'bus_zero_transfer_correctly_identified_as_availability_fallback':bus_fallback}
