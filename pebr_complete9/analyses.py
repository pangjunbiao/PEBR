"""Parameter and corpus-size analyses of the exact revised rule."""
from dataclasses import replace
from pathlib import Path
import copy
import numpy as np
from scipy import sparse
from .common import read,write,table,key
from .core import fit_nmf,shared,summary

def parameter_protocol(base,case):
    p=copy.deepcopy(base);p.update(cap=case['cap'],downside_weight=case['downside_weight'],
        positive_offered_weights=[case['cap']],policies=base['size_policies'],parameter_case=case['case'])
    return p

def parameters(study):
    from pebr_fresh_validation.references import read_references
    from pebr_fresh_validation.banks import load_bank
    from pebr_robust8.selection import select
    rows=[]
    for corpus,c in study.corpora.items():
        col=c['collection'];bank=load_bank(study.step5/'mask_banks'/corpus/'mask3711',col,study.protocol)
        refs=read_references(study.step3,col,12,study.protocol)
        for case in study.protocol['parameter_cases']:
            p=parameter_protocol(study.protocol,case)
            if case['case'] not in ('default','lambda0'):
                paths=select(col,bank,refs,p,study.work/'parameter_selection'/corpus/case['case'],study.logger)
            for seed in range(5):
                context={'case':case['case'],'cap':case['cap'],'lambda':case['downside_weight']}
                if case['case'] in ('default','lambda0'):
                    policy='preserve_forward_fixed_cap' if case['case']=='default' else 'preserve_fixed_no_downside'
                    row=shared(study.step5,study.step3,corpus,12,policy,seed,context)
                    row['policy']='preserve_forward_fixed_cap'
                else:
                    row=fit_nmf(study.work/'parameter_models'/corpus/case['case']/('s%d'%seed),c,
                        sparse.load_npz(paths['preserve_forward_fixed_cap']/'transferred.npz'),12,seed,'preserve_forward_fixed_cap',
                        dict(context,corpus=corpus),study.protocol['final_nmf'])
                rows.append(row)
            export_parameters(study.work,rows,False)
    return export_parameters(study.work,rows,True)

def export_parameters(work,rows,require):
    d=work/'parameter_results';write(d/'seed_records.json',rows)
    table(d/'seed_metrics.csv',[{k:v for k,v in r.items() if k!='top_words'} for r in rows])
    ss=summary(rows,['corpus','case','cap','lambda']);table(d/'summaries.csv',ss)
    expected={(c,case,s) for c in ('transport_grievance','chinese_pooled') for case in ('default','cap6','cap2','lambda0','lambda05','lambda2') for s in range(5)}
    actual={(r['corpus'],r['case'],int(r['seed'])) for r in rows}
    if len(actual)!=len(rows) or not actual.issubset(expected):raise ValueError('Duplicate or undeclared parameter record')
    complete=actual==expected
    write(d/'coverage.json',{'complete':complete,'records':len(rows),'expected':60,'new_fits_planned':40,
        'selection_bank':3711,'five_seed_sd_conditional_on_one_bank':True,'default_parameters_retained':True})
    if require and not complete:raise ValueError('Parameter grid incomplete')
    return complete

def sample_collection(full,indices,corpus,ladder,target,directory):
    """Refit minimum DF inside each sample; zero-row exclusion preserves nesting."""
    from transport_topics.data.prepared import PreparedCollection
    from pebr_revision_step3.io import save_receipt,complete
    indices=np.asarray(indices,int);x=full['collection'].counts[indices].tocsr()
    keep=np.asarray((x>0).sum(axis=0)).ravel()>=2
    x=x[:,keep].tocsr();positive=np.asarray(x.sum(axis=1)).ravel()>0
    retained=indices[positive];x=x[positive].tocsr();words=full['vocabulary'][keep]
    if x.shape[0]<=12 or len(words)<12:raise ValueError('A sampled collection cannot support fixed K12')
    col=full['collection'];spec=read(col.directory/'identity.json')
    ident={'original_collection':col.summary['identity_sha256'],'source_rows':col.source_rows[retained].tolist(),
        'vocabulary':words.tolist(),'minimum_df':2,'masked_views':spec.get('masked_views',spec.get('views')),
        'encoder':spec['encoder'],'nominal_target':target,'ladder':ladder,'sampled_full_indices':indices.tolist(),
        'zero_representation_source_rows':col.source_rows[indices[~positive]].tolist()}
    name=('zh' if corpus=='chinese_pooled' else 'tg')+'_l%d_n%d'%(ladder,target)
    info={'name':name,'identity_sha256':key(ident),'posts':x.shape[0],'vocabulary':x.shape[1]}
    directory.mkdir(parents=True,exist_ok=True)
    posts=read(col.directory/'posts.json')
    if complete(directory):
        if read(directory/'identity.json')!=ident:raise ValueError('Saved sample identity changed')
    else:
        sparse.save_npz(directory/'counts.npz',x)
        for n,a in [('vocabulary',words),('folds',col.folds[retained]),('families',col.families[retained]),
                    ('source_rows',col.source_rows[retained]),
                    ('embeddings',np.load(col.directory/'embeddings.npy',allow_pickle=False)[retained])]:np.save(directory/(n+'.npy'),a,allow_pickle=False)
        write(directory/'posts.json',[posts[int(i)] for i in retained]);write(directory/'identity.json',ident)
        save_receipt(directory)
    subset=PreparedCollection(info,directory,x,col.folds[retained],col.families[retained],col.source_rows[retained])
    return {'collection':subset,'vocabulary':words,'reference':full['reference']},retained,ident

def sample_bank(study,corpus,c,retained,records,encoder,directory):
    from pebr_revision_step3.io import complete,save_receipt
    from pebr_fresh_validation.banks import masked_evidence,load_bank
    from transport_topics.proposed.candidates import retrieve
    if complete(directory):return load_bank(directory,c['collection'],study.protocol)
    col=c['collection'];p=study.protocol
    view=masked_evidence(corpus,col,[records[int(i)] for i in retained],p,3711)
    vectors=encoder.encode(view['queries'],label=col.summary['name']+'_visible')
    emb=np.load(col.directory/'embeddings.npy',allow_pickle=False)
    full=np.load(study.corpora[corpus]['collection'].directory/'topic_learning/full_receiver_embeddings.npy',allow_pickle=False)[retained]
    n,v=col.counts.shape;positive=np.ones(n,dtype=bool)
    lists=np.full((n*p['views'],p['candidates_per_view']),-1,dtype=np.int32)
    lists[view['positions']]=retrieve(vectors,view['positions']//p['views'],emb,col.families,col.folds,positive,p['candidates_per_view'])
    available=retrieve(full,np.arange(n),emb,col.families,col.folds,positive,p['candidates_per_view'])
    directory.mkdir(parents=True,exist_ok=True)
    for name in ('base','visible','targets'):sparse.save_npz(directory/(name+'.npz'),view[name])
    for name,array in [('eligible',view['eligible']),('candidates',lists.reshape(n,p['views'],p['candidates_per_view'])),
                       ('full_candidates',available),('full_receiver_embeddings',full)]:np.save(directory/(name+'.npy'),array,allow_pickle=False)
    write(directory/'summary.json',{'collection':col.summary,'queries':len(vectors),'mask':3711,
                                   'donors_and_predictor_training_restricted_to_sample':True,'encoder_identity':encoder.identity})
    save_receipt(directory);return load_bank(directory,col,p)

def sizes(study):
    from pebr_revision_step3.io import complete
    from transport_topics.data.splits import nested_family_samples
    from pebr_fresh_validation.banks import query_records,make_encoder
    from transport_topics.proposed.reference import reference_bank
    from pebr_robust8.selection import select
    p=dict(study.protocol,policies=study.protocol['size_policies']);rows=[];manifests=[]
    for corpus,full in study.corpora.items():
        col=full['collection'];records=query_records(corpus,col,study.work/'tokenizer_cache');encoder=None
        for ladder in p['size_ladders']:
            samples=nested_family_samples(col.families,p['size_targets'],ladder,p['size_namespace']+':'+corpus)
            previous=set()
            for target in p['size_targets']:
                prefix=study.work/'sizes'/corpus/('l%d_n%d'%(ladder,target))
                c,retained,identity=sample_collection(full,samples[target],corpus,ladder,target,prefix/'data')
                if not previous.issubset(set(retained)):raise ValueError('Retained nested samples are not nested')
                previous=set(retained)
                bankdir=prefix/'bank'
                if not complete(bankdir) and encoder is None:encoder=make_encoder(corpus,study.step2,study.work,full['collection'],p,study.logger)
                bank=sample_bank(study,corpus,c,retained,records,encoder,bankdir)
                st=p['reference_predictor'];settings={'folds':5,'seed':0,'tolerance':st['nmf_tolerance'],
                    'max_iterations':st['nmf_max_iterations'],'smoothing':st['smoothing'],'unigram_smoothing':st['unigram_smoothing']}
                refs,audits=reference_bank(study.work,c['collection'].summary,c['collection'].counts,c['collection'].folds,12,settings,study.logger)
                for a in audits:
                    if not a['converged'] or not a['receiver_fold_excluded']:raise ValueError('Sample reference fit failed')
                paths=select(c['collection'],bank,refs,dict(p,size_identity=identity),prefix/'selection',study.logger)
                context={'corpus':corpus,'ladder':ladder,'target':target,'retained_posts':len(retained),'vocabulary':len(c['vocabulary'])}
                manifests.append(dict(context,sampled_source_rows=col.source_rows[samples[target]].tolist(),
                                      retained_source_rows=c['collection'].source_rows.tolist(),zero_rows_excluded=len(samples[target])-len(retained)))
                for policy in p['size_policies']:
                    learned=c['collection'].counts if policy=='nmf_original' else sparse.load_npz(paths[policy]/'transferred.npz')
                    for seed in range(5):
                        study.logger.info('Size fit | %s ladder%d target%d %s seed%d',corpus,ladder,target,policy,seed)
                        rows.append(fit_nmf(prefix/'models'/policy/('s%d'%seed),c,learned,12,seed,policy,context,p['final_nmf']))
                export_sizes(study.work,rows,manifests,False,p)
        del encoder
    return export_sizes(study.work,rows,manifests,True,p)

def export_sizes(work,rows,manifests,require,protocol):
    d=work/'size_results';write(d/'seed_records.json',rows);write(d/'sampling_manifests.json',manifests)
    table(d/'sampling_counts.csv',[{k:v for k,v in r.items() if not k.endswith('source_rows')} for r in manifests])
    ss=summary(rows,['corpus','ladder','target','policy']);table(d/'within_ladder_summaries.csv',ss)
    paired=[]
    for corpus in sorted({r['corpus'] for r in rows}):
        for target in sorted({r['target'] for r in rows if r['corpus']==corpus}):
            for policy in sorted({r['policy'] for r in rows if r['policy']!='nmf_original'}):
                for metric in ('npmi','cv','diversity'):
                    values=[]
                    for ladder in range(5):
                        a={r['seed']:r for r in rows if (r['corpus'],r['target'],r['ladder'],r['policy'])==(corpus,target,ladder,policy)}
                        b={r['seed']:r for r in rows if (r['corpus'],r['target'],r['ladder'],r['policy'])==(corpus,target,ladder,'nmf_original')}
                        if len(a)==len(b)==5:values.append(float(np.mean([a[s][metric]-b[s][metric] for s in range(5)])))
                    if len(values)==5:paired.append({'corpus':corpus,'target':target,'policy':policy,'metric':metric,
                        'mean_paired_gain':float(np.mean(values)),'sample_sd_across_ladder_means':float(np.std(values,ddof=1)),
                        'positive_ladder_means':sum(v>0 for v in values),'ladders':5,'confidence_interval':False})
    table(d/'paired_gains.csv',paired)
    expected={(c,l,t,p,s) for c in protocol['corpora'] for l in protocol['size_ladders'] for t in protocol['size_targets']
              for p in protocol['size_policies'] for s in range(5)}
    actual={(r['corpus'],r['ladder'],r['target'],r['policy'],r['seed']) for r in rows}
    if len(actual)!=len(rows) or not actual.issubset(expected):raise ValueError('Duplicate or undeclared size-analysis record')
    ok=actual==expected and len({(r['corpus'],r['ladder'],r['target']) for r in manifests})==40
    write(d/'coverage.json',{'complete':ok,'seed_records':len(rows),'expected':1000,'sample_conditions':len(manifests),'expected_conditions':40,
        'vocabulary_refitted_per_sample':True,'zero_representations_excluded_and_recorded':True,'samples_overlap':True})
    if require and not ok:raise ValueError('Corpus-size analysis incomplete')
    return ok
