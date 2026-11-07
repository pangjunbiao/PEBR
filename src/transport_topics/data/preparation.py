"""Reusable, checksummed preparation for the registered fitting collections."""
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
from scipy import sparse
from ..config import canonical_hash
from ..runtime.artifacts import write_json, file_hash, atomic_write
from .loaders import load_primary, load_external
from .preprocessing import deduplicate
from .families import text_families, cross_source_exclusions
from .splits import family_folds, nested_family_samples
from ..representations.counts import count_representation
from ..representations.tfidf import idf_weights, transform
from ..proposed.masks import build_views
from ..proposed.candidates import retrieve


def frozen_json(path,value):
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=json.loads(json.dumps(value)):
            raise ValueError('Frozen metadata differs; preserve and inspect: '+str(path))
    else: write_json(path,value)


def check_receipt(directory):
    receipt=directory/'complete.json'
    if not receipt.exists(): return False
    record=json.loads(receipt.read_text(encoding='utf-8'))
    for relative,sha in record['files'].items():
        if not (directory/relative).is_file() or file_hash(directory/relative)!=sha:
            raise ValueError('Prepared artifact integrity failure: '+str(directory/relative))
    return True


def complete(directory):
    entries={p.relative_to(directory).as_posix():file_hash(p) for p in sorted(directory.rglob('*'))
             if p.is_file() and p.name!='complete.json'}
    write_json(directory/'complete.json',{'files':entries},overwrite=True)


def save_array(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.partial.npy')
    np.save(temporary,np.asarray(value),allow_pickle=False); temporary.replace(path)


def save_matrix(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.partial.npz')
    sparse.save_npz(temporary,value.tocsr()); temporary.replace(path)


def parent_pool(root,posts,texts,name,source_hash,config,logger):
    spec={'schema':2,'source':source_hash,'texts':canonical_hash(texts),
          'family_threshold':config['preparation']['family_threshold'],
          'implementation':file_hash(Path(__file__).with_name('families.py'))}
    directory=root/'data/processed'/name/('pool_'+canonical_hash(spec)[:16])
    directory.mkdir(parents=True,exist_ok=True)
    if check_receipt(directory):
        logger.info('Reusing verified family pool: %s.',name)
        return np.load(directory/'families.npy',allow_pickle=False)
    families=text_families(texts,config['preparation']['family_threshold'],config['preparation']['family_block_size'],logger)
    save_array(directory/'families.npy',families)
    frozen_json(directory/'identity.json',spec)
    complete(directory)
    return families


def bank_for_condition(root,directory,counts,words,texts,rows,families,folds,embeddings,encoder,config,logger,audit):
    scope='outer_audit' if audit else 'topic_learning'
    target=directory/scope
    if check_receipt(target):
        logger.info('Reusing verified %s bank.',scope)
        return json.loads((target/'audit.json').read_text(encoding='utf-8'))
    target.mkdir(parents=True,exist_ok=True)
    view=build_views(counts,words,texts,rows,config,audit)
    M=config['method']['views']['count']; L=config['method']['candidates']['per_view']
    encoded=encoder.encode(view['queries'],scope+' masked views')
    receiver_embeddings=encoder.encode(view['full_queries'],scope+' final receiver views') if audit else embeddings
    positive=np.asarray(counts.sum(axis=1)).ravel()>0
    receivers=view['positions']//M
    lists=retrieve(encoded,receivers,embeddings,families,folds,positive,L)
    per_view=np.full((len(texts)*M,L),-1,dtype=np.int32)
    per_view[view['positions']]=lists
    full=retrieve(receiver_embeddings,np.arange(len(texts)),embeddings,families,folds,positive,L)
    for receiver,candidates in zip(receivers,lists):
        ids=candidates[candidates>=0]
        if np.any(folds[ids]==folds[receiver]) or np.any(families[ids]==families[receiver]) or np.any(~positive[ids]):
            raise AssertionError('Donor eligibility boundary failed')
    if view['visible'].multiply(view['targets']).nnz:
        raise AssertionError('Visible and hidden inner evidence overlap')
    if audit:
        if view['base'].multiply(view['outer']).nnz: raise AssertionError('Outer targets remain in base counts')
        expanded=view['outer'][np.repeat(np.arange(len(texts)),M)]
        if expanded.multiply(view['visible']+view['targets']).nnz: raise AssertionError('Outer target leaked into inner evidence')
    for label in ('base','outer','visible','targets'): save_matrix(target/(label+'.npz'),view[label])
    save_array(target/'eligible.npy',view['eligible'])
    save_array(target/'candidates.npy',per_view.reshape(len(texts),M,L))
    save_array(target/'full_candidates.npy',full)
    save_array(target/'full_receiver_embeddings.npy',receiver_embeddings)
    # Embedding cache is content addressed; avoid duplicating large masked arrays.
    query_hashes=np.array([canonical_hash(text) for text in view['queries']],dtype='U64')
    save_array(target/'query_hashes.npy',query_hashes)
    save_array(target/'query_positions.npy',view['positions'])
    summary={'scope':scope,'posts':len(texts),'eligible_receivers':int(view['eligible'].sum()),
             'masked_queries':len(view['queries']),'inner_query_leaks':0,'outer_query_leaks':0,
             'donor_boundary_violations':0,'outer_target_tokens':int(view['outer'].sum()),
             'view_count':M,'neighbors_per_view':L,'count_vocabulary':len(words),
             'protected_outer_targets':'outer.npz: evaluation-only; not supplied as fitting inputs',
             'queries_use_visible_text_only':True}
    frozen_json(target/'audit.json',summary); complete(target)
    logger.info('%s bank complete: %d eligible receivers, %d masked queries.',scope,summary['eligible_receivers'],len(view['queries']))
    return summary


def prepare_condition(root,name,posts,texts,source_hash,encoder,config,logger):
    from importlib import metadata
    prep=config['preparation']
    spec={'schema':2,'source_hash':source_hash,'source_rows':[p.source_row for p in posts],
          'implementation':{relative:file_hash(root/'src/transport_topics'/relative) for relative in (
            'data/preparation.py','data/preprocessing.py','data/families.py','data/splits.py',
            'representations/counts.py','representations/tfidf.py','proposed/masks.py','proposed/candidates.py')},
          'numerical_versions':{name:metadata.version(name) for name in ('numpy','scipy','scikit-learn')},
          'texts_hash':canonical_hash(texts),'encoder':encoder.identity,
          'min_df':prep['min_document_frequency'],'family_threshold':prep['family_threshold'],
          'views':config['method']['views'],'predictor_seeds':{k:config['method']['predictor'][k] for k in ('folds','fold_seed','mask_seed','outer_seed')},
          'neighbors':config['method']['candidates']['per_view'],'outer_fraction':config['evaluation']['prediction']['outer_fraction']}
    key=canonical_hash(spec)
    dataset='indian_transport' if name.startswith('primary') else 'dabiri_traffic'
    directory=root/'data/processed'/dataset/(name+'_'+key[:16])
    directory.mkdir(parents=True,exist_ok=True)
    if check_receipt(directory):
        logger.info('Reusing complete fitting collection: %s.',name)
        return json.loads((directory/'condition.json').read_text(encoding='utf-8'))
    logger.info('Preparing %s: %d original records.',name,len(posts))
    counts,words,stop_hash=count_representation(texts,prep['min_document_frequency'])
    families=text_families(texts,prep['family_threshold'],prep['family_block_size'])
    if name=='primary_full' and (len(posts),len(words),len(np.unique(families)))!=(619,1296,619):
        raise AssertionError('Primary preparation differs from the locked 619-post/1296-term audit')
    folds=family_folds(families,config['method']['predictor']['folds'],config['method']['predictor']['fold_seed'])
    rows=np.array([p.source_row for p in posts],dtype=np.int64)
    embeddings=encoder.encode(texts,name+' full posts')
    weights=idf_weights(counts); tfidf=transform(counts,weights)
    save_matrix(directory/'counts.npz',counts); save_matrix(directory/'tfidf.npz',tfidf)
    for label,value in (('vocabulary',words),('source_rows',rows),('families',families),('folds',folds),('idf',weights),('embeddings',embeddings)):
        save_array(directory/(label+'.npy'),value)
    frozen_json(root/'data/manifests/folds'/(key+'.json'),{'condition':name,'identity':key,
                'source_rows':rows.tolist(),'family_ids':families.tolist(),'fold_ids':folds.tolist()})
    records=[dict(asdict(p),cleaned_text=t) for p,t in zip(posts,texts)]
    frozen_json(directory/'posts.json',records); frozen_json(directory/'identity.json',spec)
    banks={scope:bank_for_condition(root,directory,counts,words,texts,rows,families,folds,embeddings,
                    encoder,config,logger,audit) for scope,audit in [('topic_learning',False),('outer_audit',True)]}
    lengths=np.asarray(counts.sum(axis=1)).ravel()
    summary={'name':name,'path':directory.relative_to(root).as_posix(),'identity_sha256':key,
             'posts':len(posts),'vocabulary':len(words),'families':len(np.unique(families)),
             'fold_sizes':np.bincount(folds,minlength=config['method']['predictor']['folds']).tolist(),
             'zero_count_posts':int((lengths==0).sum()),'mean_tokens':float(lengths.mean()),
             'median_tokens':float(np.median(lengths)),'stopword_hash':stop_hash,'banks':banks}
    if name=='primary_full':
        if (len(posts),len(words),summary['families'])!=(619,1296,619):
            raise AssertionError('Primary preparation differs from the locked 619-post/1296-term audit')
    frozen_json(directory/'condition.json',summary); complete(directory)
    return summary


def freeze_query_cohort(root,condition,annotations):
    directory=root/condition['path']
    rows=np.load(directory/'source_rows.npy',allow_pickle=False)
    families=np.load(directory/'families.npy',allow_pickle=False); folds=np.load(directory/'folds.npy',allow_pickle=False)
    categories=[annotations[int(row)] for row in rows]
    queries=[]
    for q,labels in enumerate(categories):
        if len(labels)!=1: continue
        eligible=(families!=families[q]) & (folds!=folds[q]); eligible[q]=False
        relevant=sum(eligible[i] and labels[0] in categories[i] for i in range(len(rows)))
        if relevant:
            queries.append({'document_index':q,'source_row':int(rows[q]),'category':labels[0],
                            'eligible_candidates':int(eligible.sum()),'relevant_candidates':int(relevant)})
    payload={'condition_identity':condition['identity_sha256'],'queries':queries,
             'query_count':len(queries),'catalogue_rule':'All other retained posts outside query family and fold; independent of representations',
             'zero_query_policy':'Return empty response; keep fixed catalogue and evaluation cohort',
             'relevance':'Candidate carries the query substantive service category; not identical-incident ground truth'}
    path=root/'data/manifests/queries'/('primary_'+condition['identity_sha256'][:16]+'.json')
    frozen_json(path,payload)
    return {'path':path.relative_to(root).as_posix(),'query_count':len(queries),
            'category_counts':{category:sum(q['category']==category for q in queries) for category in sorted({q['category'] for q in queries})}}


def prepare_collections(root,encoder,config,logger,sources):
    from .sources import PRIMARY_SHA, EXTERNAL_SHA
    primary,annotations=load_primary(root/sources['primary']['path'])
    primary,texts,duplicates=deduplicate(primary)
    external,external_audit=load_external(root/sources['external']['path'])
    external,ext_texts,ext_duplicates=deduplicate(external)
    duplicates_receipt={'primary':duplicates,'external':ext_duplicates}
    frozen_json(root/'data/manifests/sources'/('exact_duplicates_'+canonical_hash(duplicates_receipt)[:16]+'.json'),duplicates_receipt)
    prep=config['preparation']
    parent=parent_pool(root,external,ext_texts,'dabiri_traffic',EXTERNAL_SHA,config,logger)
    excluded=cross_source_exclusions(texts,ext_texts,parent,prep['family_threshold'],prep['family_block_size'])
    external=[p for p,drop in zip(external,excluded) if not drop]
    ext_texts=[t for t,drop in zip(ext_texts,excluded) if not drop]
    parent=parent[~excluded]
    logger.info('External source: %d rows, %d traffic rows, %d normalized unique; %d cross-source records excluded.',
                external_audit['raw_rows'],external_audit['transport_rows'],len(external),int(excluded.sum()))
    # Annotations are written separately; no preparation function receives them until query freezing.
    ann_path=root/'data/annotations/indian_transport'/('categories_'+PRIMARY_SHA[:16]+'.json')
    frozen_json(ann_path,{'source_sha256':PRIMARY_SHA,'categories_by_source_row':annotations,'evaluation_only':True})
    full_scope=prep['scope']=='full'
    primary_families=text_families(texts,prep['family_threshold'],prep['family_block_size'])
    frozen_samples={'schema':2,'external_source':EXTERNAL_SHA,'primary_source':PRIMARY_SHA,
                    'primary_ladders':{},'external_samples':{},'sample_indices':list(range(5)),
                    'family_policy':'Whole parent families in sampling; within-collection family TF-IDF refitted for folds'}
    for index in range(5):
        ladders=nested_family_samples(primary_families,config['studies']['corpus_size']['sizes'],index,'primary_ladder')
        frozen_samples['primary_ladders'][str(index)]={str(size):[primary[i].source_row for i in ids] for size,ids in ladders.items()}
        ladders=nested_family_samples(parent,config['studies']['corpus_size']['sizes'],index,'external_sample')
        frozen_samples['external_samples'][str(index)]={str(size):[external[i].source_row for i in ids] for size,ids in ladders.items()}
    sample_key=canonical_hash(frozen_samples)
    sample_path=root/'data/manifests/samples'/('registered_'+sample_key[:16]+'.json')
    frozen_json(sample_path,frozen_samples)
    collection_posts=primary if full_scope else primary[:prep['smoke_posts']]
    collection_text=texts if full_scope else texts[:prep['smoke_posts']]
    name='primary_full' if full_scope else 'primary_smoke'
    conditions=[prepare_condition(root,name,collection_posts,collection_text,PRIMARY_SHA,encoder,config,logger)]
    cohort=freeze_query_cohort(root,conditions[0],annotations)
    external_lookup={p.source_row:i for i,p in enumerate(external)}
    for index in range(5 if full_scope else 1):
        source_rows=frozen_samples['external_samples'][str(index)][str(prep['external_target_posts'])]
        ids=[external_lookup[row] for row in source_rows]
        if not full_scope: ids=ids[:prep['smoke_posts']]
        conditions.append(prepare_condition(root,'external_sample_%d'%index if full_scope else 'external_smoke',
                          [external[i] for i in ids],[ext_texts[i] for i in ids],EXTERNAL_SHA,encoder,config,logger))
    return {'scope':prep['scope'],'complete_registered_preparation':full_scope,'conditions':conditions,
            'primary_raw_rows':len(primary)+len(duplicates),'primary_unique_posts':len(primary),
            'primary_exact_duplicates':len(duplicates),'external':dict(external_audit,
               normalized_unique_after_overlap=len(external),exact_duplicates=len(ext_duplicates),
               cross_source_exclusions=int(excluded.sum())),
            'sample_manifest':sample_path.relative_to(root).as_posix(),'sample_manifest_sha256':file_hash(sample_path),
            'retrieval_cohort':cohort,'annotation_file':ann_path.relative_to(root).as_posix(),
            'topic_models_fitted':False,'performance_scores_computed':False}
