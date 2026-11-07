"""Text-only Chinese pooling and preparation for the unchanged borrowing engine."""
from __future__ import annotations

from collections import Counter
import csv
import html
from importlib import metadata
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
from scipy import sparse

from ..config import canonical_hash
from ..data.families import text_families
from ..data.splits import family_folds
from ..data.prepared import PreparedCollection
from ..data.preparation import check_receipt, complete, save_array, save_matrix
from ..runtime.artifacts import file_hash, write_json
from ..proposed.masks import mask_seed
from ..proposed.candidates import retrieve

NEGATORS={'不','没','没有','无','未','别','不能','不要','不是','不好'}
WORD=re.compile(r'[a-z\u3400-\u4dbf\u4e00-\u9fff]+',re.I)


def clean(text):
    value=unicodedata.normalize('NFKC',html.unescape(str(text))).strip()
    if value.startswith('[') and value.endswith(']'):value=value[1:-1]
    value=re.sub(r'https?://\S+|www\.\S+',' ',value)
    value=re.sub(r'@[\w\u4e00-\u9fff-]+',' ',value)
    for marker in ('网页链接','转发微博','微博视频','秒拍视频'):value=value.replace(marker,' ')
    return re.sub(r'\s+',' ',value).strip().lower()


def read_records(root, cfg, logger):
    import jieba
    stop_path=root/cfg['stopwords']
    if file_hash(stop_path)!=cfg['stopwords_sha256']:raise ValueError('The fixed Chinese stopword list changed')
    stop=set(stop_path.read_text(encoding='utf-8').splitlines())-NEGATORS
    records=[];seen={};sources=[];duplicates=[];empty=0
    for name in cfg['files']:
        path=root/cfg['raw_directory']/name
        if not path.is_file():raise FileNotFoundError('Put the original CSV here: '+str(path))
        decoded=None
        for encoding in ('utf-8-sig','gb18030'):
            try:decoded=path.read_text(encoding=encoding);break
            except UnicodeDecodeError:continue
        if decoded is None:raise ValueError('CSV is neither UTF-8 nor GB18030: '+str(path))
        import io
        reader=csv.DictReader(io.StringIO(decoded,newline=''))
        if cfg['text_column'] not in (reader.fieldnames or []):
            raise ValueError(f"Missing text column {cfg['text_column']!r} in {name}")
        count=0
        for row_number,row in enumerate(reader):
            count+=1;text=clean(row.get(cfg['text_column']) or '')
            if not text or not WORD.search(text):empty+=1;continue
            if text in seen:
                duplicates.append({'source_file':name,'source_row':row_number,'retained_pool_row':seen[text]});continue
            segments=list(jieba.cut(text,HMM=True))
            if ''.join(segments)!=text:raise AssertionError('Chinese segmentation changed original character order')
            tokens=[w for w in segments if WORD.fullmatch(w) and w not in stop]
            seen[text]=len(records)
            records.append({'pool_index':len(records),'source_file':name,'source_row':row_number,
                            'text':text,'segments':segments,'tokens':tokens})
        sources.append({'filename':name,'rows':count,'sha256':file_hash(path),'encoding':encoding,
                        'ignored_columns':[x for x in reader.fieldnames if x!=cfg['text_column']]})
        logger.info('CHINESE RAW | %s | %d rows | text column only.',name,count)
    if len(records)<10:raise ValueError('Too few nonduplicate Chinese posts')
    audit={'sources':sources,'raw_posts':sum(r['rows'] for r in sources),'empty_bodies':empty,
           'normalized_duplicates':len(duplicates),'normalized_unique_posts':len(records),
           'duplicates':duplicates,'jieba_version':metadata.version('jieba'),
           'stopwords_sha256':file_hash(stop_path),'protected_negations':sorted(NEGATORS),
           'fitting_uses_city_time_account_or_engagement':False,
           'collection_scope':'all retained posts; no 600-post pilot sample',
           'body_and_tokens_sha256':canonical_hash([(r['text'],r['tokens']) for r in records])}
    return records,audit


def make_counts(records,min_df):
    df=Counter(w for r in records for w in set(r['tokens']))
    vocabulary=np.asarray(sorted(w for w,c in df.items() if c>=min_df))
    lookup={str(w):i for i,w in enumerate(vocabulary)}
    row_ids=[];column_ids=[];values=[]
    for i,r in enumerate(records):
        for word,count in Counter(r['tokens']).items():
            if word in lookup:row_ids.append(i);column_ids.append(lookup[word]);values.append(count)
    matrix=sparse.csr_matrix((values,(row_ids,column_ids)),shape=(len(records),len(vocabulary)),dtype=np.float64)
    return matrix,vocabulary


def prepare_collection(root,cfg,protocol,records,name,logger):
    selected=records if name=='pooled' else [r for r in records if Path(r['source_file']).stem==name]
    counts,words=make_counts(selected,cfg['min_document_frequency'])
    keep=np.asarray(counts.sum(axis=1)).ravel()>0
    dropped=[r['pool_index'] for r,k in zip(selected,keep) if not k]
    selected=[r for r,k in zip(selected,keep) if k];counts=counts[keep].tocsr()
    if counts.shape[0]<10 or len(words)<max(cfg['topic_counts']):
        raise ValueError('The retained Chinese collection cannot support the fixed topic counts')
    spec={'schema':1,'language':'zh','collection':name,'count_shape':counts.shape,
          'records':canonical_hash([(r['pool_index'],r['text'],r['tokens']) for r in selected]),
          'min_df':cfg['min_document_frequency'],'family_threshold':cfg['family_threshold'],
          'predictor_seeds':{k:protocol['method']['predictor'][k] for k in ('folds','fold_seed','mask_seed','outer_seed')},
          'vocabulary':words.tolist(),'jieba_version':metadata.version('jieba'),
          'stopwords_sha256':cfg['stopwords_sha256'],'implementation':file_hash(Path(__file__)),
          'family_code':file_hash(Path(__file__).parents[1]/'data/families.py'),
          'fold_code':file_hash(Path(__file__).parents[1]/'data/splits.py'),
          'encoder':cfg['encoder'],'encoder_manifest_sha256':file_hash(root/cfg['encoder']['manifest']),
          'masked_views':protocol['method']['views'],'neighbors':protocol['method']['candidates']['per_view']}
    identity=canonical_hash(spec)
    directory=root/'data/processed/chinese_transport'/('chinese_'+name+'_'+identity[:16])
    directory.mkdir(parents=True,exist_ok=True)
    if check_receipt(directory):
        families=np.load(directory/'families.npy',allow_pickle=False)
        folds=np.load(directory/'folds.npy',allow_pickle=False)
        logger.info('Reusing prepared Chinese text collection: %s.',name)
    else:
        families=text_families([r['text'] for r in selected],cfg['family_threshold'],128,logger)
        settings=protocol['method']['predictor']
        folds=family_folds(families,settings['folds'],settings['fold_seed'])
        save_matrix(directory/'counts.npz',counts)
        for key,value in [('vocabulary',words),('source_rows',np.asarray([r['pool_index'] for r in selected],dtype=np.int64)),
                          ('families',families),('folds',folds)]:save_array(directory/(key+'.npy'),value)
        write_json(directory/'posts.json',[{'tokens':r['tokens'],'cleaned_text':' '.join(r['tokens']),
                    'original_text':r['text'],'source_file':r['source_file'],'source_row':r['source_row']} for r in selected],overwrite=True)
        write_json(directory/'identity.json',spec,overwrite=True)
        complete(directory)
    rows=np.asarray([r['pool_index'] for r in selected],dtype=np.int64)
    summary={'name':'chinese_'+name,'path':directory.relative_to(root).as_posix(),'language':'zh',
             'identity_sha256':identity,'posts':len(selected),'vocabulary':len(words),
             'text_families':len(np.unique(families)),'zero_count_rows_excluded':dropped,
             'full_retained_collection':True,'sampled_600_post_pilot':False}
    return PreparedCollection(summary,directory,counts,folds,families,rows),selected


def visible_queries(counts,words,records,source_rows,protocol):
    settings=protocol['method'];n,v=counts.shape;masks=settings['views']['count']
    visible=sparse.lil_matrix((n*masks,v),dtype=np.float64)
    targets=sparse.lil_matrix((n*masks,v),dtype=np.float64)
    eligible=np.diff(counts.indptr)>=settings['views']['min_word_types']
    queries=[];positions=[]
    for i in range(n):
        if not eligible[i]:continue
        ids=counts[i].indices
        rng=np.random.default_rng(mask_seed(int(source_rows[i]),settings['predictor']['mask_seed']))
        for m in range(masks):
            order=rng.permutation(ids)
            cut=max(2,int(np.floor(len(ids)*(1-settings['views']['hide_fraction']))))
            if len(ids)-cut<2:raise ValueError('Too few hidden Chinese word types in a valid view')
            observed,hidden=order[:cut],order[cut:]
            values=counts[i,observed].toarray().ravel()
            visible[i*masks+m,observed]=values/values.sum()
            targets[i*masks+m,hidden]=counts[i,hidden].toarray()
            blocked=set(words[hidden])
            kept=[token for token in records[i]['segments'] if token not in blocked]
            if set(kept)&blocked:raise AssertionError('A hidden retained segment entered a retrieval query')
            queries.append(''.join(' ' if token in blocked else token for token in records[i]['segments']))
            positions.append(i*masks+m)
    visible=visible.tocsr();targets=targets.tocsr()
    if visible.multiply(targets).nnz:raise AssertionError('Visible and hidden count supports overlap')
    return visible,targets,eligible,queries,np.asarray(positions,dtype=np.int64)


def prepare_bank(collection,records,protocol,encoder,logger):
    directory=collection.directory/'topic_learning'
    if check_receipt(directory):
        recorded=json.loads((directory/'encoder_identity.json').read_text(encoding='utf-8'))
        if recorded['identity']!=canonical_hash(encoder.identity):
            raise ValueError('Prepared Chinese retrieval features use another encoder backend/implementation. Preserve the existing cache and use the same backend, or move the printed Chinese collection cache aside before rebuilding: '+str(collection.directory))
        logger.info('Reusing Chinese masked-view retrieval bank: %s.',collection.summary['name']);return
    directory.mkdir(exist_ok=True)
    counts=collection.counts;n,v=counts.shape
    words=np.load(collection.directory/'vocabulary.npy',allow_pickle=False)
    visible,targets,eligible,queries,positions=visible_queries(counts,words,records,collection.source_rows,protocol)
    full=encoder.encode([r['text'] for r in records],collection.summary['name']+'_full')
    masked=encoder.encode(queries,collection.summary['name']+'_visible')
    save_array(collection.directory/'embeddings.npy',full)
    m=protocol['method']['views']['count'];neighbors=protocol['method']['candidates']['per_view']
    positive=np.asarray(counts.sum(axis=1)).ravel()>0
    candidates=np.full((n*m,neighbors),-1,dtype=np.int32)
    if len(positions):
        candidates[positions]=retrieve(masked,positions//m,full,collection.families,collection.folds,positive,neighbors)
    full_lists=retrieve(full,np.arange(n),full,collection.families,collection.folds,positive,neighbors)
    for name,matrix in [('base',counts),('visible',visible),('targets',targets)]:save_matrix(directory/(name+'.npz'),matrix)
    for name,array in [('eligible',eligible),('candidates',candidates.reshape(n,m,neighbors)),
                       ('full_candidates',full_lists),('full_receiver_embeddings',full)]:save_array(directory/(name+'.npy'),array)
    write_json(directory/'language_audit.json',{'masked_queries':len(queries),'language':'zh',
               'mask_unit':'retained Jieba word type; all matching original segments removed',
               'hidden_visible_count_overlap':0,'fitting_labels_used':False},overwrite=True)
    write_json(directory/'encoder_identity.json',{'identity':canonical_hash(encoder.identity),'encoder':encoder.identity},overwrite=True)
    complete(directory)
    complete(collection.directory)
    logger.info('CHINESE RETRIEVAL COMPLETE | %d posts | %d eligible | %d masked queries.',n,int(eligible.sum()),len(queries))
