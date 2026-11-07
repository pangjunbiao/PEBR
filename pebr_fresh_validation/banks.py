"""Rebuild masks and visible queries with the original language-specific code."""
from __future__ import annotations
import copy
from importlib import metadata
import time
import numpy as np
from scipy import sparse
from transport_topics.proposed.masks import build_views
from transport_topics.proposed.candidates import retrieve
from transport_topics.chinese.data import visible_queries
from transport_topics.data.prepared import SelectionBank
from pebr_revision_step3.io import canonical_hash,complete,read_json,save_receipt,write_json

def view_configuration(collection,protocol,mask):
    identity=read_json(collection.directory/'identity.json')
    views=copy.deepcopy(identity.get('masked_views',identity.get('views')))
    if not isinstance(views,dict) or views.get('count')!=protocol['views']:raise ValueError('Unexpected frozen view settings')
    return {'method':{'views':views,'predictor':{'mask_seed':int(mask),'outer_seed':2712}},
            'evaluation':{'prediction':{'outer_fraction':.2}}}

def query_records(corpus,collection,work):
    posts=read_json(collection.directory/'posts.json')
    if corpus=='transport_grievance':return posts
    import jieba
    if metadata.version('jieba')!='0.42.1':raise ValueError('Expected the audited Jieba 0.42.1 tokenizer')
    tokenizer=jieba.Tokenizer();work.mkdir(parents=True,exist_ok=True)
    tokenizer.tmp_dir=str(work);tokenizer.cache_file='step4_jieba.cache';tokenizer.initialize()
    result=[]
    for post in posts:
        text=post['original_text'];segments=list(tokenizer.cut(text,HMM=True))
        if ''.join(segments)!=text:raise ValueError('Chinese segmentation did not preserve the original string')
        result.append({'segments':segments,'text':text,'tokens':post['tokens']})
    return result

def masked_evidence(corpus,collection,records,protocol,mask):
    words=np.load(collection.directory/'vocabulary.npy',allow_pickle=False)
    config=view_configuration(collection,protocol,mask)
    if corpus=='transport_grievance':
        texts=[r['cleaned_text'] for r in records]
        view=build_views(collection.counts,words,texts,collection.source_rows,config,False)
        return {key:view[key] for key in ('base','visible','targets','eligible','queries','positions')}
    visible,targets,eligible,queries,positions=visible_queries(collection.counts,words,records,collection.source_rows,config)
    return {'base':collection.counts,'visible':visible,'targets':targets,'eligible':eligible,
            'queries':queries,'positions':positions}

def validate_default_reconstruction(corpus,collection,original,records,protocol,expected_query_identity=None):
    view=masked_evidence(corpus,collection,records,protocol,2711)
    for name in ('base','visible','targets'):
        difference=(view[name]-getattr(original,name)).tocsr()
        if difference.nnz and np.max(np.abs(difference.data))>2e-12:
            raise ValueError('Default masked evidence cannot be reconstructed: '+corpus+' '+name)
    if not np.array_equal(view['eligible'],original.eligible):raise ValueError('Default mask eligibility differs')
    hash_path=original.directory/'query_hashes.npy'
    if hash_path.is_file():
        expected=np.load(hash_path,allow_pickle=False)
        if not np.array_equal(expected,np.asarray([canonical_hash(q) for q in view['queries']])):
            raise ValueError('English default query text differs from its saved hashes')
    if corpus=='chinese_pooled' and expected_query_identity is not None:
        encoder=read_json(original.directory/'encoder_identity.json')['encoder']
        actual=canonical_hash({'encoder':encoder,'ordered_texts':list(view['queries'])})
        if actual!=expected_query_identity:
            raise ValueError('Chinese default query strings differ from the accepted Step 1 cache identity')
    return {'corpus':corpus,'passed':True,'default_mask_base':2711,'queries':len(view['queries']),
            'visible_targets_eligibility_match':True,'saved_query_hashes_checked':hash_path.is_file(),
            'saved_chinese_query_cache_identity_checked':expected_query_identity is not None}

def make_encoder(corpus,step2,work,collection,protocol,logger):
    """Read frozen assets in Step 2; redirect embedding caches to Step 4."""
    identity=read_json(collection.directory/'identity.json')
    runtime=protocol['runtime'];run_directory=work/'encoder_audits'/corpus
    run_directory.mkdir(parents=True,exist_ok=True)
    if corpus=='transport_grievance':
        from transport_topics.representations.embeddings import FrozenEncoder
        expected=identity['encoder']
        config={'method':{'encoder':copy.deepcopy(expected)},
            'preparation':{'embedding_batch_size':runtime['embedding_batch_size'],'gpu_memory_mb':2048},
            'runtime':{'cpu_threads':runtime['cpu_threads'],'gpu_index':runtime['gpu_index']}}
        gpu=expected['provider']=='CUDAExecutionProvider'
        encoder=FrozenEncoder(step2,config,{'device_selection':{'torch_cuda_execution_verified':gpu}},logger,run_directory)
    else:
        from transport_topics.chinese.embeddings import FrozenChineseEncoder
        expected=read_json(collection.directory/'topic_learning/encoder_identity.json')['encoder']
        encoder_spec=copy.deepcopy(identity['encoder'])
        config={'encoder':encoder_spec,'download_missing_assets':False,
            'runtime':{'cpu_threads':runtime['cpu_threads'],'gpu_index':runtime['gpu_index'],
                'embedding_batch_size':runtime['embedding_batch_size'],'gpu_memory_mb':2048}}
        device='cuda:'+str(runtime['gpu_index']) if expected['provider']=='CUDAExecutionProvider' else 'cpu'
        encoder=FrozenChineseEncoder(step2,config,device,logger,run_directory)
    if encoder.identity!=expected:
        raise ValueError('Encoder identity/backend differs from the accepted default bank: '+corpus)
    encoder.root=work
    write_json(run_directory/'identity.json',{'expected':expected,'actual':encoder.identity,'identical':True,
                'assets_read_from':str(step2),'cache_written_to':str(work),'downloads_allowed':False})
    return encoder

def load_bank(directory,collection,protocol):
    if not complete(directory):raise ValueError('Repeated mask bank is incomplete')
    bank=SelectionBank('topic_learning',directory,
        *[sparse.load_npz(directory/(n+'.npz')).tocsr() for n in ('base','visible','targets')],
        *[np.load(directory/(n+'.npy'),allow_pickle=False) for n in ('eligible','candidates','full_candidates')])
    n,v=collection.counts.shape;m=protocol['views'];limit=protocol['candidates_per_view']
    if bank.candidates.shape!=(n,m,limit) or bank.visible.shape!=(n*m,v) or bank.targets.shape!=(n*m,v):
        raise ValueError('Repeated bank dimensions differ')
    if bank.visible.multiply(bank.targets).nnz:raise ValueError('Visible/target overlap')
    if (bank.base-collection.counts).nnz:raise ValueError('A repeated fitting bank altered the original counts')
    for i in range(n):
        ids=bank.candidates[i][bank.candidates[i]>=0]
        if len(ids) and (np.any(collection.folds[ids]==collection.folds[i]) or np.any(collection.families[ids]==collection.families[i])):
            raise ValueError('A repeated bank donor crosses the receiving family/fold boundary')
    return bank

def prepare_bank(corpus,collection,original,records,protocol,mask,encoder,directory,logger):
    if complete(directory):return load_bank(directory,collection,protocol)
    directory.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    view=masked_evidence(corpus,collection,records,protocol,mask)
    logger.info('Retrieval | %s mask=%d | %d visible queries',corpus,mask,len(view['queries']))
    vectors=encoder.encode(view['queries'],label=corpus+'_mask'+str(mask))
    embeddings=np.load(collection.directory/'embeddings.npy',allow_pickle=False)
    n,v=collection.counts.shape;m=protocol['views'];limit=protocol['candidates_per_view']
    if vectors.shape!=(len(view['queries']),embeddings.shape[1]) or not np.isfinite(vectors).all():
        raise ValueError('Encoded query dimensions/values differ')
    positive=np.asarray(collection.counts.sum(axis=1)).ravel()>0
    candidates=np.full((n*m,limit),-1,dtype=np.int32)
    candidates[view['positions']]=retrieve(vectors,view['positions']//m,embeddings,
            collection.families,collection.folds,positive,limit)
    for name in ('base','visible','targets'):sparse.save_npz(directory/(name+'.npz'),view[name])
    for name,value in [('eligible',view['eligible']),('candidates',candidates.reshape(n,m,limit)),
        ('full_candidates',original.full_candidates),
        ('full_receiver_embeddings',np.load(original.directory/'full_receiver_embeddings.npy',allow_pickle=False)),
        ('query_positions',view['positions']),('query_hashes',np.asarray([canonical_hash(q) for q in view['queries']],dtype='U64'))]:
        np.save(directory/(name+'.npy'),value,allow_pickle=False)
    write_json(directory/'summary.json',{'corpus':corpus,'mask_base':int(mask),'queries':len(view['queries']),
        'eligible_receivers':int(view['eligible'].sum()),'selection_seed_changes_only':True,
        'encoder_identity':encoder.identity,'masked_retrieval_seconds':time.perf_counter()-started,
        'full_post_embeddings_and_availability_reused':True})
    save_receipt(directory)
    return load_bank(directory,collection,protocol)
