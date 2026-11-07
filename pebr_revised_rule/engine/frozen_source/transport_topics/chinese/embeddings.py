"""Pinned multilingual frozen features and language-appropriate baseline inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request

import numpy as np

from ..config import canonical_hash
from ..runtime.artifacts import file_hash, write_json
from ..data.preparation import check_receipt, complete, save_array


def prepare_assets(root,cfg,logger):
    spec=json.loads((root/cfg['encoder']['manifest']).read_text(encoding='utf-8'))
    directory=root/cfg['encoder']['directory']
    for record in spec['files']:
        path=directory/record['file'];path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            if path.stat().st_size!=record['bytes'] or file_hash(path)!=record['sha256']:
                raise ValueError('Existing Chinese encoder file differs from its pinned source: '+str(path))
            continue
        if not cfg['download_missing_assets']:
            raise FileNotFoundError('Missing pinned multilingual encoder asset: '+str(path))
        url=cfg['encoder']['base_url'].rstrip('/')+'/'+cfg['encoder']['name']+'/resolve/'+spec['revision']+'/'+record['file']
        partial=path.with_suffix(path.suffix+'.partial')
        for attempt in range(3):
            try:
                logger.info('Downloading Chinese encoder asset: %s (%.1f MiB).',record['file'],record['bytes']/2**20)
                request=urllib.request.Request(url,headers={'User-Agent':'proposed-transport-research/1.0'})
                with urllib.request.urlopen(request,timeout=60) as response,partial.open('wb') as stream:
                    last=time.monotonic()
                    while True:
                        block=response.read(2**20)
                        if not block:break
                        stream.write(block)
                        if time.monotonic()-last>=10:
                            logger.info('Chinese encoder download %s: %.1f / %.1f MiB.',record['file'],stream.tell()/2**20,record['bytes']/2**20)
                            last=time.monotonic()
                if partial.stat().st_size!=record['bytes'] or file_hash(partial)!=record['sha256']:
                    raise ValueError('Downloaded multilingual asset hash/size mismatch: '+record['file'])
                partial.replace(path);break
            except Exception:
                if attempt==2:raise
                logger.warning('Asset download attempt %d failed; retrying the same pinned file.',attempt+1)
                time.sleep(attempt+1)
    logger.info('Pinned multilingual encoder assets verified; the English encoder is unchanged.')
    return directory,spec


class FrozenChineseEncoder:
    def __init__(self,root,cfg,device,logger,run_directory):
        from tokenizers import Tokenizer
        # Load CUDA/cuDNN from the existing torch installation before ORT.
        if device.startswith('cuda'):
            import torch
        import onnxruntime as ort
        directory,spec=prepare_assets(root,cfg,logger)
        self.root,self.cfg,self.logger=root,cfg,logger
        self.tokenizer=Tokenizer.from_file(str(directory/'tokenizer.json'))
        if self.tokenizer.token_to_id('<pad>')!=1:raise ValueError('Unexpected multilingual padding token')
        self.tokenizer.enable_truncation(max_length=cfg['encoder']['max_sequence_length'])
        self.tokenizer.enable_padding(pad_id=1,pad_token='<pad>')
        options=ort.SessionOptions();options.intra_op_num_threads=cfg['runtime']['cpu_threads']
        options.inter_op_num_threads=1;options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        options.log_severity_level=3
        gpu=device.startswith('cuda')
        if gpu:
            if 'CUDAExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError('CUDA torch is available but ONNX Runtime has no CUDA provider. Preserve torch and install a compatible onnxruntime-gpu backend in this same interpreter, or explicitly use --device cpu.')
            if hasattr(ort,'preload_dlls'):ort.preload_dlls()
            options.enable_profiling=True;options.profile_file_prefix=str(run_directory/'chinese_encoder_profile')
            providers=[('CUDAExecutionProvider',{'device_id':cfg['runtime']['gpu_index'],'use_tf32':'0',
                         'gpu_mem_limit':str(cfg['runtime']['gpu_memory_mb']*1024**2)}),'CPUExecutionProvider']
        else:providers=['CPUExecutionProvider']
        self.session=ort.InferenceSession(str(directory/'onnx/model.onnx'),sess_options=options,providers=providers)
        self.session.disable_fallback()
        self.inputs={x.name for x in self.session.get_inputs()}
        probes=['公交车晚点了，乘客还在等车。','出租车司机要求额外收费。','车厢没有饮用水。','']
        first=self._batch(probes,self.session)
        if not np.isfinite(first).all() or not np.allclose(np.linalg.norm(first,axis=1),1,atol=2e-6):
            raise ArithmeticError('Frozen multilingual encoder failed its finite/unit-norm check')
        audit={'name':cfg['encoder']['name'],'revision':spec['revision'],
               'provider':self.session.get_providers()[0],'cuda_execution_verified':False,
               'max_sequence_length':cfg['encoder']['max_sequence_length'],'padding_id':1,
               'precision':'float32','mean_pooling':True,'unit_normalization':True,
               'model_sha256':file_hash(directory/'onnx/model.onnx'),'tokenizer_sha256':file_hash(directory/'tokenizer.json')}
        if gpu:
            profile=Path(self.session.end_profiling())
            events=json.loads(profile.read_text(encoding='utf-8'))
            compute=[r for r in events if r.get('cat')=='Node' and
                     r.get('args',{}).get('provider')=='CUDAExecutionProvider' and
                     r.get('args',{}).get('op_name') not in ('MemcpyFromHost','MemcpyToHost')]
            if not compute:raise RuntimeError('No actual multilingual-encoder CUDA kernels were recorded')
            cpu_options=ort.SessionOptions();cpu_options.intra_op_num_threads=cfg['runtime']['cpu_threads']
            cpu=ort.InferenceSession(str(directory/'onnx/model.onnx'),sess_options=cpu_options,providers=['CPUExecutionProvider'])
            reference=self._batch(probes,cpu)
            error=float(np.max(np.abs(first-reference)));cosine=float(np.min(np.sum(first*reference,axis=1)))
            if error>2e-5 or cosine<.99999:raise ArithmeticError('Multilingual CPU/GPU agreement check failed')
            audit.update(cuda_execution_verified=True,cuda_compute_nodes=len(compute),cpu_gpu_max_abs_error=error,cpu_gpu_min_cosine=cosine)
            del cpu
        self.audit=audit
        identity_fields=('name','revision','provider','max_sequence_length','padding_id','precision',
                         'mean_pooling','unit_normalization','model_sha256','tokenizer_sha256')
        self.identity={**{k:audit[k] for k in identity_fields},'implementation':file_hash(Path(__file__)),
                       'onnxruntime':ort.__version__,'tf32':False,'batch_size':cfg['runtime']['embedding_batch_size']}
        write_json(run_directory/'chinese_encoder_audit.json',audit,overwrite=True)
        logger.info('CHINESE ENCODER | %s | actual CUDA verified=%s.',audit['provider'],audit['cuda_execution_verified'])

    def _batch(self,texts,session):
        encoded=self.tokenizer.encode_batch(list(texts))
        ids=np.asarray([x.ids for x in encoded],dtype=np.int64)
        mask=np.asarray([x.attention_mask for x in encoded],dtype=np.int64)
        feed={'input_ids':ids,'attention_mask':mask,'token_type_ids':np.zeros_like(ids)}
        output=session.run(None,{k:v for k,v in feed.items() if k in self.inputs})[0]
        if output.ndim!=3:raise ValueError('Pinned ONNX encoder did not return token embeddings')
        pooled=(output*mask[:,:,None]).sum(axis=1)/mask.sum(axis=1,keepdims=True)
        pooled/=np.maximum(np.linalg.norm(pooled,axis=1,keepdims=True),1e-15)
        return pooled.astype(np.float32)

    def encode(self,texts,label):
        if not texts:return np.empty((0,384),dtype=np.float32)
        identity=canonical_hash({'encoder':self.identity,'ordered_texts':list(texts)})
        directory=self.root/'cache/embeddings/chinese'/identity[:24]
        directory.mkdir(parents=True,exist_ok=True)
        path=directory/'vectors.npy'
        if check_receipt(directory):
            values=np.load(path,allow_pickle=False,mmap_mode='r')
            if values.shape!=(len(texts),384):raise ValueError('Chinese embedding cache shape mismatch')
            self.logger.info('Reusing Chinese embeddings: %s (%d texts).',label,len(texts));return values
        partial=path.with_suffix('.partial.npy')
        result=np.lib.format.open_memmap(partial,mode='w+',dtype=np.float32,shape=(len(texts),384))
        batch=self.cfg['runtime']['embedding_batch_size'];last=time.monotonic()
        for start in range(0,len(texts),batch):
            result[start:start+batch]=self._batch(texts[start:start+batch],self.session)
            if start==0 or start+batch>=len(texts) or time.monotonic()-last>=10:
                self.logger.info('Encoding Chinese %s: %d/%d.',label,min(start+batch,len(texts)),len(texts));last=time.monotonic()
        result.flush();del result;partial.replace(path)
        write_json(directory/'identity.json',{'identity':identity,'encoder':self.identity,'texts':len(texts)},overwrite=True)
        complete(directory)
        return np.load(path,allow_pickle=False,mmap_mode='r')


def vector_directory(root,vocabulary,dimension):
    spec={'vocabulary':list(map(str,vocabulary)),'dimension':int(dimension),
          'encoder_manifest_sha256':file_hash(root/'assets/chinese_study/encoder_files.json'),
          'projection':'centered unwhitened PCA; full deterministic SVD; zero-pad only unavailable components',
          'adapter_sha256':file_hash(Path(__file__))}
    return root/'assets/pretrained/chinese_word_embeddings'/canonical_hash(spec)[:24],spec


def prepare_word_vectors(root,vocabulary,encoder,logger):
    from sklearn.decomposition import PCA
    from threadpoolctl import threadpool_limits
    if all(check_receipt(vector_directory(root,vocabulary,d)[0]) for d in (200,300)):
        for dimension in (200,300):
            previous=json.loads((vector_directory(root,vocabulary,dimension)[0]/'audit.json').read_text(encoding='utf-8'))
            if previous['source_model']!=encoder.audit:
                # Runtime profile node counts may vary; compare only numerical/source identity.
                keys=('name','revision','provider','max_sequence_length','precision','model_sha256','tokenizer_sha256')
                if any(previous['source_model'].get(k)!=encoder.audit.get(k) for k in keys):
                    raise ValueError('Existing Chinese baseline vectors use another encoder backend/source; preserve and rebuild the Chinese-only vector cache: '+str(vector_directory(root,vocabulary,dimension)[0]))
        return
    embeddings=np.asarray(encoder.encode(list(map(str,vocabulary)),'baseline vocabulary'),dtype=np.float64)
    rank=min(300,len(vocabulary)-1,embeddings.shape[1])
    if rank<1:raise ValueError('Not enough vocabulary for baseline word-vector projection')
    with threadpool_limits(limits=1):
        projector=PCA(n_components=rank,svd_solver='full',whiten=False)
        projected=projector.fit_transform(embeddings)
    for dimension in (200,300):
        directory,spec=vector_directory(root,vocabulary,dimension);directory.mkdir(parents=True,exist_ok=True)
        vectors=np.zeros((len(vocabulary),dimension),dtype=np.float32)
        used=min(dimension,rank);vectors[:,:used]=projected[:,:used]
        if not np.isfinite(vectors).all() or (np.linalg.norm(vectors,axis=1)<1e-12).any():
            raise ArithmeticError('The language-specific baseline vectors contain a zero or invalid row')
        save_array(directory/'vectors.npy',vectors);save_array(directory/'vocabulary.npy',vocabulary)
        save_array(directory/'pca_mean.npy',projector.mean_);save_array(directory/'pca_components.npy',projector.components_[:used])
        audit={**spec,'language':'zh','source':'frozen multilingual encoder vocabulary embeddings',
               'source_model':encoder.audit,'projection_rank':used,'zero_padded_dimensions':dimension-used,
               'annotation_use':False,'native_network_or_epoch_budget_changed':False,
               'no_English_GloVe_OOV_substitution':True}
        write_json(directory/'audit.json',audit,overwrite=True);complete(directory)
    logger.info('Chinese baseline word vectors prepared: 200 and 300 dimensions; no English OOV substitutions.')


def load_word_vectors(root,vocabulary,dimension):
    directory,_=vector_directory(root,vocabulary,dimension)
    if not check_receipt(directory):raise ValueError('Prepare the Chinese baseline vocabulary vectors before fitting')
    saved=np.load(directory/'vocabulary.npy',allow_pickle=False)
    if not np.array_equal(saved,vocabulary):raise ValueError('Chinese word-vector vocabulary alignment changed')
    vectors=np.load(directory/'vectors.npy',allow_pickle=False)
    audit=json.loads((directory/'audit.json').read_text(encoding='utf-8'))
    audit.update(aligned_values_sha256=hashlib.sha256(vectors.tobytes()).hexdigest(),dimension=dimension)
    return vectors,audit
