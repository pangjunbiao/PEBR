"""Pinned nonquantized ONNX encoder with real-provider and CPU agreement checks."""
import hashlib
import json
from pathlib import Path
import numpy as np
from ..runtime.artifacts import write_json, file_hash


class FrozenEncoder:
    def __init__(self, root, config, environment, logger, run_directory):
        from tokenizers import Tokenizer
        gpu=environment['device_selection']['torch_cuda_execution_verified']
        if gpu:
            # Import before ORT session creation to load the existing PyTorch CUDA/cuDNN DLLs.
            import torch
        import onnxruntime as ort
        self.root,self.logger,self.config=root,logger,config
        self.batch_size=config['preparation']['embedding_batch_size']
        self.model=root/'assets/pretrained/encoder/onnx/model.onnx'
        self.tokenizer=Tokenizer.from_file(str(root/'assets/pretrained/encoder/tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=config['method']['encoder']['max_sequence_length'])
        self.tokenizer.enable_padding(pad_id=0,pad_token='[PAD]')
        options=ort.SessionOptions()
        options.intra_op_num_threads=config['runtime']['cpu_threads']; options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        options.log_severity_level=3
        if gpu:
            if 'CUDAExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError('GPU ONNX provider is unavailable; no silent CPU encoder fallback is permitted')
            if hasattr(ort,'preload_dlls'): ort.preload_dlls()
            options.enable_profiling=True
            options.profile_file_prefix=str(run_directory/'encoder_profile')
            providers=[('CUDAExecutionProvider',{'device_id':config['runtime']['gpu_index'],
                         'use_tf32':'0','gpu_mem_limit':str(config['preparation']['gpu_memory_mb']*1024**2)}),
                         'CPUExecutionProvider']
        else: providers=['CPUExecutionProvider']
        self.session=ort.InferenceSession(str(self.model),sess_options=options,providers=providers)
        self.session.disable_fallback()
        self.inputs={x.name for x in self.session.get_inputs()}
        probes=['the bus is delayed and the service is not available',
                'taxi driver charged an additional fare','road closure near the station', '', 'no refund was provided']
        first=self._encode(probes,self.session)
        if not np.isfinite(first).all() or not np.allclose(np.linalg.norm(first,axis=1),1,atol=2e-6):
            raise ArithmeticError('Encoder normalization/finite-value verification failed')
        audit={'provider':self.session.get_providers()[0],'gpu_execution_verified':False,
               'precision':'float32','tf32':False,'model_sha256':file_hash(self.model),
               'tokenizer_sha256':file_hash(root/'assets/pretrained/encoder/tokenizer.json'),
               'max_sequence_length':config['method']['encoder']['max_sequence_length'],
               'runtime_version':ort.__version__}
        if gpu:
            profile=Path(self.session.end_profiling())
            events=json.loads(profile.read_text(encoding='utf-8'))
            nodes=[e for e in events if e.get('cat')=='Node']
            cuda_events=[e for e in nodes if e.get('args',{}).get('provider')=='CUDAExecutionProvider'
                         and e.get('args',{}).get('op_name') not in ('MemcpyFromHost','MemcpyToHost')]
            cuda_nodes=len(cuda_events)
            if cuda_nodes==0: raise RuntimeError('Encoder profile contains no actual CUDA kernels')
            cpu_options=ort.SessionOptions();cpu_options.intra_op_num_threads=config['runtime']['cpu_threads']
            cpu_options.log_severity_level=3
            reference=ort.InferenceSession(str(self.model),sess_options=cpu_options,providers=['CPUExecutionProvider'])
            cpu=self._encode(probes,reference)
            error=float(np.max(np.abs(cpu-first))); cosine=float(np.min(np.sum(cpu*first,axis=1)))
            audit.update(gpu_execution_verified=True,cuda_profile_nodes=cuda_nodes,
                         cuda_operations=sorted({e.get('args',{}).get('op_name','unknown') for e in cuda_events}),
                         cpu_gpu_max_abs_error=error,cpu_gpu_min_cosine=cosine)
            if error>2e-5 or cosine<.99999: raise ArithmeticError('CPU/GPU frozen encoder agreement check failed')
            del reference
        from importlib import metadata
        self.identity={**config['method']['encoder'],'provider':audit['provider'],
                       'implementation_sha256':file_hash(Path(__file__)),
                       'tokenizers_version':metadata.version('tokenizers'),'batch_size':self.batch_size,
                       'runtime_version':ort.__version__,'tf32':False,'model_sha256':audit['model_sha256']}
        self.audit=audit
        logger.info('Frozen encoder verified: %s; GPU kernels verified=%s.',audit['provider'],audit['gpu_execution_verified'])

    def _encode(self,texts,session):
        encoded=self.tokenizer.encode_batch(list(texts))
        ids=np.asarray([x.ids for x in encoded],dtype=np.int64)
        mask=np.asarray([x.attention_mask for x in encoded],dtype=np.int64)
        feed={'input_ids':ids,'attention_mask':mask,'token_type_ids':np.zeros_like(ids)}
        output=session.run(None,{k:v for k,v in feed.items() if k in self.inputs})[0]
        pooled=(output*mask[:,:,None]).sum(axis=1)/mask.sum(axis=1,keepdims=True)
        pooled=pooled/np.maximum(np.linalg.norm(pooled,axis=1,keepdims=True),1e-15)
        return pooled.astype(np.float32)

    def encode(self,texts,label='queries'):
        # Content identity includes the exact ordered texts and compute backend.
        from ..config import canonical_hash
        if not texts: return np.empty((0,384),dtype=np.float32)
        key=canonical_hash({'encoder':self.identity,'texts':texts})
        path=self.root/'cache/embeddings'/('vectors_'+key+'.npy')
        receipt=path.with_suffix('.json')
        if path.exists() and receipt.exists():
            meta=json.loads(receipt.read_text(encoding='utf-8'))
            if file_hash(path)!=meta['sha256']: raise ValueError('Embedding cache checksum mismatch')
            matrix=np.load(path,allow_pickle=False,mmap_mode='r')
            if matrix.shape!=(len(texts),384): raise ValueError('Embedding cache shape mismatch')
            self.logger.info('Reusing embeddings: %s (%d texts).',label,len(texts))
            return matrix
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.partial.npy')
        matrix=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float32,shape=(len(texts),384))
        for start in range(0,len(texts),self.batch_size):
            matrix[start:start+self.batch_size]=self._encode(texts[start:start+self.batch_size],self.session)
            if start==0 or (start//self.batch_size)%50==0:
                self.logger.info('Encoding %s: %d/%d.',label,min(start+self.batch_size,len(texts)),len(texts))
        matrix.flush(); del matrix
        temporary.replace(path)
        write_json(receipt,{'sha256':file_hash(path),'identity':key,'rows':len(texts)},overwrite=False)
        return np.load(path,allow_pickle=False,mmap_mode='r')
