"""Missing-only dependency resolution; never upgrades an existing installation."""
from importlib import metadata, import_module
from packaging.requirements import Requirement
from .bootstrap import _install


# CPU and GPU distributions do not publish every patch version in common.
# In particular, the GPU release is 1.22.0, not the CPU-only 1.22.1 pin.
MISSING_ONNX_REQUIREMENTS = {
    'onnxruntime': 'onnxruntime==1.22.1',
    'onnxruntime-gpu': 'onnxruntime-gpu==1.22.0',
}


def ensure_corpus_requirements(environment, logger, install_missing=True):
    packages=environment['watched_packages']
    gpu=environment['device_selection']['torch_cuda_execution_verified']
    if packages.get('onnxruntime') and packages.get('onnxruntime-gpu'):
        raise RuntimeError('Both ONNX Runtime distributions are installed. Preserve the environment and send inspection before changing either.')
    if gpu and packages.get('onnxruntime') and not packages.get('onnxruntime-gpu'):
        raise RuntimeError('CPU ONNX Runtime is installed while GPU execution is requested. It was preserved; select --device cpu or review backend replacement before proceeding.')
    backend='onnxruntime-gpu' if gpu else ('onnxruntime-gpu' if packages.get('onnxruntime-gpu') else 'onnxruntime')
    backend_requirement=(backend+'>=1.20,<1.27' if packages.get(backend)
                         else MISSING_ONNX_REQUIREMENTS[backend])
    roots=['numpy>=1.24,<3','scipy>=1.10,<2','scikit-learn>=1.3,<2',
           'tokenizers>=0.19,<0.22',backend_requirement]
    visited=set(); records=[]
    def ensure(text):
        requirement=Requirement(text)
        if requirement.marker and not requirement.marker.evaluate({'extra':''}): return
        name=requirement.name
        if name.lower() in ('torch','pytorch','nvidia-cuda-runtime-cu12'):
            raise RuntimeError('Automatic CUDA/PyTorch installation is outside this phase')
        try: version=metadata.version(name); installed=False
        except metadata.PackageNotFoundError:
            if not install_missing: raise RuntimeError('Missing corpus requirement %s; use main.py --install-missing' % text)
            _install(str(requirement),logger,timeout=900)
            version=metadata.version(name); installed=True
        if requirement.specifier and not requirement.specifier.contains(version,prereleases=True):
            raise RuntimeError('Existing %s==%s does not satisfy %s; preserved rather than upgraded' % (name,version,requirement.specifier))
        key=name.lower().replace('_','-')
        if key in visited: return
        visited.add(key)
        records.append({'distribution':name,'version':version,'installed_now':installed})
        logger.info('Corpus requirement OK: %s==%s (%s).',name,version,'installed missing' if installed else 'preserved')
        for dependency in metadata.requires(name) or []: ensure(dependency)
    for root in roots: ensure(root)
    for module in ('numpy','scipy','sklearn','tokenizers','onnxruntime'): import_module(module)
    return records
