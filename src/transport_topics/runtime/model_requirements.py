"""Install only missing portable baseline dependencies, preserving PyTorch/CUDA."""
from importlib import metadata, import_module
from packaging.requirements import Requirement
from .bootstrap import _install


def ensure_model_requirements(methods,logger,install_missing=True):
    roots=[]
    if 'btm' in methods:
        try:metadata.version('bitermplus'); roots.append('bitermplus>=1.0,<1.1')
        except metadata.PackageNotFoundError:roots.append('bitermplus==1.0.0')
    if 'knntm' in methods:
        try:metadata.version('POT'); roots.append('POT>=0.9.5,<0.10')
        except metadata.PackageNotFoundError:roots.append('POT==0.9.6.post1')
    if any(m in methods for m in ('glocom','knntm')):
        try:metadata.version('torch')
        except metadata.PackageNotFoundError as error:
            raise RuntimeError('PyTorch is required for published neural baselines. Select the interpreter with the previously verified PyTorch installation; this phase does not replace CUDA/PyTorch.') from error
        import_module('torch')
    visited=set();records=[]
    def ensure(text):
        requirement=Requirement(text)
        if requirement.marker and not requirement.marker.evaluate({'extra':''}):return
        if requirement.name.lower() in ('torch','torchvision') or requirement.name.lower().startswith('nvidia-'):
            raise RuntimeError('Automatic replacement/installation of CUDA or PyTorch is not permitted by this baseline setup')
        installed=False
        try:version=metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            if not install_missing:raise RuntimeError('Missing baseline requirement '+text)
            _install(str(requirement),logger,timeout=900);version=metadata.version(requirement.name);installed=True
        if not requirement.specifier.contains(version,prereleases=True):
            raise RuntimeError('Existing '+requirement.name+'=='+version+' does not satisfy '+str(requirement.specifier)+'; it was preserved')
        key=requirement.name.lower().replace('_','-')
        if key in visited:return
        visited.add(key);records.append({'distribution':requirement.name,'version':version,'installed_now':installed})
        logger.info('Model requirement OK: %s==%s (%s).',requirement.name,version,'installed missing' if installed else 'preserved')
        for dependency in metadata.requires(requirement.name) or []:ensure(dependency)
    for root in roots:ensure(root)
    if 'btm' in methods:import_module('bitermplus')
    if 'knntm' in methods:import_module('ot')
    return records
