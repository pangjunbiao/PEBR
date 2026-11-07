"""Missing-only supplementary coherence dependency checks."""
from importlib import metadata,import_module
from packaging.requirements import Requirement
from .bootstrap import _install


def ensure_evaluation_requirements(logger,install_missing=True):
    records=[];visited=set()
    def ensure(text):
        requirement=Requirement(text)
        if requirement.marker and not requirement.marker.evaluate({'extra':''}):return
        name=requirement.name
        try:version=metadata.version(name);installed=False
        except metadata.PackageNotFoundError:
            if not install_missing:raise RuntimeError('Missing evaluation requirement: '+text)
            _install(str(requirement),logger,timeout=900);version=metadata.version(name);installed=True
        if not requirement.specifier.contains(version,prereleases=True):
            raise RuntimeError('Existing '+name+'=='+version+' is outside '+str(requirement.specifier)+'; it was preserved')
        key=name.lower().replace('_','-')
        if key in visited:return
        visited.add(key);records.append({'distribution':name,'version':version,'installed_now':installed})
        logger.info('Evaluation requirement OK: %s==%s (%s).',name,version,'installed missing' if installed else 'preserved')
        for dependency in metadata.requires(name) or []:ensure(dependency)
    ensure('gensim>=4.4,<4.5' if _installed('gensim') else 'gensim==4.4.0')
    import_module('gensim')
    return records


def _installed(name):
    try:metadata.version(name);return True
    except metadata.PackageNotFoundError:return False
