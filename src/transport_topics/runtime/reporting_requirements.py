"""Small missing-only plotting requirements; fitting environments remain untouched."""
from importlib import import_module, metadata
from packaging.requirements import Requirement
from .bootstrap import _install


def ensure_reporting_requirements(logger, install_missing=True):
    roots = ('numpy>=1.26,<3', 'matplotlib>=3.8,<4')
    records, seen = [], set()
    def ensure(text):
        requirement = Requirement(text)
        if requirement.marker and not requirement.marker.evaluate({'extra': ''}):
            return
        key = requirement.name.lower().replace('_', '-')
        if key in seen:
            return
        try:
            version = metadata.version(requirement.name)
            installed = False
        except metadata.PackageNotFoundError:
            if not install_missing:
                raise RuntimeError('Missing reporting requirement: ' + text)
            pin = 'matplotlib==3.9.2' if key == 'matplotlib' else ('numpy==1.26.4' if key == 'numpy' else text)
            _install(pin, logger, timeout=900)
            version, installed = metadata.version(requirement.name), True
        if not requirement.specifier.contains(version, prereleases=True):
            raise RuntimeError('Existing %s==%s is incompatible with %s; it was preserved' % (requirement.name, version, text))
        seen.add(key)
        records.append({'distribution': requirement.name, 'version': version, 'installed_now': installed})
        logger.info('Reporting requirement OK: %s==%s (%s).', requirement.name, version, 'installed missing' if installed else 'preserved')
        for dependency in metadata.requires(requirement.name) or []:
            ensure(dependency)
    for root in roots:
        ensure(root)
    import_module('numpy')
    import_module('matplotlib')
    return records
