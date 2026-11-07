"""Pinned published model sources, obtained unchanged for local execution."""
from pathlib import Path
import sys
import types

from ..data.sources import obtain

COMMIT = '4094055b9e2d0169b0aa75d5aed7220e9509f0de'
REPOSITORY = 'https://github.com/qducnguyen/GloCOM'
FILES = {
    'models/GloCOM/GloCOM.py': '663b4078fbd730f26f045d134bf1dea518ce7a564f01459efa22bde695df87f4',
    'models/GloCOM/ECR.py': 'd8e6240f1df932c636167eaf9c627e6cd243f2339758a5e9b34261917eac1259',
    'models/KNNTM/KNNTM.py': '45865932d4e827aec462cd1e3cf703b80d8163d61aec92a6ee6a64f84fd327e8',
    'models/KNNTM/TopicDistQuant.py': '96847cd9386a3282c3f39cd792a25a8547e18cf0e983317ec71fa005fa1ed9e5',
}


def prepare_sources(root, logger, download=True):
    receipts = []
    prefix = 'third_party/published/glocom_' + COMMIT[:12]
    for relative, sha in FILES.items():
        url = 'https://raw.githubusercontent.com/qducnguyen/GloCOM/' + COMMIT + '/' + relative
        receipts.append(obtain(root, (prefix+'/'+relative, url, sha), logger, download))
    return root/prefix, {'repository': REPOSITORY, 'commit': COMMIT, 'files': receipts,
                         'license_in_repository': 'not declared; source is downloaded, not redistributed in the code update',
                         'knntm_attribution': 'GloCOM authors released reproduction of kNNTM; not original kNNTM authors code'}


def load_native(directory, method):
    # Only package-import names change, so unrelated installed 'models' modules
    # cannot shadow the pinned classes. Network computations remain unchanged.
    names = {
        'glocom': ('models/GloCOM/ECR.py', 'models/GloCOM/GloCOM.py',
                   'from models.GloCOM.ECR import ECR', 'ECR', 'GloCOM'),
        'knntm': ('models/KNNTM/TopicDistQuant.py', 'models/KNNTM/KNNTM.py',
                  'from models.KNNTM.TopicDistQuant import TopicDistQuant', 'TopicDistQuant', 'KNNTM'),
    }
    dependency, main, original_import, dependency_class, model_class = names[method]
    prefix = '_transport_published_' + method
    for name, relative, rewrite in ((prefix+'_dependency', dependency, False), (prefix, main, True)):
        module = types.ModuleType(name)
        path = Path(directory)/relative
        module.__file__ = str(path)
        source = path.read_text(encoding='utf-8')
        if rewrite:
            if source.count(original_import) != 1:
                raise ValueError('Pinned native model import no longer matches its namespace adapter')
            source = source.replace(original_import, 'from '+prefix+'_dependency import '+dependency_class)
        sys.modules[name] = module
        exec(compile(source, str(path), 'exec'), module.__dict__)
    return getattr(sys.modules[prefix], model_class)
