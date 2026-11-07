"""Short workspace names and an actual memmap write/read probe before encoding."""
from pathlib import Path
import hashlib
import os
import time

def workspace_path(project, run_key):
    """The complete Step 5 lineage stays in identity.json rather than the folder name."""
    if len(run_key) != 64 or any(c not in '0123456789abcdef' for c in run_key):
        raise ValueError('Expected the complete SHA256 run identity')
    return Path(project)/'outputs/revision_rule_lab/protected_words'/('s6_'+run_key[:12])

def verify_cache_write(work):
    """Exercise NumPy's real temporary-cache file operations, without any model call."""
    import numpy as np
    work = Path(work)
    directory = work/'cache/embeddings'
    directory.mkdir(parents=True, exist_ok=True)
    token = hashlib.sha256((str(work)+'|step6-cache-write-probe|'+str(os.getpid())+'|'+str(time.monotonic_ns())).encode()).hexdigest()
    # Same prefix and suffix lengths as the frozen English encoder's temporary file.
    probe = directory/('probe___'+token+'.partial.npy')
    path_characters = len(str(probe))
    if os.name == 'nt' and path_characters >= 260:
        raise OSError('Step 6 cache path is still too long for legacy Windows file operations: '
                      +str(path_characters)+' characters. Use a shorter project directory.')
    created = False
    try:
        # Exclusive creation ensures this probe cannot replace a pre-existing file.
        with probe.open('xb'):
            created = True
        matrix = np.lib.format.open_memmap(probe, mode='w+', dtype=np.float32, shape=(1, 2))
        matrix[0] = [1., 2.]
        matrix.flush()
        del matrix
        values = np.load(probe, allow_pickle=False)
        if values.shape != (1, 2) or not np.array_equal(values, np.asarray([[1., 2.]], dtype=np.float32)):
            raise ArithmeticError('The NumPy embedding-cache write/read check failed')
        del values
    finally:
        if created:
            probe.unlink(missing_ok=True)
    return {'passed': True, 'temporary_cache_path_characters': path_characters,
            'actual_numpy_memmap_write_read_checked': True, 'probe_removed': not probe.exists(),
            'model_or_encoder_inference_performed': False}
