"""Family-aware folds and fixed sampling ladders; never uses class annotations."""
import numpy as np
from ..runtime.reproducibility import stable_seed


def family_folds(families, folds=5, seed=2710):
    unique=np.unique(families)
    if len(unique)<folds: raise ValueError('Fewer text families than required folds')
    shuffled=np.random.default_rng(seed).permutation(unique)
    lookup={int(f):i%folds for i,f in enumerate(shuffled)}
    return np.array([lookup[int(f)] for f in families],dtype=np.int32)


def nested_family_samples(families, sizes, sample_index, namespace):
    order=np.random.default_rng(stable_seed(sample_index,namespace)).permutation(np.unique(families))
    groups={int(f):np.flatnonzero(families==f) for f in order}
    selected=[]; answer={}; cursor=0
    for target in sorted(sizes):
        while len(selected)<target and cursor<len(order):
            selected.extend(groups[int(order[cursor])].tolist()); cursor+=1
        if len(selected)<target: raise ValueError('Eligible corpus cannot support target size %d' % target)
        answer[target]=np.array(sorted(selected),dtype=np.int32)
    return answer
