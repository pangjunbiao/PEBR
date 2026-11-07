"""Paired query-family bootstrap conditional on fitted seed-averaged models."""
import numpy as np
from ..runtime.reproducibility import stable_seed


def paired_family_bootstrap(difference,categories,families,resamples=2000,seed=0):
    difference=np.asarray(difference,dtype=np.float64)
    categories=np.asarray(categories);families=np.asarray(families)
    if difference.ndim!=1 or difference.shape!=categories.shape or difference.shape!=families.shape:
        raise ValueError('Paired bootstrap queries are not aligned')
    if not np.isfinite(difference).all() or resamples<1 or not len(difference):
        raise ValueError('Invalid bootstrap observations')
    rows=[];macro=np.zeros(resamples)
    for category in sorted(set(categories.tolist())):
        keep=categories==category
        unique,inverse=np.unique(families[keep],return_inverse=True)
        sums=np.bincount(inverse,weights=difference[keep])
        counts=np.bincount(inverse)
        rng=np.random.default_rng(stable_seed(seed,'paired_query_families',category))
        draw=rng.integers(0,len(unique),size=(resamples,len(unique)))
        boot=sums[draw].sum(axis=1)/counts[draw].sum(axis=1)
        macro+=boot
        lo,hi=np.quantile(boot,[.025,.975])
        rows.append({'category':category,'queries':int(keep.sum()),'query_families':len(unique),
                     'difference':float(difference[keep].mean()),'ci_low':float(lo),'ci_high':float(hi)})
    macro/=len(rows)
    lo,hi=np.quantile(macro,[.025,.975])
    return {'difference':float(np.mean([x['difference'] for x in rows])),
            'ci_low':float(lo),'ci_high':float(hi),'resamples':resamples,
            'seed_averaged_before_resampling':True,'resampling_unit':'query family within service category',
            'conditional_on_fitted_models':True,'categories':rows}
