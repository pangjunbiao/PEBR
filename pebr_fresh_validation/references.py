from .bootstrap import read_json
def read_references(step3,collection,k,protocol):
    """Load the exact existing five predictors without refitting or writing Step 3."""
    import numpy as np
    from pebr_revision_step3.io import complete
    from transport_topics.proposed.reference import PredictiveReference
    expected=protocol['reference_predictor'];selected={}
    for path in (step3/'cache/predictors').glob('*'):
        if not (path/'identity.json').is_file():continue
        identity=read_json(path/'identity.json')
        if identity.get('condition')!=collection.summary['identity_sha256'] or identity.get('topics')!=k:continue
        settings=identity['settings']
        desired={'folds':5,'seed':0,'tolerance':expected['nmf_tolerance'],
            'max_iterations':expected['nmf_max_iterations'],'smoothing':expected['smoothing'],
            'unigram_smoothing':expected['unigram_smoothing']}
        if settings!=desired:continue
        fold=int(identity['excluded_fold'])
        if fold in selected or not complete(path):raise ValueError('Duplicated/incomplete frozen predictive reference')
        indices=np.load(path/'training_indices.npy',allow_pickle=False)
        if not np.array_equal(indices,np.flatnonzero(collection.folds!=fold)):raise ValueError('Reference fold exclusion differs')
        audit=read_json(path/'fit.json')
        if not audit['converged'] or not audit['receiver_fold_excluded']:raise ValueError('Predictive reference failed its fit audit')
        selected[fold]=PredictiveReference(np.load(path/'dictionary.npy',allow_pickle=False),
            np.load(path/'idf.npy',allow_pickle=False),np.load(path/'prior.npy',allow_pickle=False),expected['smoothing'])
    if set(selected)!=set(range(5)):raise ValueError('Expected exactly five completed frozen references')
    return [selected[f] for f in range(5)]

