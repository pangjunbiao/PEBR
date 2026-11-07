"""Maximum-weight top-word Jaccard topic matching across model seeds."""
from itertools import combinations
import numpy as np
from scipy.optimize import linear_sum_assignment


def matched_jaccard(left,right):
    if len(left)!=len(right) or not len(left):raise ValueError('Stability requires the same positive topic count')
    a,b=[set(x) for x in left],[set(x) for x in right]
    score=np.array([[len(x&y)/len(x|y) if x|y else 0. for y in b] for x in a])
    rows,columns=linear_sum_assignment(-score)
    return float(score[rows,columns].mean())


def seed_stability(seed_topics):
    pairs=[{'seed_a':a,'seed_b':b,'matched_jaccard':matched_jaccard(seed_topics[a],seed_topics[b])}
           for a,b in combinations(sorted(seed_topics),2)]
    return {'stability':float(np.mean([p['matched_jaccard'] for p in pairs])) if pairs else None,
            'seed_pairs':len(pairs),'pairs':pairs}
