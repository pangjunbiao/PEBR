"""Export revised-candidate models with memberships inferred from original posts."""
from pathlib import Path
import argparse
import sys
import shutil
from run_trial import Trial,read,write,digest,say


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',default='forward_controls_v1')
    p.add_argument('--output',type=Path,default=None)
    p.add_argument('--dependencies',type=Path,default=None)
    args=p.parse_args();root=Path(__file__).resolve().parent
    if args.dependencies:sys.path.insert(0,str(args.dependencies.resolve()))
    sys.path[:0]=[str(root/'frozen_source'),str(root/'candidate')]
    import numpy as np
    from threadpoolctl import threadpool_limits
    from transport_topics.representations.tfidf import idf_weights,transform
    from transport_topics.topics.memberships import infer_original_memberships
    trial=Trial(root,root/'model_export_check')
    output=args.output or root/'exported_models'
    count=0
    for corpus in trial.protocol['corpora']:
        c=trial.collection(corpus);weights=idf_weights(c['counts']);original=transform(c['counts'],weights)
        rows=np.load(root/'inputs/data'/corpus/'source_rows.npy',allow_pickle=False)
        for k in trial.protocol['topic_counts']:
            for seed in trial.protocol['final_model_seeds']:
                source=root/args.source_run/'fits'/corpus/('K%d'%k)/'preserve_forward_fixed_cap'/('seed%d'%seed)
                dictionary=np.load(source/'topic_scores.npy',allow_pickle=False)
                with threadpool_limits(limits=1):memberships,features,audit=infer_original_memberships(original,dictionary)
                target=output/corpus/('K%d'%k)/('seed%d'%seed);target.mkdir(parents=True,exist_ok=True)
                for name,data in [('topic_scores',dictionary),('doc_topics',memberships),('retrieval_features',features),
                                  ('source_rows',rows),('vocabulary',c['vocabulary'])]:np.save(target/(name+'.npy'),data,allow_pickle=False)
                assert np.isfinite(memberships).all() and np.isfinite(features).all()
                assert np.all(memberships>=-1e-12) and np.all(features>=-1e-12)
                assert np.all((np.abs(features.sum(axis=1)-1)<1e-10)|(features.sum(axis=1)==0))
                shutil.copyfile(source/'quality.json',target/'quality.json')
                write(target/'model.json',{'method':'PEBR_receiver_preserving_fixed_cap','status':'development_candidate',
                    'corpus':corpus,'topics':k,'seed':seed,'selection_mask_base':2711,'selection_banks':1,
                    'cap':1/3,'downside_weight':1.,'max_donors':3,'views':20,
                    'original_post_memberships_only':True,'source_dictionary_sha256':digest(source/'topic_scores.npy'),
                    'independent_selection_validation_complete':False,'original_membership_inference':audit,
                    'contributions':'retrieval_features = W times H row L1 mass, normalized per original post'})
                count+=1
    write(output/'MODEL_EXPORT_COMPLETE.json',{'models':count,'new_NMF_training':False,'original_post_inference_only':True})
    say('MODEL EXPORT COMPLETE',count,'dictionaries with original-post memberships')


if __name__=='__main__':main()
