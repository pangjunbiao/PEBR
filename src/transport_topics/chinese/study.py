"""Full pooled Chinese comparison using the unchanged borrowing and metric code."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import statistics
import time

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from ..baselines.registry import NMF_VARIANTS
from ..config import canonical_hash
from ..data.preparation import check_receipt, complete
from ..evaluation.reference import TextReference
from ..evaluation.stability import seed_stability
from ..evaluation.topic_quality import evaluate_topics
from ..experiments.sharing_controls import make_controls
from ..proposed.reference import reference_bank
from ..runtime.artifacts import file_hash, write_json
from .data import make_counts, prepare_bank
from .embeddings import FrozenChineseEncoder, prepare_word_vectors
from .models import fit_chinese_model
from .saved_results import CONTROLS, LABELS, export_table, write_csv


def original_reference(records,collections,directory):
    words=np.asarray(sorted({str(w) for c,_ in collections for w in
                            np.load(c.directory/'vocabulary.npy',allow_pickle=False)}))
    lookup={str(w):i for i,w in enumerate(words)}
    rows=[];cols=[]
    for i,r in enumerate(records):
        for w in set(r['tokens']):
            if w in lookup:rows.append(i);cols.append(lookup[w])
    binary=sparse.csr_matrix((np.ones(len(rows),dtype=np.int64),(rows,cols)),shape=(len(records),len(words)))
    identity=canonical_hash({'original_text_tokens':[r['tokens'] for r in records],'vocabulary':words.tolist(),
                            'metric_code':file_hash(Path(__file__).parents[1]/'evaluation/topic_quality.py')})
    return TextReference(identity,'Chinese original pooled text',words,len(records),
                         np.asarray(binary.sum(axis=0)).ravel(),(binary.T@binary).tocsr(),
                         [r['tokens'] for r in records],directory)


def methods_for(cfg):
    return cfg['methods']+[m for m,enabled in cfg['controls'].items() if enabled]


def export_progress(directory,seed_rows,topic_sets,models,expected):
    if seed_rows:write_csv(directory/'topic_seed_results.csv',seed_rows)
    write_json(directory/'progress.json',{'completed_evaluations':len(seed_rows),'expected_evaluations':expected,
               'complete':len(seed_rows)==expected,'models':models},overwrite=True)


def run_study(root,cfg,protocol,records,collections,device,logger,store):
    from importlib import metadata
    output=root/cfg['outputs']/store.run_id
    output.mkdir(parents=True,exist_ok=False)
    methods=methods_for(cfg)
    expected=len(collections)*len(cfg['topic_counts'])*len(cfg['model_seeds'])*len(methods)
    logger.info('FULL CHINESE STUDY | %d collection(s), K=%s, seeds=%s, %d methods, %d complete model/evaluation jobs.',
                len(collections),cfg['topic_counts'],cfg['model_seeds'],len(methods),expected)
    logger.info('Every retained post is used; no pilot sampling and no outcome-dependent setting changes.')
    encoder=FrozenChineseEncoder(root,cfg,device,logger,store.directory)
    reference=original_reference(records,collections,output)
    write_json(output/'coherence_reference.json',{'identity':reference.identity,'documents':reference.documents,
               'vocabulary':len(reference.vocabulary),'source':'original pooled Chinese text only',
               'npmi_unit':'original document presence','cv_window':110,'top_words':10,
               'fitting_or_selection_receives_this_reference':False},overwrite=True)
    seed_rows=[];models=[];all_topics={};stability_rows=[];pair_rows=[];controls_index=[]
    preparation_times=[]
    for collection,selected in collections:
        tick=time.perf_counter()
        prepare_bank(collection,selected,protocol,encoder,logger)
        vocabulary=np.load(collection.directory/'vocabulary.npy',allow_pickle=False)
        if any(m in methods for m in ('knntm','glocom')):prepare_word_vectors(root,vocabulary,encoder,logger)
        preparation_times.append({'collection':collection.summary['name'],'seconds':time.perf_counter()-tick,
                                  'scope':'masked retrieval and baseline input vectors; caches may be reused'})
    encoder_audit=dict(encoder.audit)
    del encoder
    import gc
    gc.collect()
    if device.startswith('cuda'):
        import torch
        torch.cuda.empty_cache()
    logger.info('Frozen Chinese features prepared; encoder session released before fitting topic models.')
    for collection,selected in collections:
        vocabulary=np.load(collection.directory/'vocabulary.npy',allow_pickle=False)
        for topics in cfg['topic_counts']:
            variants=[NMF_VARIANTS[m] for m in methods if m in NMF_VARIANTS]
            control_directory=None;control_summary=None
            if any(v!='no_borrowing' for v in variants):
                with threadpool_limits(limits=1):
                    _,audits=reference_bank(root,collection.summary,collection.counts,collection.folds,
                                            topics,protocol['method']['predictor'],logger)
                if not all(a['converged'] and a['receiver_fold_excluded'] for a in audits):
                    raise RuntimeError('A fixed Chinese predictive reference did not converge or crossed its receiving fold')
                write_json(output/f"reference_audits_{collection.summary['name']}_K{topics}.json",audits,overwrite=True)
                with threadpool_limits(limits=1):
                    control_directory,control_summary=make_controls(root,collection,topics,'topic_learning',protocol,logger,variants)
                controls_index.append({'collection':collection.summary['name'],'K':topics,
                                       'directory':control_directory.relative_to(root).as_posix(),**control_summary})
            topic_sets=defaultdict(dict)
            for seed in cfg['model_seeds']:
                for method in methods:
                    learning=None
                    if method in NMF_VARIANTS and method!='nmf_original':
                        learning=control_directory/(NMF_VARIANTS[method]+'.npz')
                    result=fit_chinese_model(root,collection,method,topics,seed,protocol,device,logger,
                                            learning_path=learning,
                                            control_identity=control_summary['identity'] if control_summary else None)
                    model_directory=root/result['path']
                    scores=np.load(model_directory/'topic_scores.npy',allow_pickle=False)
                    key=canonical_hash({'model_identity':result['identity'],'reference':reference.identity,
                          'top_words':10,'gensim':metadata.version('gensim'),
                          'metric_code':file_hash(Path(__file__).parents[1]/'evaluation/topic_quality.py')})
                    evaluation=root/'outputs/evaluation/chinese_topics'/key[:24]
                    evaluation.mkdir(parents=True,exist_ok=True)
                    if check_receipt(evaluation):quality=json.loads((evaluation/'quality.json').read_text(encoding='utf-8'))
                    else:
                        quality=evaluate_topics(scores,vocabulary,reference,10,True)
                        write_json(evaluation/'quality.json',quality,overwrite=True);complete(evaluation)
                    topic_sets[method][seed]=quality['top_words']
                    all_topics[f"{collection.summary['name']}/{method}/K{topics}/seed{seed}"]=quality['top_words']
                    row={'study':'chinese_full','collection':collection.summary['name'],'method':method,
                         'K':topics,'seed':seed,'posts':collection.counts.shape[0],'vocabulary':len(vocabulary),
                         'npmi':quality['npmi'],'cv':quality['cv'],'diversity':quality['diversity'],
                         'constant_score_topics':quality['constant_score_topics'],
                         'reference_identity':reference.identity,'model_identity':result['identity'],
                         'evaluation_path':evaluation.relative_to(root).as_posix(),'development_only':False}
                    seed_rows.append(row);models.append(result)
                    export_progress(output,seed_rows,topic_sets,models,expected)
                    logger.info('CHINESE RESULT | %s K=%d seed=%d | NPMI %.6f | C_v %.6f | TD %.6f | %d/%d.',
                                method,topics,seed,row['npmi'],row['cv'],row['diversity'],len(seed_rows),expected)
            for method in methods:
                stability=seed_stability(topic_sets[method])
                if stability['seed_pairs']!=10:raise AssertionError('Five-seed stability requires all ten pairs')
                stability_rows.append({'collection':collection.summary['name'],'method':method,'K':topics,
                                       'stability':stability['stability'],'seed_pairs':10})
                pair_rows += [{'collection':collection.summary['name'],'method':method,'K':topics,**p} for p in stability['pairs']]
            write_csv(output/'seed_stability.csv',stability_rows);write_csv(output/'seed_stability_pairs.csv',pair_rows)
    if len(seed_rows)!=expected:raise RuntimeError('The full declared Chinese grid is incomplete')
    summaries=[]
    for collection,_ in collections:
        name=collection.summary['name']
        for topics in cfg['topic_counts']:
            display=[]
            for method in methods:
                values=[r for r in seed_rows if r['collection']==name and r['K']==topics and r['method']==method]
                if sorted(r['seed'] for r in values)!=cfg['model_seeds']:raise AssertionError('Missing Chinese model seed')
                row={'Model':LABELS[method]};summary={'collection':name,'method':method,'K':topics,'seed_fits':5}
                for metric,label in [('npmi','NPMI'),('cv','C_v'),('diversity','TD')]:
                    mean=statistics.mean(r[metric] for r in values);sd=statistics.stdev(r[metric] for r in values)
                    row[label]=mean;row[label+'_sd']=sd;summary[metric+'_mean']=mean;summary[metric+'_seed_sd']=sd
                stable=next(r['stability'] for r in stability_rows if r['collection']==name and r['K']==topics and r['method']==method)
                row.update(Stability=stable,Stability_sd=None);summary.update(stability=stable);display.append(row);summaries.append(summary)
            note='Mean plus/minus sample SD over seeds 0--4. Stability averages all ten matched topic-set Jaccard seed pairs. All retained pooled Chinese posts are used; no category labels enter fitting.'
            main_rows=[r for r in display if r['Model'] in [LABELS[m] for m in cfg['methods']]]
            export_table(output,name+f'_K{topics}_main',main_rows,f'Chinese topic quality: {name}, $K={topics}$.',note)
            if any(cfg['controls'].values()):
                by_label={r['Model']:r for r in display}
                control_rows=[by_label[LABELS[m]] for m in CONTROLS if LABELS[m] in by_label]
                export_table(output,name+f'_K{topics}_controls',control_rows,f'Chinese borrowing controls: {name}, $K={topics}$.',note)
    write_csv(output/'topic_quality_summary.csv',summaries)
    write_json(output/'all_topic_words.json',all_topics,overwrite=True)
    write_json(output/'borrowing_controls_index.json',controls_index,overwrite=True)
    write_json(output/'model_fit_audits.json',models,overwrite=True)
    result={'complete':True,'full_retained_corpus':True,'pilot':False,'expected_evaluations':expected,
            'completed_evaluations':len(seed_rows),'topic_counts':cfg['topic_counts'],'model_seeds':cfg['model_seeds'],
            'methods':methods,'output_directory':output.relative_to(root).as_posix(),'collections':[c.summary for c,_ in collections],
            'encoder':encoder_audit,'feature_preparation_times':preparation_times,'english_model_or_results_changed':False,
            'main_setting_selected_from_outcomes':False,'single_seed_stability':'undefined'}
    write_json(output/'study_summary.json',result,overwrite=True)
    logger.info('FULL CHINESE STUDY COMPLETE | %d/%d evaluations | %s.',len(seed_rows),expected,output)
    return result
