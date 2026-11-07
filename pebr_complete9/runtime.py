"""One fresh, audited standalone PEBR pipeline per corpus; no shared-control time."""
import time
import uuid
import platform
import numpy as np
from scipy import sparse
from .common import read,write,table

def standalone(collection,bank,references,p):
    from pebr_fresh_validation.anchored_scoring import ReceiverAnchoredEvaluator
    from transport_topics.proposed.search import select_action
    from transport_topics.proposed.transfer import normalized_counts,transfer_action,check_transfer
    donors=normalized_counts(collection.counts);base=normalized_counts(bank.base)
    projections=[r.projection(donors) for r in references]
    result=np.empty_like(base);records=[]
    for i in range(len(base)):
        if bank.eligible[i] and np.any(bank.candidates[i]>=0):
            fold=int(collection.folds[i]);sl=slice(i*p['views'],(i+1)*p['views'])
            ev=ReceiverAnchoredEvaluator(references[fold],bank.visible[sl].toarray(),bank.targets[sl].toarray(),
                bank.candidates[i],donors,collection.counts[i].toarray(),projections[fold],128)
            action=select_action(ev,[p['cap']],p['max_donors'],p['downside_weight'],p['admission_margin']).action
        else:
            from transport_topics.proposed.actions import NO_BORROWING
            action=NO_BORROWING
        q,mass,available=transfer_action(base[i],action,bank.full_candidates[i],donors)
        check_transfer(base[i],q,mass,p['cap']);result[i]=q
        records.append({'donors':list(action.donors),'available_donors':list(available),'offered_beta':action.beta,'actual_beta':mass})
    return sparse.csr_matrix(result),records

def run(study):
    from pebr_revision_step3.io import complete,save_receipt
    from pebr_fresh_validation.banks import query_records,masked_evidence,make_encoder
    from transport_topics.data.prepared import SelectionBank
    from transport_topics.proposed.candidates import retrieve
    from transport_topics.proposed.reference import reference_bank
    from transport_topics.topics.learning import fit_topics
    rows=[]
    for corpus,c in study.corpora.items():
        done=study.work/'runtime_completed'/corpus
        if complete(done):rows.append(read(done/'timing.json'));continue
        col=c['collection'];p=study.protocol
        attempt=study.work/'runtime_attempts'/corpus/uuid.uuid4().hex[:8];attempt.mkdir(parents=True)
        times={};start=time.perf_counter();tick=start
        records=query_records(corpus,col,attempt/'tokenizer_cache')
        view=masked_evidence(corpus,col,records,p,3711)
        times['mask_preparation_seconds']=time.perf_counter()-tick;tick=time.perf_counter()
        encoder=make_encoder(corpus,study.step2,attempt,col,p,study.logger)
        times['audited_encoder_loading_seconds']=time.perf_counter()-tick;tick=time.perf_counter()
        texts=[r['cleaned_text'] for r in records] if corpus=='transport_grievance' else [r['text'] for r in records]
        full=encoder.encode(texts,label='runtime_full_posts')
        visible=encoder.encode(view['queries'],label='runtime_visible_queries')
        times['full_and_visible_encoding_seconds']=time.perf_counter()-tick;tick=time.perf_counter()
        n,v=col.counts.shape;positive=np.ones(n,dtype=bool)
        lists=np.full((n*p['views'],p['candidates_per_view']),-1,dtype=np.int32)
        lists[view['positions']]=retrieve(visible,view['positions']//p['views'],full,col.families,col.folds,positive,p['candidates_per_view'])
        full_lists=retrieve(full,np.arange(n),full,col.families,col.folds,positive,p['candidates_per_view'])
        bank=SelectionBank('topic_learning',attempt,view['base'],view['visible'],view['targets'],view['eligible'],
                           lists.reshape(n,p['views'],p['candidates_per_view']),full_lists)
        times['candidate_retrieval_seconds']=time.perf_counter()-tick;tick=time.perf_counter()
        st=p['reference_predictor'];settings={'folds':5,'seed':0,'tolerance':st['nmf_tolerance'],
            'max_iterations':st['nmf_max_iterations'],'smoothing':st['smoothing'],'unigram_smoothing':st['unigram_smoothing']}
        refs,audits=reference_bank(attempt,col.summary,col.counts,col.folds,12,settings,study.logger)
        if not all(a['converged'] and a['receiver_fold_excluded'] for a in audits):raise ValueError('Runtime reference fit failed')
        times['five_reference_fitting_seconds']=time.perf_counter()-tick;tick=time.perf_counter()
        learned,actions=standalone(col,bank,refs,p)
        times['standalone_selection_and_transfer_seconds']=time.perf_counter()-tick;tick=time.perf_counter()
        model=fit_topics(col.counts,learned,12,0,p['final_nmf'])
        if not model.audit['fit']['converged']:raise ValueError('Runtime final NMF did not converge')
        times['final_NMF_and_original_inference_seconds']=time.perf_counter()-tick
        times['cold_audited_pipeline_seconds']=time.perf_counter()-start
        # Verification and topic metrics are deliberately outside the runtime interval.
        old=study.step5/'selection'/corpus/'mask3711'/'K12'/'preserve_forward_fixed_cap'
        expected=read(old/'actions.json')
        for a,b in zip(actions,expected):
            if a['donors']!=b['donors'] or a['available_donors']!=b['available_donors'] or abs(a['actual_beta']-b['actual_beta'])>1e-12:
                raise ValueError('Cold runtime did not reproduce the frozen default action bank')
        delta=(learned-sparse.load_npz(old/'transferred.npz')).tocsr()
        if delta.nnz and np.max(np.abs(delta.data))>2e-12:raise ValueError('Runtime transfer bank differs')
        import torch
        hardware={'cpu':platform.processor(),'platform':platform.platform(),'torch':torch.__version__,
                  'encoder_provider':encoder.audit['provider'],'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        row=dict(corpus=corpus,K=12,final_seed=0,posts=n,vocabulary=v,**times,
            actual_encoder_queries=n+len(view['queries']),cache_reuse_in_this_attempt=False,
            evaluation_and_verification_excluded=True,audited_loading_includes_backend_verification_probes=True,
            repetitions=1,hardware=hardware,actions_reproduced=True)
        done.mkdir(parents=True,exist_ok=True);write(done/'timing.json',row);save_receipt(done)
        rows.append(row);del encoder
    directory=study.work/'runtime_results';write(directory/'timings.json',rows)
    table(directory/'timings.csv',[{k:v for k,v in r.items() if k!='hardware'} for r in rows])
    write(directory/'coverage.json',{'complete':len(rows)==2,'corpora':len(rows),'repetitions_per_corpus':1,
        'standalone_PEBr_only':True,'training_references_and_encoding_included':True,'comparative_speedup_claim':False})
    return len(rows)==2
