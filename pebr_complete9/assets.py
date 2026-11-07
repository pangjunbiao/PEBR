"""Generate publication tables, figures and a source ledger from numerical records."""
from collections import defaultdict
from pathlib import Path
import re
import shutil
import numpy as np
from .common import read,write,table,csv_rows,digest
from .core import summary

LABELS={'nmf_original':'NMF','nmf_similarity':'Similarity sharing','btm':'BTM','knntm':'kNNTM','glocom':'GloCOM',
    'preserve_forward_fixed_cap':'PEBR','matched_preserve_fixed':'Matched-amount sharing',
    'preserve_fixed_prefix':'Same-score prefix','full_pool_loo':'Full-pool LOO control',
    'weighted_hidden_fixed':'Without visible-loss charge','preserve_fixed_no_downside':'Without downside penalty',
    'preserve_fixed_individual':'Individual-donor scoring','preserve_fixed_single_view':'Single-view selection'}
CORPORA={'transport_grievance':'Transport-grievance','chinese_pooled':'Chinese transport'}
def escape(text):
    return ''.join({'&':r'\&','%':r'\%','_':r'\_','#':r'\#','$':r'\$','{':r'\{','}':r'\}',
                    '\\':r'\textbackslash{}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}.get(c,c) for c in str(text))
def value(mean,sd=None,bold=False):
    a='%.4f'%float(mean);a=r'\mathbf{'+a+'}' if bold else a
    if sd is None:return '$'+a+'$'
    b='%.4f'%float(sd)
    return r'\shortstack{$'+a+r'$\\[-1pt]{\scriptsize $\pm '+b+'$}}'
def latex_table(path,caption,label,headers,rows,note='',wide=False,group_header=None,first_column=None):
    environment='table*' if wide else 'table'
    width=r'\textwidth' if wide else r'\linewidth'
    text=[r'\begin{'+environment+r'}[!t]',r'\centering\begingroup\footnotesize',r'\caption{'+caption+'}',
          r'\label{'+label+'}',r'\setlength{\tabcolsep}{3pt}',r'\renewcommand{\arraystretch}{1.12}',
          r'\resizebox{'+width+r'}{!}{\begin{tabular}{@{}'+(first_column or 'l')+'c'*(len(headers)-1)+r'@{}}',r'\toprule']
    if group_header:text += [group_header,r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}']
    text += [' & '.join(headers)+r' \\',r'\midrule']
    text+=[' & '.join(str(v) for v in row)+r' \\[2pt]' for row in rows]
    text += [r'\bottomrule',r'\end{tabular}}']
    if note:text += [r'\par\vspace{2pt}\parbox{'+width+r'}{\scriptsize '+note+'}']
    text += [r'\endgroup\end{'+environment+'}']
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text('\n'.join(text)+'\n',encoding='utf-8')

def method_table(path,records,caption,label,metrics=('npmi','cv','diversity','stability'),wide=False):
    headers=['Method']+[{'npmi':r'NPMI $\uparrow$','cv':r'$C_v\uparrow$','diversity':r'TD $\uparrow$','stability':r'Stability $\uparrow$'}[m] for m in metrics]
    rows=[]
    for r in records:
        cells=[escape(LABELS.get(r['policy'],r['policy']))]
        for m in metrics:
            field=m if m=='stability' else m+'_mean';best=max(round(float(x[field]),4) for x in records)
            cells.append(value(r[field],None if m=='stability' else r[m+'_sd'],round(float(r[field]),4)==best))
        rows.append(cells)
    latex_table(path,caption,label,headers,rows,
        'Means and sample SDs use five final model seeds. Stability averages the ten matched seed pairs. Bold marks the highest displayed mean, including displayed ties; it is not a significance claim.',wide)

def plot_size(study,directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=csv_rows(study.work/'size_results/paired_gains.csv')
    policies=study.protocol['size_policies'][1:]
    colors=['#6b7280','#d97706','#2563eb','#047857']
    for corpus in study.corpora:
        fig,axes=plt.subplots(1,2,figsize=(7.2,2.7),layout='constrained')
        for ax,metric,title in zip(axes,['npmi','cv'],['(a) NPMI change','(b) $C_v$ change']):
            for policy,color in zip(policies,colors):
                rr=sorted([r for r in rows if r['corpus']==corpus and r['policy']==policy and r['metric']==metric],key=lambda r:int(r['target']))
                if len(rr)!=4:raise ValueError('Incomplete size plot data')
                ax.errorbar([int(r['target']) for r in rr],[float(r['mean_paired_gain']) for r in rr],
                    yerr=[float(r['sample_sd_across_ladder_means']) for r in rr],marker='o',ms=3,capsize=2,lw=1,label=LABELS[policy],color=color)
            ax.axhline(0,color='black',linestyle='--',lw=.7);ax.set_title(title,fontsize=9)
            ax.set_xlabel('Nominal collection size',fontsize=8);ax.set_ylabel('Paired change from NMF',fontsize=8)
            ax.set_xticks(study.protocol['size_targets']);ax.tick_params(labelsize=7);ax.grid(alpha=.15)
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=4,fontsize=7,frameon=False)
        name='corpus_size_'+corpus
        fig.savefig(directory/(name+'.pdf'));fig.savefig(directory/(name+'.png'),dpi=170);plt.close(fig)

def plot_parameters(study,directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=csv_rows(study.work/'parameter_results/summaries.csv')
    for corpus in study.corpora:
        fig,axes=plt.subplots(1,2,figsize=(7.2,2.5),layout='constrained')
        for ax,axis,cases in zip(axes,['cap','lambda'],[['cap6','default','cap2'],['lambda0','lambda05','default','lambda2']]):
            rr=[next(r for r in rows if r['corpus']==corpus and r['case']==case) for case in cases]
            for metric,color in [('npmi','#047857'),('cv','#2563eb')]:
                vals=[float(r[metric+'_mean']) for r in rr];base=next(float(r[metric+'_mean']) for r in rr if r['case']=='default')
                ax.plot([float(r[axis]) for r in rr],np.asarray(vals)-base,'o-',ms=3,color=color,label='NPMI' if metric=='npmi' else '$C_v$')
            ax.axhline(0,color='black',lw=.7,ls='--');ax.set_xlabel(r'$\beta_{\max}$' if axis=='cap' else r'$\lambda$',fontsize=9)
            ax.set_ylabel('Mean change from default',fontsize=8);ax.tick_params(labelsize=7);ax.grid(alpha=.15)
            ax.set_xticks([float(r[axis]) for r in rr],['1/6','1/3','1/2'] if axis=='cap' else ['0','0.5','1','2'])
        axes[0].legend(fontsize=7,frameon=False)
        name='parameter_sensitivity_'+corpus
        fig.savefig(directory/(name+'.pdf'));fig.savefig(directory/(name+'.png'),dpi=170);plt.close(fig)

def budget_diagnostics(study,root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records=read(study.work/'benchmark_results/seed_records.json');rows=[];trajectories=[]
    for r in records:
        if r['policy'] not in ('knntm','glocom') or r['epoch_budget']!=400:continue
        loss=csv_rows(Path(r['path'])/'training_losses.csv');values=np.asarray([float(x['loss']) for x in loss])
        if len(values)!=400 or not np.isfinite(values).all():raise ValueError('Incomplete budget-check loss history')
        rows.append({'corpus':r['corpus'],'K':r['K'],'policy':r['policy'],'seed':r['seed'],
            'epochs21_40_mean_loss':float(values[20:40].mean()),'epochs161_200_mean_loss':float(values[160:200].mean()),
            'epochs361_400_mean_loss':float(values[360:400].mean()),
            'within_augmented_phase_comparison':True,'formal_convergence_claim':False})
        trajectories += [{'corpus':r['corpus'],'K':r['K'],'policy':r['policy'],'seed':r['seed'],
                          'epoch':i+1,'native_loss':float(v)} for i,v in enumerate(values)]
    table(root/'source_data/neural_budget_loss_diagnostics.csv',rows)
    table(root/'source_data/neural_budget_loss_trajectories.csv',trajectories)
    fig,axes=plt.subplots(2,2,figsize=(7.2,4.2),layout='constrained')
    for i,c in enumerate(study.corpora):
        for j,method in enumerate(['knntm','glocom']):
            vals=[]
            for seed in range(5):
                rr=[r for r in trajectories if (r['corpus'],r['K'],r['policy'],r['seed'])==(c,12,method,seed)]
                vals.append([r['native_loss'] for r in rr])
            values=np.asarray(vals);x=np.arange(1,401);mean=values.mean(0);sd=values.std(0,ddof=1)
            axes[i,j].plot(x,mean,lw=1,color='#2563eb');axes[i,j].fill_between(x,mean-sd,mean+sd,color='#2563eb',alpha=.15)
            if method=='knntm':axes[i,j].axvline(20,lw=.8,ls='--',color='#6b7280')
            axes[i,j].set_title(CORPORA[c]+' / '+LABELS[method],fontsize=8)
            axes[i,j].set_xlabel('Epoch',fontsize=8);axes[i,j].set_ylabel('Native training loss',fontsize=8);axes[i,j].tick_params(labelsize=7)
    fig.savefig(root/'figures/neural_budget_loss_K12.pdf');fig.savefig(root/'figures/neural_budget_loss_K12.png',dpi=170);plt.close(fig)

def borrowing_statistics(study,root):
    out=[];policies=['nmf_original','nmf_similarity','matched_preserve_fixed','preserve_fixed_prefix','preserve_forward_fixed_cap']
    for c,corpus in study.corpora.items():
        n=corpus['collection'].counts.shape[0]
        for p in policies:
            banks=[]
            for mask in [3711,4711,5711,6711]:
                if p=='nmf_original':stats=[0.,0.,0.,0.,0.]
                else:
                    rr=read(study.step5/'selection'/c/('mask%d'%mask)/'K12'/p/'actions.json')
                    if len(rr)!=n:raise ValueError('Borrowing bank post count differs')
                    stats=[100*np.mean([r['actual_beta']>0 for r in rr]),np.mean([r['actual_beta'] for r in rr]),
                        np.mean([len(r['donors']) for r in rr]),np.mean([len(r['available_donors']) for r in rr]),
                        100*np.mean([r['offered_beta']>0 and r['actual_beta']==0 for r in rr])]
                banks.append(stats)
            a=np.asarray(banks)
            out.append({'corpus':c,'policy':p,**{f+'_mean':float(a[:,i].mean()) for i,f in enumerate(['coverage_percent','mass','selected_donors','contributing_donors','zero_transfer_positive_offer_percent'])},
                        **{f+'_sd_banks':float(a[:,i].std(ddof=1)) for i,f in enumerate(['coverage_percent','mass','selected_donors','contributing_donors','zero_transfer_positive_offer_percent'])}})
    table(root/'source_data/borrowing_behavior_K12.csv',out)
    for c in study.corpora:
        rows=[]
        for r in out:
            if r['corpus']==c:rows.append([escape(LABELS[r['policy']])]+[value(r[f+'_mean'],r[f+'_sd_banks']) for f in ['coverage_percent','mass','selected_donors','contributing_donors','zero_transfer_positive_offer_percent']])
        latex_table(root/'supplement'/('borrowing_'+c+'.tex'),'Borrowing behavior at K12 on '+CORPORA[c]+'.','tab:s9_borrowing_'+c,
            ['Policy',r'Transfer coverage (\%)','Mean mass','Selected donors','Contributing donors',r'Positive offer, zero transfer (\%)'],rows,
            'All post-level statistics include zero-borrowing rows. Means and SDs span four selection banks; they are not five final-NMF-seed experiments. Selected donors and available contributing donors are different quantities.')

def benchmark_tables(study,root):
    rows=read(study.work/'benchmark_results/seed_records.json')
    main=[r for r in rows if r['policy'] not in ('knntm','glocom') or int(r['epoch_budget'])==400]
    if len(main)!=180:raise ValueError('Main all-method benchmark does not have 180 real records')
    ss=summary(main,['corpus','K','policy'])
    table(root/'source_data/main_benchmark_summaries.csv',ss)
    policies=['nmf_original','nmf_similarity','btm','knntm','glocom','preserve_forward_fixed_cap']
    headers=['Model']+[m for c in ('transport_grievance','chinese_pooled') for m in ['NPMI','$C_v$','TD','Stability']]
    out=[]
    for policy in policies:
        row=[escape(LABELS[policy])]
        for c in ('transport_grievance','chinese_pooled'):
            rr=[r for r in ss if r['corpus']==c and r['K']==12];r=next(x for x in rr if x['policy']==policy)
            for m in ('npmi','cv','diversity','stability'):
                f=m if m=='stability' else m+'_mean';best=max(round(float(x[f]),4) for x in rr)
                row.append(value(r[f],None if m=='stability' else r[m+'_sd'],round(float(r[f]),4)==best))
        out.append(row)
    latex_table(root/'main/benchmark_K12.tex','Topic-quality comparison at $K=12$.','tab:topic_model_comparison',headers,out,
        'Values summarize five final seeds. NMF sharing methods use the predefined first fresh selection bank (3711). Neural methods use the precommitted 400-epoch budget; BTM uses 1,000 iterations. Stability is the ten-pair aggregate. The 200-epoch neural check and all topic counts are retained in the supplement.',True,
        r' & \multicolumn{4}{c}{Transport-grievance} & \multicolumn{4}{c}{Chinese transport} \\')
    for c in study.corpora:
        for k in [8,10]:
            rr=[next(r for r in ss if (r['corpus'],r['K'],r['policy'])==(c,k,p)) for p in policies]
            method_table(root/'supplement'/('benchmark_%s_K%d.tex'%(c,k)),rr,CORPORA[c]+' topic quality at $K=%d$.'%k,'tab:s9_bench_'+c+str(k))
    allss=summary(rows,['corpus','K','policy','epoch_budget']);table(root/'source_data/all_benchmark_budget_summaries.csv',allss)
    budget=[]
    for c in study.corpora:
        for k in [8,10,12]:
            for method in ['knntm','glocom']:
                a=next(r for r in allss if (r['corpus'],r['K'],r['policy'],r['epoch_budget'])==(c,k,method,200))
                b=next(r for r in allss if (r['corpus'],r['K'],r['policy'],r['epoch_budget'])==(c,k,method,400))
                budget.append([escape(CORPORA[c]),str(k),LABELS[method],value(a['npmi_mean'],a['npmi_sd']),value(b['npmi_mean'],b['npmi_sd']),
                               value(a['cv_mean'],a['cv_sd']),value(b['cv_mean'],b['cv_sd'])])
    latex_table(root/'supplement/neural_budget_check.tex','Neural training-budget check with all other settings fixed.','tab:s9_budget',
        ['Corpus','$K$','Method','NPMI, 200','NPMI, 400','$C_v$, 200','$C_v$, 400'],budget,
        'Each 400-epoch run continues its same-seed 200-epoch checkpoint. The main neural benchmark budget was fixed at 400 before these results. This checks budget sensitivity, not optimal hyperparameter tuning.')

def existing_tables(study,root):
    fresh=read(study.step5/'results/topic_words.json')
    ps=['nmf_original','nmf_similarity','matched_preserve_fixed','preserve_fixed_prefix','full_pool_loo','preserve_forward_fixed_cap']
    comp=['preserve_forward_fixed_cap','weighted_hidden_fixed','preserve_fixed_no_downside','preserve_fixed_individual','preserve_fixed_single_view']
    for c in study.corpora:
        rr=[r for r in fresh if r['corpus']==c and r['K']==12 and r['policy'] in ps and (r['policy']=='nmf_original' or int(r['mask_base'])==3711)]
        ss=summary(rr,['policy'])
        ordered=[next(r for r in ss if r['policy']==p) for p in ps]
        method_table(root/'main'/('policy_'+c+'.tex'),ordered,'Controlled borrowing policies on '+CORPORA[c]+'.','tab:s9_policy_'+c,('npmi','cv','diversity'))
        cr=[next(r for r in fresh if r['corpus']==c and r['K']==12 and r['policy']==p and int(r['mask_base'])==3711 and r['seed']==0) for p in comp]
        lines=[[escape(LABELS[r['policy']]),value(r['npmi']),value(r['cv']),value(r['diversity'])] for r in cr]
        latex_table(root/'main'/('components_seed0_'+c+'.tex'),'Fixed-seed component comparison on '+CORPORA[c]+'.','tab:s9_components_'+c,
            ['Variant','NPMI','$C_v$','TD'],lines,'Selection bank 3711 and final seed 0. The complete five-seed/four-bank study remains in the supplementary results. Single-view selection changes both retrieval coverage and scoring views.')
        # Keep the raw component pattern as well as paired differences. Aggregate
        # final seeds within each bank first; the SD unit is then the four banks.
        raw=[r for r in fresh if r['corpus']==c and r['policy'] in comp and int(r['mask_base']) in (3711,4711,5711,6711)]
        grouped=summary(raw,['K','policy','mask_base']);component_rows=[];component_source=[]
        for k in [8,10,12]:
            for policy in comp:
                rr=[r for r in grouped if int(r['K'])==k and r['policy']==policy]
                if len(rr)!=4:raise ValueError('Four-bank component coverage is incomplete')
                row=[str(k),escape(LABELS[policy])];record={'corpus':c,'K':k,'policy':policy,'selection_banks':4}
                for m in ('npmi','cv','diversity','stability'):
                    values=np.array([float(r[m if m=='stability' else m+'_mean']) for r in rr])
                    record[m+'_mean']=float(values.mean());record[m+'_sd_banks']=float(values.std(ddof=1))
                    row.append(value(values.mean(),values.std(ddof=1)))
                component_rows.append(row);component_source.append(record)
        table(root/'source_data'/('components_four_banks_'+c+'.csv'),component_source)
        latex_table(root/'supplement'/('components_four_banks_'+c+'.tex'),'Repeated-selection component results on '+CORPORA[c]+'.',
            'tab:s9_components_banks_'+c,['$K$','Variant','NPMI','$C_v$','TD','Stability'],component_rows,
            'Means and sample SDs span four fresh selection banks, after averaging five final seeds within each bank. Stability first matches the ten seed pairs within each bank. These are four conditional bank summaries, not twenty independent experiments. All variants and topic counts are retained.')
    # Four-bank effect table: means of paired five-final-seed means, not twenty independent fits.
    pairs=csv_rows(study.step5/'results/paired_fresh_bank_contrasts.csv')
    rows=[]
    for c in study.corpora:
        for k in [8,10,12]:
            for p in ['nmf_original','nmf_similarity','matched_preserve_fixed','preserve_fixed_prefix','weighted_hidden_fixed','preserve_fixed_no_downside','preserve_fixed_individual','preserve_fixed_single_view']:
                a=next(r for r in pairs if (r['corpus'],int(r['K']),r['comparator'],r['metric'])==(c,k,p,'npmi'))
                b=next(r for r in pairs if (r['corpus'],int(r['K']),r['comparator'],r['metric'])==(c,k,p,'cv'))
                rows.append([escape(CORPORA[c]),str(k),escape(LABELS[p]),value(a['mean_revised_minus_control'],a['sd_across_fresh_bank_means']),
                             str(a['positive_banks'])+'/4',value(b['mean_revised_minus_control'],b['sd_across_fresh_bank_means']),str(b['positive_banks'])+'/4'])
    for c in study.corpora:
        selected=[r for r in rows if r[0]==escape(CORPORA[c])]
        latex_table(root/'supplement'/('fresh_selection_'+c+'.tex'),'Fresh-selection paired effects: PEBR minus comparator, '+CORPORA[c]+'.','tab:s9_fresh_'+c,
            ['Corpus','$K$','Comparator',r'$\Delta$NPMI','Positive banks',r'$\Delta C_v$','Positive banks'],selected,
            'Means and sample SDs are across four fresh bank effects, each averaging five paired final seeds. The known development bank 2711 is excluded. Samples concern one collection and are not independent-corpus replicates or confidence intervals.')
    protected=csv_rows(study.step6/'results/fresh_protected_bank_summaries.csv')
    for c in study.corpora:
        lines=[]
        for k in [8,10,12]:
            for p in ['nmf_original','nmf_similarity','matched_preserve_fixed','preserve_fixed_prefix','weighted_hidden_fixed','preserve_fixed_no_downside','preserve_forward_fixed_cap']:
                r=next(x for x in protected if (x['corpus'],int(x['K']),x['policy'])==(c,k,p))
                lines.append([str(k),escape(LABELS[p]),value(r['mean_post_gain_mean'],r['mean_post_gain_sd_banks']),
                    value(100*float(r['post_loss_fraction_mean']),100*float(r['post_loss_fraction_sd_banks'])),
                    value(r['mean_negative_part_all_posts_mean'],r['mean_negative_part_all_posts_sd_banks'])])
        latex_table(root/'supplement'/('protected_words_'+c+'.tex'),'Protected-word audit on '+CORPORA[c]+'.','tab:s9_protected_'+c,
            ['$K$','Policy','Mean gain',r'Posts with loss (\%)','Mean negative part'],lines,
            'Means and SDs span four fresh outer word holdouts, excluding known bank 2712. Gain and negative-part magnitude are nats/token. Negative parts average all eligible posts. This is a fixed-predictor audit, not final-model seed replication or a guarantee of topic coherence.')
    effects=csv_rows(study.step8/'results/paired_effects.csv');ss=csv_rows(study.step8/'results/case_summaries.csv')
    for c in study.corpora:
        lines=[]
        for k in [8,10,12]:
            for case in ['default','r1','r2','f3710','f4710']:
                r=next(x for x in ss if (x['corpus'],int(x['K']),x['case'],x['policy'])==(c,k,case,'preserve_forward_fixed_cap'))
                def effect(p,m):return next(x['mean_paired_difference'] for x in effects if (x['corpus'],int(x['K']),x['case'],x['comparator'],x['metric'])==(c,k,case,p,m))
                lines.append([str(k),escape(case),value(r['npmi_mean'],r['npmi_sample_sd']),value(r['cv_mean'],r['cv_sample_sd']),
                    value(effect('matched_preserve_fixed','npmi')),value(effect('matched_preserve_fixed','cv')),
                    value(effect('preserve_fixed_prefix','npmi')),value(effect('preserve_fixed_prefix','cv'))])
        latex_table(root/'supplement'/('reference_folds_'+c+'.tex'),'Reference/fold sensitivity on '+CORPORA[c]+'.','tab:s9_ref_'+c,
            ['$K$','Setting','PEBR NPMI','PEBR $C_v$',r'$\Delta$NPMI matched',r'$\Delta C_v$ matched',r'$\Delta$NPMI prefix',r'$\Delta C_v$ prefix'],lines,
            'Each setting averages five final seeds. r1/r2 change reference initialization only; f3710/f4710 change family folds, predictive fitting partitions and eligible donors. All use mask 3711. Shared fits are not additional independent replicates.')

def transport_assets(study,root):
    source=study.step7/'transportation'
    for name in ['transport_cases_revised.tex','service_word_recurrence_revised_1.pdf','service_word_recurrence_revised_2.pdf',
                 'service_word_recurrence_revised_1.png','service_word_recurrence_revised_2.png']:
        shutil.copyfile(source/name,root/'supplement'/name)
    # Only concise original excerpts are used in the publication table; no completion text.
    counts=csv_rows(study.step8/'transportation_clarification/decision_types.csv')
    gains=csv_rows(study.step8/'transportation_clarification/case_policy_gains.csv')
    selected=[271,357,508,5];rows=[]
    titles={271:'Cab payment demand',357:'Train delay',508:'Aircraft smoking report',5:'Toll/payment complaint'}
    for s in selected:
        r=next(x for x in counts if int(x['source_row'])==s)
        # Mark omissions, remove phone-like spaced sequences and keep the statement inspectable.
        quote=re.sub(r'(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)','[NUMBER]',r['receiver_excerpt'])
        if len(quote)>170:quote=quote[:quote.rfind(' ',0,170)]+' [...]'
        def gain(policy):return next(float(x['gain_nats_per_token']) for x in gains if int(x['source_row'])==s and x['policy']==policy)
        rows.append([escape(titles[s]+' (row '+str(s)+')')+r'\par\textit{'+escape(quote)+'}',
            r'\shortstack{'+r['selected_donor_count']+r' selected\\'+r['contributing_donor_count']+' contributing}',
            value(r['realized_mass']),value(gain('nmf_similarity')),value(gain('preserve_fixed_prefix')),value(gain('preserve_forward_fixed_cap'))])
    latex_table(root/'main/transport_decisions.tex','Illustrative evidence decisions on original transport posts.','tab:transport_decisions',
        ['Original post','PEBR donors','Realized mass','Sharing gain','Prefix gain','PEBR gain'],rows,
        'Fixed protected audit: $K=12$, outer bank 3712. Gains are nats per target token. Rows 271/357 are prior fixed anchors; row 508 is the first eligible explicit empty action in source order with positive similarity transfer; row 5 is the disclosed unsuccessful example. Illustrations are not a separate validation cohort. Source row 508 is a news-like safety report. Unavailable donor shares remain with the receiver. Selected and contributing donor counts differ. The bus anchor, row 14, illustrates availability fallback, not explicit admission refusal.',True,first_column=r'p{0.45\textwidth}')
    cleaned=[]
    for row in counts:
        row=dict(row);quote=re.sub(r'(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)','[NUMBER]',row['receiver_excerpt'])
        if len(quote)>170:quote=quote[:quote.rfind(' ',0,170)]+' [...]'
        row['receiver_excerpt']=quote;cleaned.append(row)
    table(root/'source_data/transport_decision_types.csv',cleaned)
    for name in ['protected_case_word_probabilities.csv','protected_case_policy_gains.csv','fixed_anchor_protected_repeats.csv','topic_word_recurrence.csv']:
        shutil.copyfile(source/name,root/'source_data'/('transport_'+name))
    donor_rows=csv_rows(source/'protected_case_donors.csv')
    for row in donor_rows:
        quote=re.sub(r'(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)','[NUMBER]',row['donor_excerpt'])
        if len(quote)>170:quote=quote[:quote.rfind(' ',0,170)]+' [...]'
        row['donor_excerpt']=quote
    table(root/'source_data/transport_protected_case_donors.csv',donor_rows)
    shutil.copyfile(study.step8/'transportation_clarification/case_policy_gains.csv',root/'source_data/transport_all_decision_policy_gains.csv')

def retrieval_assets(study,root):
    source=study.work/'retrieval_results';rows=csv_rows(source/'summaries.csv');cohort=read(source/'cohort.json')
    policies=['nmf_original','nmf_similarity','btm','knntm','glocom',*study.protocol['retrieval_controls'],'preserve_forward_fixed_cap']
    for k in [8,10,12]:
        lines=[]
        for p in policies:
            row=[escape(LABELS[p])]
            for budget in [5,10,20]:
                r=next(x for x in rows if (int(x['K']),x['policy'],int(x['budget']))==(k,p,budget))
                row += [value(r['ndcg_mean'],r['ndcg_sd']),value(r['recall_mean'],r['recall_sd'])]
            lines.append(row)
        latex_table(root/'supplement'/('auxiliary_retrieval_K%d.tex'%k),
            'Within-collection category retrieval on transport-grievance posts at $K=%d$.'%k,'tab:s9_retrieval_'+str(k),
            ['Method','nDCG@5','Recall@5','nDCG@10','Recall@10','nDCG@20','Recall@20'],lines,
            'Macro scores first average queries within each category, then categories equally. Means and sample SDs span five fitted seeds; PEBR uses selection bank3711. Neural models use the same 400-epoch budget as the main benchmark. Labels enter this evaluation only. These frozen models were fitted on the collection containing the queries; this audit does not measure generalization to new complaints.')
    contrasts=csv_rows(source/'paired_contrasts.csv')
    lines=[[escape(LABELS[r['comparator']]),'nDCG' if r['metric']=='ndcg' else 'Recall',
        value(r['difference']),value(r['ci_low']),value(r['ci_high'])] for r in contrasts]
    latex_table(root/'supplement/auxiliary_retrieval_contrasts.tex',
        'Paired category-retrieval contrasts: PEBR minus comparator, $K=12$, $B=10$.','tab:s9_retrieval_contrasts',
        ['Comparator','Metric','Difference',r'95\% lower',r'95\% upper'],lines,
        'Intervals use 2,000 query-family resamples within each service category after averaging the five fitted seeds. They are conditional on those fitted models, not independent-corpus or unseen-query uncertainty estimates. All unfavorable differences are retained.')
    (root/'supplement/retrieval_protocol.tex').write_text(
        '\\paragraph{Auxiliary category-based complaint retrieval.} The audit uses '+str(cohort['query_count'])+
        ' original posts with exactly one substantive service category and at least one eligible relevant candidate. '
        'All methods share the same cohort and catalogues. Catalogues exclude the query family and fold; '
        'a candidate is relevant if its annotations include the query category. Miscellaneous and Not\\_applicable are excluded. '
        'Original-post topic contributions are ranked by cosine, with source-order ties. Zero query vectors return nothing while remaining in the cohort; '
        'zero candidate vectors remain eligible with score zero. Recall uses all relevant catalogue items as its denominator; '
        'nDCG uses binary relevance and the corresponding ideal ranking. Labels enter neither topic fitting nor ranking.\n',encoding='utf-8')

def run(study):
    root=study.work/'publication_assets'
    for name in ['main','supplement','figures','source_data']:(root/name).mkdir(parents=True,exist_ok=True)
    benchmark_tables(study,root);existing_tables(study,root);transport_assets(study,root);borrowing_statistics(study,root);retrieval_assets(study,root)
    params=csv_rows(study.work/'parameter_results/summaries.csv')
    for c in study.corpora:
        lines=[]
        for parameter,cases in [(r'$\beta_{\max}$',['cap6','default','cap2']),(r'$\lambda$',['lambda0','lambda05','default','lambda2'])]:
            for case in cases:
                r=next(x for x in params if x['corpus']==c and x['case']==case)
                x=r['cap'] if parameter==r'$\beta_{\max}$' else r['lambda']
                shown=({'cap6':'$1/6$','default':'$1/3$','cap2':'$1/2$'}[case] if parameter==r'$\beta_{\max}$' else '$%g$'%float(x))
                lines.append([parameter,shown,value(r['npmi_mean'],r['npmi_sd']),value(r['cv_mean'],r['cv_sd']),value(r['diversity_mean'],r['diversity_sd'])])
        latex_table(root/'main'/('parameters_'+c+'.tex'),'Parameter sensitivity on '+CORPORA[c]+'.','tab:s9_parameters_'+c,
            ['Parameter','Value','NPMI','$C_v$','TD'],lines,
            r'One parameter changes at a time. Means and SDs use five final seeds with mask3711. The default appears in both sweeps but is fitted only once. It remains $\beta_{\max}=1/3$, $\lambda=1$; these results do not retune the method.')
    plot_size(study,root/'figures');plot_parameters(study,root/'figures');budget_diagnostics(study,root)
    times=read(study.work/'runtime_results/timings.json')
    headers=['Corpus','Posts','Encoder load/verification','Encoding','References','Selection/transfer','Final fit/inference','Cold total']
    rows=[[CORPORA[r['corpus']],str(r['posts'])]+['$%.2f$'%r[f] for f in ['audited_encoder_loading_seconds','full_and_visible_encoding_seconds','five_reference_fitting_seconds','standalone_selection_and_transfer_seconds','final_NMF_and_original_inference_seconds','cold_audited_pipeline_seconds']] for r in times]
    latex_table(root/'supplement/runtime.tex','Standalone PEBR pipeline timing, in seconds.','tab:s9_runtime',headers,rows,
        'One fresh-cache, predefined seed0 pipeline per corpus at K12. Full/masked encoding and all five predictive-reference fits are included. The total also includes mask construction and candidate retrieval. Audited loading includes backend verification probes. Topic evaluation and output verification are excluded. This is a hardware-specific observation, not a comparative speedup or runtime uncertainty estimate.')
    # Preserve all numerical tables, not just manuscript-selected settings.
    for phase in ['benchmark_results','parameter_results','size_results','runtime_results','retrieval_results']:
        for p in (study.work/phase).glob('*'):
            if p.is_file():shutil.copyfile(p,root/'source_data'/(phase+'_'+p.name))
    for stage,source in [('selection',study.step5/'results'),('protected',study.step6/'results'),('reference_folds',study.step8/'results')]:
        for p in source.glob('*.csv'):shutil.copyfile(p,root/'source_data'/(stage+'_'+p.name))
    fragments=sorted((root/'supplement').glob('*.tex'),key=lambda p:(p.name!='retrieval_protocol.tex',p.name))
    text=['% Append these result assets below the corrected supplementary methods.',
          '% Requires booktabs, graphicx, array and tabularx.',r'\section{Additional results for the revised PEBR rule}']
    text += [r'\input{publication_assets/supplement/'+p.name+'}' for p in fragments]
    for c in study.corpora:
        text += [r'\begin{figure}[!htbp]\centering',r'\includegraphics[width=\linewidth]{publication_assets/figures/corpus_size_'+c+'.pdf}',
            r'\caption{Corpus-size sensitivity on '+CORPORA[c]+r': paired changes from NMF. Points average five ladder means, each averaging five paired final seeds. Bars are sample SDs across overlapping ladders, not confidence intervals. Vocabulary is refitted at minimum DF2 within each sample; unusable rows and actual sizes are recorded.}\end{figure}']
    for i,description in [(1,'cab-payment and train-delay'),(2,'coach-condition and bus-service')]:
        text += [r'\begin{figure}[!htbp]\centering',r'\includegraphics[width=\linewidth]{publication_assets/supplement/service_word_recurrence_revised_'+str(i)+'.pdf}',
            r'\caption{Topic-word recurrence for the '+description+r' anchors at $K=12$. Cells count top-ten appearances across five final model fits in the first fresh selection bank3711. Displayed words occur in at least three fits for a method. Counts concern model fits, not complaints.}\end{figure}']
    (root/'supplement_results.tex').write_text('\n'.join(text)+'\n',encoding='utf-8')
    main_fragments=sorted((root/'main').glob('*.tex'))
    (root/'main_results_tables.tex').write_text('% Table fragments only; add the revised Results prose separately.\n'+
        '\n'.join(r'\input{publication_assets/main/'+p.name+'}' for p in main_fragments)+'\n',encoding='utf-8')
    (root/'FIGURE_CAPTIONS.md').write_text(
        '# Figure assets and scope\n\n'
        '- `corpus_size_transport_grievance.pdf` / `corpus_size_chinese_pooled.pdf`: K12 paired coherence changes from no borrowing. '
        'Five nested family-aware ladders; each ladder averages five paired model seeds. Bars show sample SD across overlapping ladder means, not confidence intervals. '
        'Nominal sample targets are on the horizontal axis; actual retained sizes and refitted DF2 vocabularies are in source_data/size_results_sampling_counts.csv.\n'
        '- `parameter_sensitivity_*.pdf`: mean changes from the unchanged default cap1/3 and lambda1, one parameter at a time. '
        'The corresponding tables provide SDs across five final seeds conditional on mask3711; no parameter is retuned from these curves. '
        'Use either the table or plot where space makes duplication unnecessary.\n'
        '- `neural_budget_loss_K12.pdf`: native loss mean and sample SD across five seeds for each neural method/corpus. '
        'The kNNTM dashed line marks the warm-up boundary. Loss scales are model-specific; these are training diagnostics, not cross-method quality rankings or formal convergence guarantees.\n'
        '- `supplement/service_word_recurrence_revised_1.pdf` / `_2.pdf`: unchanged verified Step7 recurrence outputs for the same four fixed anchors, '
        'first fresh bank3711 and five final seeds. Counts are top-ten word appearances across model fits, not complaints. '
        'They describe final topic interpretation; protected decision examples use a different outer bank3712.\n',encoding='utf-8')
    ledger={}
    sources={phase:{p.relative_to(study.work).as_posix():digest(p) for p in (study.work/phase).glob('*') if p.is_file()}
             for phase in ['benchmark_results','parameter_results','size_results','runtime_results','retrieval_results']}
    for p in root.rglob('*'):
        if p.is_file():ledger[p.relative_to(root).as_posix()]={'sha256':digest(p),'phase_sources':sources,
            'existing_stage_identities':study.lineage}
    write(study.work/'asset_manifest.json',ledger)
    write(study.work/'asset_coverage.json',{'complete':True,'assets':len(ledger),'main_benchmark_rows':180,
        'main_neural_budget':400,'main_rule_selection_bank':3711,'manual_result_numbers_inserted':False,
        'manuscript_text_updated':False,'all_assets_require_final_inspection_before_submission':True})
    return True
