"""Create a standalone, source-bound English report for the completed native grid."""
from pathlib import Path
import hashlib
import json
import sys
from xml.sax.saxutils import escape

REPO=Path(__file__).resolve().parents[2]
CODE=REPO/'code'
sys.path.insert(0,str(CODE/'.cache/report_dependencies'))
sys.path.insert(0,str(CODE))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,Flowable
from pypdf import PdfReader

from experiments.laptop_comparison import atomic_json
from experiments.protected_jl_study import Archive

ROOT=CODE/'results/paper_compression_balanced_audited'
TABLES=ROOT/'paper_tables'
SUPPORT=ROOT/'new_report_20261012'
PDF=REPO/'output/pdf/native_compression_results_20261012.pdf'
NAVY='#17334A';TEAL='#007E82';GRAY='#586A7B';ORANGE='#BB6B29';RED='#A23F40'
ORDER=['gaussian_easy','gaussian_hard','dbscan_margin','mnist','fashion_mnist','gisette','arcene','20ng_tfidf']
NAMES={'gaussian_easy':'Gaussian easy','gaussian_hard':'Gaussian hard','dbscan_margin':'DBSCAN margin',
    'mnist':'MNIST','fashion_mnist':'Fashion-MNIST','gisette':'Gisette','arcene':'Arcene','20ng_tfidf':'20 Newsgroups'}
METHODS={'full':'Full','pca':'PCA: full SVD','pca_randomized':'PCA: randomized','sparse_jl':'Sparse JL',
    'protected_jl':'Protected JL','protected_only':'Protected only'}
WIDTH=A4[0]-84


def name(row):
    value=NAMES[row['dataset']]
    return value+' (nonzero)' if row.get('variant')=='nonzero' else value


def ordered(frame):
    result=frame.copy();result['_order']=result.dataset.map({v:i for i,v in enumerate(ORDER)})
    return result.sort_values(['_order','variant','k'] if 'k' in result else ['_order','variant']).drop(columns='_order')


def num(value,digits=3):return '--' if pd.isna(value) else f'{float(value):.{digits}g}'


def ci(row,prefix,scale=1,digits=3):
    value=row.get(prefix+'_mean',np.nan);lo=row.get(prefix+'_ci_low',np.nan);hi=row.get(prefix+'_ci_high',np.nan)
    if pd.isna(value):return '--'
    text=num(value*scale,digits)
    if pd.notna(lo) and pd.notna(hi):text+=f'<br/><font color="{GRAY}">[{num(lo*scale,digits)}, {num(hi*scale,digits)}]</font>'
    return text


def build():
    SUPPORT.mkdir(parents=True,exist_ok=True);PDF.parent.mkdir(parents=True,exist_ok=True)
    validation=json.loads((ROOT/'report/validation.json').read_text())
    replay=json.loads((ROOT/'report/replay_validation.json').read_text())
    manifest=json.loads((ROOT/'manifest.json').read_text());cfg=manifest['identity']['configuration']
    assert validation['complete'] and validation['valid_observed_results'] and replay['all_passed']
    source_hashes={}
    def read(filename):
        path=TABLES/filename;source_hashes[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.read_csv(path)
    data=read('01_datasets.csv');selection=read('02_selection.csv');quality=read('03_fidelity_all.csv')
    reference=read('04_fidelity_m128_k5.csv');setup=read('05_compression_setup.csv');training=read('06_model_training.csv')
    wl=read('07_downstream_quality.csv');runtimes=read('08_downstream_runtime.csv');profiles=read('10_dbscan_profiles.csv')
    ops=read('11_selected_operating_points.csv')
    hjl=ordered(ops[ops.method.eq('protected_jl')]);qualified=hjl[hjl.test_fidelity_rule_pass]
    assert len(hjl)==16 and len(qualified)==15
    archive=Archive();x=archive.load('results/paper_server_20261006/rq1/20ng_tfidf/artifacts/dataset.npz')['X']
    zero=~np.any(x!=0,axis=1)
    diagnostic_path=ROOT/'jobs/20ng_tfidf_primary/workloads/test_k5_s0_protected_jl_m256_q1_seed0/knn.npz'
    with np.load(diagnostic_path) as saved:
        zero_info=dict(queries=len(saved['query_indices']),neighbor_slots=saved['original_neighbors'].size,
            zero_queries=int(zero[saved['query_indices']].sum()),zero_neighbors=int(zero[saved['original_neighbors']].sum()),
            zero_neighbor_fraction=float(zero[saved['original_neighbors']].mean()),source_npz_sha256=hashlib.sha256(diagnostic_path.read_bytes()).hexdigest())
    atomic_json(SUPPORT/'zero_neighbor_diagnostic.json',zero_info)
    del x

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'savefig.facecolor':'white','axes.titleweight':'bold'})
    palette={'protected_jl':TEAL,'sparse_jl':ORANGE,'pca':'#476EA0','pca_randomized':'#879AB2'}
    fig,axs=plt.subplots(1,3,figsize=(9.8,3.3),layout='constrained')
    for ax,dataset in zip(axs,['mnist','gisette','20ng_tfidf']):
        cells=quality[(quality.dataset==dataset)&(quality.variant=='primary')&(quality.k==5)]
        for method in ['pca','pca_randomized','sparse_jl','protected_jl']:
            group=cells[(cells.method==method)&(cells.q==(1 if method in ('sparse_jl','protected_jl') else 0))].sort_values('m')
            ax.plot(group.m,group.pair_q95_mean,marker='o',markersize=3,color=palette[method],
                linestyle='--' if method=='pca_randomized' else '-',label=METHODS[method])
            ax.fill_between(group.m,group.pair_q95_ci_low,group.pair_q95_ci_high,color=palette[method],alpha=.10)
        ax.axhline(.2,color=GRAY,linestyle=':',linewidth=1);ax.set_xscale('log',base=2)
        ax.set_xticks([16,64,256,512],[16,64,256,512]);ax.set(title=NAMES[dataset],xlabel='Total dimension m',ylim=(0,1.08))
        ax.grid(alpha=.18)
    axs[0].set_ylabel('Pair Q95 relative error')
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='outside lower center',fontsize=8)
    geometry=SUPPORT/'geometry.png';fig.savefig(geometry,dpi=240);fig.savefig(SUPPORT/'geometry.pdf');plt.close(fig)

    k5=hjl[hjl.k.eq(5)].copy();k5['label']=[name(r) for _,r in k5.iterrows()]
    fig,ax=plt.subplots(figsize=(9.4,4.2),layout='constrained');positions=np.arange(len(k5));height=.24
    for offset,prefix,label,color in [(-height,'query_only_speedup','Query kernel only',TEAL),
        (0,'conditional_reuse_speedup','Reuse incl. preparation','#577EA4'),(height,'cold_speedup','Cold incl. fitting',ORANGE)]:
        mean=k5[prefix+'_mean'].to_numpy();lo=k5[prefix+'_ci_low'].to_numpy();hi=k5[prefix+'_ci_high'].to_numpy()
        ax.errorbar(mean,positions+offset,xerr=np.array([np.maximum(mean-lo,0),np.maximum(hi-mean,0)]),
            fmt='o',markersize=4,color=color,ecolor=color,label=label,linewidth=1,capsize=2)
    ax.axvline(1,color=RED,linestyle='--',linewidth=1);ax.set_xscale('log',base=2)
    ax.set_xticks([.25,.5,1,2,4,8,16,32],['0.25','0.5','1','2','4','8','16','32'])
    ax.set_yticks(positions,k5.label);ax.invert_yaxis();ax.set_xlabel('Speedup over paired Full; greater than 1 favors compression')
    ax.grid(axis='x',alpha=.15);fig.legend(*ax.get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=8)
    speed=SUPPORT/'timing_boundaries.png';fig.savefig(speed,dpi=240);fig.savefig(SUPPORT/'timing_boundaries.pdf');plt.close(fig)
    k5.to_csv(SUPPORT/'timing_chart_values.csv',index=False)

    fontdir=Path('C:/Windows/Fonts')
    for tag,file in [('Report','segoeui.ttf'),('ReportBold','segoeuib.ttf'),('ReportItalic','segoeuii.ttf')]:
        pdfmetrics.registerFont(TTFont(tag,str(fontdir/file)))
    pdfmetrics.registerFontFamily('Report',normal='Report',bold='ReportBold',italic='ReportItalic',boldItalic='ReportBold')
    styles={
        'body':ParagraphStyle('body',fontName='Report',fontSize=9.6,leading=14,textColor=colors.HexColor(NAVY),spaceAfter=8),
        'small':ParagraphStyle('small',fontName='Report',fontSize=8,leading=11,textColor=colors.HexColor(GRAY),spaceAfter=7),
        'h1':ParagraphStyle('h1',fontName='ReportBold',fontSize=20,leading=25,textColor=colors.HexColor(NAVY),spaceAfter=12),
        'h2':ParagraphStyle('h2',fontName='ReportBold',fontSize=12,leading=16,textColor=colors.HexColor(TEAL),spaceAfter=7,spaceBefore=8),
        'title':ParagraphStyle('title',fontName='ReportBold',fontSize=29,leading=35,textColor=colors.HexColor(NAVY),spaceAfter=16),
        'cell':ParagraphStyle('cell',fontName='Report',fontSize=7.7,leading=10.5,textColor=colors.HexColor(NAVY)),
        'head':ParagraphStyle('head',fontName='ReportBold',fontSize=7.8,leading=10.5,textColor=colors.white)}
    story=[];written_tables={}
    def p(text,style='body'):return Paragraph(text,styles[style])
    def body(text):story.append(p(text))
    def heading(text):story.append(p(text,'h2'))
    def page(title):
        if story:story.append(PageBreak())
        story.append(p(title,'h1'))
    def caption(text):story.append(p(text,'small'))
    def table(identifier,title,headers,rows,widths,note=None):
        heading(f'{identifier}. {title}')
        values=[[p(escape(str(v)),'head') for v in headers]]
        values += [[p(str(v),'cell') for v in row] for row in rows]
        obj=Table(values,colWidths=widths,repeatRows=1,hAlign='LEFT')
        obj.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor(NAVY)),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),
            ('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),3.5),
            ('BOTTOMPADDING',(0,0),(-1,-1),3.5),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F0F5F7')]),
            ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor(TEAL)),
            ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#DCE5EB'))]))
        story.append(obj);story.append(Spacer(1,7))
        if note:caption(note)
        written_tables[identifier]=dict(title=title,headers=headers,rows=rows)
    def figure(path,width=WIDTH):
        from PIL import Image as PILImage
        with PILImage.open(path) as image:w,h=image.size
        story.append(Image(str(path),width=width,height=width*h/w));story.append(Spacer(1,5))
    def lookup(frame,row):
        mask=pd.Series(True,index=frame.index)
        for key in ['dataset','variant','k','method','m','q']:mask &= frame[key].eq(row[key])
        found=frame[mask];assert len(found)==1,(row['dataset'],row['method'],len(found))
        return found.iloc[0]

    story.append(Spacer(1,12));story.append(p('Protected Sparse JL<br/>Complete Native Evaluation','title'))
    body('<b>Geometry, compression, downstream quality and computation cost</b><br/>12 October 2026 | Completed balanced experiment grid')
    body('The method is successful at a specific task: retaining the assignments of a learned fixed-center model while using a residual sketch to preserve distances. '
        'It provides a broader set of fidelity-qualified operating points than the tested PCA baselines. It does not establish a universal speed or downstream-quality advantage.')
    table('Table 1','Principal findings',['Question','Measured result'],[
        ['Is the experiment complete?','9 completed jobs; 15,216 validation/test cases; 5 projection draws and 3 centroid starts.'],
        ['Does protected JL meet the joint rule?','16/18 settings qualified on validation; 15/16 selected settings also passed on held-out test data.'],
        ['Does it always compress better than PCA?','No. On MNIST, PCA qualifies at m=128; protected JL needs m=256 for k=5. Fashion-MNIST shows the reverse pattern.'],
        ['Are reduced-space queries faster?','Yes for the selected protected-JL query kernels. Preparation and fitting change the complete cost comparison.'],
        ['Is complete execution faster than Full?','At 32 queries, conditional reuse wins on Gisette. None of the 15 held-out-qualified settings has a cold-speedup CI entirely above one.'],
        ['Does fixed-center agreement guarantee other tasks?','No. Fresh clustering and informative neighbor recall remain weak on some corpora, especially nonzero text.']],
        [160,WIDTH-160])
    heading('Interpretation')
    body('The main contribution is a geometry-aware compression trade-off, supported by optimized Full and PCA controls. '
        'A reusable learned model can make high-dimensional distance workloads cheaper. Building that model and compressing the data can still cost more than directly processing Full once.')
    caption('Scope: native float64 numerical experiments. These measurements do not include witness generation, circuit setup, proof generation or proof verification. '
        'Complete native results do not complete the new hybrid ZK evidence requirements.')

    page('Data and experiment design')
    dataset_rows=[]
    for _,r in ordered(data).iterrows():dataset_rows.append([name(r),str(int(r.n)),str(int(r.d)),
        f'{int(r.train_rows)} / {int(r.validation_rows)} / {int(r.test_rows)}',str(int(r.zero_rows))])
    table('Table 2','Preserved corpora',['Corpus','Rows','Features','Train / validation / test','Zero rows'],dataset_rows,[139,48,57,207,WIDTH-451])
    body('MNIST and Fashion-MNIST each use 5,000 of 70,000 combined source rows. Gisette uses 3,000 of 6,000 training rows; Arcene uses all 100 labeled training rows; '
        '20 Newsgroups uses 3,000 of 11,314 training documents. Synthetic fixtures are preserved in full. The text sensitivity corpus removes exactly 114 zero rows.')
    heading('Controlled comparisons')
    body('The numerical split is fixed, shuffled and disjoint. Centers, the protected basis and both PCA solvers are learned from the training fold. '
        'Validation quality chooses m and q before fresh test maps are evaluated. Both validation and test screens retain all feasible grid cells.')
    body('The grid uses k=5 and 10; m=16, 32, 64, 128, 256 and 512; and Sparse-JL q=1 and 2. '
        'The total m includes the protected head. PCA requires m no greater than the training rank; Arcene adds m=59. m must be smaller than the original d.')
    body('Native timing uses four CPU threads, five measured repetitions and an excluded warmup. Center/PCA fitting uses three measured fits; basis extraction uses five measurements. '
        'Model construction is one observation per generated map. Full and compressed task order is randomized and interleaved.')
    caption('Quality intervals independently resample projection draws and center starts, using 1,000 bootstrap samples. They are exploratory pointwise 95% intervals conditional on these fixed corpora. '
        'Repeated timer calls are not independent seeds. Previously explored datasets and a new split do not form an independent new study population.')
    caption('Images/UCI coordinates are not newly standardized. Archived text is L2-normalized TF-IDF with 2,000 features; its vocabulary predates this numerical split. '
        'This supports representation-compression analysis, not a claim about raw-text generalization.')

    page('What the method preserves')
    body('A learned set of k centers determines a small subspace through their differences. The protected head retains coordinates in that subspace exactly in real arithmetic. '
        'The residual Sparse-JL sketch retains approximate geometry outside it. q is the number of nonzeros per input column of the sparse random map.')
    body('For a fixed center set, nearest-center decisions depend on center differences. Preserving their subspace preserves those decisions. '
        'The head often has k-1 coordinates: at k=5 and total m=128, this is typically 4 protected plus 124 residual coordinates.')
    body('The residual sketch matters because label agreement alone does not preserve distances or the numerical objective. '
        'The protected-only ablation can obtain ARI=1 against Full while losing almost all residual distance energy. It did not qualify under the joint geometry rule.')
    table('Table 3','Predeclared validation rule',['Endpoint','Required confidence-bound condition'],[
        ['Fixed-center assignment ARI','Lower 95% CI bound at least 0.99'],
        ['Training-reconstructed original objective ratio','Upper 95% CI bound at most 1.10'],
        ['Pair squared-distance Q95 relative error','Upper 95% CI bound at most 0.20'],
        ['Point-to-center squared-distance Q95 error','Upper 95% CI bound at most 0.20'],
        ['Projected objective relative error','Upper 95% CI bound at most 0.10'],
        ['Completeness and correctness','Complete crossed grid; no zero-distance violations; defined ratios; no capped training fit']],
        [218,WIDTH-218])
    heading('Reading the scores')
    body('ARI=1 means identical partitions up to label naming. Here the primary ARI reference is Full, not source ground-truth classes. '
        'Q95=0.15 means that the within-case 95th percentile absolute relative error in squared distances is 15%; this is not an all-pairs guarantee.')
    caption('The displayed Q95 mean averages case-level percentiles; it is not a percentile recomputed after pooling all pairs. '
        'A global count of zero certificate violations means observed conditional checks were consistent, not that every pair was certified or every downstream task was successful.')

    page('Fidelity against optimized baselines')
    counts=selection.groupby('method').selected.agg(['sum','count'])
    pass_counts=ops.groupby('method').test_fidelity_rule_pass.sum()
    table('Table 4','Qualification across 18 dataset/k settings',['Method','Validation qualified','Selected test passes'],[
        [METHODS[m],f'{int(counts.loc[m,"sum"])}/18',f'{int(pass_counts.get(m,0))}/{int(counts.loc[m,"sum"])}' if counts.loc[m,'sum'] else 'No selected point']
        for m in METHODS],[205,151,WIDTH-356],
        'Full is an uncompressed self-reference. Text primary/nonzero and k=5/k=10 settings are correlated settings, not 18 independent populations.')
    figure(geometry)
    caption('Figure 1. Held-out pair-distance distortion for k=5. Sparse maps use q=1; the dotted line marks the 0.20 selection threshold. '
        'Only one of five required endpoints is plotted. Bands are pointwise exploratory CIs.')
    body('Protected JL supports more qualifying settings overall, particularly on high-dimensional Gisette and text. '
        'MNIST provides a counterexample to uniform superiority: at m=128, PCA has lower pair distortion (0.118 versus 0.191) and qualifies, '
        'while protected JL needs m=256 for its k=5 selected point. PCA also has better conditional reuse speed on MNIST.')
    caption('PCA is evaluated as an unscaled truncated projection, whose objective concentrates variance rather than guaranteeing JL-style near-isometry. '
        'Rescaled/whitened PCA and dedicated single-m solver fits were not evaluated. The observed rule does not rank every possible PCA formulation.')

    page('Matched-dimension ablation')
    subset=reference[reference.dataset.isin(['mnist','gisette','20ng_tfidf']) & reference.variant.eq('primary')]
    rows=[]
    for dataset in ['mnist','gisette','20ng_tfidf']:
        for method in METHODS:
            r=subset[(subset.dataset==dataset)&(subset.method==method)].iloc[0]
            rows.append([NAMES[dataset],METHODS[method],str(int(r.m)),ci(r,'ari'),ci(r,'pair_q95'),ci(r,'projected_cost_relative_error')])
    table('Table 5','k=5, total m=128 and JL q=1',['Dataset','Method','m','Assignment ARI','Pair Q95 error','Objective error'],rows,[94,110,29,88,94,WIDTH-415],
        'Means with pointwise 95% CIs. Full uses its original d; protected-only uses 4 coordinates. These exceptions are explicit, not equal-m baselines.')
    body('The head protects the chosen fixed-center decisions. The residual sketch substantially improves geometry compared with protected-only. '
        'Ordinary Sparse JL has similar projection machinery but much lower assignment agreement on these examples, which isolates the contribution of protecting center differences.')
    caption('The common m=128 examples are exploratory configurations, not all selected points. Gisette and text require m=256 under the complete selection rule. '
        'Arcene has no PCA m=128 cell because its training rank is at most 59; the DBSCAN fixture has d=128 and cannot be compressed to m=128.')

    page('How much compression is obtained?')
    hjl_select=ordered(selection[selection.method.eq('protected_jl')]);rows=[]
    for _,r in hjl_select.iterrows():
        if r.selected:
            selected=lookup(hjl,r);fraction=100*selected.m/selected.original_d
            rows.append([name(r),str(int(r.k)),str(int(r.m)),str(int(r.q)),num(fraction,4)+'%',ci(selected,'pair_q95'),
                'Pass' if selected.test_fidelity_rule_pass else 'Fail'])
        else:rows.append([name(r),str(int(r.k)),'--','--','--','--','No choice'])
    table('Table 6','Protected-JL selected operating points',['Corpus','k','m','q','Coordinates kept','Test pair Q95','Test rule'],rows,[119,23,31,22,94,141,WIDTH-430],
        'Failed validation selections remain visible. Test rule checks all five CI conditions, not pair distortion alone. Arcene k=10 fails its held-out pair-distance bound.')
    body('For Gisette, m=256 retains 5.12% of 5,000 coordinates: about 19.5 times fewer dense float64 representation bytes. '
        'With 3,000 rows, that is approximately 114.4 MiB for Full versus 5.86 MiB for projected points, excluding model and workspace storage.')
    body('At k=5, text retains 12.8% of coordinates, Fashion-MNIST 16.3%, and MNIST 32.7%. '
        'Arcene retains 1.28% at m=128 but has only 100 total rows, so representation reduction does not itself imply favorable preparation amortization.')
    caption('These are feature-coordinate and dense-array byte ratios. They are not lossless file-compression ratios or isolated process peak-memory reductions. '
        'The learned centers, protected basis, sparse map and worker retain additional memory.')

    page('Training and compression are separate costs')
    train_rows=[];compression_rows=[]
    for _,r in k5.iterrows():
        pool=training[(training.dataset==r.dataset)&(training.variant==r.variant)]
        fits=pool[pool.fit_key.fillna('').str.startswith('k5_')]
        med=lambda stage:fits.loc[fits.stage.eq(stage),'median_seconds'].median()
        pca=pool[pool.stage.eq('pca_training_largest_dimension_once')]
        randomized=pool[pool.stage.eq('randomized_pca_training_largest_dimension_once')]
        train_rows.append([name(r),num(med('centroids')),num(med('basis')),num(pca.median_seconds.median()),
            num(randomized.median_seconds.median()),str(int(pca.shared_pca_fitted_m.iloc[0]))])
        s=lookup(setup,r)
        compression_rows.append([name(r),str(int(r.m)),num(s.model_construction_seconds),num(s.compression_seconds),num(s.point_norm_preparation_seconds)])
    table('Table 7','Training stage medians in seconds (k=5)',['Corpus','Learn centers','Extract basis','Full-SVD PCA','Randomized PCA','PCA fitted m'],train_rows,[127,79,74,86,90,WIDTH-456],
        'Center/basis entries summarize per-start fit medians; randomized PCA summarizes per-draw fit medians. These are shared model-learning costs.')
    table('Table 8','Selected protected-JL setup medians in seconds',['Corpus','m','Map construction','Whole compression','Point norms'],compression_rows,[140,34,115,132,WIDTH-421],
        'Pooled stage medians for selected k=5 configurations. Point+center norm preparation is a different boundary and is not added to point-only preparation.')
    body('All methods use native optimized linear algebra. Full has no projection stage and uses the same optimized distance/assignment kernels. '
        'PCA uses one training fit at the largest feasible sweep dimension and reuses prefixes. Its recorded fit is not a dedicated fit at each smaller selected m.')
    body('Map construction is a concrete implementation target: about 0.235 seconds on Gisette and 0.467 seconds on Arcene, '
        'versus whole-corpus projection of about 0.144 and 0.00975 seconds. Improving this stage can reduce setup cost while retaining the method; '
        'it does not remove the center-learning cost in a cold execution.')
    caption('Randomized PCA is not guaranteed faster when fitting a relatively large rank. A dedicated single-m randomized fit could change cold comparisons. '
        'Saving arrays, hashing inputs and loading the archive are outside these task timers, but contribute to campaign wall time.')

    page('Why a smaller representation can cost more')
    body('A reduced query kernel reads fewer coordinates, so the computation after compression can be much faster. '
        'The complete operation must also learn or retrieve the model, construct the map, transform the corpus and prepare norms. '
        'Full can start computing on the input immediately. A small number of queries may not repay the preparation cost.')
    figure(speed)
    caption('Figure 2. Selected protected-JL k=5 configurations, 32 distinct queries. Points show case-mean speedups with pointwise 95% CIs. '
        'Kernel excludes setup; reuse includes construction/compression/norms with learned model/basis available; cold also includes model fitting. The vertical line is equal cost.')
    g=k5[k5.dataset.eq('gisette')].iloc[0]
    body(f'<b>Gisette example:</b> its reduced query kernel is {num(g.query_only_speedup_mean)} times faster than Full. '
        f'After preparation, 32-query reuse is {num(g.conditional_reuse_speedup_mean)} times faster, or about '
        f'{100*(1-1/g.conditional_reuse_speedup_mean):.1f}% less elapsed composite cost. Cold speedup is only {num(g.cold_speedup_mean)}, '
        f'so the cold compressed pipeline costs about {1/g.cold_speedup_mean:.2f} times as much as Full.')
    body('Across the 15 held-out-qualified protected-JL points, only the two Gisette settings have reuse-speedup CIs entirely above one. '
        'No such point has a cold-speedup CI entirely above one at the measured 32-query budget.')
    caption('This demonstrates a conditional reuse advantage, not universal cold acceleration. Larger query budgets might amortize more setup, but are not established by this 32-query experiment. '
        'Composite costs sum required stage medians; they are not separately measured end-to-end wall-clock totals.')

    page('Selected-point timing with uncertainty')
    rows=[]
    for _,r in hjl.iterrows():rows.append([name(r)+(' *' if not r.test_fidelity_rule_pass else ''),str(int(r.k)),str(int(r.m)),
        ci(r,'query_only_speedup'),ci(r,'conditional_reuse_speedup'),ci(r,'cold_speedup')])
    table('Table 9','Thirty-two distance queries: speedup over paired Full',['Corpus','k','m','Query kernel','Reuse incl. setup','Cold incl. fitting'],rows,[135,25,32,105,109,WIDTH-406],
        'Means [95% CI] of case-level speedup ratios. Values greater than 1 favor the compressed method. The starred Arcene k=10 point failed held-out fidelity; its speed is not a quality-qualified win.')
    body('The Gaussian-hard k=10 reuse mean is slightly above one, but its CI overlaps one. This is inconclusive under the exploratory interval criterion. '
        'The text reuse means at fidelity-selected m=256 are near or below one; faster kernels alone do not demonstrate a complete 32-query saving.')
    body('PCA is a strong control on MNIST: its selected m=128 gives about 1.30 times conditional reuse speedup over Full, while protected JL at k=5 requires m=256 and has reuse speedup about 0.73. '
        'Different selected dimensions are matched-fidelity operating points, not matched-dimension speed tests.')
    caption('Each method has its own interleaved Full control. Ratios between methods that use separate controls do not form a directly paired method-versus-method significance test. '
        'All numeric timings and their raw repetitions are retained in the source bundle.')

    page('Downstream quality is a separate requirement')
    rows=[]
    for _,r in hjl.iterrows():rows.append([name(r)+(' *' if not r.test_fidelity_rule_pass else ''),str(int(r.k)),str(int(r.m)),ci(r,'knn_recall',100),ci(r,'kmeans_ari'),ci(r,'kmeans_refine_ari')])
    table('Table 10','Selected protected-JL downstream quality',['Corpus','k','m','Top-5 recall (%)','Fresh KMeans ARI','Refinement ARI'],rows,[133,23,31,108,109,WIDTH-404],
        'Whole-corpus workloads; means [95% CI] against Full. Top-five recall is the fraction of Full neighbors retrieved after compression. '
        'kNN uses up to 128 query points and excludes self matches. Fresh KMeans starts from common point indices; refinement starts from shared learned centers. '
        'The starred Arcene k=10 setting failed held-out geometry.')
    body('Fixed-center assignment agreement is not a promise about cluster discovery. Centers move during KMeans, and the new differences need not remain inside the original protected subspace. '
        'Refinement often improves agreement, but remains weak on text. Original-objective ratios near one can coexist with very different partitions.')
    page('Why the text-neighbor score needs scrutiny')
    table('Table 11','A fixed original-text neighbor diagnostic',['Quantity','Observed count'],[
        ['Query points',str(zero_info['queries'])],['Zero-vector queries',f'{zero_info["zero_queries"]} / {zero_info["queries"]}'],
        ['Top-five neighbor slots',str(zero_info['neighbor_slots'])],
        ['Slots containing zero vectors',f'{zero_info["zero_neighbors"]} / {zero_info["neighbor_slots"]} ({100*zero_info["zero_neighbor_fraction"]:.1f}%)']],
        [300,WIDTH-300],
        'Direct replay of the original neighbor indices from the saved k=5,m=256,q=1 diagnostic. This is a deterministic property of that query set, not an independent-sample estimate.')
    body(f'The saved original text query set has {zero_info["queries"]} queries and {zero_info["neighbor_slots"]} top-five neighbor slots. '
        f'{zero_info["zero_neighbors"]} slots ({100*zero_info["zero_neighbor_fraction"]:.1f}%) point to zero vectors, although only '
        f'{zero_info["zero_queries"]} queries themselves are zero rows. A unit-normalized document is at distance 1 from a zero row; '
        'many distinct documents are farther apart. Zero rows can therefore dominate the Full neighbor reference.')
    text_original=k5[(k5.dataset=='20ng_tfidf') & (k5.variant=='primary')].iloc[0]
    text_nonzero=k5[(k5.dataset=='20ng_tfidf') & (k5.variant=='nonzero')].iloc[0]
    table('Table 12','Text sensitivity at selected k=5,m=256',['Corpus','Top-five recall (%)','Fresh KMeans ARI','Refinement ARI'],[
        ['Original text',ci(text_original,'knn_recall',100),ci(text_original,'kmeans_ari'),ci(text_original,'kmeans_refine_ari')],
        ['Nonzero text',ci(text_nonzero,'knn_recall',100),ci(text_nonzero,'kmeans_ari'),ci(text_nonzero,'kmeans_refine_ari')]],
        [140,130,120,WIDTH-390])
    body('At the selected k=5,m=256 point, protected-JL recall falls from 94.1% on the original text corpus to 27.5% on the nonzero sensitivity corpus. '
        'The latter changes the corpus, split and maps; it is a diagnostic rather than a paired causal effect. High recall of degenerate neighbors is not semantic retrieval accuracy.')
    body('This is a reference-quality issue, rather than a program error: Full itself is selecting largely uninformative zero rows. '
        'The original corpus remains in the exports for transparency. The nonzero sensitivity should carry more weight when discussing retrieval usefulness.')
    body('Even a modest relative distance error can reorder nearly tied neighbors in high dimensions. Protecting differences between a small set of centers does not protect every neighbor ordering. '
        'Improving this application therefore requires an explicit neighbor-quality objective, with dimensions and runtime validated again.')

    page('DBSCAN: use graph quality and radius sensitivity')
    rows=[]
    for _,meta in ordered(data).iterrows():
        base=wl[(wl.dataset==meta.dataset)&(wl.variant==meta.variant)&(wl.k==5)&(wl.method=='protected_jl')&(wl.m==64)&(wl.q==1)]
        assert len(base)==1
        r=base.iloc[0]
        rows.append([name(r),'Primary',ci(r,'graph_f1'),ci(r,'dbscan_ari'),f'{int(r.informative_dbscan_cases)}/{int(r.cases)}'])
        extra=profiles[(profiles.dataset==meta.dataset)&(profiles.variant==meta.variant)&(profiles.k==5)&(profiles.method=='protected_jl')&
            (profiles.m==64)&(profiles.q==1)&(profiles.profile=='validation_q75')]
        if len(extra)==1:
            r=extra.iloc[0];rows.append([name(r),'Validation Q75',ci(r,'graph_f1'),ci(r,'dbscan_ari'),f'{int(r.informative_dbscan_cases)}/{int(r.cases)}'])
        else:
            assert len(extra)==0
            saved_selection=json.loads((ROOT/'jobs'/f'{meta.dataset}_{meta.variant}'/'selection.json').read_text())
            assert np.isclose(saved_selection['radii']['validation_q75'],saved_selection['radius'],rtol=1e-12,atol=0)
            rows.append([name(meta),'Q75 = primary','See primary','See primary','Not repeated'])
    table('Table 13','Radius diagnostic at common m=64,k=5,q=1',['Corpus','Radius profile','Edge F1','Cluster ARI','Informative reference'],rows,[132,104,103,98,WIDTH-437],
        'Diagnostic cells, not selected operating points. Every graph uses at most 256 fixed corpus rows, with the same radius and min_samples=5 across methods. '
        'Original text has Q75 equal to the primary radius, so it is not rerun. Other quartiles and graph precision/recall are in the full exports.')
    body('Most absent edges in a sparse graph can make all-pairs agreement look high despite poor edge recall. Edge F1, precision/recall, core masks and noise fractions give a more informative assessment. '
        'An all-noise or one-partition reference can also yield a trivial ARI=1.')
    caption('The original synthetic margin radius=1 is retained as an all-noise certificate fixture. It is not evidence of useful cluster recovery. '
        'The run flags 168 uninformative primary DBSCAN cases. Undefined graph ratios remain blank; complete artifact checks do not convert them into successes.')

    page('What the paper can claim')
    table('Table 14','Evidence-supported scope',['Supported statement','Practical limit'],[
        ['Protected coordinates retain fixed-center Full decisions while residual JL improves distance fidelity.','This does not prove fresh-clustering, neighbor-order or ground-truth accuracy preservation.'],
        ['A complete native dimension/q/k study, ablation and conditional certificates are available.','Intervals are conditional and exploratory; Arcene has only 20 held-out test rows.'],
        ['Gisette has a quality-qualified 32-query reuse cost advantage and large representation reduction.','Cold acceleration over Full is not demonstrated at this query budget.'],
        ['Both full-SVD and randomized PCA were evaluated with the same task kernels.','PCA uses shared largest-rank fits; dedicated single-m or rescaled/whitened PCA remains an additional control.'],
        ['Archived proof results remain available under their original scopes.','These native results provide no new hybrid witness/prove/verify or proof-byte measurements.']],
        [248,WIDTH-248])
    heading('Recommended next steps')
    body('First, frame the numerical contribution around fixed-center fidelity, geometry and reusable high-dimensional workloads. '
        'Keep the negative cold-cost and fresh-clustering results in the paper. Avoid a general claim that the method beats PCA or Full.')
    body('Second, if speed is the principal objective, compare dedicated PCA fits at the selected dimensions, evaluate a predeclared larger distinct-query budget, '
        'and measure the intended deployment boundary, including model availability and storage/transfer costs when relevant. '
        'Profile and optimize map construction while preserving its deterministic seed and sampling semantics, then benchmark the complete cost again.')
    body('Third, if fresh clustering or informative neighbor retrieval is the intended application, refine the method and repeat validation on the nonzero corpus. '
        'Increasing residual dimension can change the trade-off; it should not be hidden behind fixed-center ARI=1.')
    body('Finally, align the public data-dependent protected basis with the manuscript statement and threat model, then collect fidelity-matched quantized models '
        'and verified hybrid proof records. Complete controlled native n/d scaling if it remains a paper claim; small implementation checks are not final scaling evidence.')
    caption('The observed zero floating certificate violations are a numerical consistency result. They do not establish a new integer-circuit theorem, cryptographic soundness claim or prover performance improvement.')

    page('Reproducibility and complete source tables')
    table('Table 15','Completion and audit record',['Check','Observed result'],[
        ['Completed grid','9/9 jobs; 15,216 saved cases'],['Native stage timing samples',f'{validation["timing_samples"]:,}'],
        ['Distinct-query timing records',f'{validation["query_samples"]:,}'],['Certificate / zero-distance violations','0'],
        ['Full-control / source-snapshot errors','0 / 0'],['Missing checkpoint artifacts','None reported'],
        ['Capped fresh/refinement KMeans workloads','None in either method or paired Full control'],
        ['Representative numeric replay',f'{replay["checked_cases"]} cases passed; all checkpoint artifact hashes also inspected'],
        ['Largest observed worker-tree RSS','0.773 GiB; shared worker, not isolated method allocation'],
        ['Campaign execution','About 10 hours for the resumed session, including reporting and checkpoint resumption']],
        [240,WIDTH-240])
    body('The preserved source campaign is <b>paper_compression_balanced_audited</b>. This report reads its completed exports and leaves every raw run, map, '
        'split index and model unchanged. The companion report manifest records source-file hashes and the PDF checksum.')
    heading('Complete numeric appendix')
    body('The existing table bundle contains all eleven CSV/LaTeX families: dataset metadata; validation choices; all held-out fidelity cells; '
        'common-m=128 comparisons; compression/preparation; model training; downstream quality; native task runtimes; 32-query reuse; DBSCAN radius sensitivity; '
        'and selected operating points. The PDF selects readable representative tables; it does not discard the full grid.')
    caption('Source families: paper_tables/01_datasets.csv through paper_tables/11_selected_operating_points.csv. '
        'Detailed raw exports and figures are under report/. Replay evidence is report/replay_validation.json. '
        'Native table coverage and outstanding proof evidence are recorded in report/paper_coverage.json.')
    heading('Audit cautions')
    body('A time-limited final job resumed from its saved checkpoints and completed. The worker-tree RSS monitor can miss brief peaks. '
        'Model fit, compression, preparation and task timings are separate. Shared fit costs and stage-sum composites are explicitly labeled. '
        'No timing repetition is counted as an independent geometric draw, and no smoke/pilot row is included in this completed-grid report.')

    def decorate(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#D1DEE6'));canvas.line(42,A4[1]-31,A4[0]-42,A4[1]-31)
        canvas.setFillColor(colors.HexColor(GRAY));canvas.setFont('Report',8)
        canvas.drawString(42,A4[1]-22,'PROTECTED SPARSE JL  |  NATIVE NUMERICAL EVALUATION')
        canvas.drawString(42,24,'Completed balanced study  |  12 October 2026')
        canvas.drawRightString(A4[0]-42,24,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(PDF),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=48,bottomMargin=42,
        title='Protected Sparse JL: Complete Native Evaluation',author='Research study',subject='Geometry, compression, quality and timing')
    doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
    atomic_json(SUPPORT/'report_tables.json',written_tables)
    reader=PdfReader(PDF);text='\n'.join(page.extract_text() or '' for page in reader.pages)
    assert '15,216' in text and '94.7%' in text and 'Table 15' in text and 'No choice' in text
    report_manifest=dict(date='2026-10-12',pdf=str(PDF),pages=len(reader.pages),pdf_sha256=hashlib.sha256(PDF.read_bytes()).hexdigest(),
        source_csv_sha256=source_hashes,validation_sha256=hashlib.sha256((ROOT/'report/validation.json').read_bytes()).hexdigest(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),tables=len(written_tables),
        selection_validation_passes=16,selection_test_passes=15,representative_replay_cases=216,
        zero_neighbor_diagnostic=zero_info,scope='Complete balanced native grid only; no new ZK measurements')
    atomic_json(SUPPORT/'report_manifest.json',report_manifest)
    print(json.dumps(dict(pdf=str(PDF),pages=len(reader.pages),tables=len(written_tables),source_csvs=len(source_hashes))))


if __name__=='__main__':build()
