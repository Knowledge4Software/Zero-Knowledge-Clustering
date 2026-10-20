"""Rebuild manuscript evidence from the completed bound-audit cohort only.

Run from any directory: python code/analysis/revise_paper_results.py
No experiments or server commands are launched. Legacy exports are preserved.
"""
from pathlib import Path
import hashlib
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'code/results/final_paper_review_1792295389'
OUT = ROOT / 'paper_results/bound_revision'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'figures').mkdir(exist_ok=True)
INPUTS = set()


def read(name):
    p = EVIDENCE / name
    INPUTS.add(p)
    return pd.read_csv(p)


def js(name):
    p = EVIDENCE / name
    INPUTS.add(p)
    return json.loads(p.read_text(encoding='utf-8'))


def write_table(name, caption, label, headers, rows, notes='', widths=None):
    """Full-width readable tables; exact source decimals exported separately."""
    spec = widths or ('l' + 'r' * (len(headers) - 1))
    text = [r'\begin{table*}[tp]', r'\caption{' + caption + r'}\label{tab:' + label + '}',
            r'\centering\small', r'\begin{adjustbox}{max width=\textwidth}',
            r'\begin{tabular}{@{}' + spec + r'@{}}', r'\toprule',
            ' & '.join(headers) + r' \\', r'\midrule']
    text += [' & '.join(map(str, row)) + r' \\' for row in rows]
    text += [r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}']
    if notes:
        text += [r'\par\vspace{3pt}\begin{minipage}{0.98\textwidth}\footnotesize ' + notes + r'\end{minipage}']
    text += [r'\end{table*}', '']
    (OUT / (name + '.tex')).write_text('\n'.join(text), encoding='utf-8')


def export(name, d):
    d.to_csv(OUT / (name + '.csv'), index=False)


def fmt(x, digits=3):
    return f'{x:.{digits}f}'


DS = {'gaussian_easy': 'Gaussian-easy', 'gaussian_hard': 'Gaussian-hard',
      'dbscan_margin': 'Margin-synthetic', 'mnist': 'MNIST',
      'fashion_mnist': 'Fashion-MNIST', 'gisette': 'Gisette', 'arcene': 'Arcene',
      '20ng_tfidf_nonzero': '20 Newsgroups'}
METHOD = {'full': 'Full', 'pca': 'PCA-16', 'protected_energy': 'Head + energy',
          'protected_tail': 'CommitJL-16', 'affine': 'Exact affine', 'cached': 'Cached exact'}


def main():
    summary = js('checked_review/validation_summary.json')
    assert summary['audit_checks'] == 22645 and summary['audit_failures'] == 0
    n = read('completion/campaign/native.csv')
    assert len(n) == 7776
    primary = n[n.k.isin([5, 10]) & (n.input_bits == 12) & (n.basis_bits == 12)]
    full = read('followup/campaign/full_native.csv')
    detail = read('checked_review/full_native_by_dataset_k.csv')
    ref = read('checked_review/full_native_reference_comparison.csv')
    capped = primary[((primary.method == 'pca') & (primary.budget == 16)) |
                     (primary.method == 'head_energy') |
                     ((primary.method == 'sparse_jl') & (primary.budget == 16) & (primary.q == 1))]
    capped = capped.groupby(['dataset', 'k', 'start', 'method'])[
        ['ari', 'agreement', 'ari_quantized_full', 'agreement_quantized_full', 'cost_abs_relative_error']].mean()
    capped = capped.groupby(['dataset', 'k', 'method']).mean().reset_index()
    export('capped_accuracy', capped)
    rows = []
    for dataset in DS:
        for k in [5, 10]:
            z = capped[(capped.dataset == dataset) & (capped.k == k)].set_index('method')
            rows.append([DS[dataset], k, fmt(z.loc['sparse_jl', 'ari_quantized_full'], 5),
                         fmt(z.loc['pca', 'ari_quantized_full'], 5),
                         fmt(z.loc['head_energy', 'ari_quantized_full'], 5),
                         fmt(100*z.loc['pca', 'agreement_quantized_full'], 3),
                         fmt(100*z.loc['head_energy', 'agreement_quantized_full'], 3)])
    write_table('capped_accuracy', 'Fixed-center fidelity on eight capped corpora at storage budget 16.',
                'capped', ['Corpus', '$k$', 'Sparse ARI', 'PCA ARI', 'Head/E ARI', 'PCA agree. (\\%)', 'Head/E agree. (\\%)'], rows,
                'Reference: quantized Full. Means over three fitted starts; sparse maps are averaged within a start before aggregation. '
                'Head/E and CommitJL have identical fixed-center outputs. Sparse JL uses one signed bucket per input column. '
                'ARI measures assignment fidelity, not agreement with class labels.')
    rows = []
    for dataset in ['mnist', 'fashion_mnist', '20ng_tfidf_nonzero']:
        for k in [5, 10]:
            z = detail[(detail.dataset == dataset) & (detail.k == k)].set_index('method')
            rows.append([DS[dataset], k, fmt(z.loc['pca', 'ari_quantized_full'], 5),
                         fmt(z.loc['protected_tail', 'ari_quantized_full'], 5),
                         fmt(100*z.loc['pca', 'agreement_quantized_full'], 3),
                         fmt(100*z.loc['protected_tail', 'agreement_quantized_full'], 3),
                         fmt(100*z.loc['protected_tail', 'agreement'], 3)])
    write_table('full_accuracy', 'Native fidelity on all cached image rows and nonzero cached text-training rows.',
                'full-accuracy', ['Corpus', '$k$', 'PCA ARI$_Q$', 'Ours ARI$_Q$', 'PCA agree.$_Q$ (\\%)', 'Ours agree.$_Q$ (\\%)', 'Ours agree.$_F$ (\\%)'], rows,
                'Subscripts $Q$ and $F$ denote quantized-Full and floating-Full references. Five fitted starts per corpus/$k$; '
                '60/40 fitting/audit split. MNIST and Fashion-MNIST: 70,000 rows each (28,000 audit rows). '
                'Text: 10,944 nonzero training documents, 2,000 TF--IDF features (4,378 audit rows). '
                'These are native runs, not whole-corpus proofs. Ours denotes both head/energy variants.')
    rows = [[METHOD[v.method], fmt(v.ari, 6), fmt(100*v.agreement, 4),
             fmt(v.ari_quantized_full, 6), fmt(100*v.agreement_quantized_full, 4)] for v in ref.itertuples()]
    write_table('reference', 'Effect of reference precision in the full-cached native study.', 'references',
                ['Method', 'ARI vs. floating', 'Agree. vs. floating (\\%)', 'ARI vs. quantized', 'Agree. vs. quantized (\\%)'], rows,
                'Equal-weight means over 30 corpus/$k$/start cells. Quantized Full is not identical to floating Full. '
                'Exact affine is exact relative to integer Full; cached exact answers have this same algebraic property.')
    datasets = js('main/datasets.json')
    rows = []
    catalog = {'gaussian_hard': '32 ($k=5,10$)', 'mnist': '16; 64 ($k=5$)',
               'fashion_mnist': '16 ($k=5$)', '20ng_tfidf_nonzero': '16 ($k=5$)', 'gisette': '8 ($k=5$)'}
    for d in datasets:
        name = d['dataset']
        audit = int(np.ceil(.4*d['n']))
        rows.append([DS[name], f"{d['n']:,}", f"{d['d']:,}", f'{audit:,}', catalog.get(name, '--')])
    write_table('datasets', 'Capped native corpora and actual fixed-center proof catalogs.', 'datasets',
                ['Corpus', 'Native rows', 'Features $d$', 'Audit rows/start', 'Proof catalog $N$'], rows,
                'Each capped corpus uses a 60/40 fit/audit split and $k=5,10$ with three starts. '
                'Image subsets are sampled by the loader. Text starts with at most 3,000 training documents; zero vectors are removed, leaving 2,913. '
                'Gisette uses a 3,000-row cap; Arcene uses 100 cached training rows. '
                'Margin-synthetic is used here only for fixed-center and geometry tests, not a DBSCAN proof.', widths='lrrrl')
    proof = read('corrected_handoff/main_actual_queries.csv')
    instance_quality=read('main/analysis/proof_instance_quality.csv')
    instance_quality=instance_quality[(instance_quality.q>=8)&instance_quality.method.isin(['pca','protected_energy','protected_tail','affine'])]
    assert instance_quality.groupby('method').size().eq(13).all()
    export('proved_instance_quality',instance_quality)
    export('query_verification',proof[['case','q','verify_median_seconds','proof_bytes_median','independent_bundle_verified']])
    extra = read('completion/full_budget_control/full_q8_controls.csv')
    rows = []
    for dataset, k in [('gaussian_hard',5), ('gaussian_hard',10), ('mnist',5), ('fashion_mnist',5), ('20ng_tfidf_nonzero',5), ('gisette',5)]:
        z = proof[(proof.dataset == dataset) & (proof.k == k) & (proof.q == 8)].set_index('method')
        ft = fmt(z.loc['full', 'prove_median_seconds']) if 'full' in z.index else '--'
        if dataset in set(extra.dataset):
            ft = fmt(extra.set_index('dataset').loc[dataset, 'prove_median_seconds']) + r'$^{\dagger}$'
        rows.append([DS[dataset], k, ft] + [fmt(z.loc[m, 'prove_median_seconds']) for m in ['pca','protected_energy','protected_tail','affine']])
    write_table('query8', 'Actual eight-record fixed-center query proving time (seconds).', 'query8',
                ['Corpus', '$k$', 'Full', 'PCA-16', 'Head + energy', 'CommitJL-16', 'Exact affine'], rows,
                'Compilation, setup and Stage 1 excluded. Three measured repetitions after one warmup, summarized across distinct subsets. '
                'Main $q=8$ runs have eight subsets when $N>8$; Gisette has one full-catalog subset. '
                r'$^{\dagger}$Additional Full controls use two subsets at a 1.5M-constraint ceiling. '
                'Dashes are unmeasured resource-budget exclusions, not observed out-of-memory failures.')
    large = read('checked_review/mnist_64point_query.csv')
    rows = [[METHOD[v.variant], f'{v.constraints:,}', fmt(v.prove_median_seconds)] for v in large.itertuples()]
    write_table('query64', 'Actual whole-catalog query on 64 committed MNIST records, $k=5$.', 'query64',
                ['Method', 'Constraints', 'Proving time (s)'], rows,
                'Same original catalog and centers for all methods; each saved proof independently verifies. '
                'This is one possible 64-record subset, with three timing repetitions after warmup. '
                'Full was measured only at $q=1$ for this catalog; no large-shape Full estimate is inserted. '
                'The one-constraint head/E difference comes from compiled context details, not useful tail computation.')
    cp = read('completion/campaign/proofs.csv')
    stage = cp[cp.large & (cp.phase == 'stage1') & (cp.status == 'complete')].copy()
    workload = read('checked_review/mnist_64catalog_four_distinct_queries.csv')
    rows=[]
    for m in ['pca','protected_energy','protected_tail','affine','cached']:
        a=stage[stage.variant==m].iloc[0];w=workload[workload.variant==m].iloc[0]
        rows.append([METHOD[m],f'{int(a.constraints):,}',fmt(a.prove_median_seconds),fmt(w.stage1_prove_seconds),
                     fmt(w.stage1_compile_setup_seconds),fmt(w.modeled_fresh_workload_seconds)])
    write_table('preprocessing', 'Preprocessing and reconstructed fresh-workload cost, MNIST $N=64$, $k=5$.', 'preprocessing',
                ['Method',r'\shortstack{Stage-1\\constraints/row}',r'\shortstack{Median row\\proof (s)}',
                 r'\shortstack{All row\\proofs (s)}',r'\shortstack{Stage-1\\compile/setup (s)}',r'\shortstack{Four-query\\workload (s)}'],rows,
                'The workload uses four genuinely distinct 32-record subsets. It adds native Go preparation, all 64 Stage-1 witness/proof costs, '
                'Stage-1 and query compilation/setup, and one witness/proof per subset. Public-model fitting, verification, warmups and I/O are excluded. '
                'These sums reconstruct measured components; they are not observed elapsed times or an empirical break-even threshold.')
    ab=read('checked_review/component_attribution.csv')
    rows=[]
    names=['Head only','Head + energy','Head + JL (no energy)','CommitJL-16','PCA-16']
    sizes=[r'$h$',r'$h+1$',r'$15$',r'$16$',r'$16$']
    for name,size,v in zip(names,sizes,ab.itertuples()):
        rows.append([name,size,fmt(v.ari_vs_float,5),fmt(v.agreement_vs_float_percent,3),fmt(v.mean_absolute_cost_error_percent,6)])
    write_table('ablation','Component attribution on the 48 capped corpus/$k$/start cells.', 'ablation',
                ['Representation','Statistics/row','ARI vs. floating','Agreement (\\%)','Mean absolute cost error (\\%)'],rows,
                'Maps are first averaged within a start. Cost error is the absolute relative error of each cell\'s aggregate nearest-center cost, '
                'then averaged; it is not a signed bias or a per-point mean. The no-energy control keeps the same head and tail as CommitJL-16, '
                'so it stores 15 statistics. Its tail norm approximates residual energy; it is not added to an exact energy term. '
                'Equal-budget no-energy controls are also retained in the artifact. PCA error concerns use of projected cost as an original-cost proxy.')
    geometry=read('checked_review/geometry_matched_cells.csv')
    rows=[[v.budget,METHOD[v.method].replace('PCA-16','PCA').replace('CommitJL-16','CommitJL'),v.matched_cells,
           fmt(v.mean_absolute_squared_distance_error_percent,3),fmt(v.recall10_percent,3)] for v in geometry.itertuples()]
    write_table('geometry','Matched native geometry quality at equal total statistic budgets.', 'geometry',
                ['Budget $m$','Method','Matched cells','Mean abs. squared-distance error (\\%)','Recall@10 (\\%)'],rows,
                'Eight capped corpora, $k=5,10$, three starts, ten maps per start/budget for the random tail. '
                'Cells are matched separately at each budget. Arcene is excluded from both methods at 64 because PCA exceeds the fit rank. '
                'Pairs are sampled; nearest neighbors use up to 128 held-out queries. Energy is excluded from Euclidean coordinates; '
                'CommitJL has $m-1$ such coordinates and one energy statistic. Native maps differ from the post-commit maps in proof bundles.')
    gp=read('followup/campaign/geometry_proofs.csv')
    rows=[]
    for dataset,k in [('gaussian_hard',5),('gaussian_hard',10),('mnist',5),('20ng_tfidf_nonzero',5)]:
        z=gp[(gp.dataset==dataset)&(gp.k==k)&(gp.budget==16)].set_index('method')
        rows.append([DS[dataset],k,int(z.loc['pca','constraints']),int(z.loc['protected_tail','constraints']),
                     fmt(z.loc['pca','prove_median_seconds']),fmt(z.loc['protected_tail','prove_median_seconds'])])
    write_table('pair_proofs','Actual authenticated pair-distance proofs, storage budget 16.', 'pair-proofs',
                ['Corpus','$k$','PCA constraints','Ours constraints','PCA prove (s)','Ours prove (s)'],rows,
                'Eight distinct pairs, three repetitions after warmup. All saved proofs verify. '
                'The circuit proves distance in the stored embedding, not its accuracy relative to the original space. '
                'Ours authenticates head and tail through separate roots; PCA uses one representation root. '
                'The complete 32-configuration study also includes budgets 32/64 and Full/head controls.')
    precision=read('checked_review/precision_all_tested_center_counts.csv')
    rows=[[int(v.input_bits),int(v.basis_bits),fmt(100*v.agreement,4),fmt(100*v.cost_abs_relative_error,6)] for v in precision.itertuples()]
    write_table('precision','Separate input and basis precision sensitivity for the protected representation.', 'precision',
                ['Input scale bits','Basis scale bits','Agreement vs. floating (\\%)','Mean absolute cost error (\\%)'],rows,
                'Means over all tested native center-count cells: the 48 primary cells and 18 additional Gaussian-hard/MNIST cells '
                'at $k=2,20,50$. These native sweeps do not benchmark proof time or establish a minimum safe circuit bit width.')
    sparse=read('checked_review/sparsity_separate_settings.csv')
    rows=[[int(v.budget),int(v.q),fmt(v.ari,5),fmt(100*v.agreement,3),fmt(100*v.cost_abs_relative_error,3)] for v in sparse.itertuples()]
    write_table('sparsity','Plain sparse-map density sensitivity, kept separate by nonzeros per input column.', 'sparsity',
                ['Budget','Nonzeros/column $s$','ARI vs. floating','Agreement (\\%)','Mean absolute cost error (\\%)'],rows,
                'Three starts and five maps per setting on the eight capped corpora. Increasing density does not consistently improve fidelity. '
                'This is a native sparse-JL control, not a multi-sparsity proof-performance experiment.')
    controls=primary[((primary.method.isin(['pca','rademacher'])) & primary.budget.isin([16,32,64])) |
                     ((primary.method=='sparse_jl') & (primary.q==1) & primary.budget.isin([16,32,64]))].copy()
    # PCA-64 is undefined for Arcene; compare every method on common cells.
    controls=controls[~((controls.dataset=='arcene') & (controls.budget==64))]
    controls=controls.groupby(['dataset','k','start','method','budget'])[['ari','agreement','cost_abs_relative_error']].mean().reset_index()
    controlmeans=controls.groupby(['budget','method'])[['ari','agreement','cost_abs_relative_error']].mean().reset_index()
    rows=[[int(v.budget),{'pca':'PCA','rademacher':'Rademacher','sparse_jl':'Sparse JL'}[v.method],
           fmt(v.ari,5),fmt(100*v.agreement,3),fmt(100*v.cost_abs_relative_error,3)] for v in controlmeans.itertuples()]
    write_table('random_controls','Matched native random-map baselines at three budgets.', 'random-controls',
                ['Budget','Method','ARI vs. floating','Agreement (\\%)','Mean absolute cost error (\\%)'],rows,
                '48 matched corpus/$k$/start cells at 16 and 32; 42 at 64, excluding Arcene from all methods. '
                'One PCA fit or five maps per start/setting. Sparse JL uses one nonzero per input column. '
                'Rademacher uses normalized signed dense coefficients rounded at basis scale 4096.')
    sensitivity=n[(n.k.isin([2,20,50]))&(n.method=='head_energy')].groupby(['dataset','k'])[
        ['ari_quantized_full','agreement_quantized_full','stored_columns']].mean().reset_index()
    rows=[[DS[v.dataset],int(v.k),int(v.stored_columns),fmt(v.ari_quantized_full,5),fmt(100*v.agreement_quantized_full,3)]
          for v in sensitivity.itertuples()]
    write_table('center_counts','Native protected-head sensitivity beyond five and ten centers.', 'centers',
                ['Corpus','$k$','Head/E statistics','ARI vs. quantized Full','Agreement (\\%)'],rows,
                'Three starts per cell at input/basis scale bits 12. The head rank determines the minimum budget. '
                'At $k=50$, rank 49 plus energy requires 50 values; a 16- or 32-value head/tail representation is infeasible. '
                'No high-$k$ proof-time claim is inferred from these native checks.')
    variability=full[full.method.isin(['pca','protected_tail'])].groupby(['dataset','k','method'])['ari_quantized_full'].agg(['mean','std']).reset_index()
    rows=[]
    for dataset in ['mnist','fashion_mnist','20ng_tfidf_nonzero']:
        for k in [5,10]:
            z=variability[(variability.dataset==dataset)&(variability.k==k)].set_index('method')
            rows.append([DS[dataset],k]+[f"${z.loc[m,'mean']:.5f} \\pm {z.loc[m,'std']:.5f}$" for m in ['pca','protected_tail']])
    write_table('variability','Variation across fitted starts in the full-cached native study.', 'variability',
                ['Corpus','$k$','PCA ARI$_Q$ (mean $\\pm$ SD)','CommitJL ARI$_Q$ (mean $\\pm$ SD)'],rows,
                'Five starts on each fixed corpus/$k$; these are descriptive standard deviations, not population confidence bounds. '
                'For random maps, a map seed is not an independent dataset.')
    # Complete, machine-readable source exports accompany all condensed tables.
    export('full_accuracy',detail);export('reference',ref);export('query8_source',proof[proof.q==8]);export('query64',large)
    export('preprocessing_source',stage);export('workload',workload);export('ablation',ab);export('geometry',geometry)
    export('pair_proofs',gp);export('precision',precision);export('sparsity',sparse);export('full_controls',extra)
    export('random_controls',controlmeans);export('center_counts',sensitivity);export('variability',variability)
    native=full.groupby(['dataset','k','method'])[['model_fit_seconds','compression_seconds','downstream_seconds']].median().reset_index()
    export('native_times',native)
    rows=[[DS[v.dataset],v.k,METHOD[v.method],fmt(v.model_fit_seconds,4),fmt(v.compression_seconds,4),fmt(v.downstream_seconds,6)]
          for v in native.itertuples()]
    write_table('native_times','Native model construction, statistic formation and subsequent answering on full cached datasets.', 'native-times',
                ['Corpus','$k$','Method','Model fit (s)','Form statistics (s)','Answer (s)'],rows,
                'Medians across five starts; answers cover the audit split. Common center fitting is excluded and separately logged. '
                'The native statistic pipeline uses conservative exact-integer arithmetic and is not a tuned BLAS speed benchmark. '
                'Full\'s formation time includes the native reference wrapper; the actual ZK Full path uses the original commitment directly. '
                'Native formation must not be added to Go preparation as if both were consecutive steps in one pipeline.',widths='lr l rrr')
    # Same-start paired uncertainty; small descriptive intervals are not significance claims.
    pairs=full.pivot(index=['dataset','k','start'],columns='method',values='agreement_quantized_full')
    delta=pairs['protected_tail']-pairs['pca'];rng=np.random.default_rng(20261018)
    boot=np.array([rng.choice(delta.to_numpy(),len(delta),replace=True).mean() for _ in range(10000)])
    stats={'paired_mean_agreement_gain_percentage_points':100*delta.mean(),
           'exploratory_fixed_corpus_bootstrap_95_percent':(100*np.quantile(boot,[.025,.975])).tolist(),
           'note':'Resamples fitted corpus/k/start cells; descriptive for these fixed corpora, not a population or multiplicity-adjusted test.'}
    (OUT/'paired_uncertainty.json').write_text(json.dumps(stats,indent=2))
    figures(detail,large,workload)
    manifest={'cohort':'bound-audit-v1; 17-18 October 2026','generator':str(Path(__file__).relative_to(ROOT)),
              'validation':summary,'legacy_MiMC_used':False,'inputs':{},'outputs':{}}
    INPUTS.add(Path(__file__))
    for p in sorted(INPUTS):manifest['inputs'][str(p.relative_to(ROOT)).replace('\\','/')]=hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name!='manifest.json':manifest['outputs'][str(p.relative_to(OUT)).replace('\\','/')]=hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'tables':len(list(OUT.glob('*.tex'))),'figures':len(list((OUT/'figures').glob('*.pdf'))),'paired_uncertainty':stats},indent=2))


def figures(detail,large,workload):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    blue='#285c9b';teal='#168274';gray='#627282'
    def save(fig,name):
        fig.savefig(OUT/'figures'/f'{name}.pdf',bbox_inches='tight')
        fig.savefig(OUT/'figures'/f'{name}.png',bbox_inches='tight',dpi=180)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.1,2.3))
    order=[('mnist',5),('mnist',10),('fashion_mnist',5),('fashion_mnist',10),('20ng_tfidf_nonzero',5),('20ng_tfidf_nonzero',10)]
    for m,color,marker,label in [('pca',blue,'o','PCA-16'),('protected_tail',teal,'s','CommitJL-16')]:
        vals=[detail[(detail.dataset==d)&(detail.k==k)&(detail.method==m)].iloc[0].ari_quantized_full for d,k in order]
        ax.plot(range(6),vals,marker=marker,color=color,label=label,lw=1.6)
    ax.set_xticks(range(6),[f'{DS[d]}\n$k={k}$' for d,k in order]);ax.set_ylabel('ARI vs. quantized Full');ax.set_ylim(.79,1.015)
    ax.grid(axis='y',alpha=.2);ax.legend(loc='lower left',frameon=False);save(fig,'full_accuracy')
    fig,axs=plt.subplots(1,2,figsize=(7.1,2.8),gridspec_kw={'wspace':.42})
    order=['pca','protected_energy','protected_tail','affine','cached'];labels=['PCA','Head/E','CommitJL','Affine','Cached']
    a=large.set_index('variant').loc[order];w=workload.set_index('variant').loc[order]
    axs[0].bar(range(5),a.prove_median_seconds,color=[blue,teal,teal,gray,gray]);axs[0].set_ylabel('Query proof time (s)');axs[0].set_title('(a) One 64-record query')
    axs[1].bar(range(5),w.modeled_fresh_workload_seconds,color=[blue,teal,teal,gray,gray]);axs[1].set_ylabel('Reconstructed cost (s)');axs[1].set_title('(b) Four distinct 32-record queries')
    for ax in axs:ax.set_xticks(range(5),labels,rotation=25,ha='right');ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
    save(fig,'proof_costs')
    geo=read('corrected_handoff/geometry_plot_common_cases.csv')
    fig,axs=plt.subplots(1,2,figsize=(7.1,2.65),gridspec_kw={'wspace':.38})
    for m,color,marker,label in [('pca',blue,'o','PCA'),('protected_tail',teal,'s','CommitJL')]:
        z=geo[geo.method==m].groupby('budget')[['pair_mean_abs_relative_error','recall10']].mean()
        axs[0].plot(z.index,100*z.pair_mean_abs_relative_error,marker=marker,color=color,label=label)
        axs[1].plot(z.index,100*z.recall10,marker=marker,color=color,label=label)
    axs[0].set_ylabel('Mean abs. squared-distance\nerror (%)');axs[1].set_ylabel('Recall@10 (%)')
    for ax in axs:ax.set_xlabel('Total stored statistics per row');ax.set_xticks([16,32,64]);ax.grid(alpha=.17);ax.legend(frameon=False)
    save(fig,'geometry')
    # Accurate architecture: original root is authenticated in Stage 1; tail is optional.
    fig,ax=plt.subplots(figsize=(7.1,3.1));ax.set_xlim(0,10);ax.set_ylim(0,4.3);ax.axis('off')
    boxes=[(.05,2.7,2.6,1.1,'Owner commitment\nQuantized private rows\nTrusted root $C_X$'),
           (3.05,2.7,3.05,1.1,'Stage 1: every catalog row\nProve source membership\nand deterministic statistics'),
           (6.65,2.7,3.25,1.1,'Authenticated representations\nHead + energy: root $C_H$\nOptional JL tail: root $C_T$'),
           (3.05,.45,3.05,1.3,'Stage 2: fixed-center query\nPublic distinct subset indices\nPrivate labels; public cost'),
           (6.65,.45,3.25,1.3,'Stage 2: optional pair query\nOpen head and tail of two rows\nPublic embedding distance')]
    for x,y,w,h,label in boxes:
        patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025',linewidth=.8,edgecolor=gray,facecolor='#f2f6fa')
        ax.add_patch(patch);ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=8)
    for start,end in [((2.67,3.25),(3.0,3.25)),((6.12,3.25),(6.6,3.25)),((7.1,2.65),(5.3,1.8)),((8.4,2.65),(8.4,1.8))]:
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=11,color=gray))
    ax.text(.08,1.1,'After $C_X$:\napprove public model;\nsample and pin tail map.\nRequire all Stage-1 rows.',fontsize=8,color=gray,va='center')
    save(fig,'protocol')


if __name__ == '__main__':
    main()
