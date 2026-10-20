"""Consolidate completed experiments; never turn partial cells into successes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def md(frame):
    if frame.empty:return '_No completed evidence._'
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |',
        '| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(f'{v:.5g}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |'
         for row in frame.itertuples(index=False,name=None)])


def read_csv(path):return pd.read_csv(path) if path.exists() and path.stat().st_size>2 else pd.DataFrame()


def bootstrap_starts(frame, metrics, groups):
    rows=[];rng=np.random.default_rng(7021)
    for key,g in frame.groupby(groups,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        starts=g.groupby('start')[metrics].mean()
        for metric in metrics:
            v=starts[metric].dropna().to_numpy()
            if not len(v):continue
            boot=rng.choice(v,size=(2000,len(v)),replace=True).mean(1)
            rows.append(dict(zip(groups,key),metric=metric,mean=float(v.mean()),std=float(v.std(ddof=1)) if len(v)>1 else None,
                ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),starts=len(v),
                note='Exploratory percentile bootstrap over fitted starts within a fixed corpus; not a cross-dataset or multiplicity-adjusted interval.'))
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--main',type=Path,required=True);p.add_argument('--followup',type=Path,required=True)
    p.add_argument('--supplement',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    statuses={name:json.loads(path.read_text()) for name,path in [
        ('main',a.main/'status.json'),('followup',a.followup/'campaign/status.json'),('supplement',a.supplement/'status.json')]}
    (a.out/'coverage_status.json').write_text(json.dumps(statuses,indent=2))
    full=read_csv(a.followup/'campaign/full_native.csv');geo=read_csv(a.followup/'campaign/geometry_native.csv')
    native=read_csv(a.supplement/'native.csv');proof=read_csv(a.supplement/'proofs.csv')
    mainproof=read_csv(a.main/'analysis/proof_summary.csv');pipe=read_csv(a.main/'analysis/pipeline_summary.csv')
    sections=['# Paper evidence and writing handoff', '',
        'This report summarizes retained evidence and explicit limits. Completion is not journal acceptance or a novelty proof.',
        '', '## Campaign coverage', '',md(pd.DataFrame([dict(campaign=n,status=s.get('status'),complete=s.get('complete',s.get('status')=='complete'),
            failures=len(s.get('failures',[])),exclusions=len(s.get('exclusions',[]))) for n,s in statuses.items()])),
        '', '## Fixed-center full-cached native results', '',
        'All available cached MNIST/Fashion rows and nonzero 20 Newsgroups training rows, 60/40 split, k=5/10, five starts. This is not a whole-corpus cryptographic proof.']
    if not full.empty:
        ci=bootstrap_starts(full,['ari','agreement','ari_quantized_full','agreement_quantized_full'],['dataset','k','method'])
        ci.to_csv(a.out/'full_native_intervals.csv',index=False)
        tab=full.groupby(['dataset','k','method'])[['ari','agreement','ari_quantized_full','agreement_quantized_full','model_fit_seconds','compression_seconds','downstream_seconds']].mean().reset_index()
        tab.to_csv(a.out/'full_native_table.csv',index=False);sections+=['',md(tab)]
    sections+=['','## Component attribution, precision and center-count sensitivity','',
        'Head protects the fixed-center decision, energy restores cost; the tail addresses a separate geometric workload. Compare fixed-component rows separately from equal-storage rows. Energy is not a Euclidean coordinate.']
    if not native.empty:
        metrics=['ari','agreement','r_phi','cost_relative_error','cost_abs_relative_error','compression_seconds','downstream_seconds']
        group=['dataset','k','method','budget','q','policy','input_bits','basis_bits']
        tab=native.groupby(group,dropna=False)[metrics].mean().reset_index();tab.to_csv(a.out/'component_precision_tables.csv',index=False)
        bootstrap_starts(native,metrics,group).to_csv(a.out/'component_intervals.csv',index=False)
        ordinary=native[(native.input_bits==12)&(native.basis_bits==12)&(native.k.isin([5,10]))&(native.budget<=64)]
        # Keep sparsity and error magnitude visible: signed averaging can cancel
        # over/underestimation, and different q settings are distinct experiments.
        display=ordinary.groupby(['method','budget','q','policy'])[['ari','agreement','cost_abs_relative_error']].mean().reset_index()
        sections+=['',md(display)]
        for metric,filename in [('agreement','ablation_assignment'),('cost_abs_relative_error','ablation_cost')]:
            data=ordinary[ordinary.budget.isin([16])|ordinary.method.isin(['head_only','head_energy'])]
            data=data[(data.q.isin([0,1])) & ((data.method!='head_jl') | (data.policy=='fixed_components'))]
            data=data.groupby('method')[metric].mean().sort_values()
            fig,ax=plt.subplots(figsize=(8,4));data.plot.barh(ax=ax);ax.set_xlabel(metric);fig.tight_layout()
            fig.savefig(a.out/(filename+'.png'),dpi=180);fig.savefig(a.out/(filename+'.pdf'));plt.close(fig)
    sections+=['','## Actual proof costs and reuse','',
        'Compile/setup and all Stage-1 row proofs must be counted. Different subsets are distinct workloads; repeated identical queries can cache a previous proof for every method. Affine is an exact quantized fixed-center baseline.']
    if not mainproof.empty:
        q=mainproof[(mainproof.phase=='query')&mainproof.complete]
        q.to_csv(a.out/'main_actual_queries.csv',index=False)
        sections+=['',md(q[q.q==8][['dataset','k','method','constraints','prove_median_seconds']])]
    if not pipe.empty:pipe.to_csv(a.out/'complete_workload_components.csv',index=False)
    if not proof.empty:
        proof.to_csv(a.out/'supplement_proof_phases.csv',index=False)
        q=proof[(proof.phase.str.startswith('query_'))&(proof.status=='complete')]
        sections+=['',md(q[['dataset','k','variant','n_catalog','phase','constraints','prove_median_seconds','public_export_verified']])]
    sections+=['','## JL utility and its price','',
        'The geometry proof proves the stored embedding distance, not its accuracy in original space. Pair-distance distortion and nearest-neighbor recall measure different properties.']
    if not geo.empty:
        g=geo.groupby(['dataset','k','start','method','budget'])[['recall10','pair_mean_abs_relative_error']].mean().reset_index()
        g.to_csv(a.out/'geometry_by_start.csv',index=False)
        g.groupby(['method','budget'])[['recall10','pair_mean_abs_relative_error']].mean().to_csv(a.out/'geometry_unpaired_summary.csv')
        paired=[]
        for budget in [16,32,64]:
            z=g[(g.budget==budget)&g.method.isin(['pca','protected_tail'])]
            z=z.set_index(['dataset','k','start','method'])[['recall10','pair_mean_abs_relative_error']].unstack('method').dropna()
            for method in ['pca','protected_tail']:
                paired.append(dict(method=method,budget=budget,matched_cells=len(z),
                    recall10=z['recall10'][method].mean(),pair_mean_abs_relative_error=z['pair_mean_abs_relative_error'][method].mean()))
        paired=pd.DataFrame(paired);paired.to_csv(a.out/'geometry_summary.csv',index=False)
        sections+=['','PCA/tail comparisons use matching dataset/k/start cells. PCA-64 exceeds Arcene training rank, so Arcene is excluded from both sides at width 64.',md(paired)]
        # The trend plot must ALSO keep the same cases across widths.
        common=g[g.method.isin(['pca','protected_tail'])].pivot_table(
            index=['dataset','k','start'],columns=['budget','method'],values='recall10').dropna().index
        plot_data=g.set_index(['dataset','k','start']).loc[common].reset_index()
        plot_data=plot_data[plot_data.method.isin(['pca','protected_tail'])]
        plot_data.groupby(['method','budget'])[['recall10','pair_mean_abs_relative_error']].mean().to_csv(a.out/'geometry_plot_common_cases.csv')
        sections+=['The geometry trend figure uses the same complete cases across all three widths; its averages can therefore differ from the per-width coverage table.']
        fig,ax=plt.subplots(figsize=(6,4))
        for name,part in plot_data.groupby('method'):
            z=part.groupby('budget')[['recall10','pair_mean_abs_relative_error']].mean()
            ax.plot(z.pair_mean_abs_relative_error,z.recall10,'o-',label=name)
        ax.set_xlabel('Mean absolute squared-distance relative error');ax.set_ylabel('Neighbor recall@10');ax.legend(fontsize=8);fig.tight_layout()
        fig.savefig(a.out/'geometry_tradeoff.png',dpi=180);fig.savefig(a.out/'geometry_tradeoff.pdf');plt.close(fig)
    sections+=['','## Claims supported and claims to avoid','',
        '- Supported: measured small-catalog original-data binding and authenticated subset queries, fixed-center fidelity, component attribution, and disclosed actual proof costs.',
        '- Not established: general dominance over PCA or exact affine; whole-corpus proofs; a distribution-specific high-probability guarantee from q=1 measurements alone; differential privacy; production deployment security.',
        '- Public outputs reveal centers/model, metadata, selected indices and costs/distances. Private experimental arrays and assignment logs must never be distributed as a public proof bundle.',
        '- Public exports are independently verified without private witnesses; this is engineering evidence, not a formal cryptographic audit.',
        '- The existing manuscript describes an older MiMC query-only cohort and must be rewritten to accurately distinguish the new Poseidon2/fixed-point evidence.',
        '- For a Q1 submission, position novelty against exact affine statistics, cached exact fixed-task answers, and verified prior ZK-PCA work. More runs cannot substitute for a distinct contribution.',
        '', '## Proposed research questions','',
        'RQ1: Fixed-center fidelity at equal storage and bit precision. RQ2: Complete ZK cost and genuinely varying-query reuse. RQ3: Original-data binding, trust and disclosure. RQ4: Added geometric utility and proof cost of JL. RQ5: Component ablation, sparsity, precision and center-count sensitivity.', '']
    (a.out/'report.md').write_text('\n'.join(sections),encoding='utf-8')


if __name__=='__main__':main()
