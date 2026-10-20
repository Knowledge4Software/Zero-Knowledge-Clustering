"""Offline exports of measured geometry, costs, and bounded circuit controls."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def table(frame):
    def cell(v):return f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v).replace('|','\\|')
    return '\n'.join(['| '+' | '.join(map(cell,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(map(cell,row))+' |' for row in frame.itertuples(index=False,name=None)])


def report(root):
    root=Path(root);records=[]
    manifest=json.loads((root/'manifest.json').read_text())
    for path in sorted((root/'jobs').glob('*/rows/*.json')):
        row=json.loads(path.read_text());row['job_id']=path.parents[1].name;row['record_path']=str(path.relative_to(root));records.append(row)
    data=pd.DataFrame(records)
    if data.empty:
        (root/'report.md').write_text('# Laptop comparison\n\nNo completed measurements yet.\n',encoding='utf-8');return
    data.to_csv(root/'all_results.csv',index=False)
    geometry=data[data.job_id.str.startswith('geometry_')].copy()
    precision=data[data.job_id.str.startswith('precision_')].copy()
    circuits=data[data.job_id.str.startswith(('growth_','proof_','profile','real_'))].copy()
    parts=['# Laptop comparison: geometry, objective preservation and proof cost',
        'Exploratory diagnosis. ARI measures agreement with fixed-center Full assignments, not external-label accuracy. Native geometry uses whole preserved paper subsets and saved starts 3/4; exact-integer controls use explicitly limited prefixes. No KMeans or PCA is retrained on real datasets. Prove controls use bounded engineering shapes, not complete real corpora.',
        'Native project/assign timings include three measurements after a small warm-up; basis construction and Sparse JL map generation are separately recorded. Cached PCA training is excluded. These are local engineering timings, not independent-dataset inference. Matched output dimensions do not mean matched geometry. The chosen exploratory thresholds are ARI >=0.99, centroid and sampled-pair Q95 relative error <=0.20, and normalized projected minimum-cost error <=0.10. No global JL event is inferred from a sampled Q95.']
    if len(geometry):
        geometry.to_csv(root/'geometry_raw.csv',index=False)
        keys=['dataset','k','method','total_m','q']
        metrics=['ari','agreement','centroid_q95','pair_q95','projected_cost_ratio','projected_cost_relative_error',
                 'original_assignment_cost_ratio','neighbor_recall_5','native_seconds_median']
        summary=geometry.groupby(keys,dropna=False)[metrics].agg(['mean','min','max'])
        summary.columns=['_'.join(c) for c in summary.columns];summary=summary.reset_index()
        counts=geometry.groupby(keys,dropna=False).size().reset_index(name='completed_rows')
        specs={j['dataset']:j for j in manifest['plan'] if j['kind']=='geometry'}
        expected=counts.apply(lambda r:len(specs[r.dataset]['starts'])*(1 if r.method in ['pca','full','protected_only'] else len(specs[r.dataset]['seeds'])),axis=1)
        counts['expected_rows']=expected;counts['complete_cell']=counts.completed_rows==expected
        summary=summary.merge(counts,on=keys)
        qualified=geometry.groupby(keys,dropna=False).qualified_geometry.all().reset_index(name='qualified_all_observed')
        summary=summary.merge(qualified,on=keys)
        summary['compression']=summary.apply(lambda r:r.total_m<float(geometry[geometry.dataset==r.dataset].d.iloc[0]),axis=1)
        summary['selectable']=summary.complete_cell&summary.qualified_all_observed&summary.compression
        summary.to_csv(root/'geometry_summary.csv',index=False)
        candidates=summary[summary.selectable].sort_values('total_m').groupby(['dataset','k','method']).head(1)
        candidates.to_csv(root/'qualified_candidates.csv',index=False)
        parts+=['## Geometry and directly normalized objective',f'{len(geometry)} completed rows. All artifacts contain projected centroid/pair distances and labels; original distances and bases are saved beside them. `projected_cost_ratio` directly compares the projected nearest-center cost with Full; `original_assignment_cost_ratio` evaluates chosen labels back in original space. Their meanings differ.',
            table(candidates[['dataset','k','method','total_m','q','ari_mean','centroid_q95_mean','pair_q95_mean','projected_cost_relative_error_max']]) if len(candidates) else 'No complete compressed cell passes all selected thresholds yet.']
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        names=list(geometry.dataset.unique());fig,axes=plt.subplots(2,4,figsize=(16,8),squeeze=False)
        for ax,name in zip(axes.flat,names):
            for method,g in summary[(summary.dataset==name)&(summary.k==5)].groupby('method'):
                if method=='full':continue
                envelope=g.groupby('total_m').pair_q95_mean.min()
                ax.plot(envelope.index,envelope.values,marker='o',label=method)
            ax.axhline(.2,color='gray',ls='--');ax.set_title(name);ax.set_xscale('log',base=2);ax.grid(alpha=.2)
            ax.set_xlabel('Total output dimensions');ax.set_ylabel('Pairwise Q95 relative error')
        if names:axes[0,0].legend(fontsize=6)
        fig.suptitle('Exploratory geometry envelope: best q per method and dimension, k=5')
        fig.tight_layout();fig.savefig(root/'geometry.png',dpi=160);plt.close(fig)
        parts+=['![Geometry curves](geometry.png)']
    if len(precision):
        precision.to_csv(root/'precision_raw.csv',index=False)
        keys=['dataset','k','method','total_m','scale']
        cols=['ari','ari_integer_full','centroid_q95','pair_q95','projected_cost_relative_error','projected_cost_ratio_integer_full']
        summary=precision.groupby(keys,dropna=False)[cols].agg(['mean','min','max'])
        summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(root/'precision_summary.csv',index=False)
        parts+=['## Integer precision',f'{len(precision)} rows on first min(128,n) points, starts 3, fresh maps 100/101/102, k=5/10 and T=1024/4096. Both original float and exact-integer Full references are reported. Hybrid squared costs divide by q*T^4 and PCA costs by T^2 before undoing the common input scaling. Input rounding and basis rounding are not conflated.']
    if len(circuits):
        circuits.to_csv(root/'circuits_raw.csv',index=False)
        profiling=circuits[circuits.job_id=='profile'].copy();profile_rows=[]
        stages=['range','theta','data','matrix','projection','complete']
        for method,g in profiling.groupby('method'):
            previous=0
            for stage in stages:
                s=g[g.stage==stage]
                if len(s) and s.status.iloc[0]=='ok':
                    count=int(s.constraints.iloc[0]);profile_rows.append(dict(method=method,stage=stage,cumulative_constraints=count,increment=count-previous));previous=count
        profile=pd.DataFrame(profile_rows);profile.to_csv(root/'constraint_breakdown.csv',index=False)
        parts+=['## Circuit cost breakdown','All prefix controls are solver-only and cannot generate proofs. Prefix differences are engineering diagnostics: deferred range/lookup batching can change their cost, so increments are not a mathematically exact additive decomposition.',table(profile) if len(profile) else 'No completed prefix measurements.']
        proofs=circuits[(circuits.kind=='prove')&(circuits.status=='ok')].copy()
        parts+=['## Actual proofs','Groth16, three measured proofs plus one warm-up per method (one measured proof in the smoke check). Complete authentication, range checks and tie-breaking remain enabled. Setup/compile and prove-only costs are separated. Single_run_total sums compile, setup, backend witness construction, median proving and median verification for one proof; it excludes Python preparation and cached PCA training. Synthetic PCA is learned from a separate deterministic synthetic training matrix with enough rows for the requested dimension. Specialization needs a new setup for a changed map; it is an engineering control, not a free optimization.']
        if len(proofs):
            parts+=[table(proofs[['job_id','method','total_m','constraints','prove_seconds_median','single_run_total','peak_rss_gib']])]
        ratios=[]
        for name,g in proofs.groupby('job_id'):
            full=g[g.method=='full'];pca=g[g.method=='pca']
            for method,r in g.groupby('method'):
                row=r.iloc[0];item=dict(job_id=name,method=method)
                for label,control in [('full',full),('pca',pca)]:
                    if len(control):
                        item['prove_speedup_vs_'+label]=float(control.prove_seconds_median.iloc[0]/row.prove_seconds_median)
                        item['total_speedup_vs_'+label]=float(control.single_run_total.iloc[0]/row.single_run_total)
                ratios.append(item)
        pd.DataFrame(ratios).to_csv(root/'proof_speedups.csv',index=False)
        failed=circuits[circuits.status!='ok']
        parts+=[f'{len(failed)} circuit controls hit an error or resource limit. They are retained in CSV, not counted as successful proofs. Actual proof selection is restricted to successfully solved controls <=750,000 constraints.']
    parts+=['## Scope and continuation',f'Plan complete: {manifest["complete"]}. Missing rows/resource limits are explicit; partial cells cannot qualify. Re-running the same command resumes saved jobs with identical code/binary fingerprints. A comparison between the native quality table and a synthetic proof shape does not establish a joint real-corpus speed/quality win. These experiments precede the proposed shared-routing optimization; no Artemis integration or new optimized routing is represented as implemented.']
    (root/'report.md').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
