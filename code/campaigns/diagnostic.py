"""Bounded, resumable native diagnostics; not the complete paper campaign.

Run with campaigns/diagnostic.ps1. All inference is exploratory and conditional
on the recorded dataset subset. No native timing is presented as ZK cost.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_var] = '1'

import numpy as np
import pandas as pd
from scipy.stats import norm
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from commitjl.clustering import _squared_distances, nearest_centroid_assign, phi_X
from commitjl.certificates import centroid_margin_certificate, dbscan_pair_certificate, pair_squared_distances
from commitjl.metrics import clustering_agreement, sample_pair_set
from commitjl.quantize import quantize, scale_for_precision
from experiments.pipeline import PipelineCache, run_native, normalize_for_quantization, build_projection, _centroids_from_labels
from experiments.common import row_hash, release_process_memory
from experiments.statistics import mean_std_ci95
from experiments.rq3_overhead import run_one
from campaigns.supplements.engine import measure_rq3_pair

CODE = Path(__file__).resolve().parents[1]
VERSION = 'local_diagnostics_v1'


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Distinguish an empty conditional set from an observed zero rate.
    def clean(x):
        if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, np.generic): return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x): return None
        return x
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def rate(numerator, denominator):
    return numerator / denominator if denominator else float('nan')


def wilson(success, total):
    if not total: return float('nan'), float('nan')
    z = norm.ppf(.975)
    p = success / total
    den = 1 + z*z/total
    center = (p + z*z/(2*total))/den
    half = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total))/den
    return float(max(0,center-half)), float(min(1,center+half))


def datasets(smoke=False):
    n = 64 if smoke else 1000
    specs = [
        {'loader':'gaussian_mixture', 'kwargs':{'n':n, 'd':64 if smoke else 256, 'k':5, 'gamma':.75, 'seed':12345}},
        {'loader':'dbscan_margin', 'kwargs':{'n':n, 'd':64 if smoke else 128, 'seed':12345}},
    ]
    if not smoke:
        specs += [{'loader':name, 'kwargs':{'n_max':100 if name=='arcene' else n,
                   **({'max_features':2000} if name=='20ng_tfidf' else {})}}
                  for name in ('mnist','fashion_mnist','gisette','arcene','20ng_tfidf')]
    return specs


def jobs(smoke=False):
    seeds = list(range(2 if smoke else 10))
    specs = datasets(smoke)
    work = [{'kind':'quality', 'dataset':ds, 'seeds':seeds} for ds in specs]
    # Security diagnostics use fixed datasets, independent projection draws.
    work += [{'kind':'grinding', 'dataset':ds, 'seeds':seeds} for ds in specs[:1] + ([] if smoke else [specs[2]])]
    work += [{'kind':'boundary', 'dataset':specs[1], 'seeds':seeds}]
    work += [{'kind':'gamma', 'dataset':specs[0], 'seeds':seeds}]
    for n in ([8] if smoke else [64,128,256]):
        for seed in seeds:
            sources=('nonce','beacon','commit_reveal')
            sources=sources[seed%3:]+sources[:seed%3]
            for source in sources:
                work.append({'kind':'overhead','dataset':{'loader':'gaussian_mixture',
                    'kwargs':{'n':n,'d':8 if smoke else 128,'k':3,'gamma':1.,'seed':12345}},
                    'seeds':[seed], 'source':source})
    return work


def centroid_counts(X, C, Z, Cz, eps):
    orig = _squared_distances(X,C)
    proj = _squared_distances(Z,Cz)
    rel = np.where(orig>1e-24, np.abs(proj/np.maximum(orig,1e-24)-1), np.where(proj<=1e-24,0,np.inf))
    event = np.all(rel<=eps,axis=1)
    margin = centroid_margin_certificate(X,C,eps)
    eligible = margin & event
    changed = np.argmin(orig,axis=1)!=np.argmin(proj,axis=1)
    nm, ne = int(margin.sum()), int(eligible.sum())
    return {'eps':eps, 'margin_count':nm, 'eligible_count':ne, 'total_count':len(X),
        'eligible_changes':int((eligible&changed).sum()), 'margin_changes_without_premise':int((margin&changed&~event).sum()),
        'conditional_violation_rate':rate(int((eligible&changed).sum()),ne),
        'eligible_fraction':ne/len(X), 'margin_fraction':nm/len(X),
        'pointwise_event_fraction':float(event.mean()), 'global_event':bool(event.all()),
        'assignment_stability':float((~changed).mean())}


def worker(job, root, smoke=False):
    root = Path(root)
    cache = PipelineCache(512*1024**2)
    row = {'dataset':job['dataset'], 'k':5,'precision_bits':16,'B':4096, 'projection':'sparse_jl','q':2,'m':32}
    completed = 0
    def emit(spec, fn):
        nonlocal completed
        key = row_hash(spec)
        path = root/'checkpoints'/f'{key}.json'
        if path.exists():
            existing = json.loads(path.read_text(encoding='utf-8'))
            if existing.get('_status')=='ok': return
        start = time.perf_counter()
        value = fn()
        atomic_json(path, {**spec, **value, '_status':'ok','row_id':key,
                     'row_wall_s':time.perf_counter()-start,'proof_generated':False})
        completed += 1
    with threadpool_limits(limits=1):
        data, dk = cache.dataset(row)
        X = np.asarray(data['X'],dtype=float)
        n,d = X.shape
        base = {'dataset_name':job['dataset']['loader'], 'dataset_sha256':data['meta'].get('sha256',''),
                'n':n,'d':d,'dataset_spec':job['dataset'],'campaign':VERSION}
        ms = [16,32] if smoke else [16,32,64,128,256]
        ms = [m for m in ms if m<=d]
        for seed in job['seeds']:
            rr = {**row,'seed':seed}
            if job['kind']=='quality':
                (_, C, _), _ = cache.kmeans_run(dk,X,5,seed)
                for method in ('identity','pca','sparse_jl'):
                    effective_ms = [d] if method=='identity' else [m for m in ms if method!='pca' or m<=min(n,d)]
                    for m in effective_ms:
                        for q in ([2,8] if method=='sparse_jl' else [0]):
                            spec={**base,'rq':'RQ1','seed':seed,'method':method,'m':m,'q':q,'k':5}
                            def quality(m=m,q=q,method=method):
                                out=run_native({**rr,'m':m,'q':max(1,q),'projection':method},{'pair_set_size':500,'eps':.15},cache)
                                result={key:out[key] for key in ('ari','ari_ground_truth_full','ari_ground_truth_projected','r_phi','delta_q95','t_assign_s')}
                                if method=='sparse_jl':
                                    P,_=cache.projection(dk,method,X,m,q,seed,65536)
                                    A=np.rint(P['real']*np.sqrt(q)).astype(np.int64)
                                    physical=normalize_for_quantization(X,65536,4096)
                                    factor=np.max(np.abs(physical))/np.max(np.abs(X)) if np.any(X) else 1.
                                    Xq=quantize(physical,65536,4096)
                                    Cq=quantize(C*factor,65536,4096)
                                    full=nearest_centroid_assign(X,C)
                                    qfull=nearest_centroid_assign(Xq,Cq)
                                    assigned=nearest_centroid_assign(Xq@A.T,Cq@A.T)
                                    result.update({'ari_quantized_projected':clustering_agreement(full,assigned)['ari'],
                                        'ari_quantization_only':clustering_agreement(full,qfull)['ari'],
                                        'ari_projection_given_quantization':clustering_agreement(qfull,assigned)['ari'],
                                        'quantized_path':'integer_fixed_centroids_native_seed_fixture_not_transcript_proof'})
                                return result
                            emit(spec,quality)
                            if method=='sparse_jl' and m in ([32] if smoke else [32,128,256]):
                                P,_=cache.projection(dk,method,X,m,q,seed,65536)
                                Z,Cz=X@P['real'].T,C@P['real'].T
                                for eps in (.15,.3,.5):
                                    emit({**spec,'rq':'RQ4','eps':eps,'test':'centroid'},lambda eps=eps:centroid_counts(X,C,Z,Cz,eps))
                # Fixed physical data and FIXED original centroids isolate precision.
                for bits in (8,12,16,20,24):
                    for policy in ('fixed_reference_24','range_utilized'):
                        spec={**base,'rq':'RQ5','seed':seed,'test':'precision','bits':bits,'policy':policy,'m':32,'q':2}
                        def precision(bits=bits,policy=policy):
                            S=scale_for_precision(bits)
                            reference=scale_for_precision(24) if policy=='fixed_reference_24' else S
                            physical=(normalize_for_quantization(X,reference,4096)
                                if policy=='fixed_reference_24' else X*(.9*4096/(S*max(float(np.max(np.abs(X))),1e-300))))
                            factor=np.max(np.abs(physical))/np.max(np.abs(X)) if np.any(X) else 1.
                            Xq=quantize(physical,S,4096);Cq=quantize(C*factor,S,4096)
                            P=build_projection('sparse_jl',X,32,2,seed,S)['real']
                            A=np.rint(P*np.sqrt(2)).astype(np.int64)
                            full=nearest_centroid_assign(X,C);qfull=nearest_centroid_assign(Xq,Cq)
                            assigned=nearest_centroid_assign(Xq@A.T,Cq@A.T)
                            decoded=Xq.astype(float)/S
                            cost=phi_X(X,_centroids_from_labels(X,assigned,5))
                            return {'ari':clustering_agreement(full,assigned)['ari'],
                                'ari_quantization_only':clustering_agreement(full,qfull)['ari'],
                                'ari_projection_given_quantization':clustering_agreement(qfull,assigned)['ari'],
                                'r_phi':cost/phi_X(X,C), 'zero_fraction':float(np.mean(Xq==0)),
                                'relative_quantization_error':float(np.linalg.norm(decoded-physical)/max(np.linalg.norm(physical),1e-300)),
                                'constraint_interpretation':'no_measured_constraints; bits alone do not change fixed-B circuit range checks'}
                        emit(spec,precision)
            elif job['kind']=='overhead':
                source=job['source'];spec={**base,'rq':'RQ3','test':'overhead','seed':seed,'source':source,'m':min(32,d),'q':2}
                def overhead():
                    res=run_one({**rr,'k':3,'m':min(32,d)}, {}, source,max_retries=1)
                    t=res['transcript']['stage_times_s']
                    extra=t['issue_nonce']+t['expand_sparse_and_CA']
                    out={'total_s':sum(t.values()),'after_commit_s':extra,'after_commit_fraction':extra/sum(t.values()),
                         **{'stage_'+k+'_s':v for k,v in t.items()},'rho_hex':res['rho'].hex(),'projection_seed_hex':res['seed'].hex(),
                         'timing_semantics':'serial_isolated_protocol_stage_wall_times_excludes_dataset_and_kmeans',
                         'source_label':'Simulated local trace' if source=='beacon' else source}
                    if source=='nonce':
                        paired=measure_rq3_pair({**rr,'k':3,'m':min(32,d)})
                        out.update({k:paired[k] for k in ('prepublished_transaction_s','postcommit_transaction_s','incremental_postcommit_s')})
                    return out
                emit(spec,overhead)
            elif job['kind']=='boundary':
                for m in ([32] if smoke else [32,128]):
                    P=build_projection('sparse_jl',X,m,8,seed,65536)['real'];Z=X@P.T
                    for eps in (.15,.3,.5):
                        spec={**base,'rq':'RQ4','test':'dbscan_pairs','seed':seed,'m':m,'q':8,'eps':eps}
                        def boundary(eps=eps):
                            pc=dbscan_pair_certificate(X,1.,eps)
                            sq=pair_squared_distances(Z,pc['i'],pc['j'])
                            original=pc['orig_sqdist'];event=np.abs(sq/np.maximum(original,1e-24)-1)<=eps
                            stable=(sq<=1)==(original<=1);near=~pc['certified'];eligible=pc['certified']&event
                            return {'near_count':int(near.sum()),'near_stability':rate(int((near&stable).sum()),int(near.sum())),
                                'total_count':len(sq),'eligible_count':int(eligible.sum()),'eligible_changes':int((eligible&~stable).sum()),
                                'conditional_violation_rate':rate(int((eligible&~stable).sum()),int(eligible.sum())),
                                'coverage':float(pc['certified'].mean()),'eligible_fraction':float(eligible.mean()),'all_pairs_evaluated':True}
                        emit(spec,boundary)
            elif job['kind']=='gamma':
                for gamma in (.1,.25,.5,1.,2.):
                    ds={'loader':'gaussian_mixture','kwargs':{**job['dataset']['kwargs'],'gamma':gamma}}
                    for m in ([32] if smoke else [16,64,128]):
                        spec={**base,'rq':'RQ5','test':'separation','seed':seed,'gamma':gamma,'m':m,'q':2,'dataset_spec':ds}
                        def separation():
                            out=run_native({**rr,'dataset':ds,'m':m},{'pair_set_size':500},cache)
                            return {k:out[k] for k in ('ari','ari_ground_truth_full','ari_ground_truth_projected','r_phi','dataset_sha256')}
                        emit(spec,separation)
            elif job['kind']=='grinding':
                # Once per dataset, not per kmeans repetition.
                if seed!=job['seeds'][0]:continue
                pairs=sample_pair_set(n,min(300,n*(n-1)//2),20261004)
                i,j=np.array(pairs).T;orig=pair_squared_distances(X,i,j)
                for m in ([32] if smoke else [64,256]):
                    for q in ([2] if smoke else [2,8]):
                        spec={**base,'rq':'RQ3','test':'grinding_draws','seed':seed,'m':m,'q':q}
                        def draws():
                            pilot_n,test_n=(8,32) if smoke else (32,256)
                            seq=np.random.SeedSequence([20261004,m,q]).spawn(pilot_n+test_n)
                            maxima=[]
                            for ss in seq:
                                ps=int(ss.generate_state(1,dtype=np.uint64)[0])
                                P=build_projection('sparse_jl',X,m,q,ps,65536)['real']
                                sq=pair_squared_distances(X@P.T,i,j)
                                maxima.append(float(np.max(np.abs(sq/np.maximum(orig,1e-24)-1))))
                            threshold=float(np.quantile(maxima[:pilot_n],.7))
                            return {'pilot_maxima':maxima[:pilot_n],'evaluation_maxima':maxima[pilot_n:],
                                'diagnostic_threshold':threshold,'pilot_count':pilot_n,'evaluation_count':test_n,
                                'fixed_thresholds':[.15,.3,.5,.75], 'event':'sampled_pair_distortion_exceedance_not_forged_proof',
                                'seed_policy':'disjoint_pilot_and_evaluation_projection_draws; fixed_dataset_and_pair_set'}
                        emit(spec,draws)
            release_process_memory()
    return completed


def aggregate(root):
    root=Path(root)
    records=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((root/'checkpoints').glob('*.json'))]
    if not records:raise ValueError('No successful records to analyze')
    df=pd.DataFrame(records)
    (root/'analysis').mkdir(exist_ok=True)
    df.to_parquet(root/'measurements.parquet',index=False)
    # Flat CSV excludes nested draw arrays/specs; JSON retains full evidence.
    df.drop(columns=['dataset_spec','pilot_maxima','evaluation_maxima'],errors='ignore').to_csv(root/'measurements.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'figure.dpi':140,'axes.spines.top':False,'axes.spines.right':False})
    lines=['# Local diagnostic results', '', '**Exploratory native diagnostics, not the full paper campaign or ZK benchmarks.**',
        'Fixed dataset subsets; ten seeds in the normal campaign. CIs describe projection/KMeans-seed variability on these subsets, not uncertainty over the entire source population. No good outcome is guaranteed.', '',
        'ARI is agreement with full-space fixed-centroid assignments. Ground-truth ARI is recorded separately. Component timing is not total runtime. Precision bits and column sparsity q are different parameters.', '']
    summaries={}
    for rq in ('RQ1','RQ3','RQ4','RQ5'):
        part=df[df.rq==rq]
        lines += [f'## {rq}',f'{len(part)} completed measurement records.', '']
        summaries[rq]=[]
        if rq=='RQ3':
            overhead=part[part.test=='overhead']
            for (n,source),g in overhead.groupby(['n','source']):
                st=mean_std_ci95(g.after_commit_s)
                lines.append(f'- n={n}, {source}: after-commit mean {st["mean"]:.4f} s, 95% CI [{st["ci95_low"]:.4f}, {st["ci95_high"]:.4f}]; mean share {100*g.after_commit_fraction.mean():.3f}%.')
                summaries[rq].append({'test':'overhead','n':n,'source':source,**{f'after_commit_s_{k}':v for k,v in st.items()}})
                if source=='nonce':
                    paired=mean_std_ci95(g.incremental_postcommit_s)
                    lines.append(f'  Paired whole-transaction difference: {paired["mean"]:.4f} s [{paired["ci95_low"]:.4f}, {paired["ci95_high"]:.4f}]. A CI crossing zero means the timing difference is unresolved, not a security failure.')
                    summaries[rq][-1].update({f'paired_difference_s_{k}':v for k,v in paired.items()})
            grinding=[]
            for _,g in part[part.test=='grinding_draws'].iterrows():
                maxima=np.array(g.evaluation_maxima)
                thresholds=[('fixed',t) for t in g.fixed_thresholds]+[('pilot_calibrated_diagnostic',g.diagnostic_threshold)]
                for policy,threshold in thresholds:
                    flags=maxima>threshold
                    for Q in (1,2,4,8):
                        batches=flags[:len(flags)//Q*Q].reshape(-1,Q).any(axis=1)
                        lo,hi=wilson(int(batches.sum()),len(batches))
                        grinding.append({'dataset_name':g.dataset_name,'m':g.m,'q':g.q,'policy':policy,'threshold':threshold,'Q':Q,
                            'success_rate':batches.mean(),'ci95_low':lo,'ci95_high':hi,'independent_batches':len(batches),
                            'delta_geom':flags.mean(),'plugin_union_bound':min(1,Q*flags.mean())})
            gg=pd.DataFrame(grinding);gg.to_csv(root/'analysis/rq3_grinding.csv',index=False)
            for (ds,m,q),g in gg[gg.policy=='pilot_calibrated_diagnostic'].groupby(['dataset_name','m','q']):
                one=g[g.Q==1].iloc[0];eight=g[g.Q==8].iloc[0]
                lines.append(f'- Diagnostic grinding {ds}, m={int(m)}, q={int(q)}: Q=1 rate {one.success_rate:.3f} [{one.ci95_low:.3f}, {one.ci95_high:.3f}]; Q=8 rate {eight.success_rate:.3f} [{eight.ci95_low:.3f}, {eight.ci95_high:.3f}]. Threshold {one.threshold:.3f} frozen from independent pilot draws.')
            lines += ['', 'Beacon means **Simulated local trace**. Timing jobs run serially, without campaign-worker contention. Paired prepublished comparisons are recorded for nonce; small or zero differences are legitimate outcomes.',
                'Grinding uses fixed thresholds and an independent-pilot diagnostic threshold. The calibrated curve is diagnostic only, not a paper security parameter or proof-forgery experiment. Wilson intervals use independent Q-sized batches; saturated curves and wide intervals are explicitly inconclusive.']
            fig,ax=plt.subplots(figsize=(8,4))
            for (ds,m,q),g in gg[gg.policy=='pilot_calibrated_diagnostic'].groupby(['dataset_name','m','q']):
                ax.errorbar(g.Q,g.success_rate,yerr=[np.maximum(0,g.success_rate-g.ci95_low),np.maximum(0,g.ci95_high-g.success_rate)],marker='o',label=f'{ds}, m={m:g}, q={q:g}')
            ax.set(xlabel='Allowed independent attempts Q',ylabel='Distortion-event frequency',ylim=(-.02,1.02));ax.legend(fontsize=7)
            tf,tax=plt.subplots(figsize=(9,4))
            timing=pd.DataFrame(summaries[rq])
            labels=[f'n={int(a.n)}, {a.source}' for _,a in timing.iterrows()]
            tax.bar(np.arange(len(timing)),timing.after_commit_s_mean,yerr=np.maximum(0,timing.after_commit_s_ci95_high-timing.after_commit_s_mean),capsize=3)
            tax.set_xticks(np.arange(len(timing)),labels,rotation=45,ha='right')
            tax.set(ylabel='After-commit seconds (mean and 95% CI)',title='Serial host timings; simulated local beacon')
            tf.tight_layout();tf.savefig(root/'analysis/rq3_timing.png',dpi=220);plt.close(tf)
        else:
            groups={'RQ1':['dataset_name','method','m','q'], 'RQ4':['dataset_name','test','m','q','eps'],
                    'RQ5':['dataset_name','test','m','q','bits','policy','gamma']}[rq]
            metrics={'RQ1':['ari','ari_quantized_projected','ari_quantization_only','r_phi','delta_q95','t_assign_s'],
                     'RQ4':['eligible_fraction','margin_fraction','near_stability','conditional_violation_rate','assignment_stability'],
                     'RQ5':['ari','ari_quantization_only','ari_projection_given_quantization','r_phi','zero_fraction','relative_quantization_error']}[rq]
            for key,g in part.groupby(groups,dropna=False):
                row=dict(zip(groups,key));row['seed_count']=g.seed.nunique()
                for metric in metrics:
                    if metric in g:
                        row.update({metric+'_'+k:v for k,v in mean_std_ci95(g[metric]).items()})
                summaries[rq].append(row)
            ss=pd.DataFrame(summaries[rq]);ss.to_csv(root/f'analysis/{rq.lower()}_summary.csv',index=False)
            if rq=='RQ1':
                for ds,g in ss[(ss.method=='sparse_jl')&(ss.q==2)].groupby('dataset_name'):
                    best=g.sort_values('m').iloc[-1]
                    lines.append(f'- {ds}, largest tested m={int(best.m)}: ARI {best.ari_mean:.3f} [{best.ari_ci95_low:.3f}, {best.ari_ci95_high:.3f}]; integer-projected ARI {best.ari_quantized_projected_mean:.3f}. Compare matched m with PCA in the CSV; this is not selected as an optimal paper shape.')
                lines.append('The integer path shares the sparse matrix and fixed centroids with its float control, but uses a native seed fixture: it is NOT a transcript-to-proof validation. Quantization-only and projection-given-quantization metrics separate the sources of loss. PCA fits are cached and m never exceeds the available component count.')
                names=sorted(ss.dataset_name.unique())
                fig,axes=plt.subplots(int(np.ceil(len(names)/2)),2,figsize=(10,3*int(np.ceil(len(names)/2))),squeeze=False)
                for ax,ds in zip(axes.flat,names):
                    g=ss[ss.dataset_name==ds]
                    for (method,q),curve in g[g.method!='identity'].groupby(['method','q']):
                        curve=curve.sort_values('m')
                        ax.errorbar(curve.m,curve.ari_mean,yerr=np.maximum(0,curve.ari_ci95_high-curve.ari_mean),marker='o',label=method if q==0 else f'{method}, q={int(q)}')
                    ax.axhline(1,color='gray',ls=':',label='Full reference')
                    ax.set(title=ds,xlabel='Projected dimensions m',ylabel='Assignment ARI');ax.legend(fontsize=7)
                for ax in list(axes.flat)[len(names):]:ax.set_visible(False)
            elif rq=='RQ4':
                eligible=pd.to_numeric(part.eligible_count).sum();changes=pd.to_numeric(part.eligible_changes).sum()
                lines += [f'- Conditional eligible observations: {int(eligible)}; changes within that set: {int(changes)}.',
                    '- Consult per-seed eligible counts and near-boundary denominators. Empty sets produce missing rates, never fake zero violations. Repeated observations across eps/m are not independent points.',
                    '- A failed global premise does not refute the conditional certificate. Margin-only changes without the premise are not theorem violations. Pair results audit neighborhood relations, not full DBSCAN clustering.']
                for (ds,m,q,eps),g in ss[ss.test=='centroid'].groupby(['dataset_name','m','q','eps']):
                    a=g.iloc[0]
                    lines.append(f'- {ds}, m={int(m)}, q={int(q)}, eps={eps:g}: mean eligible fraction {a.eligible_fraction_mean:.3f}; mean assignment stability {a.assignment_stability_mean:.3f}.')
                for _,a in ss[ss.test=='dbscan_pairs'].iterrows():
                    lines.append(f'- DBSCAN boundary m={int(a.m)}, eps={a.eps:g}: near-band stability {a.near_stability_mean:.3f} [{a.near_stability_ci95_low:.3f}, {a.near_stability_ci95_high:.3f}]. See raw near_count before interpreting missing rates.')
                fig,ax=plt.subplots(figsize=(8,4))
                for ds,g in ss[(ss.test=='centroid')&(ss.q==8)&(ss.eps==.3)].groupby('dataset_name'):
                    g=g.sort_values('m');ax.plot(g.m,g.eligible_fraction_mean,'o-',label=ds)
                ax.set(xlabel='Projected dimensions m',ylabel='Conditional eligible fraction');ax.legend(fontsize=7)
            else:
                lines += ['- Fixed-reference-24 keeps physical data constant and isolates rounding; range-utilized rescales each bit setting to the same integer bound. The latter should show that bit labels alone do not improve integer resolution when B is fixed.',
                    '- Neither policy establishes an optimal precision/constraint trade-off. No constraints or proof costs were measured. Separation sweeps record both full and projected ground-truth ARI.']
                for ds,g in ss[(ss.test=='precision')&(ss.policy=='fixed_reference_24')].groupby('dataset_name'):
                    for _,a in g.sort_values('bits').iterrows():
                        lines.append(f'- {ds}, {int(a.bits)} bits: ARI {a.ari_mean:.3f} [{a.ari_ci95_low:.3f}, {a.ari_ci95_high:.3f}], quantization-only ARI {a.ari_quantization_only_mean:.3f}, zero-coordinate fraction {a.zero_fraction_mean:.3f}.')
                fig,ax=plt.subplots(figsize=(8,4))
                for ds,g in ss[(ss.test=='precision')&(ss.policy=='fixed_reference_24')].groupby('dataset_name'):
                    g=g.sort_values('bits');ax.errorbar(g.bits,g.ari_mean,yerr=np.maximum(0,g.ari_ci95_high-g.ari_mean),marker='o',label=ds)
                ax.set(xlabel='Precision bits (fixed physical scale)',ylabel='Assignment ARI');ax.legend(fontsize=7)
                gf,gax=plt.subplots(figsize=(8,4))
                for m,g in ss[ss.test=='separation'].groupby('m'):
                    g=g.sort_values('gamma');gax.errorbar(g.gamma,g.ari_mean,yerr=np.maximum(0,g.ari_ci95_high-g.ari_mean),marker='o',label=f'm={int(m)}')
                gax.set(xlabel='Synthetic separation gamma',ylabel='Assignment ARI (mean and 95% CI)');gax.legend()
                gf.tight_layout();gf.savefig(root/'analysis/rq5_separation.png',dpi=220);plt.close(gf)
        pd.DataFrame(summaries[rq]).to_csv(root/f'analysis/{rq.lower()}_summary.csv',index=False)
        fig.tight_layout();fig.savefig(root/f'analysis/{rq.lower()}.png',dpi=220);plt.close(fig)
        lines.append('')
    lines += ['## Scope and interpretation', 'Real data are explicit subsets, not whole-source runs. No RQ2 proof is executed. Results can support limited statements about fidelity, sensitivity and host overhead; they cannot establish a prover advantage, public-beacon deployment, universal stability or full paper completion.',
        'Resume with the same command. Only atomically saved successful rows are skipped; a changed implementation or plan requires a new output directory. A crash during a row loses only that unfinished row.']
    (root/'analysis/interpretation.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--out',type=Path,default=CODE/'results/diagnostics/local_v1')
    args=ap.parse_args()
    if not 1<=args.workers<=6:ap.error('workers must be in [1,6]')
    root=args.out.resolve();root.mkdir(parents=True,exist_ok=True)
    science=[p for folder in ('commitjl','experiments','data') for p in (CODE/folder).rglob('*.py')]
    science += [Path(__file__), CODE/'campaigns/supplements/engine.py']
    digest=hashlib.sha256()
    for p in sorted(science):digest.update(str(p.relative_to(CODE)).encode());digest.update(p.read_bytes())
    plan={'version':VERSION,'smoke':args.smoke,'workers':args.workers,'implementation_sha256':digest.hexdigest(),
          'jobs':jobs(args.smoke),'scope':'targeted native diagnostics, not full paper campaign','timing_workers':1}
    manifest=root/'plan.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))!=plan:
        raise RuntimeError('Plan/implementation changed. Use a new --out directory; do not mix results.')
    atomic_json(manifest,plan)
    parallel=[j for j in plan['jobs'] if j['kind']!='overhead']
    serial=[j for j in plan['jobs'] if j['kind']=='overhead']
    start=time.perf_counter()
    print(f'Native diagnostic campaign: {len(parallel)} dataset jobs, {len(serial)} isolated timing jobs; checkpoints={root}',flush=True)
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(worker,j,root,args.smoke) for j in parallel]
            for f in tqdm(as_completed(futures),total=len(futures),desc='Dataset diagnostics',file=sys.stdout):f.result()
        for j in tqdm(serial,desc='Isolated RQ3 timing',file=sys.stdout):worker(j,root,args.smoke)
    except BaseException:
        print('Campaign interrupted: completed row checkpoints are preserved; repeat the same command.',flush=True)
        raise
    aggregate(root)
    atomic_json(root/'completion.json',{'elapsed_s':time.perf_counter()-start,'complete':True,'scope':plan['scope']})
    print(f'Completed. Read {root / "analysis/interpretation.md"}',flush=True)


if __name__=='__main__':main()
