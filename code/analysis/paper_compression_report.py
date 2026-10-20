"""Export native paper measurements without conflating them with ZK timings."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.laptop_comparison import atomic_json
from experiments.paper_compression import bootstrap_crossed,STOCHASTIC
from experiments.paper_artifacts import artifact_issues,job_complete

KEYS = ['dataset', 'variant', 'phase', 'k', 'start', 'method', 'm', 'q', 'seed']
METRICS = ['ari','nmi','agreement','r_phi','r_phi_train_reconstructed',
    'projected_cost_relative_error','fixed_original_assignment_cost_ratio',
    'pair_q50','pair_q95','pair_q99','pair_max','pair_exceedance',
    'centroid_q50','centroid_q95','centroid_q99','centroid_max',
    'certificate_coverage','margin_coverage','centroid_embedding_point_fraction',
    'ari_ground_truth','nmi_ground_truth']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def timing_rows(metadata, stage, measurement):
    if not isinstance(measurement, dict) or 'samples' not in measurement: return []
    return [dict(metadata, stage=stage, repetition=i, seconds=float(value),
        peak_rss_bytes=measurement.get('peak_rss_bytes'),
        rss_baseline_bytes=measurement.get('rss_baseline_bytes'),
        peak_rss_delta_bytes=measurement.get('peak_rss_delta_bytes'))
        for i, value in enumerate(measurement['samples'])]


def quality_summary(frame, config):
    records = []
    if frame.empty: return records
    for key, group in frame.groupby(['dataset','variant','phase','k','method','m','q']):
        record = dict(zip(['dataset','variant','phase','k','method','m','q'], key))
        stochastic = record['method'] in STOCHASTIC
        expected_draws = set(range(config['seeds'])) if stochastic else {-1}
        complete = (len(group) == len(expected_draws)*config['starts'] and
            set(group.seed) == expected_draws and set(group.start) == set(range(config['starts'])) and
            not group.duplicated(['seed','start']).any() and group.get('artifact_valid',pd.Series(True,index=group.index)).all())
        record.update(rows=len(group), complete=complete)
        for metric in METRICS:
            if metric not in group or group[metric].isna().all(): continue
            record[metric+'_mean'] = float(group[metric].mean())
            if complete and not group[metric].isna().any():
                matrix = group.pivot(index='seed', columns='start', values=metric).to_numpy()
                _, lo, hi = bootstrap_crossed(matrix, draws=config['bootstrap'])
                record.update({metric+'_ci_low':lo, metric+'_ci_high':hi})
        records.append(record)
    return records


def tex_table(frame, columns, destination):
    """Plain LaTeX with no optional pandas/tabulate dependency."""
    def escape(value):
        if value is None or pd.isna(value): return '--'
        if isinstance(value, (float,np.floating)): return f'{value:.5g}'
        return str(value).replace('\\', '\\textbackslash{}').replace('_', '\\_').replace('%','\\%').replace('&','\\&')
    content = ['\\begin{tabular}{'+'l'*len(columns)+'}', '\\hline',
        ' & '.join(escape(c) for c in columns)+' \\\\', '\\hline']
    for _, row in frame.iterrows(): content.append(' & '.join(escape(row.get(c)) for c in columns)+' \\\\')
    content.extend(['\\hline','\\end{tabular}'])
    destination.write_text('\n'.join(content)+'\n', encoding='utf-8')


def task_setup(row,w,task):
    """Costs required by this task; exclude unused centroids/norms."""
    centers=row['centroid_fit']['median'];pca=(row.get('pca_fit') or {}).get('median',0.)
    basis=(row.get('basis_fit') or {}).get('median',0.)
    uses_centers=task in ('assignment','heldout_assignment','kmeans_refine')
    requires_centers=row['method'] in ('protected_jl','protected_only') or uses_centers
    fit=pca+basis+(centers if requires_centers else 0.)
    full_fit=centers if uses_centers else 0.
    prep=(w['numeric_preparation']['median'] if task in ('assignment','heldout_assignment') else
        0. if task in ('kmeans','kmeans_refine') else (w.get('point_norm_preparation') or {}).get('median'))
    full_prep=(w['full_numeric_preparation']['median'] if task in ('assignment','heldout_assignment') else
        0. if task in ('kmeans','kmeans_refine') else (w.get('full_point_norm_preparation') or {}).get('median'))
    cp=w['center_projection']['median'] if uses_centers else 0.
    # Legacy v1 did not measure point-only norms: leave totals undefined rather
    # than pretending its centroid+point measurement is this task's cost.
    setup=row['model_construction']['median']+w['compression']['median']+cp+prep if prep is not None else None
    return dict(conditional_setup_seconds=setup,full_preparation_seconds=full_prep,
        method_model_fit_seconds=fit,full_model_fit_seconds=full_fit,
        task_center_compression_seconds=cp,task_preparation_seconds=prep)


def build_report(root):
    root = Path(root); manifest = read(root/'manifest.json'); cfg = manifest['identity']['configuration']
    report = root/'report'; report.mkdir(exist_ok=True)
    rows, stages, training, comparisons, queries, selections, datasets, checks, workloads = [],[],[],[],[],[],[],[],[]
    profiles=[];profile_stages=[];digest_cache={}
    require_hashes=cfg.get('version')=='paper-compression-v2'
    planned_jobs = read(root/'plan.json')['jobs']
    for job in planned_jobs:
        folder = root/'jobs'/f"{job['dataset']}_{job['variant']}"
        base = dict(dataset=job['dataset'], variant=job['variant'])
        if (folder/'dataset.json').exists(): datasets.append(read(folder/'dataset.json'))
        if (folder/'fits.json').exists():
            fits = read(folder/'fits.json')
            training += timing_rows(base, 'pca_training_largest_dimension_once', fits['pca'])
            for draw,measure in fits.get('pca_randomized',{}).items():
                training+=timing_rows(dict(base,seed=int(draw)),'randomized_pca_training_largest_dimension_once',measure)
            for kind in ('centroids','basis'):
                for key, measure in fits[kind].items():
                    training += timing_rows(dict(base, fit_key=key, iterations=measure.get('iterations'),
                        hit_iteration_cap=measure.get('hit_iteration_cap')), kind, measure)
        if (folder/'selection.json').exists():
            selections += [dict(base, **entry) for entry in read(folder/'selection.json')['choices']]
        actual = list((folder/'rows').glob('*.json'))
        expected = read(folder/'case_inventory.json')['expected_rows'] if (folder/'case_inventory.json').exists() else None
        artifact_errors = []
        for path in actual:
            row = read(path)
            problems=artifact_issues(folder,row,digest_cache,require_hashes)
            artifact_errors.extend(problems)
            row['artifact_valid']=not problems;rows.append(row); metadata = {key:row[key] for key in KEYS}
            stages += timing_rows(metadata, 'model_construction', row['model_construction'])
            stages += timing_rows(metadata, 'screening_compression', row['screening_compression'])
            for p in [folder/'arrays'/f'{path.stem}.npz', folder/row['model_artifact']]:
                if not p.exists(): artifact_errors.append(str(p))
            if not row.get('workloads') or problems: continue
            w = read(folder/row['workloads']/'timings.json')
            scalar = dict(metadata, **{k:v for k,v in w.items() if not isinstance(v,(dict,list))})
            workloads.append(scalar)
            for profile,record in w.get('dbscan_profiles',{}).items():
                profiles.append(dict(metadata,profile=profile,**{k:v for k,v in record.items() if not isinstance(v,(dict,list))}))
                for stage,measurement in record.items():profile_stages+=timing_rows(dict(metadata,profile=profile),stage,measurement)
            for stage, measurement in w.items(): stages += timing_rows(metadata, stage, measurement)
            for control, values in w['distance_queries'].items():
                queries += [dict(metadata, control=control, **item) for item in values]
            cm = w['compression']['median']; cp = w['center_projection']['median']; prep = w['numeric_preparation']['median']
            full_prep = w['full_numeric_preparation']['median']
            map_seconds = row['model_construction']['median']
            basis = (row.get('basis_fit') or {}).get('median',0.)
            pca = (row.get('pca_fit') or {}).get('median',0.)
            centers = row['centroid_fit']['median']
            # Conditional representation setup, model fitting and task costs
            # remain separately exported. Training PCA's largest requested map
            # is a shared sweep cost, not a standalone minimal-m fit estimate.
            requires_centroids = row['method'] in ('protected_jl','protected_only')
            geometry_model_fit = pca+basis+(centers if requires_centroids else 0.)
            tasks=['assignment','heldout_assignment','knn','kmeans','dbscan_pair_build','dbscan_fit','dbscan']
            if 'kmeans_refine' in w:tasks.append('kmeans_refine')
            for task in tasks:
                method_time=(w['dbscan_pair_build']['median']+w['dbscan_fit']['median']) if task=='dbscan' else w[task]['median']
                full_time=(w['full_dbscan_pair_build']['median']+w['full_dbscan_fit']['median']) if task=='dbscan' else w['full_'+task]['median']
                setup=task_setup(row,w,task)
                stage_only=task in ('dbscan_pair_build','dbscan_fit')
                valid_measurement=not(task=='heldout_assignment' and w.get('measurement_version',1)<2)
                adequate=not(w.get(task+'_hit_iteration_cap',False) or w.get('full_'+task+'_hit_iteration_cap',False))
                if task=='kmeans' and w.get('full_kmeans_iterations',0)>=cfg['kmeans_max_iter']:adequate=False
                if task=='dbscan':adequate=bool(w.get('dbscan_reference_partition_informative',
                    0<w['full_dbscan_noise_fraction']<1))
                comparisons.append(dict(metadata, task=task, workload_rows=w['workload_rows'],
                    method_seconds=method_time, full_seconds=full_time,
                    measurement_valid=valid_measurement,task_result_adequate=adequate,
                    downstream_speedup=full_time/method_time if method_time else None,
                    whole_compression_seconds=cm, center_compression_seconds=cp, preparation_seconds=prep,
                    map_construction_seconds=map_seconds, basis_fit_seconds=basis,
                    shared_pca_fit_seconds=pca, centroid_training_seconds=centers,
                    composite_method_seconds=(setup['method_model_fit_seconds']+setup['conditional_setup_seconds']+method_time)
                        if setup['conditional_setup_seconds'] is not None and not stage_only and valid_measurement else None,
                    composite_full_seconds=(setup['full_model_fit_seconds']+setup['full_preparation_seconds']+full_time)
                        if setup['full_preparation_seconds'] is not None and not stage_only and valid_measurement else None,
                    composite_scope='sum of required stage medians; PCA shared largest-m sweep fit; DBSCAN composite counts setup once',**setup))
            qm = np.array([v['seconds'] for v in w['distance_queries']['method']]).reshape(cfg['timing_repetitions'],cfg['queries'])
            qf = np.array([v['seconds'] for v in w['distance_queries']['full']]).reshape(qm.shape)
            for count in range(1,cfg['queries']+1):
                setup=task_setup(row,w,'distance_queries')
                comparisons.append(dict(metadata, task='distinct_distance_queries', queries=count,
                    pairs_per_query=w['pairs_per_query'], method_seconds=float(np.median(qm[:,:count].sum(axis=1))),
                    full_seconds=float(np.median(qf[:,:count].sum(axis=1))),
                    conditional_method_seconds=setup['conditional_setup_seconds']+float(np.median(qm[:,:count].sum(axis=1))) if setup['conditional_setup_seconds'] is not None else None,
                    conditional_full_seconds=setup['full_preparation_seconds']+float(np.median(qf[:,:count].sum(axis=1))) if setup['full_preparation_seconds'] is not None else None,
                    model_fit_seconds=geometry_model_fit,
                    whole_compression_seconds=cm,**setup,
                    scope='measured different queries; complete-result caching not used for either method'))
        checks.append(dict(base, expected_rows=expected, observed_rows=len(actual),
            complete=job_complete(folder,digest_cache,require_hashes), missing_artifacts=artifact_errors,
            execution=read(folder/'execution.json') if (folder/'execution.json').exists() else None))
    frame = pd.DataFrame(rows)
    flat = frame.drop(columns=[c for c in frame if frame[c].map(lambda v:isinstance(v,(dict,list))).any()]) if not frame.empty else frame
    exports = dict(quality_rows=flat, quality_summary=pd.DataFrame(quality_summary(frame,cfg)),
        stage_timings=pd.DataFrame(stages), model_training=pd.DataFrame(training),
        workload_quality=pd.DataFrame(workloads), workload_comparison=pd.DataFrame(comparisons),
        query_timings=pd.DataFrame(queries), selected_dimensions=pd.DataFrame(selections),
        dbscan_profile_quality=pd.DataFrame(profiles),dbscan_profile_timings=pd.DataFrame(profile_stages),
        datasets=pd.DataFrame([dict({k:v for k,v in x.items() if not isinstance(v,(dict,list))},
            train_rows=x['split_counts']['train'],validation_rows=x['split_counts']['validation'],
            test_rows=x['split_counts']['test']) for x in datasets]))
    for name, data in exports.items(): data.to_csv(report/f'{name}.csv', index=False)
    stage_summary = pd.DataFrame()
    if stages:
        stage_summary = exports['stage_timings'].groupby(['dataset','variant','phase','k','method','m','q','stage']).seconds.agg(
            count='count', median='median', q25=lambda v:v.quantile(.25), q75=lambda v:v.quantile(.75)).reset_index()
    stage_summary.to_csv(report/'stage_summary.csv',index=False)
    runtime_summary=pd.DataFrame()
    if comparisons:
        comparisons_frame=exports['workload_comparison']
        single= comparisons_frame[comparisons_frame.task != 'distinct_distance_queries']
        runtime_summary=single.groupby(['dataset','variant','k','method','m','q','task']).agg(
            cases=('method_seconds','count'),method_seconds_median=('method_seconds','median'),
            full_seconds_median=('full_seconds','median'),paired_speedup_median=('downstream_speedup','median'),
            composite_method_seconds_median=('composite_method_seconds','median'),
            composite_full_seconds_median=('composite_full_seconds','median')).reset_index()
        interval_rows=[]
        for key,g in single.groupby(['dataset','variant','k','method','m','q','task']):
            expected_draws=set(range(cfg['seeds'])) if key[3] in STOCHASTIC else {-1}
            complete=(len(g)==cfg['starts']*len(expected_draws) and set(g.seed)==expected_draws and set(g.start)==set(range(cfg['starts'])))
            item=dict(zip(['dataset','variant','k','method','m','q','task'],key))
            item.update(complete=complete,adequate_cases=int(g.task_result_adequate.sum()),valid_measurements=int(g.measurement_valid.sum()))
            if complete and g.measurement_valid.all() and g.task_result_adequate.all():
                matrix=np.log(g.pivot(index='seed',columns='start',values='downstream_speedup').to_numpy())
                if np.isfinite(matrix).all():
                    mean,low,high=bootstrap_crossed(matrix,draws=cfg['bootstrap'])
                    item.update(paired_speedup_geomean=float(np.exp(mean)),paired_speedup_ci_low=float(np.exp(low)),paired_speedup_ci_high=float(np.exp(high)))
            interval_rows.append(item)
        runtime_summary=runtime_summary.merge(pd.DataFrame(interval_rows),on=['dataset','variant','k','method','m','q','task'])
    runtime_summary.to_csv(report/'runtime_summary.csv',index=False)
    test = exports['quality_summary']
    if not test.empty:
        test = test[test.phase == 'test']
        tex_table(test, ['dataset','variant','k','method','m','q','complete','ari_mean',
            'r_phi_train_reconstructed_mean','pair_q95_mean','centroid_q95_mean'], report/'fidelity_native.tex')
        chosen=pd.DataFrame(selections)
        if len(chosen):
            chosen=chosen[chosen.selected].merge(test,on=['dataset','variant','k','method','m','q'],how='left')
            chosen.to_csv(report/'selected_test_quality.csv',index=False)
            tex_table(chosen,['dataset','variant','k','method','m','q','ari_mean','ari_ci_low','ari_ci_high',
                'r_phi_train_reconstructed_mean','pair_q95_mean'],report/'selected_fidelity.tex')
    if comparisons:
        tex_table(runtime_summary, ['dataset','variant','k','method','m','q','task',
            'method_seconds_median','full_seconds_median','paired_speedup_median'],report/'native_runtime.tex')
        setup=stage_summary[stage_summary.stage.isin(['compression','center_projection','numeric_preparation','model_construction'])]
        tex_table(setup,['dataset','variant','k','method','m','q','stage','median','q25','q75'],report/'compression_setup.tex')
    violations = sum(int(r[k]) for r in rows for k in ('certified_violations','pair_zero_violations','centroid_zero_violations'))
    violations += sum(int(w.get('dbscan_certified_violations',0)) for w in workloads)
    violations += sum(int(w.get('dbscan_certified_violations',0)) for w in profiles)
    full_errors = [f"{r['dataset']}:{r['phase']}:{r['k']}:{r['start']}" for r in rows if r['method']=='full' and
        any(r[m] is not None and abs(r[m]-1)>1e-8 for m in ('ari','agreement','r_phi','r_phi_train_reconstructed','projected_cost_ratio'))]
    source_errors = [name for name,digest in manifest['identity']['sources'].items() if
        not (root/'sources'/name).exists() or hashlib.sha256((root/'sources'/name).read_bytes()).hexdigest()!=digest]
    validation = dict(complete=bool(manifest['complete'] and all(c['complete'] for c in checks)),
        report_generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        jobs=checks, rows=len(rows), timing_samples=len(stages), query_samples=len(queries),
        certificate_or_zero_distance_violations=violations, full_control_errors=full_errors,
        source_snapshot_errors=source_errors,
        valid_observed_results=bool(rows) and not(violations or full_errors or source_errors or any(c['missing_artifacts'] for c in checks)),
        quality_not_guaranteed='A correctness pass does not mean compressed methods pass the fidelity thresholds.')
    validation['native_grid_ready_for_analysis']=bool(validation['complete'] and validation['valid_observed_results'] and
        cfg['profile'] in ('paper','balanced') and cfg['seeds']>=5 and cfg['starts']>=3)
    validation['all_paper_evidence_complete']=False
    validation['unconverged_kmeans_cases']=sum(bool(w.get('kmeans_hit_iteration_cap') or w.get('full_kmeans_hit_iteration_cap') or
        w.get('full_kmeans_iterations',0)>=cfg['kmeans_max_iter']) for w in workloads)
    validation['uninformative_dbscan_primary_cases']=sum(not w.get('dbscan_reference_partition_informative',0<w['full_dbscan_noise_fraction']<1) for w in workloads)
    atomic_json(report/'validation.json',validation)
    evidence = [
        dict(table='fidelity-a', coverage='New native held-out fixed-centroid fidelity and matched-m geometry; quantized/ZK fidelity remains separate'),
        dict(table='cert-a', coverage='New conditional float assignment and DBSCAN graph certificates; DBSCAN uses at most configured subset size'),
        dict(table='sensitivity-a', coverage='New k,m,q,seeds,starts native sweep; n,d,precision,backend proof scaling uses existing scoped campaigns'),
        dict(table='ablation-a', coverage='Full, PCA, Sparse JL, protected-only and protected+JL native ablation; no complete hybrid ZK ablation'),
        dict(table='zkperf-a', coverage='No proof generation in this suite; existing laptop_comparison_20261011 proof records remain separate; no complete fidelity-matched hybrid proof grid'),
        dict(table='ordering-a', coverage='Existing server RQ3 and laptop controls; new native timings provide no ordering proof evidence'),
        dict(table='circuit-cost', coverage='Symbolic manuscript table; measured total constraints only from verified ZK records, no measured per-gadget attribution'),
        dict(table='datasets-a / matrix-a / rq-metrics-a / metrics-a', coverage='Dataset metadata, inventory, selection thresholds and timing definitions exported here'),
        dict(table='notation / statements / threats / related-a / transforms / claims / validity-a', coverage='Conceptual/theoretical manuscript tables; require author review, not extra native timings')]
    atomic_json(report/'paper_coverage.json', dict(tables=evidence, all_paper_tables_numerically_complete=False,
        remaining=['Fidelity-matched protected+JL quantized verified proof grid',
                   'Hybrid statement, public data-dependent basis and security/theory alignment',
                   'Full native run measurements if this is a smoke/pilot/partial run',
                   'Controlled n/d scaling for the new hybrid if the manuscript claims it (legacy sweeps are separate)']))
    lines = ['# Compression and downstream experiment report','',
        f"Profile: **{cfg['profile']}**. Complete: **{validation['complete']}**. Observed rows: {len(rows)}.",
        '**Smoke fixtures are not paper results. Pilot intervals have one seed/start and cannot establish robustness.**','',
        'Every dataset uses a fixed 60/20/20 train/validation/test split. Centroids and full-SVD PCA fit on training rows only. '
        'Validation quality fixes the selected dimension before test results are measured. The corpus was previously explored; '
        'this split holds out model fitting, not a new independent study population.','',
        'Matched-dimension methods: Full reference, PCA, ordinary Sparse JL, protected centroid span plus residual Sparse JL, '
        'randomized PCA and the protected-only ablation. The protected head counts toward the total output dimension. '
        'No standardization or zero-column deletion is silently added; text zero-row removal is a separate sensitivity job.','',
        'Geometric quality is evaluated on validation/test rows. Compression times cover the entire preserved corpus. '
        'Downstream fixed-center assignment, distance queries, kNN and reclustering use the entire corpus. '
        'Held-out assignment excludes extraction/copying in v2 (legacy v1 timers included it and are flagged). '
        'KMeans reports both fresh point-index initialization and refinement from the shared trained centers. '
        'DBSCAN uses a deterministic bounded subset, with radius chosen '
        'from original validation geometry and the same radius/min_samples for all methods. Squared matrices use eps=radius².','',
        'DBSCAN also uses validation-derived radius quartiles and exports edge precision/recall/F1, core fractions, '
        'cluster counts and degenerate-reference flags. ARI against an all-noise or one-cluster reference is not evidence of useful clustering. '
        'Cap hits for either method prevent that cell from supporting a convergence-qualified speed claim.','',
        'Full ARI=1 measures agreement with itself. Ground-truth ARI/NMI are separate secondary columns. '
        'Original-space objective ratios are reported both with within-fold reconstructed centers and with training-reconstructed centers; '
        'the latter drives selection. Pair/centroid squared-distance distortion includes Q50/Q95/Q99/max, exceedance and zero-distance checks.','',
        'Each stage has raw individual measurements. Warmups are excluded. Arithmetic timing repetitions are not geometric seeds. '
        'Quality intervals independently resample projection seeds and centroid starts and are conditional, pointwise, exploratory 95% intervals. '
        'No across-dataset population inference or multiple-comparison-adjusted superiority claim is made.','',
        'PCA uses one full-SVD training fit for the largest requested feasible dimension and slices its components. '
        'Randomized PCA fits the same largest requested feasible dimension separately per projection draw, reusing each '
        'training-fitted map on validation and test. Both solver fit costs are shared sweep costs. '
        'Its shared sweep fit time is not a measured standalone fit at each m. Both Full and compressed fixed-centroid tasks '
        'include common centroid training when interpreting a cold execution. Hybrid distance tasks also require centroid training. '
        'Stage-sum composites are descriptive: they are not separately measured wall-clock totals.','',
        'RSS samples include the shared worker and resident inputs/models; they are not isolated allocator footprints. '
        'Representation byte counts are exact. Monitor intervals can miss short peaks. '
        'Iteration-cap hits, missing grid cells, failed jobs and resource limits remain in the output.','',
        'Raw rows, model parameters, source/feature/split indices, reference distances, labels, centroid reconstructions, '
        'pointwise error/certificate masks, kNN neighbors, KMeans initialization and DBSCAN graphs are retained. '
        'Queries retain every index set and timing; v2 retains two sampled original/projected answer matrices per case, '
        'and the remaining answers can be regenerated from saved inputs/maps. Artifact hashes bind every completed checkpoint. '
        'Source snapshots and original input hashes bind the run.','',
        '## Exported data','',
        '| File | Purpose |','|---|---|',
        '| quality_rows.csv / quality_summary.csv | All configurations; crossed-axis fidelity intervals |',
        '| stage_timings.csv / stage_summary.csv | Construction, compression, preparation and individual tasks |',
        '| model_training.csv | PCA fit, centroid learning and protected-basis extraction |',
        '| workload_comparison.csv / query_timings.csv | Paired Full controls and distinct-query reuse costs |',
        '| workload_quality.csv | kNN recall, reclustering ARI/objective, DBSCAN graphs/core/noise |',
        '| selected_dimensions.csv | Frozen validation-only operating points, including failed selection |',
        '| validation.json / paper_coverage.json | Completeness, correctness and manuscript evidence gaps |','',
        '## Manuscript coverage','',
        *[f"- `{entry['table']}`: {entry['coverage']}" for entry in evidence],'',
        '**Native compute time is not witness/prove/verify time. Existing proof evidence must retain its original dataset, '
        'precision, statement, backend and resource-limit scope. This suite alone cannot complete every empirical paper table.**','']
    (report/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    if not test.empty:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        primary = test[(test.variant=='primary') & (test.k==min(cfg['k']))]
        for metric, label in [('pair_q95_mean','95th percentile squared-distance distortion'),
                              ('ari_mean','Agreement ARI with Full')]:
            fig, axes = plt.subplots(2,4,figsize=(15,7),layout='constrained')
            for ax, dataset in zip(axes.flat,manifest['identity']['datasets']):
                data = primary[primary.dataset==dataset]
                for (method,q), group in data.groupby(['method','q']):
                    if method in ('full','protected_only'): continue
                    group=group.sort_values('m')
                    ax.plot(group.m,group[metric],marker='o',markersize=3,label=f'{method} q={q}')
                    prefix=metric.removesuffix('_mean')
                    if prefix+'_ci_low' in group:
                        ax.fill_between(group.m,group[prefix+'_ci_low'],group[prefix+'_ci_high'],alpha=.12)
                ax.set_title(dataset); ax.set_xlabel('Total output dimension m'); ax.set_ylabel(label); ax.grid(alpha=.2)
            handles,labels=axes.flat[0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='outside lower center',ncol=4)
            fig.suptitle(f"{cfg['profile']} — test quality (k={min(cfg['k'])}); fixed preserved corpora")
            for ext in ('png','pdf'): fig.savefig(report/f'{metric}.{ext}',dpi=150)
            plt.close(fig)
        if not stage_summary.empty:
            fig,axes=plt.subplots(2,4,figsize=(15,7),layout='constrained')
            target=cfg['timing_m'][0]
            for ax,dataset in zip(axes.flat,manifest['identity']['datasets']):
                data=stage_summary[(stage_summary.dataset==dataset)&(stage_summary.variant=='primary')&
                    (stage_summary.phase=='test')&(stage_summary.k==min(cfg['k']))&
                    (stage_summary.m==target)&(stage_summary.method.isin(['pca','sparse_jl','protected_jl']))]
                methods=sorted(data.method.unique()); bottom=np.zeros(len(methods))
                if not methods:
                    ax.set_title(dataset);ax.text(.5,.5,f'No valid compressed cell at m={target}',
                        ha='center',va='center',transform=ax.transAxes);ax.set_axis_off();continue
                for stage in ('model_construction','compression','center_projection','numeric_preparation'):
                    values=np.array([data[(data.method==method)&(data.stage==stage)]['median'].median() for method in methods])
                    values=np.nan_to_num(values);ax.bar(methods,values,bottom=bottom,label=stage);bottom+=values
                ax.set_title(dataset);ax.set_ylabel('Seconds (stage medians)');ax.tick_params(axis='x',labelrotation=25)
            handles,labels=axes.flat[0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='outside lower center',ncol=4)
            fig.suptitle(f"{cfg['profile']} — conditional setup at m={target}; training reported separately")
            for ext in ('png','pdf'): fig.savefig(report/f'compression_stages.{ext}',dpi=150)
            plt.close(fig)
    return validation


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); print(json.dumps(build_report(args.out)))


if __name__=='__main__': main()
