"""Readable native-paper tables from a completed, validated report export.

Leaves raw experiment data and frozen campaign sources unchanged.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.paper_compression import STOCHASTIC,bootstrap_crossed
from experiments.laptop_comparison import atomic_json
from analysis.paper_compression_report import tex_table

GROUP=['dataset','variant','k','method','m','q']
METHOD_NAMES={'full':'Full','pca':'PCA (full SVD)','pca_randomized':'PCA (randomized)',
    'sparse_jl':'Sparse JL','protected_jl':'Protected JL','protected_only':'Protected only'}


def ci_summary(frame,metrics,config,extra=()):
    """Conditional crossed-axis intervals; preserve undefined/incomplete cells."""
    out=[]
    for keys,group in frame.groupby(GROUP+list(extra),dropna=False):
        record=dict(zip(GROUP+list(extra),keys))
        draws=set(range(config['seeds'])) if record['method'] in STOCHASTIC else {-1}
        complete=(len(group)==len(draws)*config['starts'] and set(group.seed)==draws and
            set(group.start)==set(range(config['starts'])) and not group.duplicated(['seed','start']).any())
        record.update(cases=len(group),complete=complete)
        if 'dbscan_reference_partition_informative' in group:
            record['informative_dbscan_cases']=int(group.dbscan_reference_partition_informative.sum())
        for metric in metrics:
            if metric not in group:continue
            values=pd.to_numeric(group[metric],errors='coerce')
            finite=np.isfinite(values)
            record[metric+'_defined_cases']=int(finite.sum())
            record[metric+'_mean']=float(values[finite].mean()) if finite.any() else np.nan
            if complete and finite.all():
                matrix=group.pivot(index='seed',columns='start',values=metric).to_numpy(dtype=float)
                mean,lo,hi=bootstrap_crossed(matrix,draws=config['bootstrap'])
                record.update({metric+'_mean':mean,metric+'_ci_low':lo,metric+'_ci_high':hi})
        out.append(record)
    return pd.DataFrame(out)


def interval(row,prefix,digits=3):
    value=row.get(prefix+'_mean',row.get(prefix+'_geomean',np.nan))
    if pd.isna(value):return '--'
    lo,hi=row.get(prefix+'_ci_low',np.nan),row.get(prefix+'_ci_high',np.nan)
    fmt=lambda number:format(float(number),f'.{digits}g')
    return f'{fmt(value)} [{fmt(lo)}, {fmt(hi)}]' if pd.notna(lo) and pd.notna(hi) else fmt(value)


def readable(frame,columns,intervals=None):
    result=pd.DataFrame(index=frame.index)
    intervals=intervals or {}
    for column,label in columns:
        if column in intervals:result[label]=frame.apply(lambda row:interval(row,intervals[column]),axis=1)
        elif column in frame:result[label]=frame[column]
    if 'Method' in result:result['Method']=result.Method.map(METHOD_NAMES).fillna(result.Method)
    return result


def export(root):
    root=root.resolve();report=root/'report';target=root/'paper_tables';target.mkdir(exist_ok=True)
    manifest=json.loads((root/'manifest.json').read_text());validation=json.loads((report/'validation.json').read_text())
    if not (manifest['complete'] and validation['complete'] and validation['valid_observed_results'] and
            validation.get('native_grid_ready_for_analysis')):
        raise ValueError('Complete, validated replicated native grid required; pilot/smoke data are not paper evidence')
    cfg=manifest['identity']['configuration'];sources={}
    if cfg['version']!='paper-compression-v2':raise ValueError('Corrected v2 timing protocol required')
    def read(name):
        path=report/(name+'.csv');sources[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.read_csv(path)
    datasets=read('datasets');quality=read('quality_summary');quality=quality[quality.phase=='test'].copy()
    selection=read('selected_dimensions');stage=read('stage_summary');training=read('model_training')
    runtime=read('runtime_summary');workloads=read('workload_quality');profiles=read('dbscan_profile_quality')
    comparisons=read('workload_comparison')
    dataset_map=datasets.set_index(['dataset','variant'])
    for frame in [quality,runtime,workloads,profiles]:
        frame['original_d']=[int(dataset_map.loc[(r.dataset,r.variant),'d']) for r in frame.itertuples()]
        frame['retained_dimension_fraction']=frame.m/frame.original_d
    selected=selection.merge(quality,on=GROUP,how='left',validate='one_to_one',suffixes=('','_test'))
    t=cfg['thresholds']
    selected['test_fidelity_rule_pass']=selected['selected'] & selected['complete'].eq(True)
    for column,threshold,lower in [('ari_ci_low',t['ari'],True),
        ('r_phi_train_reconstructed_ci_high',t['r_phi'],False),('pair_q95_ci_high',t['distortion'],False),
        ('centroid_q95_ci_high',t['distortion'],False),('projected_cost_relative_error_ci_high',t['projected_cost_error'],False)]:
        selected['test_fidelity_rule_pass'] &= selected[column].ge(threshold) if lower else selected[column].le(threshold)
    selected['selection_status']=np.where(selected.selected,'Validation selected','No grid cell qualified')
    workload_metrics=['knn_recall','kmeans_ari','kmeans_r_phi','kmeans_refine_ari','kmeans_refine_r_phi',
        'dbscan_ari','graph_f1','dbscan_noise_fraction']
    wl=ci_summary(workloads,workload_metrics,cfg)
    graph=ci_summary(profiles,['dbscan_ari','graph_precision','graph_recall','graph_f1',
        'dbscan_core_agreement','dbscan_noise_fraction'],cfg,extra=['profile'])
    queries=comparisons[comparisons.task.eq('distinct_distance_queries') & comparisons.queries.eq(cfg['queries'])].copy()
    queries['query_only_speedup']=queries.full_seconds/queries.method_seconds
    queries['conditional_reuse_speedup']=queries.conditional_full_seconds/queries.conditional_method_seconds
    queries['cold_method_seconds']=queries.method_model_fit_seconds+queries.conditional_method_seconds
    queries['cold_full_seconds']=queries.full_model_fit_seconds+queries.conditional_full_seconds
    queries['cold_speedup']=queries.cold_full_seconds/queries.cold_method_seconds
    distance=ci_summary(queries,['method_seconds','full_seconds','query_only_speedup',
        'conditional_method_seconds','conditional_full_seconds','conditional_reuse_speedup',
        'cold_method_seconds','cold_full_seconds','cold_speedup'],cfg)
    distance['queries']=cfg['queries']
    speed_columns=[c for c in distance if c not in GROUP and c!='complete']
    operating=selected[selected.selected].merge(distance[GROUP+speed_columns],on=GROUP,how='left',validate='one_to_one')
    operating=operating.merge(wl[GROUP+['knn_recall_mean','knn_recall_ci_low','knn_recall_ci_high',
        'kmeans_ari_mean','kmeans_ari_ci_low','kmeans_ari_ci_high','kmeans_refine_ari_mean',
        'kmeans_refine_ari_ci_low','kmeans_refine_ari_ci_high']],on=GROUP,how='left',validate='one_to_one')
    operating['reuse_speed_ci_above_one']=operating.conditional_reuse_speedup_ci_low.gt(1)
    operating['cold_speed_ci_above_one']=operating.cold_speedup_ci_low.gt(1)
    # Pool native stage medians separately; never silently substitute screening
    # compression for the whole-corpus transform measured in workloads.
    stage_test=stage[stage.phase.eq('test')]
    setup=stage_test.pivot(index=GROUP,columns='stage',values='median').reset_index()
    keep=['model_construction','compression','center_projection','numeric_preparation','point_norm_preparation',
        'full_numeric_preparation','full_point_norm_preparation']
    setup=setup[GROUP+[key for key in keep if key in setup]].copy()
    setup['whole_corpus_timing_available']=setup.compression.notna()
    for column in keep:
        if column in setup:setup.rename(columns={column:column+'_seconds'},inplace=True)
    train=training.groupby(['dataset','variant','stage','seed','fit_key'],dropna=False).agg(
        measured_samples=('seconds','count'),median_seconds=('seconds','median'),min_seconds=('seconds','min'),
        max_seconds=('seconds','max'),iterations_max=('iterations','max')).reset_index()
    fitted=quality[quality.method.eq('pca')].groupby(['dataset','variant']).m.max()
    train['shared_pca_fitted_m']=[int(fitted.loc[(r.dataset,r.variant)]) if 'pca' in r.stage else np.nan for r in train.itertuples()]
    reference=[]
    for meta in datasets.itertuples():
        for method in METHOD_NAMES:
            candidates=quality[(quality.dataset==meta.dataset)&(quality.variant==meta.variant)&(quality.k==5)&(quality.method==method)]
            if method not in ('full','protected_only'):candidates=candidates[(candidates.m==128)&(candidates.q==(1 if method in ('sparse_jl','protected_jl') else 0))]
            if len(candidates)==1:
                row=candidates.iloc[0].to_dict();row['status']='Measured';reference.append(row)
            elif len(candidates)==0:
                reason='m >= original d' if 128>=meta.d else 'PCA training rank limit' if method in ('pca','pca_randomized') and 128>meta.train_rows-1 else 'Not measured'
                reference.append(dict(dataset=meta.dataset,variant=meta.variant,k=5,method=method,m=128,
                    q=1 if method in ('sparse_jl','protected_jl') else 0,status=reason))
            else:raise ValueError('Duplicate reference cell')
    reference=pd.DataFrame(reference)
    base=[('dataset','Dataset'),('variant','Variant'),('k','k'),('method','Method'),('m','m'),('q','q')]
    fidelity_cols=base+[('ari_mean','ARI [95% CI]'),('pair_q95_mean','Pair Q95 [95% CI]'),
        ('centroid_q95_mean','Center Q95 [95% CI]'),('r_phi_train_reconstructed_mean','Original objective ratio [95% CI]')]
    fidelity_intervals={'ari_mean':'ari','pair_q95_mean':'pair_q95','centroid_q95_mean':'centroid_q95',
        'r_phi_train_reconstructed_mean':'r_phi_train_reconstructed'}
    tables=[]
    def add(name,title,raw,display,note):tables.append(dict(name=name,title=title,raw=raw,display=display,note=note))
    add('01_datasets','Dataset sizes and splits',datasets,readable(datasets,[('dataset','Dataset'),('variant','Variant'),
        ('n','Rows'),('d','Features'),('train_rows','Train'),('validation_rows','Validation'),('test_rows','Test'),('zero_rows','Zero rows')]),
        'These are the preserved server corpora, not the full public datasets. Source hashes and preprocessing are in the CSV. Text nonzero removes exactly 114 all-zero rows.')
    add('02_selection','Validation-selected configurations',selected,readable(selected,base+[('selection_status','Selection'),
        ('test_fidelity_rule_pass','Test rule passes'),('ari_mean','Test ARI [95% CI]'),('pair_q95_mean','Test pair Q95 [95% CI]')],fidelity_intervals),
        'Choose m/q using validation only. Failed choices remain visible. Test rule uses the same five CI thresholds on fresh held-out maps; an unselected row has no selected-point test value.')
    add('03_fidelity_all','Held-out fidelity: all measured configurations',quality,readable(quality,fidelity_cols,fidelity_intervals),
        'ARI measures agreement with fixed-center Full assignments, not ground-truth accuracy or fresh clustering. Q95 errors concern squared distances. The head counts toward m. Full and protected-only use their actual, unequal dimensions.')
    add('04_fidelity_m128_k5','Reference comparison: k=5, total m=128, JL q=1',reference,
        readable(reference,fidelity_cols+[('status','Status')],fidelity_intervals),
        'Compressed baselines use total m=128. Full uses d; protected-only uses the learned head rank. Infeasible cells are explicit: DBSCAN fixture d=128 and Arcene PCA rank <=59. No substitute dimension is presented as a matched comparison.')
    add('05_compression_setup','Construction, compression and preparation',setup,readable(setup,base+[
        ('model_construction_seconds','Construction (s)'),('compression_seconds','Whole compression (s)'),
        ('center_projection_seconds','Center compression (s)'),('numeric_preparation_seconds','Point+center norms (s)'),
        ('point_norm_preparation_seconds','Point norms only (s)'),('whole_corpus_timing_available','Whole-corpus timed')]),
        'Pooled stage medians, in seconds. Training is separate. Blank whole-corpus timings were not measured at those screening-only cells. Combined norm preparation and point-only preparation are alternative boundaries, not additive costs.')
    add('06_model_training','Training and protected-basis extraction',train,readable(train,[('dataset','Dataset'),('variant','Variant'),
        ('stage','Stage'),('seed','Draw'),('fit_key','Fit'),('shared_pca_fitted_m','Shared PCA fitted m'),
        ('measured_samples','Samples'),('median_seconds','Median (s)'),('iterations_max','Iterations')]),
        'PCA trains once to the largest feasible requested m and reuses prefixes. Its sweep fit is not a separately measured fit at each smaller m. Centroid learning is shared by fixed-center tasks; hybrid distance tasks also need it.')
    wl_cols=base+[('knn_recall_mean','kNN recall [95% CI]'),('kmeans_ari_mean','Fresh KMeans ARI [95% CI]'),
        ('kmeans_refine_ari_mean','Refinement ARI [95% CI]'),('graph_f1_mean','DBSCAN edge F1 [95% CI]'),
        ('informative_dbscan_cases','Informative DBSCAN cases'),('cases','Cases')]
    wl_intervals={column:column.removesuffix('_mean') for column,_ in wl_cols if column.endswith('_mean')}
    add('07_downstream_quality','Downstream quality',wl,readable(wl,wl_cols,wl_intervals),
        'Fresh KMeans uses common point-index initialization; refinement starts from learned centers. They answer different questions. Primary DBSCAN uses at most 256 fixed rows; trivial reference partitions are counted, not treated as useful clustering evidence.')
    add('08_downstream_runtime','Downstream runtime and paired speedups',runtime,readable(runtime,base+[('task','Task'),
        ('method_seconds_median','Method (s)'),('full_seconds_median','Paired Full (s)'),
        ('paired_speedup_geomean','Speedup [95% CI]'),('composite_method_seconds_median','Cold composite method (s)'),
        ('composite_full_seconds_median','Cold composite Full (s)'),('adequate_cases','Adequate cases'),('cases','Cases')],
        {'paired_speedup_geomean':'paired_speedup'}),
        'Speedup >1 favors the compressed method. These are task-only paired speedups; compression/training are separate. Cold composites sum required stage medians and shared sweep fit costs. DBSCAN combined setup is counted once; stage-only rows have no cold composite. Missing speedup CIs are not a qualified success.')
    dcols=base+[('queries','Queries'),('query_only_speedup_mean','Query-only speedup [95% CI]'),
        ('conditional_method_seconds_mean','Reuse method (s)'),('conditional_full_seconds_mean','Reuse Full (s)'),
        ('conditional_reuse_speedup_mean','Reuse speedup [95% CI]'),('cold_method_seconds_mean','Cold method (s)'),
        ('cold_full_seconds_mean','Cold Full (s)'),('cold_speedup_mean','Cold speedup [95% CI]')]
    add('09_distance_reuse','Thirty-two distinct distance queries',distance,readable(distance,dcols,
        {column:column.removesuffix('_mean') for column,_ in dcols if 'speedup' in column}),
        'Means of per-case composites; CI intervals use independently resampled draw/start axes. Query-only excludes setup. Reuse includes construction, whole compression and point norms. Cold also includes required model fitting; Full distances need no center fitting. Every method receives identical query indices; final answers are not cached. No extrapolated break-even claim.')
    gcols=base+[('profile','Radius profile'),('graph_precision_mean','Edge precision [95% CI]'),('graph_recall_mean','Edge recall [95% CI]'),
        ('graph_f1_mean','Edge F1 [95% CI]'),('dbscan_ari_mean','ARI [95% CI]'),('informative_dbscan_cases','Informative cases'),('cases','Cases')]
    add('10_dbscan_profiles','DBSCAN radius sensitivity',graph,readable(graph,gcols,
        {column:column.removesuffix('_mean') for column,_ in gcols if column.endswith('_mean')}),
        'Radii are derived from original validation geometry and held fixed across methods. The synthetic radius=1 all-noise fixture remains in the primary table. Undefined graph ratios remain blank; incomplete cells do not receive CIs.')
    ocols=base+[('test_fidelity_rule_pass','Test fidelity passes'),('pair_q95_mean','Pair Q95 [95% CI]'),
        ('knn_recall_mean','kNN recall [95% CI]'),('kmeans_ari_mean','Fresh KMeans ARI [95% CI]'),
        ('conditional_reuse_speedup_mean','32-query reuse speedup [95% CI]'),('cold_speedup_mean','Cold speedup [95% CI]')]
    add('11_selected_operating_points','Quality-selected operating points: fidelity and performance together',operating,
        readable(operating,ocols,{column:column.removesuffix('_mean') for column,_ in ocols if column.endswith('_mean')}),
        'Dimensions come exclusively from validation selection; test failures remain flagged. These are matched-fidelity operating points, not equal-m comparisons. Reuse assumes learned model/basis available and includes construction, compression and norms. Cold includes required model fitting. Speedup >1 favors compression; fresh-clustering quality is a separate endpoint.')
    missing=['Fidelity-matched hybrid quantized models and verified ZK proof records',
        'Manuscript alignment for the public data-dependent protected basis and security/theory claims',
        'Complete controlled native n/d scaling if claimed; its small implementation checks are not paper data']
    counts=selection.groupby('method').selected.agg(['sum','count'])
    summary='; '.join(f"{METHOD_NAMES[m]}: {int(r['sum'])}/{int(r['count'])}" for m,r in counts.iterrows())
    selected_count=int(selected.selected.sum());test_pass_count=int(selected.test_fidelity_rule_pass.sum())
    hjl=selected[selected.method.eq('protected_jl')]
    hjl_selected_count=int(hjl.selected.sum());hjl_test_pass_count=int(hjl.test_fidelity_rule_pass.sum())
    sections=[];files=[]
    for table in tables:
        name=table['name'];raw=table['raw'];display=table['display']
        raw.to_csv(target/(name+'.csv'),index=False)
        tex_table(display,list(display.columns),target/(name+'.tex'))
        files.extend([name+'.csv',name+'.tex'])
        formatted=display.to_html(index=False,border=0,na_rep='--',float_format=lambda x:f'{x:.5g}',classes='data-table')
        sections.append(f'<section><h2>{html.escape(table["title"])}</h2><p>{html.escape(table["note"])}</p>'
            f'<p><a href="{name}.csv">Numeric CSV</a> · <a href="{name}.tex">LaTeX</a> · {len(raw)} rows</p>'
            '<label>Filter this table <input type="search" placeholder="Dataset, method or task" oninput="filterTable(this)"></label>'
            f'<div class="table-wrap">{formatted}</div></section>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Native compression results — balanced study</title><style>
body{font:15px/1.55 system-ui,sans-serif;color:#17283a;margin:0;background:#f4f6f8}main{max-width:1500px;margin:auto;padding:28px}
h1,h2{line-height:1.25}section,.intro{background:white;padding:22px;margin:20px 0;border:1px solid #dce3e9;border-radius:8px}
h2{font-size:21px}.table-wrap{overflow:auto;max-height:650px;border:1px solid #e1e6eb}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}
th{background:#e9f0f6;position:sticky;top:0;text-align:left;z-index:1}th,td{padding:8px 10px;border-bottom:1px solid #e2e7ec}tbody tr:nth-child(even){background:#f8fafc}
input{padding:8px;width:260px;margin:0 0 12px 10px;border:1px solid #a8b7c6;border-radius:4px}a{color:#18599c}code{background:#eaf0f5;padding:2px 5px}
@media print{main{padding:0}body{background:white}section{border:0;break-before:page}.table-wrap{max-height:none;overflow:visible}input,label{display:none}table{font-size:9px;white-space:normal}th{position:static}}
</style><main><h1>Native compression and downstream results</h1>'''
    page+=f'<div class="intro"><p><b>Complete balanced grid: {validation["rows"]:,} cases; {cfg["seeds"]} projection draws × {cfg["starts"]} centroid starts; {cfg["threads"]} CPU threads.</b></p>'
    page+='<p>Preserved offline datasets; disjoint 60/20/20 split; validation-only selection. Exploratory pointwise 95% intervals are conditional on each previously studied corpus. Timing repetitions are not independent seeds. Native numerical costs do not measure proof generation.</p>'
    page+='<p><b>Validation-qualified configurations:</b> '+html.escape(summary)+'. These counts concern the joint fixed-center/geometry rule, not every downstream task.</p>'
    page+=f'<p>The same rule passes on held-out test data for <b>{test_pass_count}/{selected_count} selected configurations</b>, '
    page+=f'including <b>{hjl_test_pass_count}/{hjl_selected_count} selected protected-JL configurations</b>. All failures remain visible in the selection table.</p>'
    page+=f'<p>Integrity checks: {validation["certificate_or_zero_distance_violations"]} certificate/zero-distance violations; no Full-control or source-snapshot errors. '
    page+=f'{validation["uninformative_dbscan_primary_cases"]} primary DBSCAN cases have uninformative references and remain flagged.</p>'
    page+='<p><a href="../report/report.md">Protocol report</a> · <a href="../report/validation.json">Integrity validation</a> · <a href="../report/">Full CSV exports and figures</a></p></div>'
    page+=''.join(sections)+'<section><h2>Paper evidence still outstanding</h2><ul>'+''.join('<li>'+html.escape(item)+'</li>' for item in missing)+'</ul></section></main>'
    page+='<script>function filterTable(input){const term=input.value.toLowerCase();input.closest("section").querySelectorAll("tbody tr").forEach(row=>row.hidden=!row.textContent.toLowerCase().includes(term));}</script></html>'
    (target/'index.html').write_text(page,encoding='utf-8');files.append('index.html')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    index=dict(created_utc=datetime.now(timezone.utc).isoformat(),complete=True,experiment_rows=validation['rows'],
        source_exports_sha256=sources,source_manifest_sha256=digest(root/'manifest.json'),validation_sha256=digest(report/'validation.json'),
        generator_sha256=digest(Path(__file__)),tables=[dict(name=t['name'],rows=len(t['raw'])) for t in tables],
        output_sha256={name:digest(target/name) for name in files},all_paper_evidence_complete=False,remaining=missing)
    qualified=operating[operating.method.eq('protected_jl') & operating.test_fidelity_rule_pass]
    reuse_wins=int(qualified.reuse_speed_ci_above_one.sum());cold_wins=int(qualified.cold_speed_ci_above_one.sum())
    setting_count=len(datasets)*len(cfg['k'])
    pca_counts={method:int(selection[selection.method.eq(method)].selected.sum()) for method in ['pca','pca_randomized']}
    failures=operating[~operating.test_fidelity_rule_pass]
    failure_labels='; '.join(f"{METHOD_NAMES[r.method]} on {r.dataset}/{r.variant}, k={r.k}" for r in failures.itertuples()) or 'None'
    findings=(f'# {cfg["profile"].capitalize()} native study: findings\n\nAll {validation["rows"]:,} cases completed. '
        f'The joint fidelity rule selected protected JL in {hjl_selected_count}/{setting_count} dataset/k settings; '
        f'{hjl_test_pass_count}/{hjl_selected_count} selected settings also passed that rule on held-out test data. '
        f'Full-SVD PCA qualified in {pca_counts["pca"]}/{setting_count}, and randomized PCA in {pca_counts["pca_randomized"]}/{setting_count}, validation settings. '
        'These counts concern fixed-center labels, original objective and squared-distance fidelity; '
        'they do not imply success at fresh clustering or every downstream task.\n\n'
        f'Among the {len(qualified)} held-out-qualified protected-JL operating points, {reuse_wins} had an exploratory '
        f'32-query reuse speedup CI entirely above one; {cold_wins} had a cold speedup CI entirely above one. '
        'Reuse assumes the learned model and protected basis already exist, and includes map construction, compression and point norms. '
        'Cold additionally includes model fitting. Query-only timings exclude those costs. Faster reduced-space kernels therefore do not establish '
        'a faster complete cold pipeline. See table 11 before interpreting selected configurations as performance wins.\n\n'
        f'The same held-out rule passed for {test_pass_count}/{selected_count} selected configurations across all methods. '
        f'Selected settings failing the held-out rule: {failure_labels}. '
        'The dataset table records small-sample limits. Primary DBSCAN includes degenerate references; radius sensitivity and edge F1 must accompany its ARI.\n\n'
        'These are native numerical measurements with pointwise exploratory intervals on previously studied fixed corpora. '
        'They cannot replace verified hybrid ZK proof measurements or establish a universal superiority claim over PCA or Full.\n')
    (target/'findings.md').write_text(findings,encoding='utf-8')
    (target/'README.md').write_text('# Balanced native paper tables\n\nOpen index.html for readable, searchable tables. Each family also has numeric CSV and LaTeX.\n\n'
        +'The original detailed CSV exports and figures remain in ../report/. Compression, model training, preparation and task-only computation are separate. '
        +'Cold and reuse composites are descriptive sums, not directly timed whole pipelines. PCA uses the recorded largest-m sweep fit.\n\n'
        +'All dimensions and q values are retained. The m=128 reference table makes infeasible cells explicit. All failed validation selections remain visible. '
        +'This is the native numerical study; existing ZK evidence retains its original scope.\n',encoding='utf-8')
    for name in ['README.md','findings.md']:index['output_sha256'][name]=digest(target/name)
    atomic_json(target/'export_manifest.json',index)
    result=dict(complete=True,experiment_rows=validation['rows'],tables=len(tables),folder=str(target),validation_selected=summary)
    print(json.dumps(result));return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();export(args.out)


if __name__=='__main__':main()
