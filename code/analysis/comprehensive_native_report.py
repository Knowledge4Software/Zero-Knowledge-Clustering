"""Integrity audit and English report for the full current-method campaign."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from commitjl.reusable_native import file_hash
from experiments.laptop_comparison import atomic_json


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])


def audit(root):
    manifest=json.loads((root/'manifest.json').read_text())
    rows=[json.loads(p.read_text()) for p in sorted((root/'rows').glob('*.json'))]
    assert len(rows)==manifest['completed_cases']
    if manifest['complete']:assert len(rows)==manifest['expected_cases']
    for relative,digest in manifest['identity']['sources'].items():
        assert file_hash(root/'sources'/relative)==digest,relative
    checked={};seen=set();caps=0
    for r in rows:
        key=tuple(r[a] for a in ['dataset','k','start','method','m','seed'])
        assert key not in seen;seen.add(key)
        for relative,digest in r['artifacts'].items():
            if relative not in checked:checked[relative]=file_hash(root/relative)
            assert checked[relative]==digest,relative
        case=f"{r['dataset']}_k{r['k']}_s{r['start']}_{r['method']}_m{r['m']}_seed{r['seed']}"
        t=json.loads((root/'cases'/case/'timings.json').read_text())
        setup=r['model_fit_seconds']+r['construction_seconds']+r['compression_seconds']
        assert np.isclose(r['cold_fresh_seconds'],setup+t['kmeans']['median'],rtol=1e-12)
        assert np.isclose(r['cold_query_seconds'],setup+t['point_norm_preparation']['median']+r['query_seconds'],rtol=1e-12)
        assert np.isclose(r['cold_knn_seconds'],setup+t['point_norm_preparation']['median']+t['knn']['median'],rtol=1e-12)
        assert np.isclose(r['cold_dbscan_seconds'],setup+t['point_norm_preparation']['median']+
            t['dbscan_pair_build']['median']+t['dbscan_fit']['median'],rtol=1e-12)
        needs_centers=r['method'] not in ['protected_jl','protected_plus12','protected_only']
        centers=r['centroid_training_seconds'] if needs_centers else 0.
        assert np.isclose(r['cold_assignment_seconds'],setup+centers+t['center_projection']['median']+
            t['numeric_preparation']['median']+t['assignment']['median'],rtol=1e-12)
        assert np.isclose(r['cold_refine_seconds'],setup+centers+t['center_projection']['median']+
            t['kmeans_refine']['median'],rtol=1e-12)
        ds=root/'datasets'/r['dataset'];fit_root=ds/f"k{r['k']}"/'models'
        if r['method'].startswith('pca_'):
            assert r['pca_spec']['fitted_m']==r['m']==r['pca_spec']['options']['n_components']
            fit=json.loads((ds/'models'/f"{r['method']}_m{r['m']}_seed{r['seed']}.json").read_text())
            expected=fit['timing']['median']
        elif r['method'] in ['protected_jl','protected_plus12','protected_only']:
            name='enriched' if r['method']=='protected_plus12' else 'basis'
            fit=json.loads((fit_root/f"{name}_s{r['start']}.json").read_text())
            expected=r['centroid_training_seconds']+fit['timing']['median']
            assert r['fixed_test_ari']==1.
        else:expected=0.
        assert np.isclose(r['model_fit_seconds'],expected,rtol=1e-12)
        assert r['pair_error']['zero_violations']==0
        assert r['centroid_error']['zero_violations']==0
        if r['method']=='full':assert r['construction_seconds']==r['compression_seconds']==0.
        caps+=int(r['any_iteration_cap'])
    result=dict(complete=True,campaign_complete=manifest['complete'],cases_checked=len(rows),
        unique_artifacts_checked=len(checked),source_snapshots_verified=True,
        dedicated_pca_fits_verified=True,task_specific_cost_accounting_verified=True,
        capped_cases=caps,scope='All case/source/artifact hashes and cost definitions; no fresh clustering reruns')
    atomic_json(root/'audit.json',result);return result


def build(root,allow_partial=False):
    manifest=json.loads((root/'manifest.json').read_text())
    if not manifest['complete'] and not allow_partial:raise ValueError('Campaign is incomplete')
    checks=audit(root)
    summary=pd.read_csv(root/'summary.csv');comp=pd.read_csv(root/'comparisons.csv')
    rows=pd.read_csv(root/'rows.csv')
    stages=rows.groupby(['dataset','k','method','m'],as_index=False)[[
        'model_fit_seconds','construction_seconds','compression_seconds','fresh_seconds',
        'query_seconds','knn_seconds','refine_seconds','dbscan_pair_seconds','dbscan_fit_seconds']].mean()
    stages.to_csv(root/'stage_means.csv',index=False)
    joint=comp[comp.competitor.str.startswith('pca_')&comp.joint_pointwise_win.eq(True)]
    focus=summary[summary.method.isin(['pca_randomized_fast','protected_plus12'])]
    lines=['# Full native comparison of the optimized current method','',
        f'Status: {"complete" if manifest["complete"] else "PARTIAL"}. '+
        f'{manifest["completed_cases"]} / {manifest["expected_cases"]} cases saved. '+
        f'Latest invocation compute wall time: {manifest.get("elapsed_seconds",0)/3600:.2f} hours.','',
        'All eight original archived corpus subsets are evaluated, plus the nonzero-text sensitivity set. '+
        'No extra row or feature subsampling is used. k=5 and 10; three new projection draws crossed with three new centroid/clustering starts. '+
        'Every PCA is fitted separately at its actual feasible output rank. Full SVD, default randomized and fast two-power-iteration PCA are retained. '+
        'An additional matched-rank m=59 control is provided for Arcene; PCA cannot be fitted at 128/256/512 on its 60 training rows.', '',
        'The current method is the optimized, unchanged centroid span +12 residual PCA directions + canonical q=1 Sparse JL residual. '+
        'Plain Sparse JL, original Protected JL and protected-only are explicit ablations. '+
        'Candidate text fixes are measured separately and are not silently substituted into this campaign.', '',
        '## Dataset and dimension coverage','']
    plan=json.loads((root/'plan.json').read_text())
    lines.append(table(['Dataset','Rows','Features','Training rows','Widths','Widths without PCA match','Cases'],
        [[p['dataset'],p['n'],p['d'],p['training_rows'],str(p['widths']),str(p['no_pca_match']),p['cases']] for p in plan['datasets']]))
    recovery_path=root/'recovery_latest.json'
    if recovery_path.exists():
        recovery=json.loads(recovery_path.read_text())
        lines+=['','## Shutdown recovery','',
            f'The interrupted campaign retained {recovery["verified_cases"]} fully verified cases. '+
            f'Recovery checked {recovery["unique_artifacts_checked"]} referenced artifact hashes and '+
            f'{recovery["fit_caches_checked"]} fit-cache records. '+
            'Damaged checkpoints and dependent training caches were quarantined and recomputed. '+
            'The frozen numerical implementation and experiment configuration were retained.', '',
            f'Logged elapsed time before the interruption: {recovery["interrupted_elapsed_seconds"]/3600:.2f} hours. '+
            'The invocation time above covers the resumed process; it does not include that earlier interval. '+
            'Checkpoint writing now flushes contents before replacement and retains the previous valid manifest. '+
            'This evidence I/O is outside the measured numerical task stages. See recovery_latest.json and execution_io.json.','']
    lines+=['','## Current method versus fast dedicated-rank PCA','',
        'Fresh ARI compares whole-corpus clustering with a matched Full run. It is not ground-truth classification accuracy. '+
        'Cold time includes all learning needed by that task, model construction, corpus compression and the task itself. '+
        'Fitting caches retain and charge their complete measured cost in every cold use. '+
        'Totals sum stage medians and exclude shared archive I/O, Python startup and evidence writes.','']
    lines.append(table(['Dataset','k','Method','m','Fresh ARI','Test fresh ARI','Pair Q95','Cold fresh s','Reused fresh s'],
        [[r.dataset,r.k,r.method,r.m,f'{r.fresh_ari_mean:.4f}',f'{r.test_fresh_ari_mean:.4f}',
          f'{r.pair_test_q95_mean:.4f}',f'{r.cold_fresh_seconds_mean:.4f}',f'{r.fresh_seconds_mean:.4f}'] for r in focus.itertuples()]))
    lines+=['','## Uncertainty and scope','',
        f'Exploratory pointwise joint comparisons favoring +12 against an individual PCA control: {len(joint)}. '+
        'A joint cell requires a positive lower 95% bound for fresh ARI difference and a speedup lower bound above one. '+
        'Bootstrap resamples draw/start axes conditional on each fixed corpus. No multiplicity adjustment or new-data confirmation is implied. '+
        'An iteration-cap case cannot count as a joint win.', '',
        f'Integrity checks: {checks["cases_checked"]} cases; {checks["unique_artifacts_checked"]} unique artifact hashes. '+
        f'Cases flagged with a centroid/fresh/refinement/held-out iteration cap: {checks["capped_cases"]}. '+
        'Uninformative DBSCAN partitions are flagged and retained; ARI=1 for all-noise results is not evidence of useful clustering.', '',
        '## Saved evidence','',
        '- `rows.csv`, `summary.csv`, `comparisons.csv`: raw endpoints, crossed confidence intervals and paired contrasts.',
        '- `stage_means.csv`: separate learning, construction, conversion and downstream times.',
        '- `cases/`: individual timing samples, query indices/answers, fixed/held-out labels, fresh/refinement centers and labels, kNN and DBSCAN arrays.',
        '- `projection_models/`, `datasets/`: deduplicated projection parameters, original source bindings, splits and training fits.',
        '- `sources/`, `manifest.json`, `audit.json`: frozen implementation and integrity/cost checks.', '',
        'DBSCAN evaluates at most 256 explicitly saved corpus rows to bound quadratic work; this is the only downstream row restriction. '+
        'All compression, fixed assignment, fresh/refinement clustering, and kNN tasks use the complete saved corpus. '+
        'The archived text vocabulary predates this numerical train/validation/test split. '+
        'This is a representation-compression study, not a raw-text generalization or cryptographic proof benchmark. '+
        'No privacy guarantee for training-derived directions or new ZK speed claim is established.','']
    (root/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    html='<!doctype html><meta charset="utf-8"><title>Full native comparison</title><style>body{font:14px system-ui;margin:28px}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ddd}th{background:#e8f2f4}</style>'
    html+=('<h1>Full native comparison</h1><p>'+('Complete' if manifest['complete'] else 'PARTIAL')+
        f': {manifest["completed_cases"]} cases. All required fitting and compression costs are included.</p>')
    html+=focus[['dataset','k','method','m','fresh_ari_mean','test_fresh_ari_mean','pair_test_q95_mean',
        'cold_fresh_seconds_mean','fresh_seconds_mean']].to_html(index=False,float_format=lambda a:f'{a:.4f}')
    (root/'index.html').write_text(html,encoding='utf-8')
    atomic_json(root/'report_manifest.json',dict(complete=True,campaign_complete=manifest['complete'],
        builder_sha256=file_hash(Path(__file__)),outputs={name:file_hash(root/name) for name in [
            'report.md','index.html','rows.csv','summary.csv','comparisons.csv','stage_means.csv','audit.json']}))
    print(json.dumps(checks))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
    build(args.root,args.allow_partial)


if __name__=='__main__':main()
