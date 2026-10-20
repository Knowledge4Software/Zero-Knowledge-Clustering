"""Audit, numerical replay, and complete tables for residual-map experiments."""
import argparse
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
from commitjl.durable_io import atomic_json
from commitjl.reusable_native import file_hash, array_hash, stable_squared_distances
from commitjl.residual_maps import ResidualProjection, VARIANTS
from experiments.residual_map_comparison import load_dataset, split_indices
HYBRIDS = list(VARIANTS)


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])


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
        needs_centers=r['method'] not in HYBRIDS
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
        elif r['method'] in HYBRIDS:
            name='enriched'
            fit=json.loads((fit_root/f"{name}_s{r['start']}.json").read_text())
            expected=r['centroid_training_seconds']+fit['timing']['median']
            assert r['fixed_test_ari']==1.
        else:expected=0.
        assert np.isclose(r['model_fit_seconds'],expected,rtol=1e-12)
        assert r['pair_error']['zero_violations']==0
        assert r['centroid_error']['zero_violations']==0
        if r['method']=='full':assert r['construction_seconds']==r['compression_seconds']==0.
        if r['method'] in HYBRIDS:
            with np.load(root/r['model_file'],allow_pickle=False) as a:
                assert array_hash(a['basis'])==r['basis_sha256']
                assert int(a['m'])==r['m'] and int(a['q'])==r['q']
                assert str(a['kind'])==r['residual_kind']
                assert a['basis'].shape[1]+r['residual_m']==r['m']
        caps+=int(r['any_iteration_cap'])
    result=dict(complete=True,campaign_complete=manifest['complete'],cases_checked=len(rows),
        unique_artifacts_checked=len(checked),source_snapshots_verified=True,
        dedicated_pca_fits_verified=True,task_specific_cost_accounting_verified=True,
        capped_cases=caps,scope='All case/source/artifact hashes and cost definitions; no fresh clustering reruns')
    atomic_json(root/'audit.json',result);return result


def replay(root):
    manifest=json.loads((root/'manifest.json').read_text())
    config=manifest['identity']['configuration']; chosen={}; checked=[]
    for path in sorted((root/'rows').glob('*.json')):
        row=json.loads(path.read_text())
        if row['method'] in HYBRIDS:
            chosen.setdefault((row['dataset'],row['method']),row)
    for dataset in sorted({key[0] for key in chosen}):
        x,_,_=load_dataset(dataset,config['smoke']); folds=split_indices(len(x))
        test=x[folds['test']]
        ds=root/'datasets'/dataset
        with np.load(ds/'geometry_indices.npz',allow_pickle=False) as a: pairs=a['test_pair_indices']
        for (name,method),row in chosen.items():
            if name!=dataset:continue
            key=f"{name}_k{row['k']}_s{row['start']}_{method}_m{row['m']}_seed{row['seed']}"
            with np.load(root/row['model_file'],allow_pickle=False) as a:
                model=ResidualProjection.from_parameters(a)
            assert model.fingerprint==row['model_fingerprint']
            with np.load(ds/f"k{row['k']}"/'models'/f"centers_s{row['start']}.npz",allow_pickle=False) as a:
                centers=a['centers']
            z=model.transform(test);zc=model.transform(centers)
            pair=np.sum((z[pairs[:,0]]-z[pairs[:,1]])**2,axis=1)
            with np.load(root/'cases'/key/'quality.npz',allow_pickle=False) as a:
                np.testing.assert_allclose(pair,a['projected_pair_squared_distances'],rtol=1e-10,atol=1e-10)
                np.testing.assert_array_equal(stable_squared_distances(z,zc).argmin(1),a['fixed_test_labels'])
            checked.append(key)
        del x,test
    result=dict(complete=True,cases=len(checked),case_ids=checked,
        scope='One case per dataset and residual variant: restore actual map, fingerprint, held-out pairs and fixed-center labels; not a fresh K-means rerun')
    atomic_json(root/'replay.json',result);return result


def build(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    if not manifest['complete']:raise ValueError('Report requires a completed campaign')
    checks=audit(root);numerical=replay(root)
    summary=pd.read_csv(root/'summary.csv');comparisons=pd.read_csv(root/'comparisons.csv')
    config=manifest['identity']['configuration']
    limited=config['seeds']<3 or config['starts']<3 or config['smoke']
    lines=['# Residual-map comparison','',
        f'Completed {manifest["completed_cases"]} cases. '+('PILOT / correctness evidence; insufficient repetitions for superiority claims.' if limited else
        'Three or more projection draws crossed with three or more clustering starts; exploratory pointwise intervals.'),'',
        'All eight variants keep the same training-derived centroid span plus up to twelve residual PCA directions. '+
        'Only the residual map changes. Every output dimension includes both protected and residual coordinates. '+
        'Canonical q=1/2/4/8, four hashing blocks, randomized normalized Hadamard mixing followed by q=1/4, and dense Gaussian control are retained regardless of outcome.','',
        'PCA is fitted separately at each actual rank using full SVD, default randomized, and fast randomized solvers. '+
        'Only requested compression sizes are used. A size not smaller than the input dimension is excluded. '+
        'Centered PCA above training_rows-1 is unavailable, not treated as a loss or replaced by a smaller rank.','',
        '## Timing and quality definitions','',
        'Cold fresh time includes training centers and protected directions for every hybrid, model construction including cached matrix products, corpus conversion, and one fresh K-means run. '+
        'PCA and Full do not pay for centers they do not need for fresh clustering. Each task has its own setup accounting. '+
        'Fit caches charge their full measured cost. Times sum stage medians. Archive loading, Python startup, and evidence writes are excluded. '+
        'Fresh ARI compares matched Full point-index initialization; it is not true-class accuracy. Pair Q95 is the 95th percentile of absolute relative squared-distance error on saved held-out pairs.','',
        'The synthetic smoke fixture is not a real MNIST result. Real pilots and full runs use complete archived subsets, with a training/validation/test split of 60/20/20. '+
        'DBSCAN alone uses at most the configured 256 saved corpus rows. kNN has 128 query rows against the whole corpus. All raw labels, model parameters, timing samples and selected distance answers are retained.','',
        '## Per-dataset results','']
    columns=['dataset','k','method','m','cases','fresh_ari_mean','pair_test_q95_mean',
        'cold_fresh_seconds_mean','fresh_seconds_mean']
    for dataset in config['datasets']:
        group=summary[summary.dataset.eq(dataset)]
        lines += [f'### {dataset}','',table(['k','Method','Total m','Cases','Fresh ARI','Pair Q95','Cold fresh s','Reused fresh s'],
            [[v.k,v.method,v.m,v.cases,f'{v.fresh_ari_mean:.4f}',f'{v.pair_test_q95_mean:.4f}',
              f'{v.cold_fresh_seconds_mean:.4f}',f'{v.fresh_seconds_mean:.4f}'] for v in group.itertuples()]),'']
    wins=comparisons[comparisons.competitor.eq('pca_randomized_fast')&comparisons.joint_pointwise_win.eq(True)]
    lines += ['## Interpretation and audit','',
        f'Joint pointwise wins versus fast PCA: {len(wins)}. A win requires a complete grid with at least three draws and three starts, '+
        'a positive lower ARI-difference bound, a cold-speedup lower bound above one, and no iteration-limit flags. '+
        'Pilot wins are never declared. Intervals are conditional on these fixed datasets and have no multiplicity adjustment; all candidates are exploratory.','',
        f'Integrity audit: {checks["cases_checked"]} cases, {checks["unique_artifacts_checked"]} unique artifacts; '+
        f'{checks["capped_cases"]} iteration-limit cases. Numerical map replay: {numerical["cases"]} cases.','',
        'These maps are native research variants. The existing sparse commitment generator and proof circuit were not replaced. '+
        'No proof-cost advantage, private-basis guarantee, or theorem for arbitrary empirical dimensions is established. '+
        'Improved pair distances alone do not prove improved ARI. Full cold cost and reused costs must remain visible.','',
        'Files: rows.csv, summary.csv, comparisons.csv, audit.json, replay.json; cases/ for evidence; projection_models/ for all actual maps; datasets/ for frozen splits and fit caches.','']
    (root/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    page='<!doctype html><meta charset="utf-8"><title>Residual-map comparison</title><style>body{font:14px system-ui;margin:28px}table{border-collapse:collapse}td,th{padding:5px;border:1px solid #ddd}th{background:#edf3f7}</style>'
    page+='<h1>Residual-map comparison</h1><p>'+html.escape(lines[2])+'</p>'
    page+=summary[columns].to_html(index=False,float_format=lambda x:f'{x:.4f}')
    (root/'index.html').write_text(page,encoding='utf-8')
    atomic_json(root/'report_manifest.json',dict(complete=True,campaign_complete=True,pilot=limited,
        builder_sha256=file_hash(Path(__file__)),outputs={name:file_hash(root/name) for name in
        ['report.md','index.html','rows.csv','summary.csv','comparisons.csv','audit.json','replay.json']}))
    print(json.dumps(dict(audit=checks,replay_cases=numerical['cases'],report=str(root/'report.md'))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    build(parser.parse_args().root)

