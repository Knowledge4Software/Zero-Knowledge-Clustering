"""Audit and visualize a completed dedicated-rank +12-direction native screen."""
from pathlib import Path
import argparse
import json
import shutil

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from commitjl.reusable_native import NativeProjection, array_hash, file_hash, stable_squared_distances
from experiments.protected_jl_study import Archive
from experiments.laptop_comparison import atomic_json

CODE = Path(__file__).resolve().parents[1]
METHODS = ['full', 'pca_full', 'pca_randomized', 'pca_randomized_fast', 'protected_jl', 'protected_plus12']
LABELS = dict(full='Full', pca_full='PCA full SVD', pca_randomized='PCA randomized',
              pca_randomized_fast='PCA randomized, 2 iterations', protected_jl='Original Protected JL',
              protected_plus12='Protected JL +12')
NAMES = dict(mnist='MNIST', fashion_mnist='Fashion-MNIST', gisette='Gisette',
             **{'20ng_tfidf_nonzero':'20 Newsgroups (nonzero)'})


def audit(root):
    manifest = json.loads((root/'manifest.json').read_text())
    assert manifest['complete'], 'Only audit a completed screen'
    records = [json.loads(p.read_text()) for p in sorted((root/'rows').glob('*.json'))]
    assert len(records) == manifest['expected_cases'] == manifest['completed_cases']
    for relative, digest in manifest['identity']['source_hashes'].items():
        assert file_hash(CODE/relative) == digest, relative
        snapshot=root/'source_snapshot'/relative
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        if not snapshot.exists():
            shutil.copyfile(CODE/relative,snapshot)
        assert file_hash(snapshot)==digest
    config = manifest['identity']['configuration']
    assert not config['smoke']
    seen=set()
    for row in records:
        key=tuple(row[k] for k in ['dataset','method','m','seed','start'])
        assert key not in seen
        seen.add(key)
    for dataset in config['datasets']:
        for fit_meta in (root/'datasets'/dataset/'models').glob('*.json'):
            fit=json.loads(fit_meta.read_text())
            assert file_hash(fit_meta.with_suffix('.npz'))==fit['sha256']
        for start in range(config.get('start_offset', 0), config.get('start_offset', 0)+config['starts']):
            full=[r for r in records if r['dataset']==dataset and r['start']==start and r['method']=='full']
            assert len(full)==1
            for m in config['m']:
                for method in METHODS[1:]:
                    group=[r for r in records if r['dataset']==dataset and r['start']==start and r['method']==method and r['m']==m]
                    assert {r['seed'] for r in group}==({-1} if method=='pca_full' else
                        set(range(config.get('seed_offset', 0), config.get('seed_offset', 0)+config['seeds'])))
    for row in records:
        key=f"{row['dataset']}_s{row['start']}_{row['method']}_m{row['m']}_seed{row['seed']}"
        assert file_hash(root/'artifacts'/(key+'.npz')) == row['artifact_sha256'], key
        if row['method'].startswith('pca_'):
            assert row['pca_fitted_m'] == row['m'] == row['pca_spec']['n_components']
            assert row['m'] in config['m']
            path=root/'datasets'/row['dataset']/'models'/f"{row['method']}_m{row['m']}_seed{row['seed']}.json"
            fit=json.loads(path.read_text())
            assert fit['description']['fitted_n_components'] == row['m']
            assert fit['description']['options']['n_components'] == row['m']
            assert fit['description']['options'] == row['pca_spec']
            assert np.isclose(row['model_fit_seconds'],fit['timing']['median'],rtol=1e-12)
        elif row['method'] in ['protected_jl','protected_plus12']:
            models=root/'datasets'/row['dataset']/'models'
            center=json.loads((models/f"centers_s{row['start']}.json").read_text())
            basis_key='enriched' if row['method']=='protected_plus12' else 'basis'
            basis=json.loads((models/f"{basis_key}_s{row['start']}.json").read_text())
            assert np.isclose(row['model_fit_seconds'],
                center['timing']['median']+basis['timing']['median'],rtol=1e-12)
        else:
            assert row['model_fit_seconds']==0 and row['construction_seconds']==0 and row['compression_seconds']==0
        if row['method'] == 'protected_plus12':
            assert row['protected_rank'] == 16 and row['residual_m'] == row['m']-16
        if row['method'] == 'protected_jl':
            assert row['protected_rank'] == 4 and row['residual_m'] == row['m']-4
        assert row['pair_zero_violations'] == 0
        expected=row['model_fit_seconds']+row['construction_seconds']+row['compression_seconds']
        assert np.isclose(row['cold_compression_seconds'],expected,rtol=1e-12)
        assert np.isclose(row['cold_fresh_seconds'],expected+row['fresh_seconds'],rtol=1e-12)
        assert not row['training_hit_cap'] and not row['fresh_hit_cap'] and not row['heldout_hit_cap']
    archive=Archive(); replay=[]
    for dataset in config['datasets']:
        raw_name='20ng_tfidf' if dataset.endswith('_nonzero') else dataset
        saved=archive.load(f'results/paper_server_20261006/rq1/{raw_name}/artifacts/dataset.npz')
        original=saved['X']
        meta=json.loads((root/'datasets'/dataset/'dataset.json').read_text())
        assert array_hash(original)==meta['original_sha256']
        with np.load(root/'datasets'/dataset/'indices.npz',allow_pickle=False) as ids:
            indices={k:ids[k] for k in ids.files}
        rows=indices['source_indices'];x=np.ascontiguousarray(original[rows],dtype=np.float64)
        assert array_hash(x)==meta['used_sha256']
        assert meta['n']==len(x) and meta['d']==x.shape[1]
        assert x.shape==dict(mnist=(5000,784),fashion_mnist=(5000,784),gisette=(3000,5000),
                             **{'20ng_tfidf_nonzero':(2886,2000)})[dataset]
        assert len(np.unique(np.concatenate([indices[k] for k in ['train','validation','test']])))==len(x)
        subset=[r for r in records if r['dataset']==dataset]
        selected={}
        for row in subset:
            selected.setdefault((row['method'],row['m']),row)
        for row in selected.values():
            key=f"{dataset}_s{row['start']}_{row['method']}_m{row['m']}_seed{row['seed']}"
            with np.load(root/'artifacts'/(key+'.npz'),allow_pickle=False) as saved_case:
                arrays={k:saved_case[k] for k in saved_case.files}
            options=dict(origin=arrays['model_origin'])
            if row['method']=='full':
                model=NativeProjection('full',x.shape[1])
            elif row['method'].startswith('pca_'):
                model=NativeProjection('pca',x.shape[1],components=arrays['model_components'],**options)
            else:
                sparse=csr_matrix((arrays['model_a_data'],arrays['model_a_indices'],arrays['model_a_indptr']),
                                  shape=tuple(arrays['model_a_shape']))
                model=NativeProjection('hybrid',x.shape[1],basis=arrays['model_basis'],sparse_map=sparse,**options)
            assert model.fingerprint==row['model_fingerprint']
            z=model.transform(x)
            labels=stable_squared_distances(z,arrays['fresh_centers']).argmin(axis=1)
            np.testing.assert_array_equal(labels,arrays['fresh_labels'])
            full_key=f"{dataset}_s{row['start']}_full_m{x.shape[1]}_seed-1"
            with np.load(root/'artifacts'/(full_key+'.npz'),allow_pickle=False) as full:
                ari=adjusted_rand_score(full['fresh_labels'],arrays['fresh_labels'])
                assert abs(ari-row['fresh_ari'])<1e-12
                for phase in ['validation','test']:
                    value=adjusted_rand_score(full[phase+'_fresh_labels'],arrays[phase+'_fresh_labels'])
                    assert abs(value-row[phase+'_fresh_ari'])<1e-12
            with np.load(root/'datasets'/dataset/'models'/f"centers_s{row['start']}.npz",allow_pickle=False) as c:
                centers=c['centers']
            test=indices['test']
            fixed=stable_squared_distances(z[test],model.transform(centers)).argmin(axis=1)
            np.testing.assert_array_equal(fixed,arrays['fixed_test_labels'])
            pair=indices['pair_indices'];zz=z[test]
            values=np.sum((zz[pair[:,0]]-zz[pair[:,1]])**2,axis=1)
            np.testing.assert_allclose(values,arrays['projected_pair_squared_distances'],rtol=1e-12,atol=1e-12)
            positive=indices['pair_reference']>0
            error=np.quantile(np.abs(values[positive]/indices['pair_reference'][positive]-1),.95)
            assert abs(error-row['pair_q95'])<1e-12
            replay.append(dict(case=key,passed=True))
    result=dict(complete=True,artifact_hashes_checked=len(records),source_hashes_checked=True,
        dedicated_target_pca_fits_verified=True,extra_twelve_directions_verified=True,
        cost_accounting_verified=True,all_cases_converged=True,representative_numeric_replays=len(replay),
        checks=replay,scope='All source/artifact checksums; representative projection, assignment, final-cluster consistency, ARI and geometry replay. Not all KMeans optimizations rerun.')
    atomic_json(root/'audit.json',result)
    return result


def figures_and_findings(root):
    manifest=json.loads((root/'manifest.json').read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    summary=pd.read_csv(root/'summary.csv')
    comp=pd.read_csv(root/'comparisons.csv')
    # This field combines independent diagnostic scopes. Give it an explicit
    # name and never present it as a same-workload speed-and-ARI win.
    comp=comp.rename(columns={'ci_speed_and_test_ari_win':'ci_whole_corpus_speed_and_test_quality_gain'})
    comp.to_csv(root/'reviewed_comparisons.csv',index=False)
    colors=dict(full='#657384',pca_full='#4D65A8',pca_randomized='#A06FBC',
                pca_randomized_fast='#C46C35',protected_jl='#92ADA9',protected_plus12='#007D80')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,dataset in zip(axes.flat,NAMES):
        cells=summary[summary.dataset.eq(dataset)]
        for _,r in cells.iterrows():
            x=r.cold_fresh_seconds_mean;y=r.fresh_ari_mean
            lo=r.cold_fresh_seconds_ci_low;hi=r.cold_fresh_seconds_ci_high
            yl=r.fresh_ari_ci_low;yh=r.fresh_ari_ci_high
            marker='*' if r.method=='full' else ('o' if r.m==128 else '^')
            ax.errorbar(x,y,xerr=[[max(0,x-lo)],[max(0,hi-x)]],yerr=[[max(0,y-yl)],[max(0,yh-y)]],
                        fmt=marker,color=colors[r.method],markersize=8 if r.method=='full' else 5,
                        capsize=2,elinewidth=.7,alpha=.9)
        ax.set(xscale='log',ylim=(-.05,1.055),title=NAMES[dataset],
               xlabel='Total setup + fresh KMeans (seconds)',ylabel='Fresh-clustering ARI vs Full')
        ax.grid(alpha=.15)
    handles=[Line2D([],[],color=colors[m],marker='o',linestyle='None',label=LABELS[m]) for m in METHODS]
    fig.legend(handles=handles,loc='outside lower center',ncol=3,fontsize=9)
    fig.suptitle('Dedicated-rank comparison: circles m=128; triangles m=256; stars Full\nAll necessary setup is included; pointwise conditional 95% intervals',fontsize=11)
    fig.savefig(root/'speed_quality.png',dpi=180);plt.close(fig)
    # Show that gains in one endpoint do not imply gains in all endpoints.
    rows=[]
    for dataset in NAMES:
        for m in [128,256]:
            old=summary[(summary.dataset==dataset)&summary.m.eq(m)&summary.method.eq('protected_jl')].iloc[0]
            new=summary[(summary.dataset==dataset)&summary.m.eq(m)&summary.method.eq('protected_plus12')].iloc[0]
            rows.append(dict(dataset=dataset,m=m,fresh_ari_before=old.fresh_ari_mean,fresh_ari_after=new.fresh_ari_mean,
                test_ari_before=old.test_fresh_ari_mean,test_ari_after=new.test_fresh_ari_mean,
                knn_before=old.knn_recall_mean,knn_after=new.knn_recall_mean,
                pair_q95_before=old.pair_q95_mean,pair_q95_after=new.pair_q95_mean,
                cold_fresh_before=old.cold_fresh_seconds_mean,cold_fresh_after=new.cold_fresh_seconds_mean))
    changes=pd.DataFrame(rows);changes.to_csv(root/'upgrade_changes.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,before,after,title in [(axes[0],'test_ari_before','test_ari_after','Held-out fresh clustering'),
                                (axes[1],'knn_before','knn_after','Five-neighbor recall'),
                                (axes[2],'pair_q95_before','pair_q95_after','Pair squared-distance Q95 error')]:
        for _,r in changes.iterrows():
            ax.plot([0,1],[r[before],r[after]],marker='o',markersize=4,label=f'{NAMES[r.dataset]} / {r.m}')
        ax.set(xticks=[0,1],xticklabels=['Original','+12 directions'],title=title);ax.grid(alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=7)
    fig.savefig(root/'upgrade_quality.png',dpi=180);plt.close(fig)

    def table(headers,rows):
        return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                         ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])
    focused=[]
    for dataset in NAMES:
        for m in [128,256]:
            subset=summary[(summary.dataset==dataset)&summary.m.eq(m)]
            ours=subset[subset.method.eq('protected_plus12')].iloc[0]
            fast=subset[subset.method.eq('pca_randomized_fast')].iloc[0]
            contrast=comp[(comp.dataset==dataset)&comp.m.eq(m)&comp.competitor.eq('pca_randomized_fast')].iloc[0]
            focused.append([NAMES[dataset],m,f'{fast.fresh_ari_mean:.4f}',f'{ours.fresh_ari_mean:.4f}',
                            f'{fast.cold_fresh_seconds_mean:.3f}',f'{ours.cold_fresh_seconds_mean:.3f}',
                            f'{contrast.cold_fresh_speedup_mean:.2f}',str(contrast.ci_speed_and_fresh_ari_win)])
    robust=comp[comp.competitor.str.startswith('pca_')&comp.ci_speed_and_fresh_ari_win.eq(True)]
    all_three=comp[comp.competitor.str.startswith('pca_')].groupby(['dataset','m']).ci_speed_and_fresh_ari_win.agg(['sum','count'])
    all_three=all_three[(all_three['sum']==3)&(all_three['count']==3)]
    lines=['# Findings: extra 12 protected directions', '',
        f'Completed {manifest["completed_cases"]}-case exploratory native screen. '+
        'Each PCA128 and PCA256 is fitted independently at that target dimension. '+
        'The fast randomized PCA control uses two power iterations, matching the residual-basis solver effort setting. '+
        'Each cold result charges all learning, map construction, conversion and fresh KMeans costs. Common input I/O and artifact writes are excluded.', '',
        'The primary joint endpoint uses speed and ARI for the SAME whole-corpus fresh-clustering workload. '+
        'Held-out test-fold fresh clustering is a separate diagnostic; it is not paired with a timed test-fold workload in this screen.', '',
        '## Against the faster randomized PCA control', '',
        table(['Dataset','m','PCA fresh ARI','+12 fresh ARI','PCA total s','+12 total s','Paired speedup','Both pointwise lower CIs pass'],focused), '',
        '## All PCA controls', '',
        f'Pointwise joint wins versus an individual PCA control: {len(robust)} of {len(comp[comp.competitor.str.startswith("pca_")])} comparisons. '+
        f'Dataset/dimension cells beating all three PCA controls with both lower CI bounds: {len(all_three)} of '+
        f'{len(manifest["identity"]["configuration"]["datasets"])*len(manifest["identity"]["configuration"]["m"])}.', '',
        'Use reviewed_comparisons.csv for all paired differences and intervals. Speedup is the mean of paired baseline/method ratios, '+
        'so it may differ from a ratio of the mean times printed in this compact table. These are exploratory comparisons with '+
        f'{manifest["identity"]["configuration"]["seeds"]} draws and {manifest["identity"]["configuration"]["starts"]} starts, '+
        'conditional on fixed data and without multiplicity correction; they are not independent confirmation of a superiority claim.', '',
        'Fresh KMeans is nonconvex: the same initial point indices can follow different optimization paths after projection. '+
        'More output dimensions therefore need not monotonically improve ARI against one converged Full solution. '+
        'Alternate converged partitions are possible; iteration caps and saved final-label consistency are checked separately. '+
        'ARI against Full is not ground-truth classification accuracy or a guarantee that Full found the global optimum.', '',
        '## What changed relative to our original method', '',
        table(['Dataset','m','Fresh ARI before -> after','Test ARI before -> after','Pair Q95 before -> after','Cold fresh s before -> after'],
            [[NAMES[r.dataset],r.m,f'{r.fresh_ari_before:.3f} -> {r.fresh_ari_after:.3f}',
              f'{r.test_ari_before:.3f} -> {r.test_ari_after:.3f}',f'{r.pair_q95_before:.3f} -> {r.pair_q95_after:.3f}',
              f'{r.cold_fresh_before:.3f} -> {r.cold_fresh_after:.3f}'] for r in changes.itertuples()]), '',
        '## Integrity and scope', '',
        'audit.json verifies all case artifacts and source hashes, every actual PCA fitting dimension, preservation of the centroid span, '+
        'cost accounting and representative saved-result numeric replay. Fixed-center ARI=1 is expected structurally and is not used as the fresh-clustering quality win. '+
        'The text corpus has exactly 114 original zero rows removed. No extra row/feature truncation is used. '+
        'Original native/PDF reports are preserved. The new private-training-dependent basis has no demonstrated ZK/privacy guarantee.', '',
        '![Total cost and fresh ARI](speed_quality.png)', '', '![Changes from the original method](upgrade_quality.png)', '']
    (root/'findings.md').write_text('\n'.join(lines),encoding='utf-8')
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Extra 12 directions</title>'+
        '<style>body{font:15px system-ui;max-width:1300px;margin:32px auto;color:#17334a}table{border-collapse:collapse;width:100%;font-size:13px}'+
        'th,td{padding:7px;border-bottom:1px solid #ddd;text-align:right}th{background:#edf5f6}img{max-width:100%}</style>'+
        '<h1>Protected JL +12: dedicated-rank PCA comparison</h1><p>All necessary setup is charged. '+
        'Exploratory screen; fresh ARI is agreement with Full, not true-label accuracy. Every PCA fits its actual target dimension.</p>'+
        '<h2>Faster randomized PCA comparison</h2>'+pd.DataFrame(focused,columns=['Dataset','m','PCA fresh ARI','+12 fresh ARI','PCA total s','+12 total s','Speedup','Both CI bounds pass']).to_html(index=False)+
        '<img src="speed_quality.png"><h2>All method summaries</h2>'+summary[['dataset','method','m','fixed_ari_mean','fresh_ari_mean','test_fresh_ari_mean','pair_q95_mean','cold_compression_seconds_mean','cold_fresh_seconds_mean']].to_html(index=False,float_format=lambda x:f'{x:.4f}')+
        '<img src="upgrade_quality.png"><p>Full paired intervals: reviewed_comparisons.csv. Detailed provenance: manifest.json and audit.json.</p>',encoding='utf-8')
    return dict(pointwise_joint_pca_comparisons=len(robust),joint_cells_against_all_three=len(all_three))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=4):
        result=audit(args.root)
        findings=figures_and_findings(args.root)
    outputs=['rows.csv','summary.csv','comparisons.csv','reviewed_comparisons.csv',
             'upgrade_changes.csv','findings.md','index.html','speed_quality.png',
             'upgrade_quality.png','audit.json','manifest.json']
    reviewer_snapshot=args.root/'source_snapshot'/'analysis'/'review_enriched_jl.py'
    reviewer_snapshot.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(Path(__file__),reviewer_snapshot)
    outputs.append(str(reviewer_snapshot.relative_to(args.root)))
    atomic_json(args.root/'review_manifest.json',dict(complete=True,
        reviewer_sha256=file_hash(Path(__file__)),
        files={name:file_hash(args.root/name) for name in outputs},
        artifact_cases=result['artifact_hashes_checked'],
        numeric_replays=result['representative_numeric_replays'],**findings))
    print(json.dumps(dict(audit_cases=result['artifact_hashes_checked'],replays=result['representative_numeric_replays'],**findings)))


if __name__=='__main__':
    main()
