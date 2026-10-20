"""Offline full native comparison, with dedicated-rank PCA and every task timed.

This is deliberately separate from previous campaigns and from ZK proof runs.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import psutil
import scipy
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from threadpoolctl import threadpool_limits

from commitjl.enriched_jl import enrich_centroid_basis
from commitjl.protected_jl import centroid_basis
from commitjl.reusable_native import NativeProjection, array_hash, file_hash, stable_squared_distances
from commitjl.sparse_gen import sparse_matrix_gen_csr
from experiments.enriched_jl_comparison import cached_fit, pca_spec, PCA_METHODS
from experiments.laptop_comparison import atomic_json, save_arrays
from experiments.paper_compression import (split_indices, measured_workloads, distortion,
    timed, bootstrap_crossed, reconstructed_centers)
from experiments.protected_jl_study import Archive
from campaigns.paper_compression import exclusive_run

CODE=Path(__file__).resolve().parents[1]
DATASETS=['gaussian_easy','gaussian_hard','dbscan_margin','mnist','fashion_mnist',
          'gisette','arcene','20ng_tfidf','20ng_tfidf_nonzero']
METHODS=['full','pca_full','pca_randomized','pca_randomized_fast',
         'sparse_jl','protected_jl','protected_plus12','protected_only']
HYBRIDS=['sparse_jl','protected_jl','protected_plus12']
STOCHASTIC=HYBRIDS+PCA_METHODS[1:]


def load_dataset(name,smoke=False):
    if smoke:
        rng=np.random.default_rng(816)
        labels=np.arange(180)%10;c=rng.normal(size=(10,64))
        x=np.ascontiguousarray(c[labels]+.4*rng.normal(size=(180,64)))
        return x,labels,dict(source='synthetic smoke fixture',original_sha256=array_hash(x),
            source_rows=np.arange(len(x)),removed_zero_rows=0)
    raw='20ng_tfidf' if name.endswith('_nonzero') else name
    source=f'results/paper_server_20261006/rq1/{raw}/artifacts/dataset.npz'
    saved=Archive().load(source);original=saved['X'];labels=saved['y']
    digest=array_hash(original)
    manifest=json.loads((CODE/f'results/server_paper_20261006/rq1/{raw}/manifest.json').read_text())
    assert digest==manifest['dataset']['sha256']
    ids=np.flatnonzero(np.any(original!=0,axis=1)) if name.endswith('_nonzero') else np.arange(len(original))
    x=np.ascontiguousarray(original[ids],dtype=float)
    return x,labels[ids] if len(labels) else labels,dict(source=source,original_sha256=digest,
        source_rows=ids,removed_zero_rows=len(original)-len(ids))


def dimension_plan(requested,n_train,d):
    widths=sorted({int(m) for m in requested if 1<=m<d})
    cap=min(n_train-1,d)
    if cap<max(requested) and 1<=cap<d:widths=sorted(set(widths+[cap]))
    return widths,cap


def case_plan(widths,cap,d,k,start,seeds,rank=None):
    rank=k-1 if rank is None else rank
    yield dict(method='full',m=d,seed=-1)
    if rank:yield dict(method='protected_only',m=rank,seed=-1)
    for m in widths:
        for method in PCA_METHODS+HYBRIDS:
            if method in PCA_METHODS and m>cap:continue
            if method=='protected_jl' and m<=rank:continue
            if method=='protected_plus12' and m<=rank+12:continue
            for seed in seeds if method in STOCHASTIC else [-1]:
                yield dict(method=method,m=m,seed=seed)


def make_hybrid(method,d,m,seed,source,k,start,centers,bases):
    u=bases[method] if method!='sparse_jl' else np.empty((d,0))
    payload=dict(version='comprehensive-native-v1',source=source,k=k,start=start,seed=seed,
        method=method,m=m,q=1,basis=array_hash(u) if method!='sparse_jl' else None)
    map_seed=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).digest()
    _,a=sparse_matrix_gen_csr(map_seed,m-u.shape[1],d,1)
    return NativeProjection('hybrid',d,basis=u,sparse_map=a.astype(float),
        origin=centers[0] if method!='sparse_jl' else np.zeros(d))


def inventory(args):
    archive=None if args.smoke else Archive()
    result=[]
    for name in args.datasets:
        raw='20ng_tfidf' if name.endswith('_nonzero') else name
        n,d=(180,64) if args.smoke else archive.index[
            f'results/paper_server_20261006/rq1/{raw}/artifacts/dataset.npz']['arrays']['X']['shape']
        if name.endswith('_nonzero'):n=2886
        widths,cap=dimension_plan(args.m,int(.6*n),d)
        cases=sum(len(list(case_plan(widths,cap,d,k,0,range(args.seeds)))) for k in args.k)*args.starts
        result.append(dict(dataset=name,n=n,d=d,training_rows=int(.6*n),widths=widths,pca_rank_cap=cap,
            no_pca_match=[m for m in widths if m>cap],cases=cases))
    return result


def aggregate(root,draws=1000):
    records=[json.loads(p.read_text()) for p in sorted((root/'rows').glob('*.json'))]
    if not records:return
    frame=pd.DataFrame(records);frame.to_csv(root/'rows.csv',index=False)
    metrics=['fresh_ari','fresh_nmi','ground_truth_ari','fixed_test_ari','test_fresh_ari',
        'validation_fresh_ari','pair_test_q95','centroid_test_q95','knn_recall',
        'fresh_original_partition_sse_ratio','fresh_r_phi','refine_ari','refine_r_phi','dbscan_ari',
        'model_fit_seconds','construction_seconds','compression_seconds','fresh_seconds',
        'assignment_seconds','query_seconds','knn_seconds','refine_seconds','dbscan_pair_seconds',
        'dbscan_fit_seconds','cold_fresh_seconds','cold_assignment_seconds','cold_query_seconds',
        'cold_knn_seconds','cold_refine_seconds','cold_dbscan_seconds']
    entries=[]
    for keys,group in frame.groupby(['dataset','k','method','m']):
        row=dict(zip(['dataset','k','method','m'],keys),cases=len(group))
        for name in metrics:
            if name not in group:continue
            grid=group.pivot(index='seed',columns='start',values=name).to_numpy()
            if np.isfinite(grid).all():
                mean,lo,hi=bootstrap_crossed(grid,draws=draws)
                row.update({name+'_mean':mean,name+'_ci_low':lo,name+'_ci_high':hi})
        row['any_iteration_cap']=bool(group.any_iteration_cap.any());entries.append(row)
    pd.DataFrame(entries).to_csv(root/'summary.csv',index=False)
    contrasts=[]
    for keys,ours in frame[frame.method.eq('protected_plus12')].groupby(['dataset','k','m']):
        for method in PCA_METHODS+['sparse_jl','protected_jl']:
            base=frame[frame.dataset.eq(keys[0])&frame.k.eq(keys[1])&frame.m.eq(keys[2])&frame.method.eq(method)]
            if not len(base):continue
            join=['dataset','k','m','start']
            if method in STOCHASTIC:
                join+=['seed'];matched=ours.merge(base,on=join,suffixes=('_ours','_base'),validate='one_to_one')
            else:matched=ours.merge(base.drop(columns='seed'),on=join,suffixes=('_ours','_base'),validate='many_to_one')
            entry=dict(dataset=keys[0],k=keys[1],m=keys[2],competitor=method,cases=len(matched))
            derived=dict(fresh_ari_difference=matched.fresh_ari_ours-matched.fresh_ari_base,
                cold_fresh_speedup=matched.cold_fresh_seconds_base/matched.cold_fresh_seconds_ours,
                cold_query_speedup=matched.cold_query_seconds_base/matched.cold_query_seconds_ours)
            for name,values in derived.items():
                matched[name]=values
                grid=matched.pivot(index='seed',columns='start',values=name).to_numpy()
                if np.isfinite(grid).all():
                    mean,lo,hi=bootstrap_crossed(grid,draws=draws)
                    entry.update({name+'_mean':mean,name+'_ci_low':lo,name+'_ci_high':hi})
            entry['joint_pointwise_win']=bool(not matched[['any_iteration_cap_ours','any_iteration_cap_base']].any().any()
                and entry.get('fresh_ari_difference_ci_low',-1)>0 and entry.get('cold_fresh_speedup_ci_low',0)>1)
            contrasts.append(entry)
    pd.DataFrame(contrasts).to_csv(root/'comparisons.csv',index=False)


def run(args):
    root=args.out.resolve();root.mkdir(parents=True,exist_ok=True)
    (root/'rows').mkdir(exist_ok=True)
    config=vars(args).copy();config['out']=str(root);config.update(q=1,extra=12,version='comprehensive-native-v1')
    sources=[Path(__file__),CODE/'experiments/enriched_jl_comparison.py',CODE/'experiments/paper_compression.py',
        CODE/'experiments/laptop_comparison.py',CODE/'experiments/protected_jl_study.py',
        CODE/'campaigns/paper_compression.py',CODE/'experiments/paper_artifacts.py',
        CODE/'commitjl/enriched_jl.py',CODE/'commitjl/protected_jl.py',CODE/'commitjl/reusable_native.py',
        CODE/'commitjl/sparse_gen.py',CODE/'commitjl/hashing.py',CODE/'commitjl/encoding.py']
    identity=dict(configuration=config,sources={str(p.relative_to(CODE)):file_hash(p) for p in sources},
        archive_index_sha256=file_hash(CODE/'results/server_retirement_20261007/compact_arrays/array_index.json'))
    path=root/'manifest.json'
    if path.exists():
        previous=json.loads(path.read_text())
        if previous['identity']!=identity:raise ValueError('Source/configuration changed: use a new output folder')
        if previous.get('complete'):print('Already complete; existing results preserved.');return
    for p in sources:
        target=root/'sources'/p.relative_to(CODE);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(p.read_bytes())
    started=time.monotonic();process=psutil.Process()
    planned=inventory(args)
    atomic_json(root/'plan.json',dict(datasets=planned,expected_cases=sum(a['cases'] for a in planned),
        methods=METHODS,scope='Current method only; text candidate upgrades kept in a separate validation-only ablation'))
    manifest=dict(identity=identity,complete=False,completed_cases=len(list((root/'rows').glob('*.json'))),
        expected_cases=sum(a['cases'] for a in planned),
        scope='Native complete workload comparison; every PCA fit at actual rank; no proof/privacy claim',
        environment=dict(python=sys.version,platform=platform.platform(),total_ram_gib=psutil.virtual_memory().total/2**30,
            numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__))
    atomic_json(path,manifest)
    def budget():
        if time.monotonic()-started>args.max_seconds:raise TimeoutError('Checkpointed time budget reached')
        if process.memory_info().rss>args.max_rss_gib*2**30:raise MemoryError('RSS budget reached')
        if psutil.virtual_memory().available<512*2**20:raise MemoryError('Free RAM below 512 MiB')
    try:
        with threadpool_limits(limits=args.threads):
            expected=0
            for dataset in args.datasets:
                budget();x,y,origin=load_dataset(dataset,args.smoke);x.setflags(write=False)
                n,d=x.shape;folds=split_indices(n);train=x[folds['train']]
                widths,cap=dimension_plan(args.m,len(train),d)
                ds=root/'datasets'/dataset;ds.mkdir(parents=True,exist_ok=True)
                source=array_hash(x)
                save_arrays(ds/'indices.npz',source_indices=origin.pop('source_rows'),**folds)
                atomic_json(ds/'dataset.json',dict(n=n,d=d,sha256=source,widths=widths,pca_rank_cap=cap,
                    training_rows=len(train),validation_rows=len(folds['validation']),test_rows=len(folds['test']),
                    unavailable_requested=[m for m in args.m if m>=d],
                    unmatched_pca_widths=[m for m in widths if m>cap],**origin))
                rng=np.random.default_rng(20261101);pairs=rng.integers(len(folds['test']),size=(2000,2));pairs=pairs[pairs[:,0]!=pairs[:,1]]
                test_x=x[folds['test']];pair_ref=np.sum((test_x[pairs[:,0]]-test_x[pairs[:,1]])**2,axis=1)
                validation=x[folds['validation']]
                vd=stable_squared_distances(validation[:256],validation[:256])
                positive=vd[np.triu_indices(len(vd),1)];positive=positive[positive>0]
                radii={f'q{q}':float(np.sqrt(np.quantile(positive,q/100))) for q in [10,25,50]}
                if dataset=='dbscan_margin':radii['certificate_fixture']=1.
                selection=dict(radius=radii['q25'],radii=radii)
                save_arrays(ds/'geometry_indices.npz',test_pair_indices=pairs,test_pair_reference=pair_ref)
                atomic_json(ds/'dbscan_selection.json',dict(selection,scope='original validation geometry only'))
                for k in args.k:
                    for start in range(args.start_offset,args.start_offset+args.starts):
                        budget();model_root=ds/f'k{k}';model_root.mkdir(exist_ok=True)
                        def centers_fit():
                            fit=KMeans(n_clusters=k,n_init=1,random_state=start,algorithm='lloyd',tol=1e-4,max_iter=300).fit(train)
                            return dict(centers=fit.cluster_centers_),dict(iterations=int(fit.n_iter_))
                        ca,cm=cached_fit(model_root,f'centers_s{start}',centers_fit,args.fit_repetitions)
                        centers=ca['centers'];center_seconds=cm['timing']['median']
                        ua,um=cached_fit(model_root,f'basis_s{start}',lambda:(dict(basis=centroid_basis(centers)),{}),args.fit_repetitions)
                        def enriched_fit():
                            u,meta=enrich_centroid_basis(train,centers,12,random_state=1701+start)
                            return dict(basis=u),meta
                        ea,em=cached_fit(model_root,f'enriched_s{start}',enriched_fit,args.fit_repetitions)
                        bases=dict(protected_jl=ua['basis'],protected_plus12=ea['basis'])
                        fit_costs=dict(protected_jl=center_seconds+um['timing']['median'],
                            protected_plus12=center_seconds+em['timing']['median'],
                            protected_only=center_seconds+um['timing']['median'])
                        seeds=range(args.seed_offset,args.seed_offset+args.seeds)
                        cases=list(case_plan(widths,cap,d,k,start,seeds,rank=ua['basis'].shape[1]));expected+=len(cases)
                        shuffled=np.random.default_rng(801+k+start).permutation(len(cases)-1)+1
                        cases=[cases[0]]+[cases[i] for i in shuffled]
                        for case in cases:
                            budget();method,m,seed=(case[a] for a in ['method','m','seed'])
                            key=f'{dataset}_k{k}_s{start}_{method}_m{m}_seed{seed}'
                            target=root/'rows'/(key+'.json');folder=root/'cases'/key
                            if target.exists():
                                row=json.loads(target.read_text())
                                for relative,digest in row['artifacts'].items():
                                    if file_hash(root/relative)!=digest:raise ValueError('Corrupt completed case: '+key)
                                continue
                            folder.mkdir(parents=True,exist_ok=True);fit_seconds=0.;pca_meta=None
                            if method=='full':make=lambda:NativeProjection('full',d)
                            elif method in PCA_METHODS:
                                spec=pca_spec(method,m,seed)
                                def pca_fit():
                                    p=PCA(**spec).fit(train)
                                    return dict(components=p.components_,mean=p.mean_),dict(options=spec,fitted_m=int(p.n_components_))
                                pa,pm=cached_fit(ds,f'{method}_m{m}_seed{seed}',pca_fit,args.fit_repetitions)
                                assert pm['description']['fitted_m']==m
                                fit_seconds=pm['timing']['median'];pca_meta=pm['description']
                                make=lambda:NativeProjection('pca',d,components=pa['components'],origin=pa['mean'])
                            elif method=='protected_only':
                                fit_seconds=fit_costs[method]
                                make=lambda:NativeProjection('pca',d,components=ua['basis'].T,origin=centers[0])
                            else:
                                fit_seconds=fit_costs.get(method,0.)
                                make=lambda:make_hybrid(method,d,m,seed,source,k,start,centers,bases)
                            model,construction=timed(make,args.timing_repetitions,warmup=True)
                            if model.m!=m:raise ValueError('Unexpected total output dimension')
                            models=root/'projection_models';models.mkdir(exist_ok=True)
                            model_file=models/(model.fingerprint+'.npz')
                            if not model_file.exists():save_arrays(model_file,**model.parameters())
                            cfg=dict(timing_repetitions=args.timing_repetitions,query_points=512,landmarks=64,
                                queries=args.queries,query_artifact_samples=2,kmeans_max_iter=300,
                                dbscan_points=args.dbscan_points,eps=.15)
                            workload=measured_workloads(dict(case,k=k,start=start),model,x,folds['test'],centers,None,cfg,selection,folder)
                            budget();z=model.transform(x);zc=model.transform(centers)
                            fixed=stable_squared_distances(z[folds['test']],zc).argmin(axis=1)
                            full_fixed=stable_squared_distances(test_x,centers).argmin(axis=1)
                            if method in ['protected_jl','protected_plus12','protected_only'] and not np.array_equal(fixed,full_fixed):
                                raise ValueError('Protected-center preservation failure')
                            held_labels={};held_quality={}
                            for phase in ['validation','test']:
                                ids=folds[phase];initial=np.random.default_rng(900+start).choice(len(ids),k,replace=False)
                                def fit(points):
                                    return KMeans(n_clusters=k,init=points[initial].copy(),n_init=1,max_iter=300,
                                        tol=1e-4,algorithm='lloyd',random_state=start).fit(points)
                                original_fold=fit(x[ids]);projected_fold=fit(z[ids])
                                held_labels.update({phase+'_original_labels':original_fold.labels_,phase+'_labels':projected_fold.labels_})
                                held_quality[phase]=dict(ari=float(adjusted_rand_score(original_fold.labels_,projected_fold.labels_)),
                                    hit_cap=bool(original_fold.n_iter_>=300 or projected_fold.n_iter_>=300))
                            projected_pairs=np.sum((z[folds['test'][pairs[:,0]]]-z[folds['test'][pairs[:,1]]])**2,axis=1)
                            pair_error=distortion(pair_ref,projected_pairs,.15,1e-12)
                            original_centroid=stable_squared_distances(test_x,centers)
                            projected_centroid=stable_squared_distances(z[folds['test']],zc)
                            centroid_error=distortion(original_centroid.ravel(),projected_centroid.ravel(),.15,1e-12)
                            with np.load(folder/'kmeans.npz') as a:
                                original_labels=a['original_labels'];labels=a['projected_labels']
                                def partition_sse(label):
                                    reconstructed=reconstructed_centers(x,label,k)
                                    # Partition SSE, distinct from nearest-reconstructed-center Phi.
                                    unique=np.unique(label);lookup={v:i for i,v in enumerate(unique)}
                                    assigned=np.asarray([lookup[v] for v in label])
                                    return float(np.sum((x-reconstructed[assigned])**2))
                                sse_ratio=partition_sse(labels)/partition_sse(original_labels)
                                fresh_nmi=float(normalized_mutual_info_score(original_labels,labels))
                                ground_truth=float(adjusted_rand_score(y,labels)) if len(y) else None
                            save_arrays(folder/'quality.npz',fixed_test_labels=fixed,original_fixed_test_labels=full_fixed,
                                projected_pair_squared_distances=projected_pairs,original_centroid_distances=original_centroid,
                                projected_centroid_distances=projected_centroid,**held_labels)
                            atomic_json(folder/'timings.json',workload)
                            con=construction['median'] if method!='full' else 0.
                            compression=workload['compression']['median'];setup=fit_seconds+con+compression
                            shared_centers=center_seconds if method not in ['protected_jl','protected_plus12','protected_only'] else 0.
                            query_seconds=float(sum(np.median([a['seconds'] for a in workload['distance_queries']['method'] if a['query']==q])
                                for q in range(args.queries)))
                            cap_hit=bool(cm['description']['iterations']>=300 or any(held_quality[a]['hit_cap'] for a in held_quality)
                                or any(workload.get(a,False) for a in ['kmeans_hit_iteration_cap','full_kmeans_hit_iteration_cap',
                                    'kmeans_refine_hit_iteration_cap','full_kmeans_refine_hit_iteration_cap']))
                            row=dict(dataset=dataset,k=k,start=start,**case,n=n,d=d,source_sha256=source,
                                model_fingerprint=model.fingerprint,model_file=str(model_file.relative_to(root)),
                                q=1 if method in HYBRIDS else 0,protected_rank=bases[method].shape[1] if method in bases else (ua['basis'].shape[1] if method=='protected_only' else 0),
                                pca_spec=pca_meta,centroid_training_seconds=center_seconds,model_fit_seconds=fit_seconds,
                                construction_seconds=con,construction_timing=construction,compression_seconds=compression,
                                fresh_seconds=workload['kmeans']['median'],fresh_ari=workload['kmeans_ari'],fresh_nmi=fresh_nmi,
                                ground_truth_ari=ground_truth,fixed_test_ari=float(adjusted_rand_score(full_fixed,fixed)),
                                validation_fresh_ari=held_quality['validation']['ari'],test_fresh_ari=held_quality['test']['ari'],
                                pair_test_q95=pair_error['q95'],centroid_test_q95=centroid_error['q95'],
                                pair_error=pair_error,centroid_error=centroid_error,knn_recall=workload['knn_recall'],
                                fresh_original_partition_sse_ratio=sse_ratio,fresh_r_phi=workload['kmeans_r_phi'],
                                refine_ari=workload['kmeans_refine_ari'],refine_r_phi=workload['kmeans_refine_r_phi'],
                                dbscan_ari=workload['dbscan_ari'],dbscan_informative=workload['dbscan_reference_partition_informative'],
                                assignment_seconds=workload['assignment']['median'],query_seconds=query_seconds,
                                knn_seconds=workload['knn']['median'],refine_seconds=workload['kmeans_refine']['median'],
                                dbscan_pair_seconds=workload['dbscan_pair_build']['median'],dbscan_fit_seconds=workload['dbscan_fit']['median'],
                                cold_fresh_seconds=setup+workload['kmeans']['median'],
                                cold_query_seconds=setup+workload['point_norm_preparation']['median']+query_seconds,
                                cold_knn_seconds=setup+workload['point_norm_preparation']['median']+workload['knn']['median'],
                                cold_assignment_seconds=setup+shared_centers+workload['center_projection']['median']+
                                    workload['numeric_preparation']['median']+workload['assignment']['median'],
                                cold_refine_seconds=setup+shared_centers+workload['center_projection']['median']+workload['kmeans_refine']['median'],
                                cold_dbscan_seconds=setup+workload['point_norm_preparation']['median']+
                                    workload['dbscan_pair_build']['median']+workload['dbscan_fit']['median'],any_iteration_cap=cap_hit)
                            files=list(folder.rglob('*.npz'))+[folder/'timings.json',model_file]
                            row['artifacts']={str(p.relative_to(root)):file_hash(p) for p in files}
                            atomic_json(target,row);manifest['completed_cases']=len(list((root/'rows').glob('*.json')))
                            manifest['elapsed_seconds']=time.monotonic()-started;atomic_json(path,manifest)
                            print(json.dumps(dict(event='case',case=key,done=manifest['completed_cases'],elapsed=manifest['elapsed_seconds'])),flush=True)
                            del model,z;gc.collect()
                        aggregate(root,draws=300)
                del x,train,test_x,validation;gc.collect()
            assert manifest['completed_cases']==expected
            manifest.update(complete=True,expected_cases=expected,elapsed_seconds=time.monotonic()-started)
            aggregate(root);atomic_json(path,manifest)
            print(json.dumps(dict(event='complete',cases=expected,seconds=manifest['elapsed_seconds'])),flush=True)
    except BaseException as error:
        manifest.update(complete=False,stop_reason=f'{type(error).__name__}: {error}',elapsed_seconds=time.monotonic()-started)
        atomic_json(path,manifest);aggregate(root,draws=300);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,default=DATASETS)
    parser.add_argument('--k',nargs='+',type=int,default=[5,10])
    parser.add_argument('--m',nargs='+',type=int,default=[32,64,128,256,512])
    parser.add_argument('--seeds',type=int,default=3);parser.add_argument('--starts',type=int,default=3)
    parser.add_argument('--seed-offset',type=int,default=30);parser.add_argument('--start-offset',type=int,default=30)
    parser.add_argument('--fit-repetitions',type=int,default=3);parser.add_argument('--timing-repetitions',type=int,default=3)
    parser.add_argument('--queries',type=int,default=32);parser.add_argument('--dbscan-points',type=int,default=256)
    parser.add_argument('--threads',type=int,default=4);parser.add_argument('--max-seconds',type=float,default=10800)
    parser.add_argument('--max-rss-gib',type=float,default=2.5);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if min(args.k+args.m+[args.seeds,args.starts,args.fit_repetitions,args.timing_repetitions,args.queries,args.threads])<1:
        parser.error('Counts must be positive')
    if min(args.seed_offset,args.start_offset)<0:parser.error('Offsets must be nonnegative')
    if args.smoke:
        args.datasets=['mnist'];args.m=[24,32];args.seeds=1;args.starts=1
        args.fit_repetitions=1;args.timing_repetitions=1;args.queries=2;args.dbscan_points=32
    with exclusive_run(args.out.resolve()):run(args)


if __name__=='__main__':main()
