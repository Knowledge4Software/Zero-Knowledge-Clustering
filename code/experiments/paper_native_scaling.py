"""Controlled native n/d scaling at equal m, with a fixed trained model per d.

This engineering supplement separates fit, construction, compression, norms,
assignment and query-to-corpus distance costs. It measures no ZK proof costs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from campaigns.paper_compression import CODE,identity,configuration
from commitjl.reusable_native import NativeProjection,PreparedDataset,array_hash,file_hash
from experiments.laptop_comparison import atomic_json,save_arrays
from experiments.paper_compression import (centroid_basis,create_model,projection_seed,split_indices,timed,
    paired_timed,assignment,distortion,sqdist)
from experiments.protected_jl_study import Archive


def run(args):
    root=args.out.resolve();root.mkdir(parents=True,exist_ok=True)
    source=identity(configuration('pilot'),args.datasets)
    source['sources']['experiments/paper_native_scaling.py']=file_hash(__file__)
    cfg=dict(profile=args.profile,m=32 if args.profile=='check' else 128,seeds=1 if args.profile=='check' else 3,
        repetitions=1 if args.profile=='check' else 3,threads=4,k=5,max_iter=300,
        n_grid=[128,512] if args.profile=='check' else [128,256,512,1000,2000,4000],
        d_grid=[64,128] if args.profile=='check' else [256,512,1024,2000,4000],
        add_full_shape=args.profile!='check')
    binding=dict(configuration=cfg,source=source)
    manifest_path=root/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())['binding']!=binding:
        raise ValueError('Scaling source/configuration changed; use a fresh directory')
    manifest=dict(binding=binding,complete=False,scope='Conditional reuse scaling; model fitted on fixed source training rows per d, not refitted per n; paired seed/start draws, not crossed-grid confidence intervals')
    atomic_json(manifest_path,manifest)
    for name in source['sources']:
        target=root/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((CODE/name).read_bytes())
    archive=Archive();began=time.monotonic();deadline=began+args.max_minutes*60;rows=[];expected=[]
    methods=('full','pca','pca_randomized','sparse_jl','protected_jl','protected_only')
    def budget():
        if time.monotonic()>deadline:raise TimeoutError('Scaling budget; completed rows are preserved')
    try:
        for dataset in args.datasets:
            original=archive.load(f'results/paper_server_20261006/rq1/{dataset}/artifacts/dataset.npz')['X']
            source_manifest=json.loads((CODE/f'results/server_paper_20261006/rq1/{dataset}/manifest.json').read_text())
            original_sha=array_hash(original)
            if original_sha!=source_manifest['dataset']['sha256']:raise ValueError('Archive input binding failed')
            folds=split_indices(len(original));columns=np.random.default_rng(1901).permutation(original.shape[1])
            order=np.random.default_rng(1902).permutation(len(original))
            ns=sorted(set([n for n in cfg['n_grid'] if n<=len(original)]+([len(original)] if cfg['add_full_shape'] else [])))
            ds=sorted(set([d for d in cfg['d_grid'] if cfg['m']<d<=original.shape[1]]+
                ([original.shape[1]] if cfg['add_full_shape'] else [])))
            expected += [f'{dataset}_d{d}_s{draw}_{method}_n{n}' for d in ds
                for draw in range(cfg['seeds']) for method in methods for n in ns]
            for d in ds:
                budget();feature_ids=np.sort(columns[:d]);x=original[:,feature_ids];train=x[folds['train']];x_sha=array_hash(x)
                scope=root/dataset/f'd{d}';scope.mkdir(parents=True,exist_ok=True)
                if not (scope/'indices.npz').exists():
                    save_arrays(scope/'indices.npz',feature_indices=feature_ids,source_row_order=order,**folds)
                with np.load(scope/'indices.npz') as stored:
                    for key,value in dict(feature_indices=feature_ids,source_row_order=order,**folds).items():
                        np.testing.assert_array_equal(stored[key],value)
                for draw in range(cfg['seeds']):
                    budget();fit_path=scope/f'fit{draw}.json';params_path=scope/f'fit{draw}.npz'
                    if fit_path.exists():
                        fits=json.loads(fit_path.read_text())
                        if file_hash(params_path)!=fits['artifact_sha256']:raise ValueError('Corrupt shared scaling fit')
                        with np.load(params_path) as p:
                            c,u,pca_c,mean,random_c,random_mean=[p[key] for key in ('centers','basis','components','mean','random_components','random_mean')]
                    else:
                        km,ct=timed(lambda:KMeans(n_clusters=cfg['k'],n_init=1,random_state=draw,max_iter=cfg['max_iter']).fit(train),cfg['repetitions'],warmup=True)
                        c=km.cluster_centers_;u,bt=timed(lambda:centroid_basis(c),cfg['repetitions'],warmup=True)
                        pca,pt=timed(lambda:PCA(n_components=cfg['m'],svd_solver='full').fit(train),cfg['repetitions'],warmup=True)
                        randomized,rpt=timed(lambda:PCA(n_components=cfg['m'],svd_solver='randomized',random_state=draw).fit(train),cfg['repetitions'],warmup=True)
                        pca_c,mean=pca.components_,pca.mean_;random_c,random_mean=randomized.components_,randomized.mean_
                        save_arrays(params_path,centers=c,basis=u,components=pca_c,mean=mean,random_components=random_c,random_mean=random_mean)
                        fits=dict(centroid_training=ct,basis_fit=bt,pca_fit=pt,randomized_pca_fit=rpt,
                            training_rows=len(train),centroid_cap_hit=bool(km.n_iter_>=cfg['max_iter']),artifact_sha256=file_hash(params_path))
                        atomic_json(fit_path,fits)
                    for method in methods:
                        m=d if method=='full' else u.shape[1] if method=='protected_only' else cfg['m']
                        q=int(method in ('sparse_jl','protected_jl'));residual=m-u.shape[1] if method=='protected_jl' else m
                        seed=projection_seed(x_sha,c if method=='protected_jl' else None,'scaling',draw,residual,q,draw)
                        components,origin=(random_c,random_mean) if method=='pca_randomized' else (pca_c,mean)
                        model,map_t=timed(lambda:create_model(method,d,m,q,seed,c,u,components,origin))
                        model_path=scope/f'{method}_{draw}_model.npz'
                        parameters=model.parameters();valid=False
                        if model_path.exists():
                            try:
                                with np.load(model_path,allow_pickle=False) as stored:
                                    valid=set(stored.files)==set(parameters) and all(np.array_equal(stored[p],v) for p,v in parameters.items())
                            except (OSError,ValueError):pass
                        if not valid:save_arrays(model_path,**parameters)
                        pc=model.transform(c)
                        for n in ns:
                            budget();key=f'{dataset}_d{d}_s{draw}_{method}_n{n}'
                            path=root/'rows'/f'{key}.json'
                            if path.exists():
                                row=json.loads(path.read_text())
                                if all((root/p).exists() and file_hash(root/p)==h for p,h in row['artifact_sha256'].items()):rows.append(row);continue
                            ids=np.sort(order[:n]);points=x[ids]
                            if method=='full':z=points;compression=dict(samples=[0.]*cfg['repetitions'],median=0.)
                            else:z,compression=timed(lambda:model.transform(points),cfg['repetitions'],warmup=True)
                            zn,zt=timed(lambda:np.einsum('ij,ij->i',z,z),cfg['repetitions'],warmup=True)
                            xn,xt=timed(lambda:np.einsum('ij,ij->i',points,points),cfg['repetitions'],warmup=True)
                            cn=np.einsum('ij,ij->i',c,c);pcn=np.einsum('ij,ij->i',pc,pc)
                            rng=np.random.default_rng(711)
                            labels,old_labels,at,fat=paired_timed(lambda:assignment(z,pc,zn,pcn),lambda:assignment(points,c,xn,cn),cfg['repetitions'],rng)
                            state=PreparedDataset(model,z,zn,array_hash(points));full=PreparedDataset(NativeProjection('full',d),points,xn,array_hash(points))
                            left=np.arange(min(64,n));right=np.arange(n)
                            values,reference,dt,fdt=paired_timed(lambda:state.squared_distances(left,right),lambda:full.squared_distances(left,right),cfg['repetitions'],rng)
                            errors=distortion(reference,values,.15,64*np.finfo(float).eps*max(1.,float(xn.max())))
                            artifact=scope/f'{method}_{draw}_n{n}.npz'
                            save_arrays(artifact,source_indices=ids,left_indices=left,right_indices=right,
                                original_query_sq=reference,projected_query_sq=values,original_labels=old_labels[0],projected_labels=labels[0])
                            row=dict(dataset=dataset,d=d,n=n,m=m,method=method,draw=draw,q=q,source_sha256=original_sha,
                                model_sha256=model.fingerprint,training_rows=len(train),centroid_cap_hit=fits['centroid_cap_hit'],
                                map_construction=map_t,compression=compression,point_norm_preparation=zt,full_point_norm_preparation=xt,
                                assignment=at,full_assignment=fat,distance_query=dt,full_distance_query=fdt,
                                agreement=float(np.mean(labels[0]==old_labels[0])),ari=float(adjusted_rand_score(old_labels[0],labels[0])),
                                query_distortion=errors,representation_bytes=z.nbytes,original_bytes=points.nbytes,
                                fits=fit_path.relative_to(root).as_posix(),artifact_sha256={p.relative_to(root).as_posix():file_hash(p) for p in (artifact,model_path,params_path,scope/'indices.npz',fit_path)})
                            atomic_json(path,row);rows.append(row)
                print(json.dumps(dict(dataset=dataset,d=d,completed_rows=len(rows))),flush=True)
        manifest['complete']=True
    except TimeoutError as exc:manifest['limit_reason']=str(exc)
    except Exception as exc:
        manifest['failure_reason']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        manifest.update(wall_seconds=time.monotonic()-began,rows=len(rows),expected_processed_cells=len(expected),
            expected_cells=expected,inventory_scope='Datasets enumerated before the budget ended; complete requires all requested datasets')
        atomic_json(manifest_path,manifest)
        records=[]
        for row in rows:
            records.append(dict({k:v for k,v in row.items() if not isinstance(v,(dict,list))},
                **{stage+'_seconds':row[stage]['median'] for stage in ('map_construction','compression','point_norm_preparation',
                    'full_point_norm_preparation','assignment','full_assignment','distance_query','full_distance_query')},pair_q95=row['query_distortion']['q95']))
        pd.DataFrame(records).to_csv(root/'summary.csv',index=False)
    print(json.dumps(dict(complete=manifest['complete'],rows=len(rows),out=str(root))))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--profile',choices=['check','paper'],default='paper')
    parser.add_argument('--datasets',nargs='+',choices=['mnist','gisette','20ng_tfidf'],default=['mnist','gisette','20ng_tfidf'])
    parser.add_argument('--max-minutes',type=float,default=30);args=parser.parse_args()
    if args.max_minutes<=0:parser.error('Positive budget required')
    from campaigns.paper_compression import exclusive_run
    with exclusive_run(args.out),threadpool_limits(limits=4):run(args)


if __name__=='__main__':main()
