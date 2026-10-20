"""Validation-only text ablation. Never promotes a variant using test labels.

Weights below one deliberately change residual geometry, and are not claimed
to preserve the original Euclidean JL guarantee. This is a separate candidate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from commitjl.enriched_jl import enrich_centroid_basis
from commitjl.reusable_native import NativeProjection, array_hash, file_hash
from commitjl.sparse_gen import sparse_matrix_gen_csr
from experiments.comprehensive_native_comparison import load_dataset
from experiments.enriched_jl_comparison import cached_fit
from experiments.laptop_comparison import atomic_json, save_arrays
from experiments.paper_compression import split_indices, bootstrap_crossed


def cluster(points,initials):
    fits=[KMeans(n_clusters=5,init=points[ids].copy(),n_init=1,max_iter=300,tol=1e-4,
        algorithm='lloyd',random_state=0).fit(points) for ids in initials]
    return min(fits,key=lambda fit:fit.inertia_),any(f.n_iter_>=300 for f in fits)


def run(args):
    root=args.out.resolve();root.mkdir(parents=True,exist_ok=True);(root/'rows').mkdir(exist_ok=True)
    code=Path(__file__).resolve().parents[1]
    files=[Path(__file__),code/'commitjl/enriched_jl.py',code/'commitjl/reusable_native.py',
        code/'commitjl/sparse_gen.py',code/'commitjl/hashing.py',code/'experiments/comprehensive_native_comparison.py']
    identity=dict(version='text-residual-ablation-v1',widths=[128,256],extra=[12,32,64],q=[1,4],
        weight=[1.,.5,.25],seeds=list(range(40,40+args.seeds)),starts=list(range(40,40+args.starts)),
        init_budgets=[1,5],scope='Training-only learning; validation-only evaluation; test rows unused',
        sources={str(p.relative_to(code)):file_hash(p) for p in files})
    manifest_path=root/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())['identity']!=identity:
        raise ValueError('Changed source/configuration; choose new folder')
    started=time.monotonic();atomic_json(manifest_path,dict(identity=identity,complete=False))
    x,y,meta=load_dataset('20ng_tfidf_nonzero');folds=split_indices(len(x))
    train=x[folds['train']];validation=x[folds['validation']];labels=y[folds['validation']]
    save_arrays(root/'indices.npz',**folds,source_indices=meta['source_rows'])
    rng=np.random.default_rng(777);pairs=rng.integers(len(validation),size=(2000,2));pairs=pairs[pairs[:,0]!=pairs[:,1]]
    reference=np.sum((validation[pairs[:,0]]-validation[pairs[:,1]])**2,axis=1);keep=reference>0
    with threadpool_limits(limits=4):
        for start in identity['starts']:
            initials=[np.random.default_rng(900+start+1000*i).choice(len(validation),5,replace=False) for i in range(5)]
            full={budget:cluster(validation,initials[:budget]) for budget in [1,5]}
            def fit_centers():
                fit=KMeans(n_clusters=5,n_init=1,random_state=start,max_iter=300,algorithm='lloyd').fit(train)
                return dict(centers=fit.cluster_centers_),dict(iterations=int(fit.n_iter_))
            ca,cm=cached_fit(root,f'centers_s{start}',fit_centers,1);centers=ca['centers']
            learned={}
            for extra in identity['extra']:
                def fit_basis():
                    u,description=enrich_centroid_basis(train,centers,extra,random_state=1701+start)
                    return dict(basis=u),description
                a,b=cached_fit(root,f'extra{extra}_s{start}',fit_basis,1)
                learned[extra]=(a['basis'],cm['timing']['median']+b['timing']['median'])
            for budget in [1,5]:
                save_arrays(root/'full'/f's{start}_init{budget}.npz',labels=full[budget][0].labels_)
            specs=[]
            for m in identity['widths']:
                for seed in identity['seeds']:
                    specs += [(method,m,seed,0,0,1.) for method in ['pca_fast','lsa']]
                    specs += [('protected',m,seed,extra,q,weight) for extra in identity['extra']
                        for q in identity['q'] for weight in identity['weight']]
            for index in np.random.default_rng(start).permutation(len(specs)):
                method,m,seed,extra,q,weight=specs[index]
                key=f's{start}_{method}_m{m}_seed{seed}_extra{extra}_q{q}_w{weight}'
                if all((root/'rows'/f'{key}_init{budget}.json').exists() for budget in [1,5]):continue
                fit_seconds=0.;begin=time.perf_counter()
                if method in ['pca_fast','lsa']:
                    def fit_reduction():
                        if method=='pca_fast':
                            p=PCA(n_components=m,svd_solver='randomized',iterated_power=2,n_oversamples=10,random_state=90125+seed).fit(train)
                            origin=p.mean_
                        else:
                            p=TruncatedSVD(n_components=m,n_iter=2,random_state=90125+seed).fit(train);origin=np.zeros(x.shape[1])
                        return dict(components=p.components_,mean=origin),dict(method=method,m=m)
                    a,b=cached_fit(root,f'{method}_m{m}_seed{seed}',fit_reduction,1);fit_seconds=b['timing']['median']
                    begin=time.perf_counter();model=NativeProjection('pca',x.shape[1],components=a['components'],origin=a['mean'])
                else:
                    u,fit_seconds=learned[extra]
                    payload=dict(source=array_hash(x),start=start,seed=seed,m=m,q=q,basis=array_hash(u))
                    map_seed=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).digest()
                    begin=time.perf_counter();_,a=sparse_matrix_gen_csr(map_seed,m-u.shape[1],x.shape[1],q)
                    a=a.astype(float);a.data*=weight/np.sqrt(q)
                    model=NativeProjection('hybrid',x.shape[1],basis=u,sparse_map=a,origin=centers[0])
                construction=time.perf_counter()-begin;begin=time.perf_counter();z=model.transform(validation)
                conversion=time.perf_counter()-begin
                projected_pairs=np.sum((z[pairs[:,0]]-z[pairs[:,1]])**2,axis=1)
                pair_q95=float(np.quantile(np.abs(projected_pairs[keep]/reference[keep]-1),.95))
                for budget in [1,5]:
                    begin=time.perf_counter();fitted,capped=cluster(z,initials[:budget]);seconds=time.perf_counter()-begin
                    counts=np.bincount(fitted.labels_,minlength=5)
                    sums=np.zeros((5,x.shape[1]));np.add.at(sums,fitted.labels_,validation)
                    valid=counts>0;partition=float(np.sum(validation**2)-np.sum(sums[valid]**2/counts[valid,None]))
                    full_fit=full[budget][0];full_counts=np.bincount(full_fit.labels_,minlength=5)
                    full_sums=np.zeros_like(sums);np.add.at(full_sums,full_fit.labels_,validation);valid=full_counts>0
                    full_partition=float(np.sum(validation**2)-np.sum(full_sums[valid]**2/full_counts[valid,None]))
                    row=dict(method=method,m=m,seed=seed,start=start,extra=extra,q=q,weight=weight,init_budget=budget,
                        validation_ari=float(adjusted_rand_score(full_fit.labels_,fitted.labels_)),
                        ground_truth_ari=float(adjusted_rand_score(labels,fitted.labels_)),pair_q95=pair_q95,
                        original_partition_sse_ratio=partition/full_partition,
                        model_fit_seconds=fit_seconds,construction_seconds=construction,conversion_seconds=conversion,
                        clustering_seconds=seconds,any_iteration_cap=capped or full[budget][1],
                        scope='validation-only; single observed timings are diagnostic, not paper speed evidence')
                    save_arrays(root/'artifacts'/f'{key}_init{budget}.npz',labels=fitted.labels_,centers=fitted.cluster_centers_,
                        initial_indices=np.stack(initials[:budget]),**{'model_'+n:v for n,v in model.parameters().items()})
                    atomic_json(root/'rows'/f'{key}_init{budget}.json',row)
            print(json.dumps(dict(event='start_complete',start=start,rows=len(list((root/'rows').glob('*.json'))))),flush=True)
    rows=[json.loads(p.read_text()) for p in (root/'rows').glob('*.json')];frame=pd.DataFrame(rows)
    frame.to_csv(root/'rows.csv',index=False);summary=[]
    for values,g in frame.groupby(['method','m','extra','q','weight','init_budget']):
        entry=dict(zip(['method','m','extra','q','weight','init_budget'],values),cases=len(g))
        for metric in ['validation_ari','ground_truth_ari','pair_q95','original_partition_sse_ratio']:
            mean,lo,hi=bootstrap_crossed(g.pivot(index='seed',columns='start',values=metric).to_numpy(),draws=1000)
            entry.update({metric:mean,metric+'_ci_low':lo,metric+'_ci_high':hi})
        summary.append(entry)
    pd.DataFrame(summary).to_csv(root/'summary.csv',index=False)
    expected=args.starts*args.seeds*2*(2+3*2*3)*2
    assert len(rows)==expected
    atomic_json(manifest_path,dict(identity=identity,complete=True,cases=len(rows),expected_cases=expected,
        elapsed_seconds=time.monotonic()-started,test_evaluations=0))
    print(json.dumps(dict(event='complete',cases=len(rows),seconds=time.monotonic()-started)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seeds',type=int,default=3);parser.add_argument('--starts',type=int,default=3)
    args=parser.parse_args()
    if min(args.seeds,args.starts)<1:parser.error('Counts must be positive')
    run(args)


if __name__=='__main__':main()
