"""Bounded downstream test: compress once, then compute all pair distances."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiments.laptop_comparison import archived, atomic_json, relative_errors, save_arrays
from commitjl.protected_jl import protected_project
from commitjl.sparse_gen import sparse_matrix_gen_csr


def all_pairs(x):
    norms=np.sum(x*x,axis=1)
    return np.maximum(norms[:,None]+norms[None,:]-2*(x@x.T),0)


def run(root):
    root=Path(root).resolve();out=root/'pair_workload';out.mkdir(exist_ok=True)
    summary=pd.read_csv(root/'geometry_summary.csv');records=[]
    for folder in sorted((root/'jobs').glob('geometry_*')):
        dataset=folder.name.removeprefix('geometry_')
        candidate=summary[(summary.dataset==dataset)&(summary.k==5)&
            (summary.method=='hybrid_r4')&summary.selectable].sort_values(['total_m','q'])
        if not len(candidate):continue
        choice=candidate.iloc[0]
        row_path=folder/'rows'/f'k5_s3_m{int(choice.total_m)}_r4_q{int(choice.q)}_seed100.json'
        original=json.loads(row_path.read_text())
        archive,prefix,data,pca=archived(dataset)
        assert hashlib.sha256(np.ascontiguousarray(data['X']).tobytes()).hexdigest()==original['dataset_sha256']
        indices=np.sort(np.random.default_rng(333).choice(len(data['X']),min(1024,len(data['X'])),replace=False))
        x=data['X'][indices].copy();del data
        with np.load(folder/'centers_k5_start3.npz') as saved:
            u=saved['basis'];origin=saved['centroids'][0]
        _,a=sparse_matrix_gen_csr(bytes.fromhex(original['map_seed']),original['residual_m'],original['d'],original['q'])
        a=a.astype(float);a.data/=np.sqrt(original['q'])
        projectors=dict(full=lambda:x,hybrid=lambda:protected_project(x,u,a,origin),
            protected_only=lambda:protected_project(x,u,None,origin))
        if original['total_m']<=len(pca):
            matrix=pca[:original['total_m']];projectors['pca_same_m']=lambda:x@matrix.T
        reference=all_pairs(x);triangle=np.triu_indices(len(x),1);reference_values=reference[triangle]
        save_arrays(out/'arrays'/f'{dataset}_reference.npz',indices=indices,
            pair_i=triangle[0],pair_j=triangle[1],original_pair_sq=reference_values)
        for method,projector in projectors.items():
            destination=out/'rows'/f'{dataset}_{method}.json'
            if destination.exists():records.append(json.loads(destination.read_text()));continue
            all_pairs(projector()) # warm-up
            samples=[];projection_samples=[];pair_samples=[]
            for _ in range(3):
                begin=time.perf_counter();z=projector();middle=time.perf_counter()
                projected=all_pairs(z);end=time.perf_counter()
                samples.append(end-begin);projection_samples.append(middle-begin);pair_samples.append(end-middle)
            values=projected[triangle];errors=relative_errors(reference_values,values)
            item=dict(dataset=dataset,method=method,n=len(x),d=x.shape[1],total_m=z.shape[1],
                pair_count=len(values),q=original['q'] if method=='hybrid' else 0,
                seconds_median=statistics.median(samples),seconds_samples=samples,
                projection_seconds_median=statistics.median(projection_samples),
                pair_seconds_median=statistics.median(pair_samples),pair_q95=errors['q95'],
                zero_reference_violations=errors['zero_reference_violations'],
                pair_cost_ratio=float(values.sum()/reference_values.sum()),
                parent_geometry_row=str(row_path.relative_to(root)),
                scope='All pairs on deterministic sample min(1024,n); BLAS; 3 warm measurements; preprocessing excluded',
                full_subset_hybrid_qualified=bool(choice.selectable))
            save_arrays(out/'arrays'/f'{dataset}_{method}.npz',Z=z,projected_pair_sq=values)
            atomic_json(destination,item);records.append(item)
        print(f'pair workload {dataset}: {len(x)} points, {len(reference_values)} pairs',flush=True)
        del reference,reference_values,x,z,projected,pca
    frame=pd.DataFrame(records)
    for dataset,g in frame.groupby('dataset'):
        full=float(g[g.method=='full'].seconds_median.iloc[0])
        frame.loc[g.index,'speedup_vs_full']=full/g.seconds_median
    frame.to_csv(out/'measurements.csv',index=False)
    atomic_json(out/'validation.json',dict(complete=True,rows=len(records),threads=4,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='Native downstream workload, not a proof benchmark; selected points are a subset.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('results/laptop_comparison_20261011'))
    with threadpool_limits(limits=4):run(parser.parse_args().out)
