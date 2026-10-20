"""Exact-integer diagnostic on explicit prefixes of preserved paper subsets."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
from commitjl.protected_jl import centroid_basis,integer_protected_project,weighted_integer_assignment
from commitjl.sparse_gen import sparse_matrix_gen_csr
from commitjl.clustering import nearest_centroid_assign
from experiments.protected_jl_study import Archive,DATASETS,phi_from_labels

CODE=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--n',type=int,default=128)
    ap.add_argument('--scales',nargs='+',type=int,default=[64,256,1024])
    ap.add_argument('--starts',type=int,default=2)
    ap.add_argument('--seeds',type=int,default=3)
    ap.add_argument('--k',nargs='+',type=int,default=[5,10])
    ap.add_argument('--full-rank-only',action='store_true')
    args=ap.parse_args()
    if args.out.exists() or min(args.n,args.starts,args.seeds,*args.scales)<1 or args.starts>5: ap.error('Use a new directory and valid limits')
    args.out.mkdir(parents=True);archive=Archive();rows=[];begin=time.perf_counter()
    with threadpool_limits(limits=4):
        for name in DATASETS:
            prefix=f'results/paper_server_20261006/rq1/{name}/artifacts/'
            x=archive.load(prefix+'dataset.npz')['X'];pca_all=archive.load(prefix+'pca.npz')['components']
            for k in args.k:
                for start in range(args.starts):
                    baseline=archive.load(prefix+f'full_k{k}_start{start}.npz');c=baseline['centroids']
                    factor=4096/max(float(np.max(np.abs(x))),float(np.max(np.abs(c))),1e-15)
                    xq=np.rint(x[:args.n]*factor).astype(np.int64);cq=np.rint(c*factor).astype(np.int64)
                    original=baseline['labels'][:len(xq)];integer_full=nearest_centroid_assign(xq,cq)
                    u=centroid_basis(cq);ranks=sorted(set([0,1,min(2,u.shape[1]),u.shape[1]]))
                    if args.full_rank_only:ranks=[u.shape[1]]
                    physical=x[:len(xq)];norm_sum=float(np.sum(physical*physical))
                    original_distances=baseline['squared_distances'][:len(xq)]
                    original_cost=float(np.sum(original_distances.min(axis=1)))
                    pca_m=16+u.shape[1] if args.full_rank_only else 16
                    pca=pca_all[:pca_m]
                    def record(method,labels,scale,seed,r):
                        rows.append(dict(dataset=name,n=len(xq),d=xq.shape[1],k=k,start=start,method=method,
                            scale=scale,seed=seed,protected_rank=r,physical_scale=factor,
                            ari_original=float(adjusted_rand_score(original,labels)),
                            ari_integer_full=float(adjusted_rand_score(integer_full,labels)),
                            agreement_original=float(np.mean(original==labels)),
                            agreement_integer_full=float(np.mean(integer_full==labels)),
                            r_phi=phi_from_labels(physical,labels,k,norm_sum)/original_cost,
                            fixed_center_ratio=float(np.sum(original_distances[np.arange(len(labels)),labels]))/original_cost))
                    record('integer_full',integer_full,0,-1,0)
                    for scale in args.scales:
                        p=np.rint(scale*pca).astype(np.int64)
                        record(f'integer_pca{pca_m}',nearest_centroid_assign(xq@p.T,cq@p.T),scale,-1,0)
                        for r in ranks[1:]:
                            b=np.rint(scale*u[:,:r].T).astype(np.int64)
                            record('protected_only',nearest_centroid_assign(xq@b.T,cq@b.T),scale,-1,r)
                        for seed in range(args.seeds):
                            map_seed=hashlib.sha256(f'protected-precision:{name}:{k}:{start}:{seed}'.encode()).digest()
                            _,a=sparse_matrix_gen_csr(map_seed,16,xq.shape[1],1)
                            for r in ranks:
                                b=np.rint(scale*u[:,:r].T).astype(np.int64)
                                z=integer_protected_project(xq,b,a,scale);centers=integer_protected_project(cq,b,a,scale)
                                labels,_=weighted_integer_assignment(z,centers,r,1)
                                record('integer_sparse' if r==0 else 'integer_hybrid',labels,scale,seed,r)
            pd.DataFrame(rows).to_csv(args.out/'rows.csv',index=False)
            print(json.dumps(dict(dataset=name,rows=len(rows),elapsed_seconds=time.perf_counter()-begin)),flush=True)
    frame=pd.DataFrame(rows)
    frame.groupby(['dataset','k','method','protected_rank','scale'])[['ari_original','ari_integer_full','agreement_original','agreement_integer_full']].agg(['mean','min']).to_csv(args.out/'summary.csv')
    (args.out/'manifest.json').write_text(json.dumps(dict(complete=True,rows=len(rows),wall_seconds=time.perf_counter()-begin,
        scope=f'Precision diagnostic, first min({args.n},n) rows of each preserved paper subset; {args.starts} saved starts, {args.seeds} fresh maps, k={args.k}; no proof timing.',
        input_bound=4096,residual_m=16,q=1,scales=args.scales,
        sources={str(p.relative_to(CODE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),CODE/'commitjl/protected_jl.py']}),indent=2))

if __name__=='__main__':main()
