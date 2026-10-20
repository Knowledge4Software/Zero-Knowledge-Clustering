"""Bounded, offline research screen using exact saved server inputs/centers.

No clustering retraining, no changes to server records, no paper superiority
claim. Paired fixed-centroid assignment, crossed map seeds and saved starts.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
from commitjl.protected_jl import centroid_basis, protected_project
from commitjl.sparse_gen import sparse_matrix_gen_csr
from commitjl.clustering import _squared_distances
from experiments.rq1_complete import bootstrap_crossed

CODE = Path(__file__).resolve().parents[1]
DATASETS = ['gaussian_easy','gaussian_hard','dbscan_margin','mnist','fashion_mnist','gisette','arcene','20ng_tfidf']


class Archive:
    def __init__(self):
        self.base = CODE/'results/server_retirement_20261007/compact_arrays'
        self.index = json.loads((self.base/'array_index.json').read_text())['files']

    def load(self, relative):
        result = {}
        for key, entry in self.index[relative]['arrays'].items():
            with gzip.open(self.base/'objects'/(entry['object']+'.npy.gz'),'rb') as stream:
                payload = stream.read()
            if hashlib.sha256(payload).hexdigest() != entry['object']:
                raise ValueError('Preserved array hash mismatch')
            result[key] = np.load(io.BytesIO(payload),allow_pickle=False)
        return result


def phi_from_labels(x, labels, k, norm_sum):
    # Sufficient statistics, no n*k*d difference tensor.
    weights = (labels[:,None] == np.arange(k)).astype(float)
    counts = weights.sum(axis=0)
    sums = weights.T @ x
    keep = counts > 0
    centers = sums[keep]/counts[keep,None]
    # Same paper metric: Phi_X of reconstructed centroids, allowing nearest
    # center reassignment; this differs from partition SSE.
    sq=np.sum(x*x,axis=1)[:,None] + np.sum(centers*centers,axis=1)[None,:] - 2*(x@centers.T)
    return float(np.sum(np.maximum(sq.min(axis=1),0)))


def summarize(out, draws):
    data = pd.read_csv(out/'rows.csv')
    entries = []
    keys = ['dataset','k','method','protected_rank','residual_m','q','total_m']
    for values, group in data.groupby(keys,dropna=False):
        entry = dict(zip(keys,values))
        for metric in ['ari','r_phi','fixed_center_ratio','distortion_q95','agreement','projection_seconds','assignment_seconds']:
            matrix = group.pivot(index='seed',columns='start',values=metric).to_numpy()
            mean,low,high = bootstrap_crossed(matrix,draws=draws)
            entry.update({metric+'_mean':mean,metric+'_ci_low':low,metric+'_ci_high':high})
        entry.update(rows=len(group),qualified=entry['ari_ci_low']>=.8 and entry['r_phi_ci_high']<=1.1)
        entries.append(entry)
    summary = pd.DataFrame(entries)
    summary.to_csv(out/'summary.csv',index=False)
    selected = summary[summary.qualified].sort_values(['total_m','q']).groupby(['dataset','k','method']).head(1)
    selected.to_csv(out/'selected.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(16,8),sharey=True)
    for ax,dataset in zip(axes.flat,DATASETS):
        g=summary[(summary.dataset==dataset)&(summary.k==5)]
        for method,part in g.groupby('method'):
            # Plot the best mean at each dimension; explicit exploratory envelope.
            envelope=part.groupby('total_m').ari_mean.max()
            ax.plot(envelope.index,envelope.values,marker='o',label=method)
        ax.set_title(dataset); ax.set_xscale('log',base=2); ax.set_ylim(0,1.03); ax.grid(alpha=.2)
        ax.set_xlabel('Total output dimensions')
    axes[0,0].set_ylabel('ARI against saved Full labels')
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Exploratory quality screen, k=5 (best configuration per dimension)')
    fig.tight_layout(); fig.savefig(out/'quality.png',dpi=180); plt.close(fig)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=DATASETS)
    p.add_argument('--seeds',type=int,default=5)
    p.add_argument('--starts',type=int,default=5)
    p.add_argument('--m',nargs='+',type=int,default=[16,32,64,128])
    p.add_argument('--q',nargs='+',type=int,default=[1,2])
    p.add_argument('--k',nargs='+',type=int,default=[5,10])
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--bootstrap',type=int,default=500)
    p.add_argument('--max-seconds',type=float,default=3600)
    args=p.parse_args()
    if min(args.seeds,args.starts,args.threads,args.bootstrap,*args.m,*args.q)<1 or args.starts>5 or args.max_seconds<=0:
        p.error('Invalid limits; only five original starts exist')
    if args.out.exists(): p.error('Use a new output directory')
    args.out.mkdir(parents=True)
    archive=Archive(); rows=[]; begin=time.perf_counter()
    manifest=dict(configuration={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()},
        complete=False,scope='Exploratory fixed-centroid fidelity screen; pointwise crossed bootstrap; not a proof benchmark',
        sources={str(path.relative_to(CODE)):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                 [Path(__file__),CODE/'commitjl/protected_jl.py',CODE/'commitjl/sparse_gen.py']},
        seed_order='Basis determined from saved public centroids before fresh deterministic challenge; native simulation only',
        environment=dict(numpy=np.__version__,threads=args.threads),datasets={})
    def persist():
        pd.DataFrame(rows).to_csv(args.out/'rows.csv',index=False)
        (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    with threadpool_limits(limits=args.threads):
        for name in args.datasets:
            prefix=f'results/paper_server_20261006/rq1/{name}/artifacts/'
            saved=archive.load(prefix+'dataset.npz'); x=saved['X']; pairs=saved['pairs']; n,d=x.shape
            old_manifest=json.loads((CODE/f'results/server_paper_20261006/rq1/{name}/manifest.json').read_text())
            x_hash=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
            if x_hash!=old_manifest['dataset']['sha256']: raise ValueError('Wrong original data')
            manifest['datasets'][name]=dict(shape=list(x.shape),sha256=x_hash)
            norm_sum=float(np.sum(x*x))
            original_pair_sq=np.sum((x[pairs[:,0]]-x[pairs[:,1]])**2,axis=1)
            components=archive.load(prefix+'pca.npz')['components']
            with gzip.open(args.out/f'{name}_labels.u16.gz','wb') as label_file:
                label_offset=0
                for k in args.k:
                    for start in range(args.starts):
                        full=archive.load(prefix+f'full_k{k}_start{start}.npz')
                        c=full['centroids']; labels=full['labels']; distances=full['squared_distances']
                        fixed_ref=float(np.sum(distances[np.arange(n),labels]))
                        phi_ref=fixed_ref
                        basis_begin=time.perf_counter(); complete_basis=centroid_basis(c); basis_seconds=time.perf_counter()-basis_begin
                        ranks=sorted(set([0,min(1,complete_basis.shape[1]),min(2,complete_basis.shape[1]),complete_basis.shape[1]]))
                        artifact=args.out/'bases'/name; artifact.mkdir(parents=True,exist_ok=True)
                        np.savez_compressed(artifact/f'k{k}_start{start}.npz',basis=complete_basis,origin=c[0],centroids=c)
                        def evaluate(method,u,a,m,q,seed,map_seconds=0.):
                            nonlocal label_offset
                            if time.perf_counter()-begin>args.max_seconds:
                                manifest.update(complete=False,stop_reason='wall_time_limit',wall_seconds=time.perf_counter()-begin)
                                persist()
                                raise TimeoutError('Study wall-time limit; partial rows preserved')
                            t=time.perf_counter()
                            if method=='pca': z=x@u; centers=c@u
                            else:
                                z=protected_project(x,u,a,c[0]); centers=protected_project(c,u,a,c[0])
                            projection_seconds=time.perf_counter()-t
                            t=time.perf_counter(); sq=_squared_distances(z,centers); assigned=np.argmin(sq,axis=1)
                            assignment_seconds=time.perf_counter()-t
                            projected_pairs=np.sum((z[pairs[:,0]]-z[pairs[:,1]])**2,axis=1)
                            keep=original_pair_sq>0
                            distortion=np.abs(projected_pairs[keep]/original_pair_sq[keep]-1)
                            row=dict(dataset=name,n=n,d=d,k=k,start=start,seed=seed,method=method,
                                protected_rank=0 if method=='pca' else u.shape[1],residual_m=m,q=q,total_m=z.shape[1],
                                ari=float(adjusted_rand_score(labels,assigned)),agreement=float(np.mean(labels==assigned)),
                                r_phi=phi_from_labels(x,assigned,k,norm_sum)/phi_ref,
                                fixed_center_ratio=float(np.sum(distances[np.arange(n),assigned]))/fixed_ref,
                                distortion_q95=float(np.quantile(distortion,.95)),
                                projection_seconds=projection_seconds,assignment_seconds=assignment_seconds,
                                basis_seconds=0. if method=='pca' else basis_seconds,map_construction_seconds=map_seconds,
                                labels_byte_offset=label_offset)
                            label_file.write(assigned.astype('<u2').tobytes()); label_offset+=2*n
                            rows.append(row)
                        for m in args.m:
                            if m<=len(components): evaluate('pca',components[:m].T,None,m,0,-1)
                        for r in ranks[1:]: evaluate('protected_only',complete_basis[:,:r],None,0,0,-1)
                        for m in args.m:
                            if m>d: continue
                            for q in args.q:
                                if q>m: continue
                                for seed in range(args.seeds):
                                    rho=hashlib.sha256(f'protected-jl-pilot-v1:{seed}:{m}:{q}'.encode()).digest()
                                    # Same fresh sparse map for all ranks within this paired cell.
                                    transcript=x_hash.encode()+hashlib.sha256(c.tobytes()).digest()+rho
                                    map_seed=hashlib.sha256(transcript).digest()
                                    t=time.perf_counter(); _, a=sparse_matrix_gen_csr(map_seed,m,d,q); a=a.astype(float); a.data/=np.sqrt(q)
                                    map_seconds=time.perf_counter()-t
                                    for r in ranks:
                                        evaluate('sparse' if r==0 else f'hybrid_r{r}',complete_basis[:,:r],a,m,q,seed,map_seconds)
            persist()
            print(json.dumps(dict(dataset=name,rows=len(rows),elapsed_seconds=time.perf_counter()-begin)),flush=True)
    summary=summarize(args.out,args.bootstrap)
    manifest.update(complete=True,rows=len(rows),configurations=len(summary),wall_seconds=time.perf_counter()-begin,
        limitations='Basis SVD and PCA training outside projection timers; shared construction costs separate. No ZK security theorem for hybrid, no physical-precision guarantee, no independent confirmation split.')
    persist()
    print(json.dumps(dict(complete=True,rows=len(rows),wall_seconds=manifest['wall_seconds'])),flush=True)


if __name__=='__main__': main()
