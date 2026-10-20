"""Replay a native laptop geometry row and optionally export projected arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from commitjl.protected_jl import protected_project
from commitjl.sparse_gen import sparse_matrix_gen_csr
from commitjl.clustering import _squared_distances
from experiments.protected_jl_study import Archive


def rebuild(record):
    record=Path(record);row=json.loads(record.read_text());job=record.parents[1]
    archive=Archive();prefix=f'results/paper_server_20261006/rq1/{row["dataset"]}/artifacts/'
    x=archive.load(prefix+'dataset.npz')['X'][:row['n']]
    assert hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()==row['dataset_sha256']
    with np.load(job/f'centers_k{row["k"]}_start{row["start"]}.npz') as basis_file:
        c=basis_file['centroids'];u=basis_file['basis'][:,:row['rank']]
    with threadpool_limits(limits=4):
        if row['method']=='full':z,cc=x,c
        elif row['method']=='pca':
            matrix=archive.load(prefix+'pca.npz')['components'][:row['total_m']]
            z,cc=x@matrix.T,c@matrix.T
        else:
            a=None
            if row['residual_m']:
                _,a=sparse_matrix_gen_csr(bytes.fromhex(row['map_seed']),row['residual_m'],row['d'],row['q'])
                a=a.astype(float);a.data/=np.sqrt(row['q'])
            z,cc=protected_project(x,u,a,c[0]),protected_project(c,u,a,c[0])
        squared=_squared_distances(z,cc);labels=squared.argmin(axis=1)
    with np.load(job/'arrays'/(record.stem+'.npz')) as saved:
        np.testing.assert_array_equal(labels,saved['labels'])
        np.testing.assert_allclose(squared,saved['projected_centroid_sq'],rtol=1e-12,atol=1e-12)
    return dict(Z=z,projected_centroids=cc,labels=labels,basis=u,centroids=c)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record',type=Path,required=True);parser.add_argument('--out',type=Path)
    args=parser.parse_args();arrays=rebuild(args.record)
    if args.out:
        if args.out.exists():parser.error('Output already exists')
        args.out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.out,**arrays)
    print(json.dumps(dict(replay_ok=True,shape=list(arrays['Z'].shape))))


if __name__=='__main__':main()
