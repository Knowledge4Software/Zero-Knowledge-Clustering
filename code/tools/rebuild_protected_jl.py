"""Recover a study row's labels and projected arrays without the server."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from commitjl.protected_jl import protected_project
from commitjl.sparse_gen import sparse_matrix_gen_csr
from commitjl.clustering import nearest_centroid_assign
from experiments.protected_jl_study import Archive


def rebuild(study,row_index):
    study=Path(study);rows=pd.read_csv(study/'rows.csv');row=rows.iloc[row_index]
    archive=Archive();prefix=f'results/paper_server_20261006/rq1/{row.dataset}/artifacts/'
    x=archive.load(prefix+'dataset.npz')['X']
    with np.load(study/'bases'/row.dataset/f'k{int(row.k)}_start{int(row.start)}.npz') as saved:
        c=saved['centroids'];origin=saved['origin'];u=saved['basis'][:,:int(row.protected_rank)]
    if row.method=='pca':
        matrix=archive.load(prefix+'pca.npz')['components'][:int(row.total_m)]
        z=x@matrix.T;cc=c@matrix.T
    else:
        a=None
        if row.residual_m>0:
            x_hash=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
            rho=hashlib.sha256(f'protected-jl-pilot-v1:{int(row.seed)}:{int(row.residual_m)}:{int(row.q)}'.encode()).digest()
            seed=hashlib.sha256(x_hash.encode()+hashlib.sha256(c.tobytes()).digest()+rho).digest()
            _,a=sparse_matrix_gen_csr(seed,int(row.residual_m),x.shape[1],int(row.q))
            a=a.astype(float);a.data/=np.sqrt(row.q)
        z=protected_project(x,u,a,origin);cc=protected_project(c,u,a,origin)
    assigned=nearest_centroid_assign(z,cc)
    with gzip.open(study/f'{row.dataset}_labels.u16.gz','rb') as f:
        f.seek(int(row.labels_byte_offset))
        saved_labels=np.frombuffer(f.read(2*len(x)),dtype='<u2')
    if not np.array_equal(assigned,saved_labels): raise ValueError('Reconstructed assignments differ from recorded labels')
    return dict(Z=z,projected_centroids=cc,labels=assigned,centroids=c,basis=u)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--study',type=Path,required=True);ap.add_argument('--row',type=int,required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if args.out.exists():ap.error('Output exists')
    arrays=rebuild(args.study,args.row)
    args.out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.out,**arrays)
    print(json.dumps(dict(verified=True,shape=list(arrays['Z'].shape),out=str(args.out))))

if __name__=='__main__':main()
