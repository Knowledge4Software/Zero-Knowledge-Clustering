"""Recreate a saved RQ1 Z cache from preserved inputs, PCA components and seed.

Run from code/: python -m tools.rebuild_rq1_map --dataset-dir
results/server_paper_20261006/rq1/gaussian_easy --method sparse_jl --m 32 --q 2 --seed 0
This computes only the projection, not clustering, metrics or proofs.
"""
import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits
from commitjl.sparse_gen import sparse_matrix_gen
from experiments.pipeline import build_projection
from experiments.rq1_complete import save_npz


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir', required=True, type=Path)
    p.add_argument('--method', required=True,
                   choices=['sparse_jl','pca','gaussian','rademacher','achlioptas','coordinate'])
    p.add_argument('--m', required=True, type=int)
    p.add_argument('--q', default=0, type=int)
    p.add_argument('--seed', default=0, type=int)
    p.add_argument('--threads', default=4, type=int)
    args = p.parse_args()
    with np.load(args.dataset_dir/'artifacts/dataset.npz') as saved:
        x = saved['X']
    n, d = x.shape
    if not 1 <= args.m <= d or args.threads < 1:
        p.error('Invalid dimension or thread count')
    if args.method == 'sparse_jl' and not 1 <= args.q <= args.m:
        p.error('Sparse projection requires 1 <= q <= m')
    if args.method != 'sparse_jl' and args.q != 0:
        p.error('Saved non-sparse map names use q=0')
    seed = -1 if args.method == 'pca' else args.seed
    if args.method != 'pca' and seed < 0:
        p.error('Randomized projection seeds must be nonnegative')
    target = args.dataset_dir/f'artifacts/maps/{args.method}_m{args.m}_q{args.q}_seed{seed}.npz'
    if target.exists():
        p.error('Cache already exists; keep the preserved file')
    started = time.perf_counter()
    with threadpool_limits(limits=args.threads):
        if args.method == 'sparse_jl':
            seed_bytes = hashlib.sha256(b'CommitJL-RQ1-native-v2'+seed.to_bytes(8,'little')).digest()
            _, dense = sparse_matrix_gen(seed_bytes, args.m, d, args.q)
            projection = csr_matrix(dense.astype(float)/np.sqrt(args.q))
        elif args.method == 'pca':
            with np.load(args.dataset_dir/'artifacts/pca.npz') as saved:
                components = saved['components']
            if args.m > components.shape[0]:
                p.error('Requested dimension exceeds preserved PCA rank')
            projection = components[:args.m]
        else:
            projection = build_projection(args.method, x, args.m, max(args.q,1), seed, 2**16)['real']
        z = np.asarray((projection @ x.T).T)
        save_npz(target, Z=z)
    print(f'Rebuilt {target} ({n} x {args.m}) in {time.perf_counter()-started:.2f} s')
    print('Use the recorded numerical environment for identical floating-point behavior.')


if __name__ == '__main__':
    main()
