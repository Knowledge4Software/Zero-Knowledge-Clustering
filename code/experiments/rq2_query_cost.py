"""RQ2, per-query proof cost over a committed compressed matrix (Stage 2).

For each dataset and k, public centres come from k-means on a 60% training
split and the audited points are drawn from the remaining 40%. Every method
is turned into one integer matrix Z (n x W) plus integer centres, using the
same coordinate budget (|v| <= 2^COORD_BITS, one scale per method/instance):

- full      : Z = X - c_0,                        W = d,   assignment on all d
- pca       : PCA(m) fitted on the training split, W = m,   assignment on all m
- sparse    : canonical Sparse JL (q=1),          W = m,   assignment on all m
- protected : head (k-1, from public centres) + Sparse JL residual tail,
              W = m,   assignment on the head only; cost from head + tail
- protected_energy : head + exact residual energy (1 column) + Sparse JL tail,
              W = m,   assignment on the head; exact cost from head + energy
              (the tail stays committed for geometric queries, never opened here)

Queries: "assign" (prove every point's nearest centre; opens only the
assignment block) and "cost" (assignments plus the total cost; opens all of
Z). Centres are either public inputs bound by a hash, or compiled into the
relation as constants. Each configuration is compiled at two sizes n so the
per-point slope is exact; Groth16 setup/prove/verify run when the relation
is small enough for the host (MAX_PROVE_CONSTRAINTS).

python -m experiments.rq2_query_cost --bin zk/bin/commitjl-query-linux --out results/rq2_query_cost_<date>
"""
from __future__ import annotations

import argparse, gc, hashlib, json, os, platform, subprocess, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from data.loaders import load_dataset
from commitjl.protected_jl import centroid_basis, protected_project
from commitjl.sparse_gen import sparse_matrix_gen_csr

DATASETS = [
    ('gaussian_hard', 'gaussian_mixture', dict(n=2000, d=512, k=10, gamma=0.35, imbalance=3.0, seed=0)),
    ('mnist', 'mnist', dict(n_max=5000, seed=0)),
    ('fashion_mnist', 'fashion_mnist', dict(n_max=5000, seed=0)),
    ('20ng_tfidf_nonzero', '20ng_tfidf', dict(n_max=3000, seed=0)),
    ('gisette', 'gisette', dict(n_max=3000, seed=0)),
]
KS = (5, 10)
M = 16
SIZES = (1, 2, 4, 8)
# Smaller point counts where the full-dimension relation would exceed host RAM.
SIZES_BY_DATASET = {'20ng_tfidf_nonzero': (1, 2, 4), 'gisette': (1, 2)}
CONSTRAINTS_PER_LEAF = 1400          # measured full-width cost per opened coordinate, guard only
MAX_COMPILE_CONSTRAINTS = 2_500_000  # gnark compile needs ~3 KB/constraint here; 8 GB host (OOMs seen above ~2.3M)
COORD_BITS = 16
MAX_PROVE_CONSTRAINTS = 1_000_000  # Groth16 proving needs ~7 KB/constraint on this host
METHODS = ('full', 'pca', 'sparse', 'protected', 'protected_energy')


def nearest(z, c):
    return ((z**2).sum(1)[:, None]-2*z@c.T+(c**2).sum(1)).argmin(1)


def representations(x, xa, c, k, start):
    """Float compressed matrices for the audited points and the centres."""
    seed = hashlib.sha256(json.dumps(dict(v='rq2-query-v1', k=k, s=start)).encode()).digest()
    out = {'full': (xa-c[0], c-c[0], x.shape[1])}
    pca = PCA(M, svd_solver='randomized', random_state=start).fit(x)
    out['pca'] = (pca.transform(xa), pca.transform(c), M)
    _, a = sparse_matrix_gen_csr(seed, M, x.shape[1], 1); a = a.astype(float)
    out['sparse'] = (np.asarray(a@(xa-c[0]).T).T, np.asarray(a@(c-c[0]).T).T, M)
    u = centroid_basis(c); h = u.shape[1]
    _, a = sparse_matrix_gen_csr(seed, M-h, x.shape[1], 1); a = a.astype(float)
    out['protected'] = (protected_project(xa, u, a, c[0]), protected_project(c, u, a, c[0]), h)
    _, a = sparse_matrix_gen_csr(seed+b'energy', M-h-1, x.shape[1], 1); a = a.astype(float)
    z = protected_project(xa, u, a, c[0]); zc = protected_project(c, u, a, c[0])
    energy = np.maximum(((xa-c[0])**2).sum(1)-(z[:, :h]**2).sum(1), 0.0)  # exact ||(I-UU^T)(x-c_0)||^2
    out['protected_energy'] = (z, zc, h, energy)
    return out


def quantize(z, zc, energy=None, h=0):
    """One scale s per instance; an energy column is stored in s^2 units."""
    scale = (2**COORD_BITS-1)/max(np.abs(z).max(), np.abs(zc).max(), 1e-12)
    zi, ci = np.rint(z*scale).astype(np.int64), np.rint(zc*scale).astype(np.int64)
    if energy is None:
        return zi, ci, scale, 0
    e = np.rint(energy*scale**2).astype(np.int64)
    zi = np.column_stack((zi[:, :h], e, zi[:, h:])); ci = np.column_stack((ci[:, :h], np.zeros(len(ci), np.int64), ci[:, h:]))
    return zi, ci, scale, max(int(e.max()).bit_length(), 1)


def run_bin(binary, inst, workdir):
    path = workdir/f"{inst['label']}.json"; path.write_text(json.dumps(inst))
    t = time.perf_counter()
    proc = subprocess.run([binary, '-instance', str(path)], capture_output=True, text=True)
    wall = time.perf_counter()-t
    path.unlink()
    try:
        res = json.loads(proc.stdout)
    except json.JSONDecodeError:
        res = dict(error=(proc.stderr or proc.stdout)[-500:] or f'process failed (return code {proc.returncode}, likely out of memory)')
    res['wall_seconds'] = wall
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--only', nargs='*'); ap.add_argument('--reps', type=int, default=3)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    a.bin = str(Path(a.bin).resolve())
    work = out/'instances'; work.mkdir(exist_ok=True)
    if not (out/'protocol.json').exists():
        json.dump(dict(datasets=DATASETS, ks=KS, m=M, sizes=SIZES, sizes_by_dataset=SIZES_BY_DATASET, coord_bits=COORD_BITS,
                       max_prove_constraints=MAX_PROVE_CONSTRAINTS, reps=a.reps, host=platform.node(),
                       cpu_count=os.cpu_count(), gomaxprocs=os.environ.get('GOMAXPROCS')),
                  open(out/'protocol.json', 'w'), indent=1, default=str)
    path = out/'rows.csv'
    rows = pd.read_csv(path).to_dict('records') if path.exists() else []
    done = {r['label'] for r in rows}
    for name, loader, kw in DATASETS:
        if a.only and name not in a.only: continue
        x = np.asarray(load_dataset(loader, **kw)['X'], float); x = x[np.linalg.norm(x, axis=1) > 0]
        gc.collect()  # release the loader's full corpus before the Go process allocates
        for k in KS:
            rng = np.random.default_rng(500+k)
            order = rng.permutation(len(x)); tr, au = order[:int(.6*len(x))], order[int(.6*len(x)):]
            c = KMeans(k, n_init=5, random_state=0).fit(x[tr]).cluster_centers_
            sizes = SIZES_BY_DATASET.get(name, SIZES)
            xa = x[au[:max(sizes)]]
            exact = nearest(xa, c)
            true_cost = np.cumsum(((xa-c[exact])**2).sum(1))  # original-space cost of the first n points
            reps = representations(x[tr], xa, c, k, 0)
            for method in METHODS:
                zf, cf, assign_dims, *energy = reps[method]
                zi, ci, scale, energy_bits = quantize(zf, cf, energy[0] if energy else None, assign_dims)
                for n in sizes:
                    for cost in (False, True):
                        for const in (False, True):
                            label = f"{name}_k{k}_{method}_n{n}_{'cost' if cost else 'assign'}_{'const' if const else 'public'}"
                            if label in done: continue
                            inst = dict(label=label, n=n, w=zi.shape[1], k=k, assign_dims=int(assign_dims),
                                        coord_bits=COORD_BITS, cost=cost, const_centers=const,
                                        z=zi[:n].tolist(), centers=ci.tolist(), salt=str(10**30+n),
                                        compile_only=True, n_repetitions=a.reps, n_warmup=1,
                                        energy_col=bool(energy), energy_bits=energy_bits)
                            opened = (assign_dims+1 if energy else zi.shape[1]) if cost else assign_dims
                            leaves = n*opened
                            if leaves*CONSTRAINTS_PER_LEAF > MAX_COMPILE_CONSTRAINTS:
                                res = dict(error='skipped: estimated relation exceeds host RAM', wall_seconds=0.0)
                            else:
                                res = run_bin(a.bin, inst, work)
                            prove_error = ''
                            if not res.get('error') and n == min(sizes) and res['num_constraints'] <= MAX_PROVE_CONSTRAINTS:  # one proved size per configuration
                                inst['compile_only'] = False
                                proved_res = run_bin(a.bin, inst, work)
                                if proved_res.get('error'):
                                    prove_error = proved_res['error']  # keep the compile-only measurement
                                else:
                                    res = proved_res
                            proved = res.get('prove_seconds_samples') or []
                            assigned = np.asarray(res.get('assign') or [-1]*n)
                            rows.append(dict(label=label, dataset=name, k=k, method=method, n=n, d=x.shape[1],
                                w=zi.shape[1], assign_dims=int(assign_dims), query='cost' if cost else 'assign',
                                centers='const' if const else 'public', constraints=res.get('num_constraints'),
                                compile_seconds=res.get('compile_seconds'), setup_seconds=res.get('setup_seconds') if proved else np.nan,
                                prove_seconds=float(np.median(proved)) if proved else np.nan,
                                verify_seconds=float(np.median(res['verify_seconds_samples'])) if proved else np.nan,
                                proof_bytes=res.get('proof_bytes') if proved else np.nan, verify_ok=res.get('verify_ok') if proved else np.nan,
                                peak_rss_bytes=res.get('peak_rss_bytes_os_maxrss'), wall_seconds=res['wall_seconds'],
                                agreement_with_exact=float(np.mean(assigned == exact[:n])), error=res.get('error', ''),
                                prove_error=prove_error, opened_columns=opened,
                                cost_claim_rel_error=(int(res['psi'])/scale**2/true_cost[n-1]-1) if cost and res.get('psi') else np.nan))
                            tmp = out/'rows.tmp.csv'; pd.DataFrame(rows).to_csv(tmp, index=False); tmp.replace(path)
                            print(label, rows[-1]['constraints'], rows[-1]['prove_seconds'], rows[-1]['error'][:80], flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
