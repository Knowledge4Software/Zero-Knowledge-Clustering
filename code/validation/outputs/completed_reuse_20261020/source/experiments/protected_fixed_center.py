"""Fixed-centroid audit: Protected CommitJL vs Full, PCA and plain Sparse JL.

Setting of the manuscript (fixed public centres). For each start, centres C
come from k-means on a 60% training split; the remaining 40% (never used to
form C, the PCA basis or anything else) is the audited set X.

Methods, all with the same total width m:
- full           : exact original-space assignment to C (reference)
- pca            : PCA fitted on the training split (public-matrix baseline)
- sparse         : canonical Sparse JL, q=1, on x - c_0
- protected      : head = basis of centre differences (k-1, from PUBLIC C),
                   tail = canonical Sparse JL of the residual, width m-(k-1)
- protected_int  : the circuit arithmetic: inputs quantized to [-4096,4096],
                   basis rounded with scale T=4096, integer projection
- head_only      : protected head without tail (ablation; no objective estimate)

Per (dataset, k, start, map, m) the audit records assignment agreement and ARI
with Full, the original-space objective ratio r_phi of the projected labels,
the error of the objective the proof would claim (psi_hat, computed in the
compressed space), pair distortion and the downstream cost after compression
(assignment + objective on the compressed audit set, median of 5 timings).

python -m experiments.protected_fixed_center --out results/protected_fixed_center_<date>
"""
from __future__ import annotations

import argparse, hashlib, json, os, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ari

from data.loaders import load_dataset
from commitjl.protected_jl import centroid_basis, protected_project, integer_protected_project
from commitjl.sparse_gen import sparse_matrix_gen_csr

DATASETS = [  # (name, loader, kwargs)
    ('gaussian_easy', 'gaussian_mixture', dict(n=2000, d=256, k=10, gamma=1.0, seed=0)),
    ('gaussian_hard', 'gaussian_mixture', dict(n=2000, d=512, k=10, gamma=0.35, imbalance=3.0, seed=0)),
    ('dbscan_margin', 'dbscan_margin', dict(n=1000, d=128)),
    ('mnist', 'mnist', dict(n_max=5000, seed=0)),
    ('fashion_mnist', 'fashion_mnist', dict(n_max=5000, seed=0)),
    ('gisette', 'gisette', dict(n_max=3000, seed=0)),
    ('arcene', 'arcene', dict()),
    ('20ng_tfidf_nonzero', '20ng_tfidf', dict(n_max=3000, seed=0)),
]
KS = (5, 10)
WIDTHS = (16, 32, 64, 128)
STARTS = range(3)
MAPS = range(10)
SCALE = 4096
PAIRS = 4000


def nearest(z, c):
    return ((z**2).sum(1)[:, None]-2*z@c.T+(c**2).sum(1)).argmin(1)


def objective(x, c, labels):
    return float(((x-c[labels])**2).sum())


def distortion(x, z, rng):
    i, j = rng.integers(0, len(x), (2, PAIRS)); keep = i != j; i, j = i[keep], j[keep]
    a = ((x[i]-x[j])**2).sum(1); b = ((z[i]-z[j])**2).sum(1)
    ok = a > 0; r = np.abs(b[ok]/a[ok]-1)
    return float(np.quantile(r, .95)), float(np.mean(r > .15))


def timed(fn, reps=5):
    out = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); out.append(time.perf_counter()-t)
    return float(np.median(out))


def run_start(name, x, y, k, start, out):
    rng = np.random.default_rng(10_000+start)
    order = rng.permutation(len(x)); tr, au = order[:int(.6*len(x))], order[int(.6*len(x)):]
    c = KMeans(k, n_init=5, random_state=start).fit(x[tr]).cluster_centers_
    xa = x[au]
    ref = nearest(xa, c); phi = objective(xa, c, ref)
    full_s = timed(lambda: (lambda l: objective(xa, c, l))(nearest(xa, c)))
    u = centroid_basis(c); h = u.shape[1]
    # Integer circuit arithmetic: one common input scaling into [-4096, 4096].
    s = 4096/max(np.abs(x).max(), 1e-12)
    xi = np.rint(xa*s).astype(np.int64); ci = np.rint(c*s).astype(np.int64)
    bi = np.rint(SCALE*u.T).astype(np.int64)
    base = dict(dataset=name, k=k, start=start, n_audit=len(xa), d=x.shape[1], head_rank=h,
                full_seconds=full_s)

    def record(method, m, mp, z, zc, labels, psi_hat, down_s, build_s):
        q95, exc = distortion(xa, z, np.random.default_rng(77+start)) if z is not None else (np.nan, np.nan)
        out.append(dict(base, method=method, m=m, map=mp,
            agreement=float(np.mean(labels == ref)), ari=ari(ref, labels),
            r_phi=objective(xa, c, labels)/phi,
            psi_rel_error=(psi_hat/phi-1) if psi_hat is not None else np.nan,
            q95=q95, exceed=exc, downstream_seconds=down_s, build_seconds=build_s,
            downstream_speedup=full_s/down_s if down_s else np.nan))

    for m in WIDTHS:
        if m >= x.shape[1] or m <= h: continue
        if m < len(tr):  # PCA width is capped by the training rows (Arcene); no substitute size
            t = time.perf_counter(); pca = PCA(m, svd_solver='randomized', random_state=start).fit(x[tr])
            z = pca.transform(xa); zc = pca.transform(c); b_s = time.perf_counter()-t
            lab = nearest(z, zc)
            record('pca', m, -1, z, zc, lab, objective(z, zc, lab),
                   timed(lambda: (lambda l: objective(z, zc, l))(nearest(z, zc))), b_s)
        for mp in MAPS:
            seed = hashlib.sha256(json.dumps(dict(v='protected-fixed-center-v1', d=name, k=k, s=start,
                                                  m=m, map=mp)).encode()).digest()
            t = time.perf_counter(); _, a = sparse_matrix_gen_csr(seed, m, x.shape[1], 1); a = a.astype(float)
            z = np.asarray(a@(xa-c[0]).T).T; zc = np.asarray(a@(c-c[0]).T).T; b_s = time.perf_counter()-t
            lab = nearest(z, zc)
            record('sparse', m, mp, z, zc, lab, objective(z, zc, lab),
                   timed(lambda: (lambda l: objective(z, zc, l))(nearest(z, zc))), b_s)

            t = time.perf_counter(); _, a = sparse_matrix_gen_csr(seed, m-h, x.shape[1], 1); a = a.astype(float)
            z = protected_project(xa, u, a, c[0]); zc = protected_project(c, u, a, c[0]); b_s = time.perf_counter()-t
            # Assignment needs only the head; the tail only enters the objective.
            hz, hc = z[:, :h], zc[:, :h]; tail2 = (z[:, h:]**2).sum(1)
            def protected_downstream():
                l = nearest(hz, hc); return float(((hz-hc[l])**2).sum()+tail2.sum()), l
            psi, lab = protected_downstream()
            record('protected', m, mp, z, zc, lab, psi, timed(protected_downstream), b_s)
            if mp == 0:
                record('head_only', h, -1, None, None, nearest(hz, hc), None, timed(lambda: nearest(hz, hc)), b_s)

            ai = a.astype(np.int64)
            zi = integer_protected_project(xi, bi, ai, SCALE).astype(float)
            zci = integer_protected_project(ci, bi, ai, SCALE).astype(float)
            lab = nearest(zi[:, :h], zci[:, :h])
            # integer_protected_project returns head and tail both scaled by T^2 (input scale s).
            psi_i = ((zi-zci[lab])**2).sum()/(SCALE**4*s**2)
            record('protected_int', m, mp, None, None, lab, psi_i, np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--only', nargs='*')
    ap.add_argument('--starts', type=int, default=len(STARTS))
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if not (out/'protocol.json').exists():
        json.dump(dict(datasets=DATASETS, ks=KS, widths=WIDTHS, starts=list(range(a.starts)), maps=list(MAPS),
                       basis_scale=SCALE, host=platform.node(), cpu_count=os.cpu_count(),
                       threads=os.environ.get('OMP_NUM_THREADS')), open(out/'protocol.json', 'w'), indent=1, default=str)
    path = out/'rows.csv'
    rows = pd.read_csv(path).to_dict('records') if path.exists() else []
    done = {(r['dataset'], r['k'], r['start']) for r in rows}
    for name, loader, kw in DATASETS:
        if a.only and name not in a.only: continue
        data = load_dataset(loader, **kw); x = np.asarray(data['X'], float)
        y = data.get('y'); x = x[np.linalg.norm(x, axis=1) > 0]
        for k in KS:
            for start in range(a.starts):
                if (name, k, start) in done: continue
                new = []; run_start(name, x, None, k, start, new); rows += new
                tmp = out/'rows.tmp.csv'; pd.DataFrame(rows).to_csv(tmp, index=False); tmp.replace(path)
            print(name, k, 'done', flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
