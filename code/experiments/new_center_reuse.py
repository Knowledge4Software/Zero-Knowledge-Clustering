"""New-centre reuse: build the sketch once from deployed centres, audit later centre sets.

The fixed-centre identity ||x-c||^2 = ||h(x)-h(c)||^2 + e(x) needs c in the span of
the deployed centres C0. For a new public centre c' (refit, model update, other k,
Lloyd iterate, perturbation) put r' = (I-UU^T)(c'-c0). Then exactly

    ||x-c'||^2 = ||h(x)-h(c')||^2 + e(x) + ||r'||^2 - 2<r(x), r'>.

Head, energy and ||r'||^2 are exact; only the cross term is unknown. The sparse JL
tail estimates it: <r(x), r'> ~ <A r(x), A r'>/q. The sketch is never rebuilt.

Methods at total budget B stored values per row:
  ours_cross    head h + energy + tail B-h-1; cross term from the tail (proposed)
  ours_tailnorm same head and tail, no energy: ||h-h'||^2 + ||T-T'||^2   (B-1 values)
  ours_nocross  head + energy, cross term 0 (h+1 values; no-JL ablation, budget-free)
  jl            plain sparse JL of x-c0, width B
  pca           PCA fitted on the training split, width B, projected distances
  pca_energy    first B-1 PCA directions + residual energy, cross term 0
Reference: exact original-space nearest new centre. Rebuilding the head for each new
centre set is exact but repeats preprocessing over all rows; that is a cost, not a row.

Centres, bases and PCA see only the 60% training split; the 40% audit split is only
sketched and audited. Native float64; no proofs are generated.

python -m experiments.new_center_reuse --out results/new_center_reuse_<date> \
    [--worker i --workers w] [--only dataset ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ari

from commitjl.protected_jl import centroid_basis
from commitjl.sparse_gen import sparse_matrix_gen_csr
from data.loaders import load_dataset
from experiments.protected_fixed_center import DATASETS as CAPPED

DATASETS = CAPPED + [
    ('cifar10_resnet50', 'cifar10_resnet50', dict(n_max=5000, seed=0)),
    ('20ng_minilm', '20ng_minilm', dict(n_max=5000, seed=0)),
]
KS = (5, 10)
STARTS = range(3)
BUDGETS = (16, 32, 64, 128, 256)
QS = (1, 4)
MAPS = range(10)
LLOYD_STEPS = (0, 1, 3, 10)
PERTURB = (0.1, 0.3)


def sqdist(x, c):
    return np.maximum((x*x).sum(1)[:, None] - 2*x@c.T + (c*c).sum(1)[None, :], 0.)


def split_residual(points, origin, basis):
    """Head coordinates, residual vectors and residual energy w.r.t. span(basis)."""
    w = points - origin
    h = w @ basis
    r = w - h @ basis.T
    return h, r, (r*r).sum(1)


def reuse_distances(h, e, t, hc, ec, tc):
    """Estimated ||x-c'||^2 = head term + e(x) + ||r'||^2 - 2 <tail(x), tail(c')>.

    Exact when the tail inner product equals <r(x), r'> (e.g. t=r, tc=r')."""
    return sqdist(h, hc) + e[:, None] + ec[None, :] - 2*t@tc.T


def lloyd(x, c, steps):
    c = np.array(c, dtype=float)
    for _ in range(steps):
        lab = sqdist(x, c).argmin(1)
        for j in range(len(c)):
            members = lab == j
            if members.any():
                c[j] = x[members].mean(0)
    return c


def center_sets(xtr, c0, k, start):
    """Public new-centre sets derived from the training split only."""
    rng = np.random.default_rng(20_000+start)
    sets = [('same', 'same', c0)]
    for s in (1, 2):
        sets.append((f'refit{s}', 'refit', KMeans(k, n_init=1, random_state=100*s+start).fit(xtr).cluster_centers_))
    for s in (1, 2):
        idx = rng.choice(len(xtr), len(xtr)//2, replace=False)
        sets.append((f'subsample{s}', 'subsample',
                     KMeans(k, n_init=3, random_state=200+10*s+start).fit(xtr[idx]).cluster_centers_))
    for kk in sorted({max(2, k//2), 2*k}):
        sets.append((f'k{kk}', 'k_change', KMeans(kk, n_init=3, random_state=300+start).fit(xtr).cluster_centers_))
    init, _ = kmeans_plusplus(xtr, k, random_state=400+start)
    for steps in LLOYD_STEPS:
        sets.append((f'lloyd{steps}', 'lloyd', lloyd(xtr, init, steps)))
    pair = np.sqrt(sqdist(c0, c0)[np.triu_indices(len(c0), 1)])
    spacing = float(np.median(pair))
    for f in PERTURB:
        noise = rng.normal(size=c0.shape)
        noise *= f*spacing/np.linalg.norm(noise, axis=1, keepdims=True)
        sets.append((f'perturb{f}', 'perturb', c0+noise))
    return sets, spacing


def map_seed(**fields):
    return hashlib.sha256(json.dumps(dict(v='new-center-reuse-v1', **fields), sort_keys=True).encode()).digest()


def sparse_map(seed, m, d, q):
    _, a = sparse_matrix_gen_csr(seed, m, d, q)
    return a.astype(float) / np.sqrt(q)


def run_job(name, x, k, start):
    rng = np.random.default_rng(10_000+start)
    order = rng.permutation(len(x))
    tr, au = order[:int(.6*len(x))], order[int(.6*len(x)):]
    xtr, xa = x[tr], x[au]
    d = x.shape[1]
    c0 = KMeans(k, n_init=5, random_state=start).fit(xtr).cluster_centers_
    u = centroid_basis(c0)
    hrank = u.shape[1]
    origin = c0[0]
    hx, rx, ex = split_residual(xa, origin, u)
    wx = xa - origin
    sets, spacing = center_sets(xtr, c0, k, start)

    # Per centre set: exact reference and public decompositions.
    prepared = []
    for sname, family, cn in sets:
        dist = sqdist(xa, cn)
        ref = dist.argmin(1)
        phi = float(dist[np.arange(len(xa)), ref].sum())
        hc, rc, ec = split_residual(cn, origin, u)
        wc = cn - origin
        meta = dict(scenario=sname, family=family, k_new=len(cn),
                    residual_fraction=float(ec.sum()/max((wc*wc).sum(), 1e-300)),
                    shift=float(np.sqrt(sqdist(cn, c0).min(1)).mean()/spacing),
                    audit_residual_fraction=float(ex.sum()/max((wx*wx).sum(), 1e-300)))
        prepared.append((meta, cn, dist, ref, phi, hc, rc, ec, wc))

    rows = []
    base = dict(dataset=name, k=k, start=start, n_audit=len(xa), n_train=len(xtr), d=d, head_rank=hrank)

    def record(method, budget, q, mp, stored, meta, dist, ref, phi, est):
        lab = est.argmin(1)
        ar = np.arange(len(lab))
        rows.append(dict(base, **meta, method=method, budget=budget, q=q, map=mp, stored=stored,
                         agreement=float(np.mean(lab == ref)), ari=float(ari(ref, lab)),
                         r_phi=float(dist[ar, lab].sum()/phi),
                         psi_rel_error=float(est[ar, lab].sum()/phi - 1)))

    for meta, cn, dist, ref, phi, hc, rc, ec, wc in prepared:
        est = sqdist(hx, hc) + ex[:, None] + ec[None, :]
        record('ours_nocross', hrank+1, 0, -1, hrank+1, meta, dist, ref, phi, est)

    for budget in BUDGETS:
        if budget >= d:
            continue
        if budget < len(xtr):
            pca = PCA(budget, svd_solver='randomized', random_state=start).fit(xtr)
            zx = pca.transform(xa)
            centred = ((xa - pca.mean_)**2).sum(1)
            ev = np.maximum(centred - (zx[:, :-1]**2).sum(1), 0.)
            for meta, cn, dist, ref, phi, *_ in prepared:
                zc = pca.transform(cn)
                record('pca', budget, 0, -1, budget, meta, dist, ref, phi, sqdist(zx, zc))
                evc = np.maximum(((cn - pca.mean_)**2).sum(1) - (zc[:, :-1]**2).sum(1), 0.)
                est = sqdist(zx[:, :-1], zc[:, :-1]) + ev[:, None] + evc[None, :]
                record('pca_energy', budget, 0, -1, budget, meta, dist, ref, phi, est)
        tail = budget - hrank - 1
        for q in QS:
            for mp in MAPS:
                a = sparse_map(map_seed(d=name, k=k, s=start, b=budget, q=q, map=mp, kind='plain'), budget, d, q)
                zx = np.asarray(a @ wx.T).T
                if tail >= 1:
                    at = sparse_map(map_seed(d=name, k=k, s=start, b=budget, q=q, map=mp, kind='tail'), tail, d, q)
                    tx = np.asarray(at @ rx.T).T
                for meta, cn, dist, ref, phi, hc, rc, ec, wc in prepared:
                    record('jl', budget, q, mp, budget, meta, dist, ref, phi, sqdist(zx, np.asarray(a @ wc.T).T))
                    if tail >= 1:
                        tc = np.asarray(at @ rc.T).T
                        record('ours_cross', budget, q, mp, budget, meta, dist, ref, phi,
                               reuse_distances(hx, ex, tx, hc, ec, tc))
                        record('ours_tailnorm', budget, q, mp, budget-1, meta, dist, ref, phi,
                               sqdist(hx, hc) + sqdist(tx, tc))
    return rows


def jobs():
    return [(name, loader, kw, k, s) for name, loader, kw in DATASETS for k in KS for s in STARTS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--worker', type=int, default=0)
    ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out)
    (out/'jobs').mkdir(parents=True, exist_ok=True)
    proto = out/'protocol.json'
    if not proto.exists():
        proto.write_text(json.dumps(dict(datasets=DATASETS, ks=KS, starts=list(STARTS), budgets=BUDGETS,
                                         qs=QS, maps=list(MAPS), lloyd_steps=LLOYD_STEPS, perturb=PERTURB,
                                         host=platform.node(), cpu_count=os.cpu_count(),
                                         threads=os.environ.get('OMP_NUM_THREADS')), indent=1, default=str))
    mine = [j for i, j in enumerate(jobs()) if i % a.workers == a.worker and (not a.only or j[0] in a.only)]
    cache = {}
    for name, loader, kw, k, s in mine:
        path = out/'jobs'/f'{name}_k{k}_s{s}.csv'
        if path.exists():
            continue
        if name not in cache:
            cache.clear()
            try:
                x = np.asarray(load_dataset(loader, **kw)['X'], float)
            except FileNotFoundError as err:   # missing embedding cache: skip, keep other jobs
                print(f'SKIP {name}: {err}', flush=True)
                cache[name] = None
                continue
            cache[name] = x[np.linalg.norm(x, axis=1) > 0]
        if cache[name] is None:
            continue
        t0 = time.time()
        rows = run_job(name, cache[name], k, s)
        tmp = path.with_suffix('.tmp')
        pd.DataFrame(rows).to_csv(tmp, index=False)
        tmp.replace(path)
        print(f'{name} k={k} s={s} rows={len(rows)} {time.time()-t0:.0f}s', flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
