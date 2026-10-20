"""Last test of data-obliviousness: distribution shift and multiple data owners.

Learned bases (PCA, centre span + PCA residual) can only be fitted on data that
is available when the sketch is committed; the JL map needs no data at all.

shift   : basis data = training rows of the first half of the classes ('old').
          Deployed centres C0 = k-means on old training rows. Later scenarios:
            new_only  - centres refit on new-class training rows, audit new-class rows
            mixed     - centres refit on all training rows, audit all rows
owners  : training rows split over 5 owners by a Dirichlet(0.3) class mix. Basis data =
          owner 0 only (no pooling). C0 = k-means on pooled training rows (public model).
          Later scenarios: the new-centre sets of experiments.new_center_reuse (all rows).

Methods at budget B: pca_energy, learned_energy (C0 span + PCA residual, cross 0),
ours_cross (C0 head + energy + sparse JL q=4 tail), ours_nocross, jl (plain, q=4),
and pca_energy_oracle (PCA fitted on all training rows; unavailable in practice).

python -m experiments.shift_owner_reuse --out results/shift_owner_reuse_<date> [--worker i --workers w]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ari

from commitjl.enriched_jl import enrich_centroid_basis
from commitjl.protected_jl import centroid_basis
from data.loaders import load_dataset
from experiments.new_center_reuse import center_sets, map_seed, sparse_map, split_residual, sqdist

DATASETS = [
    ('mnist', 'mnist', dict(n_max=5000, seed=0)),
    ('fashion_mnist', 'fashion_mnist', dict(n_max=5000, seed=0)),
    ('cifar10_resnet50', 'cifar10_resnet50', dict(n_max=5000, seed=0)),
    ('20ng_tfidf_nonzero', '20ng_tfidf', dict(n_max=3000, seed=0)),
    ('20ng_minilm', '20ng_minilm', dict(n_max=5000, seed=0)),
]
SETTINGS = ('shift', 'owners')
KS = (5, 10)
STARTS = range(3)
BUDGETS = (16, 32, 64, 128)
MAPS = range(5)
Q = 4
OWNERS, ALPHA = 5, 0.3


def pca_energy_fn(fit_rows, budget, seed):
    pca = PCA(budget - 1, svd_solver='randomized', random_state=seed).fit(fit_rows)

    def rep(v):
        z = pca.transform(v)
        return z, np.maximum(((v - pca.mean_)**2).sum(1) - (z**2).sum(1), 0.)
    return rep


def evaluate(name, setting, k, start, basis_rows, all_rows, c0, scenarios, rows):
    """scenarios: list of (scenario, family, centres, audit rows)."""
    d = c0.shape[1]
    hrank = centroid_basis(c0).shape[1]
    origin = c0[0]
    u0 = centroid_basis(c0)
    base = dict(dataset=name, setting=setting, k=k, start=start, d=d, head_rank=hrank,
                n_basis=len(basis_rows))
    prep = []
    for sname, family, cn, xa in scenarios:
        dist = sqdist(xa, cn)
        ref = dist.argmin(1)
        prep.append((dict(scenario=sname, family=family, n_audit=len(xa)), cn, xa, dist, ref,
                     float(dist[np.arange(len(xa)), ref].sum())))

    def put(method, budget, meta, dist, ref, phi, est, **info):
        lab = est.argmin(1)
        ar = np.arange(len(lab))
        rows.append(dict(base, **meta, **info, method=method, budget=budget,
                         agreement=float(np.mean(lab == ref)), ari=float(ari(ref, lab)),
                         r_phi=float(dist[ar, lab].sum()/phi), psi_rel_error=float(est[ar, lab].sum()/phi - 1)))

    for meta, cn, xa, dist, ref, phi in prep:
        hx, _, ex = split_residual(xa, origin, u0)
        hc, _, ec = split_residual(cn, origin, u0)
        put('ours_nocross', hrank + 1, meta, dist, ref, phi, sqdist(hx, hc) + ex[:, None] + ec[None, :])

    for budget in BUDGETS:
        free = budget - hrank - 1
        if budget >= d or free < Q or budget >= len(basis_rows):
            continue
        for method, fit in (('pca_energy', basis_rows), ('pca_energy_oracle', all_rows)):
            rep = pca_energy_fn(fit, budget, start)
            for meta, cn, xa, dist, ref, phi in prep:
                zx, ex = rep(xa)
                zc, ec = rep(cn)
                put(method, budget, meta, dist, ref, phi, sqdist(zx, zc) + ex[:, None] + ec[None, :])
        u, _ = enrich_centroid_basis(basis_rows, c0, extra=free, random_state=start)
        for meta, cn, xa, dist, ref, phi in prep:
            hx, _, ex = split_residual(xa, origin, u)
            hc, _, ec = split_residual(cn, origin, u)
            put('learned_energy', budget, meta, dist, ref, phi, sqdist(hx, hc) + ex[:, None] + ec[None, :])
        for mp in MAPS:
            seed = dict(d=name, setting=setting, k=k, s=start, b=budget, map=mp, study='shift-owner')
            tail = sparse_map(map_seed(kind='tail', **seed), free, d, Q)
            plain = sparse_map(map_seed(kind='plain', **seed), budget, d, Q)
            for meta, cn, xa, dist, ref, phi in prep:
                hx, rx, ex = split_residual(xa, origin, u0)
                hc, rc, ec = split_residual(cn, origin, u0)
                tx, tc = np.asarray(tail @ rx.T).T, np.asarray(tail @ rc.T).T
                put('ours_cross', budget, meta, dist, ref, phi,
                    sqdist(hx, hc) + ex[:, None] + ec[None, :] - 2*tx@tc.T, map=mp)
                zx, zc = np.asarray(plain @ (xa - origin).T).T, np.asarray(plain @ (cn - origin).T).T
                put('jl', budget, meta, dist, ref, phi, sqdist(zx, zc), map=mp)


def run_job(name, x, y, setting, k, start):
    rng = np.random.default_rng(10_000 + start)
    order = rng.permutation(len(x))
    tr, au = order[:int(.6*len(x))], order[int(.6*len(x)):]
    xtr, ytr, xa, ya = x[tr], y[tr], x[au], y[au]
    rows = []
    if setting == 'shift':
        classes = np.unique(y)
        old = classes[:len(classes)//2]
        otr, ntr = np.isin(ytr, old), ~np.isin(ytr, old)
        nau = ~np.isin(ya, old)
        c0 = KMeans(k, n_init=5, random_state=start).fit(xtr[otr]).cluster_centers_
        new_c = KMeans(k, n_init=5, random_state=start + 50).fit(xtr[ntr]).cluster_centers_
        mix_c = KMeans(k, n_init=5, random_state=start + 60).fit(xtr).cluster_centers_
        scen = [('new_only', 'shift', new_c, xa[nau]), ('mixed', 'shift', mix_c, xa)]
        evaluate(name, setting, k, start, xtr[otr], xtr, c0, scen, rows)
    else:
        classes = np.unique(ytr)
        mix = np.random.default_rng(30_000 + start).dirichlet([ALPHA]*OWNERS, len(classes))
        owner = np.empty(len(ytr), int)
        orng = np.random.default_rng(40_000 + start)
        for ci, c in enumerate(classes):
            idx = np.flatnonzero(ytr == c)
            owner[idx] = orng.choice(OWNERS, len(idx), p=mix[ci])
        c0 = KMeans(k, n_init=5, random_state=start).fit(xtr).cluster_centers_
        sets, _ = center_sets(xtr, c0, k, start)
        scen = [(s, f, cn, xa) for s, f, cn in sets if f != 'same']
        evaluate(name, setting, k, start, xtr[owner == 0], xtr, c0, scen, rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--worker', type=int, default=0)
    ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out)
    (out/'jobs').mkdir(parents=True, exist_ok=True)
    (out/'protocol.json').write_text(json.dumps(dict(datasets=DATASETS, settings=SETTINGS, ks=KS,
        starts=list(STARTS), budgets=BUDGETS, maps=list(MAPS), q=Q, owners=OWNERS, alpha=ALPHA), indent=1))
    jobs = [(n, l, kw, st, k, s) for n, l, kw in DATASETS for st in SETTINGS for k in KS for s in STARTS]
    cache = {}
    for i, (name, loader, kw, setting, k, s) in enumerate(jobs):
        path = out/'jobs'/f'{name}_{setting}_k{k}_s{s}.csv'
        if i % a.workers != a.worker or path.exists():
            continue
        if name not in cache:
            cache.clear()
            data = load_dataset(loader, **kw)
            x, y = np.asarray(data['X'], float), np.asarray(data['y'])
            keep = np.linalg.norm(x, axis=1) > 0
            cache[name] = (x[keep], y[keep])
        t0 = time.time()
        rows = run_job(name, *cache[name], setting, k, s)
        tmp = path.with_suffix('.tmp')
        pd.DataFrame(rows).to_csv(tmp, index=False)
        tmp.replace(path)
        print(f'{name} {setting} k={k} s={s} rows={len(rows)} {time.time()-t0:.0f}s', flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
