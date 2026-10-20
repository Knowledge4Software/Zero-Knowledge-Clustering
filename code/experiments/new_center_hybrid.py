"""Hybrid learned-head + JL-tail estimators for new public centres.

Same corpora, splits, deployed centres and new-centre scenarios as
experiments.new_center_reuse. The head is the deployed-centre span enriched with
the leading PCA directions of the training residual (commitjl.enriched_jl); the
row stores head, exact residual energy and a JL tail of the remaining residual.
For a new centre the only unknown is <r(x), r'>, estimated from the tail.

Variants at budget B (stored values per row):
  split f      fraction of the B-h-1 free values given to PCA directions:
               f in {0.5, 0.75, 1.0}; f=1.0 has no tail (learned head + energy,
               cross term 0: the same-basis no-JL ablation)
  map          sparse q=4, sparse q=8, srht (signed randomized Hadamard, subsampled)
  estimator    raw | clip (Cauchy-Schwarz: |<r,r'>| <= sqrt(e(x) ||r'||^2), both exact)
               | shrink (scale fitted on training rows by least squares, in [0,1])
               | clip_shrink
Baselines repeated here for matched comparison: pca_energy and ours_cross_q4.

python -m experiments.new_center_hybrid --out results/new_center_hybrid_<date> [--worker i --workers w]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
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
from experiments.new_center_reuse import (BUDGETS, DATASETS, KS, STARTS, center_sets, map_seed,
                                          sparse_map, split_residual, sqdist)

SPLITS = (0.5, 0.75, 1.0)
MAP_KINDS = ('q4', 'q8', 'srht')
MAPS = range(5)
ESTIMATORS = ('raw', 'clip', 'shrink', 'clip_shrink')


def fwht(a):
    """Unnormalized fast Walsh-Hadamard transform along axis 1 (length a power of two)."""
    a = np.array(a, dtype=float, copy=True)
    n, length = a.shape
    h = 1
    while h < length:
        v = a.reshape(n, -1, 2, h)
        x, y = v[:, :, 0, :].copy(), v[:, :, 1, :]
        v[:, :, 0, :] += y
        v[:, :, 1, :] = x - y
        h *= 2
    return a


class Tail:
    """A t x d JL map with E||Pv||^2 = ||v||^2."""

    def __init__(self, kind, seed, t, d):
        self.kind, self.t, self.d = kind, t, d
        if kind in ('q4', 'q8'):
            self.a = sparse_map(seed, t, d, int(kind[1:]))
        else:
            rng = np.random.default_rng(int.from_bytes(seed[:8], 'little'))
            self.p = 1 << int(np.ceil(np.log2(d)))
            self.signs = rng.choice([-1., 1.], d)
            self.rows = rng.choice(self.p, t, replace=False)

    def __call__(self, v):
        if self.kind != 'srht':
            return np.asarray(self.a @ v.T).T
        out = np.zeros((len(v), self.p))
        out[:, :self.d] = v * self.signs
        for s in range(0, len(v), 256):           # bounded memory
            out[s:s+256] = fwht(out[s:s+256])
        return out[:, self.rows] / np.sqrt(self.t)


def cross_estimate(tx, tc, ex, ec, estimator, lam):
    est = tx @ tc.T
    if estimator in ('shrink', 'clip_shrink'):
        est = est * lam[None, :]
    if estimator in ('clip', 'clip_shrink'):
        bound = np.sqrt(np.maximum(ex[:, None] * ec[None, :], 0.))
        est = np.clip(est, -bound, bound)
    return est


def shrink_factor(t_tr, r_tr, tc, rc):
    """Per-centre least-squares scale of the tail inner product on training rows."""
    est, true = t_tr @ tc.T, r_tr @ rc.T
    den = (est * est).sum(0)
    lam = np.where(den > 0, (est * true).sum(0) / np.maximum(den, 1e-300), 1.)
    return np.clip(lam, 0., 1.)


def run_job(name, x, k, start):
    rng = np.random.default_rng(10_000+start)
    order = rng.permutation(len(x))
    tr, au = order[:int(.6*len(x))], order[int(.6*len(x)):]
    xtr, xa = x[tr], x[au]
    d = x.shape[1]
    c0 = KMeans(k, n_init=5, random_state=start).fit(xtr).cluster_centers_
    hrank = centroid_basis(c0).shape[1]
    origin = c0[0]
    sets, _ = center_sets(xtr, c0, k, start)
    prepared = []
    for sname, family, cn in sets:
        if family == 'same':
            continue
        dist = sqdist(xa, cn)
        ref = dist.argmin(1)
        prepared.append((dict(scenario=sname, family=family, k_new=len(cn)), cn, dist, ref,
                         float(dist[np.arange(len(xa)), ref].sum())))
    rows = []
    base = dict(dataset=name, k=k, start=start, n_audit=len(xa), d=d, head_rank=hrank)

    def record(method, budget, extra=None, **info):
        def put(meta, dist, ref, phi, est):
            lab = est.argmin(1)
            ar = np.arange(len(lab))
            rows.append(dict(base, **meta, **info, method=method, budget=budget,
                             agreement=float(np.mean(lab == ref)), ari=float(ari(ref, lab)),
                             r_phi=float(dist[ar, lab].sum()/phi),
                             psi_rel_error=float(est[ar, lab].sum()/phi - 1)))
        return put

    for budget in BUDGETS:
        free = budget - hrank - 1
        if budget >= d or free < 1:
            continue
        # Baseline 1: PCA (budget-1 directions) + energy, cross term 0.
        if budget < len(xtr):
            pca = PCA(budget, svd_solver='randomized', random_state=start).fit(xtr)
            zx = pca.transform(xa)[:, :-1]
            ev = np.maximum(((xa - pca.mean_)**2).sum(1) - (zx**2).sum(1), 0.)
            put = record('pca_energy', budget)
            for meta, cn, dist, ref, phi in prepared:
                zc = pca.transform(cn)[:, :-1]
                evc = np.maximum(((cn - pca.mean_)**2).sum(1) - (zc**2).sum(1), 0.)
                put(meta, dist, ref, phi, sqdist(zx, zc) + ev[:, None] + evc[None, :])
        for f in SPLITS:
            extra = int(round(f * free))
            u, info = enrich_centroid_basis(xtr, c0, extra=extra, random_state=start)
            h = u.shape[1]
            t = budget - h - 1
            hx, rx, ex = split_residual(xa, origin, u)
            hr, rr, _ = split_residual(xtr, origin, u)
            cent = [split_residual(cn, origin, u) for _, cn, *_ in prepared]
            tag = dict(split=f, learned=h - hrank, tail=t)
            if t < 1:      # learned head + energy, no JL
                put = record('learned_energy', budget, **tag)
                for (meta, cn, dist, ref, phi), (hc, rc, ec) in zip(prepared, cent):
                    put(meta, dist, ref, phi, sqdist(hx, hc) + ex[:, None] + ec[None, :])
                continue
            for kind in MAP_KINDS:
                if kind != 'srht' and int(kind[1:]) > t:
                    continue
                for mp in MAPS:
                    tail = Tail(kind, map_seed(d=name, k=k, s=start, b=budget, f=f, kind=kind, map=mp,
                                               study='hybrid'), t, d)
                    tx, ttr = tail(rx), tail(rr)
                    puts = {e: record('hybrid', budget, estimator=e, map_kind=kind, map=mp, **tag)
                            for e in ESTIMATORS}
                    for (meta, cn, dist, ref, phi), (hc, rc, ec) in zip(prepared, cent):
                        tc = tail(rc)
                        lam = shrink_factor(ttr, rr, tc, rc)
                        head = sqdist(hx, hc) + ex[:, None] + ec[None, :]
                        for e, put in puts.items():
                            put(meta, dist, ref, phi, head - 2*cross_estimate(tx, tc, ex, ec, e, lam))
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
                                         splits=SPLITS, map_kinds=MAP_KINDS, maps=list(MAPS),
                                         estimators=ESTIMATORS, host=platform.node(),
                                         cpu_count=os.cpu_count(), threads=os.environ.get('OMP_NUM_THREADS')),
                                    indent=1, default=str))
    mine = [j for i, j in enumerate(jobs()) if i % a.workers == a.worker and (not a.only or j[0] in a.only)]
    cache = {}
    for name, loader, kw, k, s in mine:
        path = out/'jobs'/f'{name}_k{k}_s{s}.csv'
        if path.exists():
            continue
        if name not in cache:
            cache.clear()
            x = np.asarray(load_dataset(loader, **kw)['X'], float)
            cache[name] = x[np.linalg.norm(x, axis=1) > 0]
        t0 = time.time()
        rows = run_job(name, cache[name], k, s)
        tmp = path.with_suffix('.tmp')
        pd.DataFrame(rows).to_csv(tmp, index=False)
        tmp.replace(path)
        print(f'{name} k={k} s={s} rows={len(rows)} {time.time()-t0:.0f}s', flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
