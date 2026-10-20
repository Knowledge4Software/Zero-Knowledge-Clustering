"""Tail-weight sweep: w grid x head size (12 / 32 extra PCA directions) vs fast PCA.

Records every grid point so data-dependent rules can be evaluated afterwards:
- best-cost rules pick, per case, the partition with the lowest ORIGINAL-space
  k-means cost among a set of weights (label-free, no Full reference used);
- head_share (training variance captured by the head) is recorded for
  spectrum rules.
Same splits, Full reference and seeds as experiments.tail_weight_study.

python -m experiments.tail_weight_sweep --out results/tail_weight_sweep_20261016
"""
from __future__ import annotations

import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ari

from data.loaders import load_dataset
from commitjl.enriched_jl import enrich_centroid_basis
from commitjl.residual_maps import make_residual_model
from experiments.tail_weight_study import DATASETS, WIDTHS, sse, pair_q95

WEIGHTS = (0.25, 0.35, 0.5, 0.6, 0.75, 1.0)
EXTRAS = (12, 32)


def run_case(name, x, y, k, m, draw, out):
    rng = np.random.default_rng(1000+draw)
    train = rng.permutation(len(x))[:int(.6*len(x))]
    ref = KMeans(k, n_init=10, random_state=draw).fit(x).labels_; full_sse = sse(x, ref)

    def record(method, z, fit_s, **extra):
        t = time.perf_counter(); lab = KMeans(k, n_init=5, random_state=500+draw).fit(z).labels_
        km_s = time.perf_counter()-t
        out.append(dict(dataset=name, k=k, m=m, draw=draw, method=method, ari_full=ari(ref, lab),
            ari_truth=ari(y, lab) if y is not None else np.nan, sse_ratio=sse(x, lab)/full_sse,
            pair_q95=pair_q95(x, z, np.random.default_rng(7+draw)), fit_seconds=fit_s,
            kmeans_seconds=km_s, total_seconds=fit_s+km_s, **extra))

    t = time.perf_counter()
    zp = PCA(m, svd_solver='randomized', iterated_power=2, random_state=draw).fit(x[train]).transform(x)
    record('pca_fast', zp, time.perf_counter()-t)
    centers = KMeans(k, n_init=1, random_state=2000+draw).fit(x[train]).cluster_centers_
    for extra in EXTRAS:
        t = time.perf_counter()
        basis, _ = enrich_centroid_basis(x[train], centers, extra=extra, random_state=draw)
        rank = basis.shape[1]
        if m-rank < 1: continue
        seed = hashlib.sha256(json.dumps(dict(v='tail-weight-sweep-v1', d=name, k=k, m=m, draw=draw,
                                              extra=extra)).encode()).digest()
        z1 = make_residual_model('protected_plus12', x.shape[1], m, seed, centers, basis).transform(x)
        fit_s = time.perf_counter()-t
        r = x[train]-centers[0]
        share = float((z1[train, :rank]**2).sum()/(r**2).sum())
        for w in WEIGHTS:
            z = z1.copy(); z[:, rank:] *= w
            record(f'ours_e{extra}', z, fit_s, extra=extra, tail_weight=w, head_share=share)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True)
    ap.add_argument('--draws', type=int, default=5); ap.add_argument('--only', nargs='*')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, loader, kw, k in DATASETS:
        if a.only and name not in a.only: continue
        data = load_dataset(loader, **kw); x = np.asarray(data['X'], float)
        y = data.get('y'); y = None if y is None or not len(y) else np.asarray(y)
        keep = np.linalg.norm(x, axis=1) > 0; x = x[keep]; y = None if y is None else y[keep]
        for m in WIDTHS:
            if m >= x.shape[1]: continue
            for draw in range(a.draws):
                run_case(name, x, y, k, m, draw, rows)
                pd.DataFrame(rows).to_csv(out/'rows.csv', index=False)
            print(name, m, 'done', flush=True)
    json.dump(dict(weights=WEIGHTS, extras=EXTRAS, draws=a.draws), open(out/'protocol.json', 'w'))


if __name__ == '__main__':
    main()
