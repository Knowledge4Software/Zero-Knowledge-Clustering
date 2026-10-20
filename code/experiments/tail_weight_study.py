"""Tail-weight study: protected head + canonical sparse JL tail with tail weight
1 (original), 0.25 (fixed) and label-free auto selection, versus fast PCA.

Exploratory native screen. Representation learning (centers, head, PCA, auto
weight) uses the 60% training split only. Full reference = best-SSE KMeans of
10 initializations on the whole corpus. Fresh clustering on each projection
uses 5 initializations; labels are evaluated in the original space.

python -m experiments.tail_weight_study --out results/tail_weight_study_20261016
"""
from __future__ import annotations

import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi

from data.loaders import load_dataset
from commitjl.enriched_jl import enrich_centroid_basis
from commitjl.residual_maps import make_residual_model, select_tail_weight, select_tail_weight_by_cost

DATASETS = [  # (label, loader, kwargs, k)
    ('gaussian_hard', 'gaussian_mixture', dict(n=2000, d=512, k=10, gamma=0.35, imbalance=3.0, seed=0), 10),
    ('mnist', 'mnist', dict(n_max=5000, seed=0), 10),
    ('fashion_mnist', 'fashion_mnist', dict(n_max=5000, seed=0), 10),
    ('gisette', 'gisette', dict(n_max=3000, seed=0), 5),
    ('20ng_tfidf_nonzero', '20ng_tfidf', dict(n_max=3000, seed=0), 5),
    ('20ng_tfidf_nonzero_k20', '20ng_tfidf', dict(n_max=3000, seed=0), 20),
]
WIDTHS = (128, 256)


def sse(x, labels):
    return float(sum(((x[labels == j]-x[labels == j].mean(0))**2).sum() for j in np.unique(labels)))


def pair_q95(x, z, rng, pairs=4000):
    i, j = rng.integers(0, len(x), (2, pairs)); keep = i != j; i, j = i[keep], j[keep]
    a = ((x[i]-x[j])**2).sum(1); b = ((z[i]-z[j])**2).sum(1)
    return float(np.quantile(np.abs(b/a-1), .95))


def run_case(name, x, y, k, m, draw, out_rows):
    rng = np.random.default_rng(1000+draw)
    order = rng.permutation(len(x)); train, val = order[:int(.6*len(x))], order[int(.6*len(x)):int(.8*len(x))]
    t = time.perf_counter(); full = KMeans(k, n_init=10, random_state=draw).fit(x); full_s = time.perf_counter()-t
    ref, full_sse = full.labels_, sse(x, full.labels_)

    def record(method, z, fit_s, extra=None):
        t = time.perf_counter(); lab = KMeans(k, n_init=5, random_state=500+draw).fit(z).labels_
        km_s = time.perf_counter()-t
        out_rows.append(dict(dataset=name, k=k, m=m, draw=draw, method=method,
            ari_full=ari(ref, lab), nmi_full=nmi(ref, lab), ari_truth=ari(y, lab) if y is not None else np.nan,
            full_ari_truth=ari(y, ref) if y is not None else np.nan, sse_ratio=sse(x, lab)/full_sse,
            pair_q95=pair_q95(x, z, np.random.default_rng(7+draw)), fit_seconds=fit_s, kmeans_seconds=km_s,
            total_seconds=fit_s+km_s, full_seconds=full_s, **(extra or {})))

    t = time.perf_counter()
    pca = PCA(m, svd_solver='randomized', iterated_power=2, random_state=draw).fit(x[train])
    zp = pca.transform(x); record('pca_fast', zp, time.perf_counter()-t)

    t = time.perf_counter()
    centers = KMeans(k, n_init=1, random_state=2000+draw).fit(x[train]).cluster_centers_
    basis, _ = enrich_centroid_basis(x[train], centers, extra=12, random_state=draw)
    seed = hashlib.sha256(json.dumps(dict(v='tail-weight-study-v1', d=name, k=k, m=m, draw=draw)).encode()).digest()
    model = make_residual_model('protected_plus12', x.shape[1], m, seed, centers, basis)
    z1 = model.transform(x); base_s = time.perf_counter()-t
    rank = basis.shape[1]
    record('ours_w1', z1, base_s, dict(tail_weight=1.0))
    zq = z1.copy(); zq[:, rank:] *= .25
    record('ours_w025', zq, base_s, dict(tail_weight=.25))
    t = time.perf_counter(); w, scores = select_tail_weight(z1[train], rank, k, seed=3000+draw); sel_s = time.perf_counter()-t
    za = z1.copy(); za[:, rank:] *= w
    record('ours_auto', za, base_s+sel_s, dict(tail_weight=w, selection_seconds=sel_s,
                                               selection_scores=json.dumps(scores)))
    t = time.perf_counter()
    w, scores = select_tail_weight_by_cost(z1[train], x[train], z1[val], x[val], rank, k, seed=4000+draw)
    sel_s = time.perf_counter()-t
    zc = z1.copy(); zc[:, rank:] *= w
    record('ours_cost', zc, base_s+sel_s, dict(tail_weight=w, selection_seconds=sel_s,
                                               selection_scores=json.dumps(scores)))


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
    json.dump(dict(datasets=[(n, l, kw, k) for n, l, kw, k in DATASETS], widths=WIDTHS, draws=a.draws),
              open(out/'protocol.json', 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
