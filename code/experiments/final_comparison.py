"""Confirmatory comparison: adaptive-weight protected JL vs PCA on NEW draws.

Rules are frozen before this run (chosen on tail_weight_sweep_20261016, draws 0-4):
- ours_quality : best original-space k-means cost over w in {0.25,0.35,0.5,0.6,0.75,1}
- ours_fast    : best original-space k-means cost over w in {0.25,1}
- ours_spectrum: w=0.25 if training head share < 0.27 else w=1
- ours_w1      : original method (distance-guarantee mode)
Head: centroid span + 12 residual PCA directions. Controls: fast randomized PCA
(2 power iterations) and exact full-SVD PCA, same output width. Draws 100-109
were never used to design the rules. Full self-agreement is measured per case.

python -m experiments.final_comparison --out results/final_comparison_20261016
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
from commitjl.residual_maps import make_residual_model
from experiments.tail_weight_study import DATASETS, WIDTHS, sse, pair_q95

WEIGHTS = (0.25, 0.35, 0.5, 0.6, 0.75, 1.0)
RULES = {'ours_quality': WEIGHTS, 'ours_fast': (0.25, 1.0)}
SHARE_THRESHOLD, LOW_W = 0.27, 0.25
DRAWS = range(100, 110)


def knn_recall(x, z, rng, queries=200, k=10):
    q = rng.choice(len(x), min(queries, len(x)), replace=False); hits = 0
    for i in q:
        a = np.argsort(((x-x[i])**2).sum(1))[1:k+1]; b = np.argsort(((z-z[i])**2).sum(1))[1:k+1]
        hits += len(np.intersect1d(a, b))
    return hits/(len(q)*k)


def run_case(name, x, y, k, m, draw, out):
    rng = np.random.default_rng(1000+draw)
    train = rng.permutation(len(x))[:int(.6*len(x))]
    t = time.perf_counter(); ref = KMeans(k, n_init=10, random_state=draw).fit(x).labels_; full_s = time.perf_counter()-t
    ref2 = KMeans(k, n_init=10, random_state=draw+7777).fit(x).labels_
    full_sse = sse(x, ref)
    base = dict(dataset=name, k=k, m=m, draw=draw, full_seconds=full_s, full_self_ari=ari(ref, ref2),
                full_ari_truth=ari(y, ref) if y is not None else np.nan)

    def cluster(z):
        t = time.perf_counter(); lab = KMeans(k, n_init=5, random_state=500+draw).fit(z).labels_
        return lab, time.perf_counter()-t

    def record(method, z, lab, fit_s, km_s, sel_s=0.0, **extra):
        out.append(dict(base, method=method, ari_full=ari(ref, lab), nmi_full=nmi(ref, lab),
            ari_truth=ari(y, lab) if y is not None else np.nan, sse_ratio=sse(x, lab)/full_sse,
            pair_q95=pair_q95(x, z, np.random.default_rng(7+draw)),
            knn_recall=knn_recall(x, z, np.random.default_rng(9+draw)), fit_seconds=fit_s,
            kmeans_seconds=km_s, select_seconds=sel_s, total_seconds=fit_s+km_s+sel_s, **extra))

    for method, solver in [('pca_fast', dict(svd_solver='randomized', iterated_power=2)), ('pca_exact', dict(svd_solver='full'))]:
        t = time.perf_counter(); z = PCA(m, random_state=draw, **solver).fit(x[train]).transform(x)
        fit_s = time.perf_counter()-t; lab, km_s = cluster(z); record(method, z, lab, fit_s, km_s)

    t = time.perf_counter()
    centers = KMeans(k, n_init=1, random_state=2000+draw).fit(x[train]).cluster_centers_
    basis, _ = enrich_centroid_basis(x[train], centers, extra=12, random_state=draw); rank = basis.shape[1]
    seed = hashlib.sha256(json.dumps(dict(v='final-comparison-v1', d=name, k=k, m=m, draw=draw)).encode()).digest()
    z1 = make_residual_model('protected_plus12', x.shape[1], m, seed, centers, basis).transform(x)
    r = x[train]-centers[0]; share = float((z1[train, :rank]**2).sum()/(r**2).sum())
    fit_s = time.perf_counter()-t
    cand = {}
    for w in WEIGHTS:
        z = z1.copy(); z[:, rank:] *= w; lab, km_s = cluster(z)
        t = time.perf_counter(); cost = sse(x, lab); cost_s = time.perf_counter()-t
        cand[w] = (z, lab, km_s, cost, cost_s)
    z, lab, km_s, _, _ = cand[1.0]
    record('ours_w1', z, lab, fit_s, km_s, tail_weight=1.0, head_share=share)
    for rule, ws in RULES.items():
        w = min(ws, key=lambda v: cand[v][3]); z, lab = cand[w][0], cand[w][1]
        record(rule, z, lab, fit_s, sum(cand[v][2] for v in ws), sum(cand[v][4] for v in ws),
               tail_weight=w, head_share=share)
    w = LOW_W if share < SHARE_THRESHOLD else 1.0; z, lab, km_s, _, _ = cand[w]
    record('ours_spectrum', z, lab, fit_s, km_s, tail_weight=w, head_share=share)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--only', nargs='*')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if not (out/'protocol.json').exists(): json.dump(dict(rules={k: list(v) for k, v in RULES.items()}, spectrum=dict(threshold=SHARE_THRESHOLD, low_w=LOW_W),
                   weights=WEIGHTS, draws=list(DRAWS), datasets=[(n, l, kw, k) for n, l, kw, k in DATASETS],
                   frozen_before_run=True, design_source='results/tail_weight_sweep_20261016 (draws 0-4)'),
              open(out/'protocol.json', 'w'), indent=1, default=str)
    prev = out/'rows.csv'
    rows = pd.read_csv(prev).to_dict('records') if prev.exists() else []
    # Resume: a case counts as done only when all its method rows were written.
    counts = pd.DataFrame(rows).groupby(['dataset', 'm', 'draw']).size() if rows else pd.Series(dtype=int)
    done = {key for key, n in counts.items() if n == 6}
    rows = [r for r in rows if (r['dataset'], r['m'], r['draw']) in done]
    for name, loader, kw, k in DATASETS:
        if a.only and name not in a.only: continue
        data = load_dataset(loader, **kw); x = np.asarray(data['X'], float)
        y = data.get('y'); y = None if y is None or not len(y) else np.asarray(y)
        keep = np.linalg.norm(x, axis=1) > 0; x = x[keep]; y = None if y is None else y[keep]
        for m in WIDTHS:
            if m >= x.shape[1]: continue
            for draw in DRAWS:
                if (name, m, draw) in done: continue
                run_case(name, x, y, k, m, draw, rows)
                pd.DataFrame(rows).to_csv(out/'rows.csv', index=False)
            print(name, m, 'done', flush=True)


if __name__ == '__main__':
    main()
