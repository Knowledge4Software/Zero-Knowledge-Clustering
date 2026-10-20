"""Search-denoised / score-faithful clustering vs PCA, on the confirmatory draws.

Methods (one committed w=1 sketch for every "ours" row; distances are its distances):
- ours_w1          : plain k-means on the w=1 sketch (original method)
- ours_cont        : continuation 0.25 -> 0.5 -> 1 (warm start), pick min cost IN THE SKETCH
- ours_multi_sk    : independent k-means at w in {0.25,...,1}, pick min cost IN THE SKETCH
- ours_multi_orig  : same candidates, pick min cost in ORIGINAL space (= earlier quality mode)
Controls: fast randomized PCA. Same datasets, splits and draws 100-109 as final_comparison.

python -m experiments.continuation_comparison --out results/continuation_comparison_20261016
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
from commitjl.residual_maps import make_residual_model, continuation_cluster, sketch_partition_cost
from experiments.tail_weight_study import DATASETS, WIDTHS, sse, pair_q95
from experiments.final_comparison import knn_recall, DRAWS, WEIGHTS


def run_case(name, x, y, k, m, draw, out):
    rng = np.random.default_rng(1000+draw)
    train = rng.permutation(len(x))[:int(.6*len(x))]
    ref = KMeans(k, n_init=10, random_state=draw).fit(x).labels_; full_sse = sse(x, ref)

    def record(method, z, lab, seconds, **extra):
        out.append(dict(dataset=name, k=k, m=m, draw=draw, method=method, ari_full=ari(ref, lab),
            ari_truth=ari(y, lab) if y is not None else np.nan, sse_ratio=sse(x, lab)/full_sse,
            pair_q95=pair_q95(x, z, np.random.default_rng(7+draw)),
            knn_recall=knn_recall(x, z, np.random.default_rng(9+draw)), total_seconds=seconds, **extra))

    t = time.perf_counter(); zp = PCA(m, svd_solver='randomized', iterated_power=2, random_state=draw).fit(x[train]).transform(x)
    lab = KMeans(k, n_init=5, random_state=500+draw).fit(zp).labels_; record('pca_fast', zp, lab, time.perf_counter()-t)

    t = time.perf_counter()
    centers = KMeans(k, n_init=1, random_state=2000+draw).fit(x[train]).cluster_centers_
    basis, _ = enrich_centroid_basis(x[train], centers, extra=12, random_state=draw); rank = basis.shape[1]
    seed = hashlib.sha256(json.dumps(dict(v='final-comparison-v1', d=name, k=k, m=m, draw=draw)).encode()).digest()
    z = make_residual_model('protected_plus12', x.shape[1], m, seed, centers, basis).transform(x)
    fit_s = time.perf_counter()-t

    t = time.perf_counter(); lab, w, cands = continuation_cluster(z, rank, k, seed=500+draw)
    record('ours_cont', z, lab, fit_s+time.perf_counter()-t, chosen_weight=w,
           stage_ari=json.dumps([round(ari(ref, c['labels']), 4) for c in cands]))

    cand = {}
    for wt in WEIGHTS:
        zw = z.copy(); zw[:, rank:] *= wt
        t = time.perf_counter(); l = KMeans(k, n_init=5, random_state=500+draw).fit(zw).labels_; km_s = time.perf_counter()-t
        t = time.perf_counter(); sk = sketch_partition_cost(z, l); sk_s = time.perf_counter()-t
        t = time.perf_counter(); orig = sse(x, l); orig_s = time.perf_counter()-t
        cand[wt] = (l, km_s, sk, sk_s, orig, orig_s)
    record('ours_w1', z, cand[1.0][0], fit_s+cand[1.0][1])
    km = sum(c[1] for c in cand.values())
    w = min(WEIGHTS, key=lambda v: cand[v][2])
    record('ours_multi_sk', z, cand[w][0], fit_s+km+sum(c[3] for c in cand.values()), chosen_weight=w)
    w = min(WEIGHTS, key=lambda v: cand[v][4])
    record('ours_multi_orig', z, cand[w][0], fit_s+km+sum(c[5] for c in cand.values()), chosen_weight=w)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--only', nargs='*')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    prev = out/'rows.csv'; rows = pd.read_csv(prev).to_dict('records') if prev.exists() else []
    counts = pd.DataFrame(rows).groupby(['dataset', 'm', 'draw']).size() if rows else pd.Series(dtype=int)
    done = {key for key, n in counts.items() if n == 5}
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
