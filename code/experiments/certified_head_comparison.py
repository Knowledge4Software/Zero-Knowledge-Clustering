"""Certified-head projection vs PCA: quality, geometry, setup cost and per-use cost.

Methods (same output width m for all):
- pca_fast / pca_exact : randomized (2 power iterations) / full-SVD PCA
- ours_orig            : current method, centroid span + 12 PCA, tail weight 1
- ours_ch{t}           : certified head of rank m-t, sparse JL tail of width t, weight 1
- ours_ch{t}_head      : same map; k-means uses the head coordinates only,
                         geometry metrics use the whole map
Learning uses the 60% training split. The 20% test split is never learned from
and serves as the stream of "new points" for the use phase. Draws 100-109.

Phases timed separately:
- build   : learn basis/PCA + construct map + compress the whole corpus
- cluster : k-means (5 inits) on the compressed corpus
- use     : per new point, compress + assign to the final centers
            (batch over the test split, and one-at-a-time over 200 points)

python -m experiments.certified_head_comparison --out results/certified_head_<date>
"""
from __future__ import annotations

import argparse, hashlib, json, os, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi

from data.loaders import load_dataset
from commitjl.enriched_jl import enrich_centroid_basis
from commitjl.certified_head import learn_certified_head, make_certified_head_model
from commitjl.residual_maps import make_residual_model
from experiments.tail_weight_study import DATASETS, WIDTHS, sse, pair_q95
from experiments.final_comparison import knn_recall

DRAWS = range(100, 110)
TAILS = (16, 32)
SINGLE_QUERIES = 200


def nearest(z, centers):
    return ((z**2).sum(1)[:, None]-2*z@centers.T+(centers**2).sum(1)).argmin(1)


def run_case(name, x, y, k, m, draw):
    rng = np.random.default_rng(1000+draw)
    order = rng.permutation(len(x)); train, test = order[:int(.6*len(x))], order[int(.8*len(x)):]
    t = time.perf_counter(); full = KMeans(k, n_init=10, random_state=draw).fit(x); full_s = time.perf_counter()-t
    ref = full.labels_; full_sse = sse(x, ref)
    ref2 = KMeans(k, n_init=10, random_state=draw+7777).fit(x).labels_
    base = dict(dataset=name, k=k, m=m, draw=draw, n=len(x), d=x.shape[1], full_seconds=full_s,
                full_self_ari=ari(ref, ref2))
    rows = []

    def evaluate(method, z, transform, build_s, cluster_cols=None, **extra):
        zc = z if cluster_cols is None else z[:, :cluster_cols]
        t = time.perf_counter(); km = KMeans(k, n_init=5, random_state=500+draw).fit(zc); cl_s = time.perf_counter()-t
        lab, centers = km.labels_, km.cluster_centers_
        cols = slice(None) if cluster_cols is None else slice(0, cluster_cols)
        # Use phase on unseen test rows: batch, then one point at a time.
        q = x[test]
        t = time.perf_counter(); qa = nearest(transform(q)[:, cols], centers); use_batch = (time.perf_counter()-t)/len(q)
        single = q[:SINGLE_QUERIES]; t = time.perf_counter()
        for p in single: nearest(transform(p[None, :])[:, cols], centers)
        use_single = (time.perf_counter()-t)/len(single)
        orig_centers = np.array([x[lab == j].mean(0) if np.any(lab == j) else x[0] for j in range(k)])
        rows.append(dict(base, method=method, ari_full=ari(ref, lab), nmi_full=nmi(ref, lab),
            ari_truth=ari(y, lab) if y is not None else np.nan, sse_ratio=sse(x, lab)/full_sse,
            pair_q95=pair_q95(x, z, np.random.default_rng(7+draw)),
            knn_recall=knn_recall(x, z, np.random.default_rng(9+draw)),
            use_agreement=float(np.mean(qa == nearest(q, orig_centers))),
            build_seconds=build_s, cluster_seconds=cl_s, setup_seconds=build_s+cl_s,
            use_batch_us=1e6*use_batch, use_single_us=1e6*use_single, **extra))

    for method, kw in [('pca_fast', dict(svd_solver='randomized', iterated_power=2)), ('pca_exact', dict(svd_solver='full'))]:
        t = time.perf_counter(); pca = PCA(m, random_state=draw, **kw).fit(x[train]); z = pca.transform(x)
        evaluate(method, z, pca.transform, time.perf_counter()-t)

    t0 = time.perf_counter()
    centers = KMeans(k, n_init=1, random_state=2000+draw).fit(x[train]).cluster_centers_
    centers_s = time.perf_counter()-t0
    seed_of = lambda tag: hashlib.sha256(json.dumps(dict(v='certified-head-v1', d=name, k=k, m=m, draw=draw, tag=tag)).encode()).digest()

    t = time.perf_counter()
    u, _ = enrich_centroid_basis(x[train], centers, extra=12, random_state=draw)
    model = make_residual_model('protected_plus12', x.shape[1], m, seed_of('orig'), centers, u); z = model.transform(x)
    evaluate('ours_orig', z, model.transform, centers_s+time.perf_counter()-t, head_rank=u.shape[1])

    for tail in TAILS:
        if m-tail < k: continue
        t = time.perf_counter()
        u, _ = learn_certified_head(x[train], centers, m-tail, random_state=draw)
        model = make_certified_head_model(x.shape[1], m, seed_of(tail), centers, u); z = model.transform(x)
        build_s = centers_s+time.perf_counter()-t; r = u.shape[1]
        evaluate(f'ours_ch{tail}', z, model.transform, build_s, head_rank=r)
        evaluate(f'ours_ch{tail}_head', z, model.transform, build_s, cluster_cols=r, head_rank=r)
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--only', nargs='*')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta = out/'protocol.json'
    if not meta.exists():
        json.dump(dict(draws=list(DRAWS), tails=TAILS, widths=WIDTHS, single_queries=SINGLE_QUERIES,
                       datasets=[(n, l, kw, k) for n, l, kw, k in DATASETS], host=platform.node(),
                       cpu_count=os.cpu_count(), threads={v: os.environ.get(v) for v in
                       ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']}),
                  open(meta, 'w'), indent=1, default=str)
    path = out/'rows.csv'
    rows = pd.read_csv(path).to_dict('records') if path.exists() else []
    done = {(r['dataset'], r['m'], r['draw']) for r in rows}
    for name, loader, kw, k in DATASETS:
        if a.only and name not in a.only: continue
        data = load_dataset(loader, **kw); x = np.asarray(data['X'], float)
        y = data.get('y'); y = None if y is None or not len(y) else np.asarray(y)
        keep = np.linalg.norm(x, axis=1) > 0; x = x[keep]; y = None if y is None else y[keep]
        for m in WIDTHS:
            if m >= x.shape[1]: continue
            for draw in DRAWS:
                if (name, m, draw) in done: continue
                rows += run_case(name, x, y, k, m, draw)   # a case is written only when complete
                tmp = out/'rows.tmp.csv'; pd.DataFrame(rows).to_csv(tmp, index=False); tmp.replace(path)
            print(name, m, 'done', flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
