"""Adaptive-prover attacks: projections known before commitment vs sampled after.

A cheating prover perturbs the committed data inside the kernel of the
projection it expects. If the projection P is known before the data is
committed (public PCA, deterministic Pisot, pre-published Sparse JL), the
perturbation is invisible in the projected space. Under commit-then-sample
the prover must guess; we model the best oblivious guess as the kernel of an
independent draw of the same family, then evaluate with the real P.

Attacks (fixed public centres C from a 60% training split; audited points
from the held-out 40%):

- cost hiding: X' = X + V, rows of V in the (guessed) kernel, scaled so the
  true fixed-centre cost grows by about (1 + alpha). Reported: detected
  inflation = claimed(X') / claimed(X) (ideal 1 + alpha; 1.0 = invisible).
- assignment flip: each point x moves along the kernel component u of
  (c_b - c_a), b its runner-up centre, just far enough that its true nearest
  centre becomes b. Reported: success = the projected (proved) assignment
  differs from the true one, and the required step relative to ||x - c_a||.
  If the kernel component of c_b - c_a is zero the flip is impossible.

python -m experiments.adversarial_attack --out results/adversarial_attack_<date>
"""
from __future__ import annotations

import argparse, hashlib, json, os, platform
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from data.loaders import load_dataset
from commitjl.protected_jl import centroid_basis
from commitjl.projections import pisot_beta_projection
from commitjl.sparse_gen import sparse_matrix_gen_csr

DATASETS = [
    ('gaussian_hard', 'gaussian_mixture', dict(n=2000, d=512, k=10, gamma=0.35, imbalance=3.0, seed=0)),
    ('mnist', 'mnist', dict(n_max=5000, seed=0)),
    ('fashion_mnist', 'fashion_mnist', dict(n_max=5000, seed=0)),
    ('20ng_tfidf_nonzero', '20ng_tfidf', dict(n_max=3000, seed=0)),
    ('gisette', 'gisette', dict(n_max=3000, seed=0)),
]
KS = (5, 10)
WIDTHS = (16, 64)
ALPHAS = (1.0, 10.0)
STARTS = range(2)
DRAWS = range(5)
MAX_FLIP_POINTS = 500


def sparse_matrix(tag, m, d):
    _, a = sparse_matrix_gen_csr(hashlib.sha256(tag.encode()).digest(), m, d, 1)
    return a.toarray().astype(float)


def matrix(method, x_train, c, m, tag):
    """Real-valued map P (m x d) with JL-style normalisation."""
    d = x_train.shape[1]
    if method == 'pca_public':
        return PCA(m, svd_solver='randomized', random_state=0).fit(x_train).components_
    if method == 'pisot':
        return pisot_beta_projection(m, d, seed=int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16), S=1)['real']*np.sqrt(3/m)
    if method.startswith('sparse'):
        return sparse_matrix(tag, m, d)
    u = centroid_basis(c)  # protected: head rows U^T, tail rows A (I - U U^T)
    a = sparse_matrix(tag, m-u.shape[1], d)
    return np.vstack((u.T, a-(a@u)@u.T))


def kernel_part(p, y):
    """Component of each row of y in ker(p)."""
    coef = np.linalg.lstsq(p@p.T, p@y.T, rcond=None)[0]
    return y-(p.T@coef).T


def nearest(z, zc):
    return ((z**2).sum(1)[:, None]-2*z@zc.T+(zc**2).sum(1)).argmin(1)


def claimed_cost(p, x, c):
    z, zc = x@p.T, c@p.T
    lab = nearest(z, zc)
    return float(((z-zc[lab])**2).sum())


def run_case(name, x, k, start, out):
    rng = np.random.default_rng(900+start)
    order = rng.permutation(len(x)); tr, au = order[:int(.6*len(x))], order[int(.6*len(x)):]
    c = KMeans(k, n_init=5, random_state=start).fit(x[tr]).cluster_centers_
    xa = x[au]-c[0]; cc = c-c[0]  # common origin c_0 (any origin gives the same costs)
    true_lab = nearest(xa, cc); phi = float(((xa-cc[true_lab])**2).sum())
    known = ('pca_public', 'pisot', 'sparse_prepublished', 'protected_prepublished')
    fresh = ('sparse_commit', 'protected_commit')
    for m in WIDTHS:
        if m >= x.shape[1]: continue
        for draw in DRAWS:
            for method in known+fresh:
                if method in ('pca_public',) and draw > 0: continue  # deterministic given the data
                base = f'{name}-k{k}-s{start}-m{m}-d{draw}-{method}'
                p = matrix(method, x[tr]-c[0], cc, m, base)
                # The attacker's view: the real P if known in advance, otherwise an independent guess.
                guess = p if method in known else matrix(method, x[tr]-c[0], cc, m, base+'-guess')
                honest = claimed_cost(p, xa, cc)
                row = dict(dataset=name, k=k, start=start, m=m, draw=draw, method=method,
                           known_before_commit=method in known, honest_claim_ratio=honest/phi)
                # Cost hiding.
                noise = kernel_part(guess, np.random.default_rng(draw).standard_normal(xa.shape))
                for alpha in ALPHAS:
                    v = noise*np.sqrt(alpha*phi/(noise**2).sum())
                    xs = xa+v; lab = nearest(xs, cc); true_new = float(((xs-cc[lab])**2).sum())
                    row[f'true_inflation_a{alpha:g}'] = true_new/phi
                    row[f'detected_inflation_a{alpha:g}'] = claimed_cost(p, xs, cc)/honest
                # Assignment flip.
                pts = np.arange(min(MAX_FLIP_POINTS, len(xa)))
                d2 = ((xa[pts, None, :]-cc[None])**2).sum(2)
                a_idx = true_lab[pts]; srt = np.argsort(d2, axis=1); b_idx = np.where(srt[:, 0] == a_idx, srt[:, 1], srt[:, 0])
                diff = cc[b_idx]-cc[a_idx]; u = kernel_part(guess, diff)
                un = np.linalg.norm(u, axis=1); possible = un > 1e-9*np.linalg.norm(diff, axis=1)
                margin = d2[pts, b_idx]-d2[pts, a_idx]
                # Moving t along the unit kernel direction u/|u| changes the margin by
                # -2 t <u/|u|, diff> = -2 t |u| (u is the orthogonal kernel part of diff).
                step = np.where(possible, 1.05*margin/(2*np.maximum(un, 1e-300)), 0.0)
                moved = xa[pts]+np.where(possible[:, None], step[:, None]*u/np.maximum(un, 1e-300)[:, None], 0.0)
                new_true = nearest(moved, cc); proved = nearest(moved@p.T, cc@p.T)
                flipped = possible & (new_true != a_idx)
                row.update(flip_possible=float(possible.mean()),
                           flip_success=float(np.mean(flipped & (proved != new_true))),
                           flip_step_rel=float(np.median((step/np.linalg.norm(xa[pts]-cc[a_idx], axis=1))[possible])) if possible.any() else np.nan,
                           honest_assignment_error=float(np.mean(nearest(xa[pts]@p.T, cc@p.T) != a_idx)))
                out.append(row)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--only', nargs='*')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if not (out/'protocol.json').exists():
        json.dump(dict(datasets=DATASETS, ks=KS, widths=WIDTHS, alphas=ALPHAS, starts=list(STARTS), draws=list(DRAWS),
                       max_flip_points=MAX_FLIP_POINTS, host=platform.node(), cpu_count=os.cpu_count()),
                  open(out/'protocol.json', 'w'), indent=1, default=str)
    path = out/'rows.csv'
    rows = pd.read_csv(path).to_dict('records') if path.exists() else []
    done = {(r['dataset'], r['k'], r['start']) for r in rows}
    for name, loader, kw in DATASETS:
        if a.only and name not in a.only: continue
        x = np.asarray(load_dataset(loader, **kw)['X'], float); x = x[np.linalg.norm(x, axis=1) > 0]
        for k in KS:
            for start in STARTS:
                if (name, k, start) in done: continue
                new = []; run_case(name, x, k, start, new); rows += new
                tmp = out/'rows.tmp.csv'; pd.DataFrame(rows).to_csv(tmp, index=False); tmp.replace(path)
            print(name, k, 'done', flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
