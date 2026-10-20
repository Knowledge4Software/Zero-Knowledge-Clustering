"""Evaluate fixed weights and data-dependent w rules from experiments.tail_weight_sweep rows."""
import sys
import numpy as np
import pandas as pd

rows = pd.read_csv(sys.argv[1]); key = ['dataset', 'm', 'draw']
pca = rows[rows.method == 'pca_fast'].set_index(key)
ours = rows[rows.method != 'pca_fast'].copy()
pd.set_option('display.width', 250)

print('== Mean ARI vs Full by weight (rows: dataset/m, cols: extra, w) ==')
grid = ours.pivot_table(index=['dataset', 'm'], columns=['extra', 'tail_weight'], values='ari_full')
grid[('PCA', '')] = pca.groupby(['dataset', 'm']).ari_full.mean()
print(grid.round(3).to_string())
print('\nhead share:', ours.groupby(['dataset', 'extra']).head_share.mean().round(3).to_dict())


def best_cost(g, weights, extras):
    c = g[g.tail_weight.isin(weights) & g.extra.isin(extras)]
    pick = c.loc[c.sse_ratio.idxmin()].copy()
    # charge every clustering that the rule had to run
    pick['total_seconds'] = c.fit_seconds.groupby(c.extra).first().sum()+c.kmeans_seconds.sum()
    return pick


rules = {}
for e in (12, 32):
    for w in sorted(ours.tail_weight.unique()):
        rules[f'fixed e{e} w={w}'] = ours[(ours.extra == e) & (ours.tail_weight == w)].set_index(key)
for name, ws, es in [('bestcost e12 {0.25,1}', [.25, 1], [12]), ('bestcost e12 all-w', None, [12]),
                     ('bestcost e12+e32 {0.25,1}', [.25, 1], [12, 32])]:
    ws = ws or list(ours.tail_weight.unique())
    rules[name] = pd.DataFrame([best_cost(g, ws, es) for _, g in ours.groupby(key)]).set_index(key)

# Spectrum rule, leave-one-dataset-out: w = 1 if head_share >= tau else w_low (tau, w_low fitted on other datasets)
e12 = ours[ours.extra == 12]; picks = []
for ds in e12.dataset.unique():
    other = e12[e12.dataset != ds]; best = None
    for tau in np.linspace(.05, .6, 23):
        for wl in (.25, .35, .5):
            sel = other[np.where(other.head_share >= tau, 1.0, wl) == other.tail_weight]
            s = sel.ari_full.mean()
            if best is None or s > best[0]: best = (s, tau, wl)
    _, tau, wl = best; mine = e12[e12.dataset == ds]
    p = mine[np.where(mine.head_share >= tau, 1.0, wl) == mine.tail_weight].copy(); p['rule'] = f'tau={tau:.2f},w={wl}'
    picks.append(p)
rules['spectrum rule (LODO)'] = pd.concat(picks).set_index(key)
print('\nspectrum rule thresholds:', rules['spectrum rule (LODO)'].groupby('dataset').rule.first().to_dict())

rng = np.random.default_rng(0); out = []
for name, r in rules.items():
    wins = losses = 0; speed = []; ari_m = []; per = {}
    for (ds, m), p in pca.groupby(['dataset', 'm']):
        o = r.loc[p.index]; d = (o.ari_full-p.ari_full).to_numpy()
        b = [rng.choice(d, len(d)).mean() for _ in range(3000)]; lo, hi = np.quantile(b, [.025, .975])
        v = 'W' if lo > 0 else 'L' if hi < 0 else '='; wins += v == 'W'; losses += v == 'L'
        per[f"{ds.replace('20ng_tfidf_nonzero','txt').replace('fashion_mnist','fm').replace('gaussian_hard','gh')[:7]}{m}"] = v; speed.append((p.total_seconds/o.total_seconds).mean()); ari_m.append(o.ari_full.mean())
    out.append(dict(rule=name, wins=wins, losses=losses, mean_ari=np.mean(ari_m),
                    mean_speedup=np.mean(speed), q95=r.pair_q95.mean(), **per))
res = pd.DataFrame(out).sort_values(['losses', 'wins'], ascending=[True, False])
print(f"\nPCA mean ARI {pca.groupby(['dataset','m']).ari_full.mean().mean():.3f}, PCA q95 {pca.pair_q95.mean():.3f}")
print(res.round(3).to_string(index=False))
