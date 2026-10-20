"""Summarize experiments.tail_weight_study rows: means and paired bootstrap CIs vs fast PCA."""
import sys
import numpy as np
import pandas as pd

rows = pd.read_csv(sys.argv[1])
metrics = ['ari_full', 'ari_truth', 'sse_ratio', 'pair_q95', 'total_seconds']
print(rows.groupby(['dataset', 'm', 'method'])[metrics+['tail_weight']].mean().round(3).to_string())
rng = np.random.default_rng(0)
out = []
for (ds, m), g in rows.groupby(['dataset', 'm']):
    p = g[g.method == 'pca_fast'].set_index('draw')
    for method in [m for m in ['ours_w1', 'ours_w025', 'ours_auto', 'ours_cost'] if m in set(g.method)]:
        o = g[g.method == method].set_index('draw').loc[p.index]
        diff = (o.ari_full-p.ari_full).to_numpy()
        boot = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
        lo, hi = np.quantile(boot, [.025, .975])
        out.append(dict(dataset=ds, m=m, method=method, ari_diff=diff.mean(), ci_low=lo, ci_high=hi,
                        verdict='WIN' if lo > 0 else 'LOSS' if hi < 0 else 'tie'))
print(pd.DataFrame(out).round(3).to_string(index=False))
