"""Summarize experiments.certified_head_comparison (all worker folders under one root).

python analysis/certified_head_summary.py results/certified_head_20261017
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(sys.argv[1])
rows = pd.concat([pd.read_csv(p) for p in sorted(root.glob('worker*/rows.csv'))], ignore_index=True)
rows.to_csv(root/'all_rows.csv', index=False)
BASE = 'pca_fast'
OURS = [m for m in ['ours_orig', 'ours_ch16', 'ours_ch16_head', 'ours_ch32', 'ours_ch32_head'] if m in set(rows.method)]
HIGHER = dict(ari_full=1, ari_truth=1, nmi_full=1, knn_recall=1, use_agreement=1,
              sse_ratio=0, pair_q95=0, setup_seconds=0, use_single_us=0, use_batch_us=0)
rng = np.random.default_rng(0)
cells = sorted(rows.groupby(['dataset', 'm']).groups)
print('cases per cell:', rows.groupby(['dataset', 'm']).draw.nunique().to_dict())


def verdict(meth, metric, d, m):
    g = rows[(rows.dataset == d) & (rows.m == m)]
    b = g[g.method == BASE].set_index('draw')[metric]; o = g[g.method == meth].set_index('draw')[metric].loc[b.index]
    if metric.endswith('_seconds') or metric.endswith('_us'):
        r = (b/o).to_numpy(); bs = [rng.choice(r, len(r)).mean() for _ in range(2000)]; lo, hi = np.quantile(bs, [.025, .975])
        return 'W' if lo > 1 else 'L' if hi < 1 else '='
    diff = (o-b).to_numpy(); bs = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
    lo, hi = np.quantile(bs, [.025, .975]); up = HIGHER[metric]
    return 'W' if (lo > 0 if up else hi < 0) else 'L' if (hi < 0 if up else lo > 0) else '='


pd.set_option('display.width', 250)
for metric in ['ari_full', 'pair_q95', 'knn_recall', 'use_agreement']:
    t = rows.pivot_table(index=['dataset', 'm'], columns='method', values=metric).round(3)
    print(f'\n== {metric} (mean)'); print(t.to_string())
print('\n== scoreboard vs randomized PCA: wins / same / losses over', len(cells), 'settings')
board = []
for meth in OURS:
    r = dict(method=meth)
    for metric in HIGHER:
        v = [verdict(meth, metric, d, m) for d, m in cells]; r[metric] = f"{v.count('W')}/{v.count('=')}/{v.count('L')}"
    board.append(r)
print(pd.DataFrame(board).to_string(index=False))
print('\n== mean ARI vs Full:', rows.groupby('method').ari_full.mean().round(3).to_dict())
print('== mean full self-ARI by dataset:', rows.groupby('dataset').full_self_ari.mean().round(3).to_dict())

# Break-even: setup difference repaid by cheaper per-use cost (single-point latency)
tm = rows.groupby(['dataset', 'm', 'method'])[['setup_seconds', 'use_single_us', 'use_batch_us']].mean()
out = []
for d, m in cells:
    p = tm.loc[(d, m, BASE)]
    for meth in OURS:
        o = tm.loc[(d, m, meth)]; ds = o.setup_seconds-p.setup_seconds; du = (p.use_single_us-o.use_single_us)/1e6
        be = 0 if ds <= 0 and du >= 0 else (ds/du if du > 0 else np.inf)
        out.append(dict(dataset=d, m=m, method=meth, setup_s=o.setup_seconds, pca_setup_s=p.setup_seconds,
                        use_us=o.use_single_us, pca_use_us=p.use_single_us, use_speedup=p.use_single_us/o.use_single_us,
                        break_even_points=be))
be = pd.DataFrame(out); be.to_csv(root/'break_even.csv', index=False)
print('\n== setup / per-use timing and break-even (single-point use)')
print(be.round(3).to_string(index=False))
