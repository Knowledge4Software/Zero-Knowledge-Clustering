"""Build the PDF comparison report from results/final_comparison_*/rows.csv.

python analysis/final_comparison_report.py results/final_comparison_20261016 ../output/pdf/commitjl_vs_pca_final.pdf
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

src, OUT = Path(sys.argv[1]), Path(sys.argv[2]); FIG = src/'figures'; FIG.mkdir(exist_ok=True)
rows = pd.read_csv(src/'rows.csv')
DS = {'gaussian_hard': 'Gaussian (hard)', 'mnist': 'MNIST', 'fashion_mnist': 'Fashion-MNIST', 'gisette': 'Gisette',
      '20ng_tfidf_nonzero': 'Text (k=5)', '20ng_tfidf_nonzero_k20': 'Text (k=20)'}
ORDER = [d for d in DS if d in set(rows.dataset)]
OURS = {'ours_quality': 'Ours – quality', 'ours_fast': 'Ours – fast', 'ours_spectrum': 'Ours – spectrum',
        'ours_w1': 'Ours – w=1 (original)'}
PCAS = {'pca_fast': 'PCA (randomized)', 'pca_exact': 'PCA (exact SVD)'}
COL = {'pca_fast': '#8c8c8c', 'pca_exact': '#4d4d4d', 'ours_quality': '#d62728', 'ours_fast': '#ff7f0e',
       'ours_spectrum': '#2ca02c', 'ours_w1': '#1f77b4'}
CELLS = [(d, m) for d in ORDER for m in sorted(rows[rows.dataset == d].m.unique())]
key = ['dataset', 'm', 'draw']; idx = rows.set_index(key)
rng = np.random.default_rng(0)
HIGHER = {'ari_full': True, 'ari_truth': True, 'nmi_full': True, 'knn_recall': True, 'sse_ratio': False, 'pair_q95': False}


def paired(method, base, metric, d, m):
    a = rows[(rows.method == method) & (rows.dataset == d) & (rows.m == m)].set_index('draw')[metric]
    b = rows[(rows.method == base) & (rows.dataset == d) & (rows.m == m)].set_index('draw')[metric].loc[a.index]
    diff = (a-b).to_numpy(); bs = [rng.choice(diff, len(diff)).mean() for _ in range(4000)]
    lo, hi = np.quantile(bs, [.025, .975]); good = HIGHER[metric]
    v = 'better' if (lo > 0 if good else hi < 0) else 'worse' if (hi < 0 if good else lo > 0) else 'same'
    return diff.mean(), lo, hi, v


def speedup(method, base, d, m):
    a = rows[(rows.method == method) & (rows.dataset == d) & (rows.m == m)].set_index('draw').total_seconds
    b = rows[(rows.method == base) & (rows.dataset == d) & (rows.m == m)].set_index('draw').total_seconds.loc[a.index]
    r = (b/a).to_numpy(); bs = [rng.choice(r, len(r)).mean() for _ in range(4000)]
    return r.mean(), *np.quantile(bs, [.025, .975])


def mean(method, metric, d, m):
    return rows[(rows.method == method) & (rows.dataset == d) & (rows.m == m)][metric].mean()


# ---------- scoreboard ----------
score = {}
for meth in OURS:
    for base in PCAS:
        for metric in ['ari_full', 'ari_truth', 'sse_ratio', 'pair_q95', 'knn_recall']:
            vs = [paired(meth, base, metric, d, m)[3] for d, m in CELLS]
            score[(meth, base, metric)] = (vs.count('better'), vs.count('same'), vs.count('worse'))
        sp = [speedup(meth, base, d, m) for d, m in CELLS]
        score[(meth, base, 'speed')] = (sum(s[1] > 1 for s in sp), sum(s[1] <= 1 <= s[2] for s in sp), sum(s[2] < 1 for s in sp))

# ---------- figures ----------
def bar_metric(metric, title, ylabel, fname, methods=('pca_fast', 'pca_exact', 'ours_quality', 'ours_fast', 'ours_spectrum', 'ours_w1')):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, m in zip(axs, (128, 256)):
        cells = [d for d, mm in CELLS if mm == m]; x = np.arange(len(cells)); w = .8/len(methods)
        for i, meth in enumerate(methods):
            ax.bar(x+(i-(len(methods)-1)/2)*w, [mean(meth, metric, d, m) for d in cells], w,
                   color=COL[meth], label={**PCAS, **OURS}[meth])
        ax.set_xticks(x, [DS[d].replace(' (', '\n(') for d in cells], fontsize=7.5); ax.set_title(f'{title}, m={m}', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
    axs[0].set_ylabel(ylabel); axs[1].legend(fontsize=7, frameon=False, loc='lower right')
    fig.tight_layout(); fig.savefig(FIG/fname, dpi=200); plt.close(fig)

bar_metric('ari_full', 'Agreement with Full k-means', 'ARI (higher = better)', 'ari.png')
bar_metric('pair_q95', 'Distance distortion', '95th pct relative error (lower = better)', 'dist.png')

fig, ax = plt.subplots(figsize=(11, 3.8)); x = np.arange(len(CELLS)); meths = ['ours_quality', 'ours_fast', 'ours_spectrum', 'ours_w1']; w = .2
for i, meth in enumerate(meths):
    v = [speedup(meth, 'pca_fast', d, m)[0] for d, m in CELLS]
    ax.bar(x+(i-1.5)*w, v, w, color=COL[meth], label=OURS[meth])
ax.axhline(1, ls='--', c='k', lw=1); ax.set_yscale('log')
ax.set_xticks(x, [f'{DS[d]}\nm={m}' for d, m in CELLS], fontsize=6.5); ax.set_ylabel('PCA time / our time  (>1 = faster)')
ax.set_title('End-to-end speed vs randomized PCA (learn + compress + cluster + any weight selection)', fontsize=10)
ax.legend(fontsize=7.5, frameon=False, ncol=4); ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig(FIG/'speed.png', dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 4.4))
for meth in list(PCAS)+meths:
    t = np.mean([mean(meth, 'total_seconds', d, m)/mean('pca_fast', 'total_seconds', d, m) for d, m in CELLS])
    a = np.mean([mean(meth, 'ari_full', d, m) for d, m in CELLS])
    ax.scatter(t, a, s=90, color=COL[meth], zorder=3); ax.annotate({**PCAS, **OURS}[meth], (t, a), xytext=(6, 4), textcoords='offset points', fontsize=8)
ax.set_xlabel('Relative time (randomized PCA = 1, lower = faster)'); ax.set_ylabel('Mean ARI vs Full (higher = better)')
ax.set_title('Quality / speed trade-off, averaged over all settings', fontsize=10); ax.grid(alpha=.3)
ax.spines[['top', 'right']].set_visible(False); fig.tight_layout(); fig.savefig(FIG/'tradeoff.png', dpi=200); plt.close(fig)

# ---------- PDF ----------
ss = getSampleStyleSheet()
body = ParagraphStyle('b', parent=ss['BodyText'], fontSize=10, leading=14, spaceAfter=5)
small = ParagraphStyle('s', parent=body, fontSize=8, leading=10, textColor=colors.HexColor('#555555'))
cellst = ParagraphStyle('c', parent=body, fontSize=7.8, leading=9.5, spaceAfter=0)
boxst = ParagraphStyle('x', parent=body, backColor=colors.HexColor('#f3f6fb'), borderPadding=7,
                       borderColor=colors.HexColor('#9db3d6'), borderWidth=.6, spaceBefore=6, spaceAfter=10)
h1, h2 = ss['Heading1'], ss['Heading2']; P = lambda t, s=body: Paragraph(t, s)
GOOD, BAD = colors.HexColor('#d6ecd2'), colors.HexColor('#f6d2d2')


def table(data, widths, shade=True):
    data = [[Paragraph(str(v), cellst) for v in r] for r in data]
    st = [('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dfe7f3')), ('GRID', (0, 0), (-1, -1), .3, colors.grey),
          ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('TOPPADDING', (0, 0), (-1, -1), 2), ('BOTTOMPADDING', (0, 0), (-1, -1), 2)]
    if shade:
        for i, r in enumerate(data):
            for j, c in enumerate(r):
                t = c.getPlainText()
                if 'better' in t or t.startswith('▲'): st.append(('BACKGROUND', (j, i), (j, i), GOOD))
                if 'worse' in t or t.startswith('▼'): st.append(('BACKGROUND', (j, i), (j, i), BAD))
    t = Table(data, colWidths=widths, repeatRows=1); t.setStyle(TableStyle(st)); return t


def cell_table(metric, fmt='{:.3f}', methods=('ours_quality', 'ours_fast', 'ours_spectrum', 'ours_w1')):
    head = ['Dataset', 'm', 'PCA rand.', 'PCA exact'] + [OURS[x].replace('Ours – ', 'Ours\n') for x in methods]
    data = [head]
    for d, m in CELLS:
        r = [DS[d], m, fmt.format(mean('pca_fast', metric, d, m)), fmt.format(mean('pca_exact', metric, d, m))]
        for meth in methods:
            v = paired(meth, 'pca_fast', metric, d, m)[3]
            r.append(('▲ ' if v == 'better' else '▼ ' if v == 'worse' else '') + fmt.format(mean(meth, metric, d, m)))
        data.append(r)
    return table(data, [2.6*cm, 0.9*cm, 1.7*cm, 1.7*cm]+[2.15*cm]*len(methods))


n_draws = rows.draw.nunique(); n_cells = len(CELLS)
s = [P('Adaptive Protected JL vs PCA for Verifiable Clustering', ss['Title']),
     P(f'<i>Confirmatory comparison on {n_draws} fresh random draws, {n_cells} dataset/width settings. '
       'Generated automatically from the result files; every number below is computed, not hand-written.</i>', small),
     Spacer(1, 6)]

# headline
q = score[('ours_quality', 'pca_fast', 'ari_full')]; f = score[('ours_fast', 'pca_fast', 'ari_full')]
sp = score[('ours_spectrum', 'pca_fast', 'ari_full')]
qx = score[('ours_quality', 'pca_exact', 'ari_full')]
avg = {mm: np.mean([mean(mm, 'ari_full', d, m) for d, m in CELLS]) for mm in list(PCAS)+list(OURS)}
spd = {mm: np.mean([speedup(mm, 'pca_fast', d, m)[0] for d, m in CELLS]) for mm in OURS}
s += [P('Summary', h1),
      P(f'<b>Clustering agreement (ARI vs Full), against randomized PCA, over {n_cells} settings:</b> '
        f'quality mode {q[0]} better / {q[1]} same / {q[2]} worse; fast mode {f[0]} / {f[1]} / {f[2]}; '
        f'spectrum mode {sp[0]} / {sp[1]} / {sp[2]}. Against exact PCA, quality mode is {qx[0]} / {qx[1]} / {qx[2]}.', boxst),
      P(f'<b>Average ARI:</b> randomized PCA {avg["pca_fast"]:.3f}, exact PCA {avg["pca_exact"]:.3f}, '
        f'ours quality {avg["ours_quality"]:.3f}, fast {avg["ours_fast"]:.3f}, spectrum {avg["ours_spectrum"]:.3f}, '
        f'original w=1 {avg["ours_w1"]:.3f}.'),
      P(f'<b>Average speed relative to randomized PCA</b> (>1 = faster): quality {spd["ours_quality"]:.2f}×, '
        f'fast {spd["ours_fast"]:.2f}×, spectrum {spd["ours_spectrum"]:.2f}×, original w=1 {spd["ours_w1"]:.2f}×.'),
      P('<b>Unique property:</b> every variant of our map is a linear map generated from a public challenge seed plus a '
        'committed head, so it can be authenticated in a zero-knowledge proof. PCA has no such construction.')]

s += [P('Scoreboard against randomized PCA', h2)]
data = [['Metric', *[OURS[x] for x in OURS]]]
names = {'ari_full': 'ARI vs Full', 'ari_truth': 'ARI vs true labels', 'sse_ratio': 'k-means cost ratio',
         'pair_q95': 'Distance distortion', 'knn_recall': '10-NN recall', 'speed': 'End-to-end speed'}
for metric, nm in names.items():
    data.append([nm]+[f'{score[(x, "pca_fast", metric)][0]} better / {score[(x, "pca_fast", metric)][1]} same / '
                      f'{score[(x, "pca_fast", metric)][2]} worse' for x in OURS])
s += [table(data, [3.3*cm]+[3.45*cm]*4, shade=False),
      P(f'Counts are over {n_cells} settings. "Better/worse": paired 95% bootstrap interval over {n_draws} draws '
        'excludes zero difference (for speed: excludes ratio 1). Otherwise "same".', small), PageBreak()]

s += [P('1. Method in brief', h1),
      P('Each point is split into a <b>protected head</b> (directions between cluster centers + 12 residual PCA directions, '
        'copied exactly) and a <b>residual tail</b>, compressed with a sparse random Johnson–Lindenstrauss sketch derived '
        'from a public seed. The tail is multiplied by a weight <b>w</b>. A small w suppresses residual noise (helps sparse, '
        'flat-spectrum data such as text); w = 1 keeps the JL distance guarantee (helps dense data such as MNIST).'),
      table([['Mode', 'How w is chosen', 'Cost'],
             ['Quality', 'Cluster with w ∈ {0.25, 0.35, 0.5, 0.6, 0.75, 1}; keep the partition with the lowest k-means cost in the original space (label-free).', '6 clusterings'],
             ['Fast', 'Same rule over w ∈ {0.25, 1}.', '2 clusterings'],
             ['Spectrum', 'w = 0.25 if the head captures < 27% of training variance, else w = 1.', '1 clustering'],
             ['Original', 'w = 1 always (distance-guarantee mode).', '1 clustering']], [2*cm, 11.5*cm, 3*cm], shade=False),
      P('2. Protocol', h1),
      P(f'Datasets: Gaussian mixture (n=2000, d=512, k=10, hard), MNIST and Fashion-MNIST (5000×784, k=10), Gisette '
        f'(3000×5000, k=5), 20 Newsgroups TF-IDF (2000 features, zero rows removed, k=5 and k=20). Output width m = 128 and '
        f'256. {n_draws} draws per setting (ids {rows.draw.min()}–{rows.draw.max()}). All representation learning uses '
        'a 60% training split only. "Full" = best of 10 k-means initializations on the uncompressed data; every compressed '
        'method then runs k-means with 5 initializations. Times include learning, compression, k-means and any weight '
        'selection; data loading is excluded. Single machine, float64.'),
      P('<b>Pre-registration:</b> the weight grid, the two cost rules and the spectrum threshold (0.27) were fixed on '
        'earlier draws 0–4 (results/tail_weight_sweep_20261016) and frozen in protocol.json before this run. '
        'None of the draws reported here were used to design the method.', boxst), PageBreak()]

s += [P('3. Clustering quality', h1), P('3.1 Agreement with Full k-means (ARI)', h2),
      Image(str(FIG/'ari.png'), width=17.5*cm, height=6*cm), cell_table('ari_full'),
      P('▲ / green = significantly better than randomized PCA, ▼ / red = significantly worse.', small)]

fsa = rows.groupby(['dataset', 'm']).full_self_ari.mean()
s += [P('3.2 Noise floor: how much does Full agree with itself?', h2),
      P('Two Full k-means runs (each best of 10 starts, different seeds) on the same uncompressed data. A compressed '
        'method cannot be expected to agree with Full more reliably than Full agrees with itself.'),
      table([['Setting', 'Full vs Full ARI']]+[[f'{DS[d]}, m={m}', f'{fsa.loc[(d, m)]:.3f}'] for d, m in CELLS],
            [6*cm, 3*cm], shade=False), PageBreak()]

s += [P('3.3 Agreement with true class labels (ARI)', h2), cell_table('ari_truth'),
      P('Ground-truth labels are not what k-means optimizes; values are low for text by nature.', small),
      P('3.4 k-means cost of the found partition, measured on the original data (1.000 = as good as Full)', h2),
      cell_table('sse_ratio', '{:.4f}'), PageBreak()]

s += [P('4. Speed', h1), Image(str(FIG/'speed.png'), width=17.5*cm, height=6*cm)]
data = [['Dataset', 'm', 'PCA rand. s', 'PCA exact s', 'Quality s', 'Fast s', 'Spectrum s', 'Spectrum speed-up [95% CI]']]
for d, m in CELLS:
    r, lo, hi = speedup('ours_spectrum', 'pca_fast', d, m)
    data.append([DS[d], m] + [f'{mean(x, "total_seconds", d, m):.3f}' for x in ['pca_fast', 'pca_exact', 'ours_quality', 'ours_fast', 'ours_spectrum']]
                + [('▲ ' if lo > 1 else '▼ ' if hi < 1 else '') + f'{r:.2f}× [{lo:.2f}, {hi:.2f}]'])
s += [table(data, [2.6*cm, .9*cm, 1.8*cm, 1.8*cm, 1.7*cm, 1.5*cm, 1.8*cm, 3.6*cm])]
br = rows.groupby('method')[['fit_seconds', 'kmeans_seconds', 'select_seconds', 'total_seconds']].mean()
s += [P('Where the time goes (mean over all settings, seconds)', h2),
      table([['Method', 'Learn + compress', 'k-means', 'Weight selection', 'Total']] +
            [[{**PCAS, **OURS}[x]] + [f'{br.loc[x, c]:.3f}' for c in br.columns] for x in list(PCAS)+list(OURS)],
            [4.2*cm, 3*cm, 3*cm, 3*cm, 3*cm], shade=False),
      Image(str(FIG/'tradeoff.png'), width=12*cm, height=8.1*cm), PageBreak()]

s += [P('5. Geometry preservation', h1), Image(str(FIG/'dist.png'), width=17.5*cm, height=6*cm),
      P('5.1 Pairwise distance distortion (95th percentile of |compressed/original − 1|, lower = better)', h2),
      cell_table('pair_q95'),
      P('5.2 Nearest-neighbour recall (fraction of true 10 nearest neighbours kept, higher = better)', h2),
      cell_table('knn_recall'), PageBreak()]

# conclusions generated from scoreboard
best = max(OURS, key=lambda x: (score[(x, 'pca_fast', 'ari_full')][0]-score[(x, 'pca_fast', 'ari_full')][2], avg[x]))
bw = score[(best, 'pca_fast', 'ari_full')]
worse_cells = [f'{DS[d]} m={m}' for d, m in CELLS if paired(best, 'pca_fast', 'ari_full', d, m)[3] == 'worse']
s += [P('6. Conclusions', h1),
      P(f'<b>Best mode on clustering agreement: {OURS[best]}</b> — {bw[0]} better, {bw[1]} same, {bw[2]} worse than '
        f'randomized PCA over {n_cells} settings (average ARI {avg[best]:.3f} vs {avg["pca_fast"]:.3f}).'
        + (f' Remaining losses: {", ".join(worse_cells)}.' if worse_cells else ' No setting is significantly worse.')),
      P(f'<b>Speed:</b> on average the spectrum mode runs at {spd["ours_spectrum"]:.2f}× and the fast mode at '
        f'{spd["ours_fast"]:.2f}× the speed of randomized PCA; the quality mode trades speed for accuracy '
        f'({spd["ours_quality"]:.2f}×).'),
      P('<b>Geometry:</b> the original w = 1 map preserves pairwise distances and neighbours far better than PCA; '
        'adaptive modes give up part of that advantage on noisy data in exchange for better cluster agreement.'),
      P('<b>Verifiability:</b> all modes keep a linear, seed-derived map compatible with the zero-knowledge proof. '
        'The cost-based modes additionally need the proof to cover the minimum-cost selection among a fixed, small '
        'set of weights; this is not yet implemented in the circuit.'),
      P('6.1 Limitations', h2),
      P('• Native float64 measurements on fixed public datasets and subsets; not whole-corpus ZK proofs.<br/>'
        '• Bootstrap intervals are per setting, without multiplicity correction.<br/>'
        '• Timings are single-machine wall-clock and exclude data loading.<br/>'
        '• Agreement with Full is bounded by Full\'s own instability (Section 3.2).<br/>'
        '• The Gaussian (hard) dataset uses study-specific generator settings.'),
      Spacer(1, 8),
      P(f'Source: {src.as_posix()}/rows.csv and protocol.json; code: experiments/final_comparison.py, '
        'analysis/final_comparison_report.py, commitjl/residual_maps.py.', small)]

OUT.parent.mkdir(parents=True, exist_ok=True)
SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.8*cm, rightMargin=1.8*cm, topMargin=1.6*cm, bottomMargin=1.6*cm,
                  title='Adaptive Protected JL vs PCA', author='CommitJL project').build(s)
print('wrote', OUT)
