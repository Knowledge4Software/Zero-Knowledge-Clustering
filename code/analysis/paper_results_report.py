"""Final comparison report (PDF): Protected CommitJL vs Full, PCA and Sparse JL.

Inputs (all produced on the research server):
  results/protected_fixed_center_20261017/all_rows.csv   accuracy at scale (float + integer)
  results/rq2_query_cost_v2_20261017/rows.csv            per-query proof cost + cost-claim accuracy
  results/adversarial_attack_20261017/rows.csv           adaptive-prover attacks

python analysis/paper_results_report.py <out.pdf>
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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

R = Path('results'); OUT = Path(sys.argv[1]); FIG = R/'paper_results_figures'; FIG.mkdir(exist_ok=True)
DS = {'gaussian_easy': 'Gaussian easy', 'gaussian_hard': 'Gaussian hard', 'dbscan_margin': 'DBSCAN margin',
      'mnist': 'MNIST', 'fashion_mnist': 'Fashion-MNIST', 'gisette': 'Gisette', 'arcene': 'Arcene',
      '20ng_tfidf_nonzero': '20 Newsgroups'}
NAME = {'full': 'Full dimension', 'pca': 'PCA', 'sparse': 'Sparse JL', 'protected': 'Protected (JL tail)',
        'protected_int': 'Protected, integer', 'protected_energy': 'Protected + energy (ours)', 'head_only': 'Head only'}
COL = {'full': '#4d4d4d', 'pca': '#8c8c8c', 'sparse': '#1f77b4', 'protected': '#ff7f0e', 'protected_energy': '#d62728'}

acc = pd.read_csv(R/'protected_fixed_center_20261017/all_rows.csv')
cost = pd.read_csv(R/'rq2_query_cost_v2_20261017/rows.csv')
cost_ok = cost[cost.error.fillna('') == '']  # skipped/OOM rows carry no answer
att_path = R/'adversarial_attack_20261017/rows.csv'
att = pd.read_csv(att_path) if att_path.exists() else None

# ---------------- per-point proof cost (exact slope between the two largest compiled sizes) -------------
ok = cost[cost.constraints.notna() & (cost.error.fillna('') == '')]
slopes = []
for key, g in ok.groupby(['dataset', 'k', 'method', 'query', 'centers']):
    g = g.sort_values('n')
    if len(g) >= 2:
        a, b = g.iloc[-2], g.iloc[-1]
        slopes.append(dict(zip(['dataset', 'k', 'method', 'query', 'centers'], key),
                           per_point=(b.constraints-a.constraints)/(b.n-a.n), opened=b.opened_columns, estimated=False))
slopes = pd.DataFrame(slopes)
# Full dimension where even n=1-2 exceeds host RAM: per-opened-coordinate cost measured on other datasets.
full = slopes[slopes.method == 'full']
per_coord = (full.per_point/full.opened).groupby([full['query'], full['centers']]).median()
extra = []
for (d_, k_), g in cost.groupby(['dataset', 'k']):
    for q in ('assign', 'cost'):
        for cmode in ('public', 'const'):
            if not ((slopes.dataset == d_) & (slopes.k == k_) & (slopes.method == 'full') & (slopes['query'] == q) & (slopes.centers == cmode)).any() and (q, cmode) in per_coord:
                dim = int(g.d.iloc[0])
                extra.append(dict(dataset=d_, k=k_, method='full', query=q, centers=cmode,
                                  per_point=per_coord[(q, cmode)]*dim, opened=dim, estimated=True))
slopes = pd.concat([slopes, pd.DataFrame(extra)], ignore_index=True)
proved = cost[cost.prove_seconds.notna()]
sec_per_constraint = float(np.median(proved.prove_seconds/proved.constraints)) if len(proved) else np.nan

def per_point(d_, k_, m_, q, c='const'):
    s = slopes[(slopes.dataset == d_) & (slopes.k == k_) & (slopes.method == m_) & (slopes['query'] == q) & (slopes.centers == c)]
    return (float(s.per_point.iloc[0]), bool(s.estimated.iloc[0])) if len(s) else (np.nan, False)

# ---------------- figures ----------------
cells = sorted(cost.groupby(['dataset', 'k']).groups, key=lambda t: (list(DS).index(t[0]), t[1]))
methods = ['full', 'pca', 'sparse', 'protected', 'protected_energy']
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for ax, q in zip(axs, ('assign', 'cost')):
    x = np.arange(len(cells)); w = .16
    for i, m_ in enumerate(methods):
        v = [per_point(d_, k_, m_, q)[0] for d_, k_ in cells]
        ax.bar(x+(i-2)*w, v, w, color=COL[m_], label=NAME[m_])
    ax.set_yscale('log'); ax.set_xticks(x, [f'{DS[d_]}\nk={k_}' for d_, k_ in cells], fontsize=6.5)
    ax.set_title(f'{"Assignment" if q == "assign" else "Cost"} query: constraints per audited point', fontsize=9.5)
    ax.spines[['top', 'right']].set_visible(False)
axs[0].set_ylabel('constraints per point (log)'); axs[1].legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(FIG/'per_point.png', dpi=200); plt.close(fig)

agree = acc[acc.method.isin(['pca', 'sparse', 'protected', 'protected_int'])].groupby(['method', 'm']).agreement.mean().unstack(0)
fig, ax = plt.subplots(figsize=(6, 3.6))
for m_ in agree.columns:
    ax.plot(agree.index, agree[m_], marker='o', label=NAME[m_], color=COL.get(m_, '#2ca02c'))
ax.set_xscale('log', base=2); ax.set_xlabel('output width m'); ax.set_ylabel('assignment agreement with exact')
ax.set_title('Fixed-centre assignment accuracy (mean over 8 datasets, k=5,10)', fontsize=9); ax.legend(fontsize=7, frameon=False)
ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(FIG/'agreement.png', dpi=200); plt.close(fig)

# ---------------- PDF ----------------
ss = getSampleStyleSheet()
body = ParagraphStyle('b', parent=ss['BodyText'], fontSize=9.5, leading=13, spaceAfter=5)
small = ParagraphStyle('s', parent=body, fontSize=7.5, leading=9.5, textColor=colors.HexColor('#555555'))
cell = ParagraphStyle('c', parent=body, fontSize=7.5, leading=9, spaceAfter=0)
box = ParagraphStyle('x', parent=body, backColor=colors.HexColor('#f3f6fb'), borderPadding=6,
                     borderColor=colors.HexColor('#9db3d6'), borderWidth=.6, spaceBefore=4, spaceAfter=8)
h1, h2 = ss['Heading1'], ss['Heading2']; P = lambda t, s=body: Paragraph(t, s)
GOOD, BAD = colors.HexColor('#d6ecd2'), colors.HexColor('#f6d2d2')


def table(rows, widths, good=(), bad=()):
    data = [[Paragraph(str(v), cell) for v in r] for r in rows]
    st = [('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dfe7f3')), ('GRID', (0, 0), (-1, -1), .3, colors.grey),
          ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('TOPPADDING', (0, 0), (-1, -1), 2), ('BOTTOMPADDING', (0, 0), (-1, -1), 2)]
    st += [('BACKGROUND', c, c, GOOD) for c in good] + [('BACKGROUND', c, c, BAD) for c in bad]
    t = Table(data, colWidths=widths, repeatRows=1); t.setStyle(TableStyle(st)); return t


s = [P('Protected CommitJL vs PCA and Sparse JL: Accuracy, Proof Cost and Security', ss['Title']),
     P('Fixed-centre auditing with zero-knowledge proofs. All measurements from the research server '
       '(16 cores, 8 GB RAM, Groth16 on BN254). Generated from the result files.', small)]

# Summary numbers
a16 = acc[acc.m == 16].groupby('method')[['agreement', 'r_phi']].mean()
psi = acc.assign(e=acc.psi_rel_error.abs()).groupby('method').e.agg(['min', 'max'])
cq = cost_ok[(cost_ok['query'] == 'cost') & cost_ok.cost_claim_rel_error.notna()].assign(e=lambda d: d.cost_claim_rel_error.abs()).groupby('method').e.agg(['median', 'max'])
ratio_full, ratio_pca = [], []
for d_, k_ in cells:
    ours, _ = per_point(d_, k_, 'protected_energy', 'cost'); f, _ = per_point(d_, k_, 'full', 'cost'); p_, _ = per_point(d_, k_, 'pca', 'cost')
    if ours > 0:
        ratio_full.append(f/ours); ratio_pca.append(p_/ours)
s += [P('Summary', h1), P(
    f"<b>Accuracy.</b> At m=16, assignments to the public centres agree with exact assignment "
    f"{a16.loc['protected','agreement']:.4f} (float) and {a16.loc['protected_int','agreement']:.4f} (circuit integer arithmetic) for "
    f"Protected CommitJL, versus {a16.loc['pca','agreement']:.3f} for PCA and {a16.loc['sparse','agreement']:.3f} for Sparse JL. "
    f"The proved cost is exact with the energy column (median error {cq.loc['protected_energy','median']:.1e}), "
    f"while PCA's claimed cost is off by {cq.loc['pca','median']:.0%} (median).", box),
    P(f"<b>Proof cost per audited point (cost query, constant centres).</b> Protected + energy needs "
      f"{np.nanmedian(ratio_full):.0f}x fewer constraints than full dimension (range {np.nanmin(ratio_full):.0f}-{np.nanmax(ratio_full):.0f}x) "
      f"and {np.nanmedian(ratio_pca):.1f}x fewer than PCA at the same total width.", box)]
if att is not None:
    known = att[att.known_before_commit]; fresh = att[~att.known_before_commit]
    s += [P(f"<b>Security.</b> With a projection known before commitment (PCA, Pisot, pre-published JL) a cheating prover "
            f"hides a 11x cost increase completely (detected inflation {known['detected_inflation_a10'].median():.2f}x). "
            f"Under commit-then-sample it is detected ({fresh['detected_inflation_a10'].median():.1f}x). Assignment-flip attacks "
            f"succeed on {known[known.method!='protected_prepublished'].flip_success.mean():.0%} of points for known PCA/Pisot/JL and are "
            f"impossible for Protected CommitJL (centre directions are kept exactly).", box)]
s += [PageBreak()]

# Method
s += [P('1. Method', h1), P(
    'The prover commits the private data X; the k centres C are public. A head basis U (k-1 directions) is '
    'computed from the centre differences only, before any challenge. A fresh challenge then selects a sparse '
    '+-1 JL matrix S. Each point is stored once (Stage 1) as z = [ U<sup>T</sup>(x-c<sub>1</sub>) | ||r||<sup>2</sup> | S r ], with r the '
    'part of x-c<sub>1</sub> outside the head. Per query (Stage 2) the proof opens only what it needs: the head '
    'for assignments, head + energy for the exact cost, the JL tail for geometric queries (distances, DBSCAN).'),
      table([['Component', 'Size', 'Used for', 'Guarantee'],
             ['Head U<sup>T</sup>(x-c<sub>1</sub>)', 'k-1', 'assignment to the centres', 'exact (tail adds the same amount to every centre)'],
             ['Residual energy ||r||<sup>2</sup>', '1', 'exact cost', 'exact'],
             ['Sparse JL tail S r', 'm-k', 'distances, DBSCAN, new centres', 'JL, sampled after commitment']],
            [4.2*cm, 1.4*cm, 4.4*cm, 7.3*cm]), PageBreak()]

# Accuracy
s += [P('2. Accuracy (fixed public centres, held-out audit points)', h1),
      P('8 datasets, k = 5 and 10, 3 centre sets x 10 maps, m = 16-128. Centres come from a 60% training split; '
        'all metrics are on the remaining 40%.', small), Image(str(FIG/'agreement.png'), width=12*cm, height=7.2*cm)]
rows = [['Method', 'Agreement m=16', 'r_phi m=16', 'Claimed-cost error (range)']]
for m_ in ['pca', 'sparse', 'protected', 'protected_int']:
    rows.append([NAME[m_], f"{a16.loc[m_,'agreement']:.4f}", f"{a16.loc[m_,'r_phi']:.4f}",
                 f"{psi.loc[m_,'min']:.1%} - {psi.loc[m_,'max']:.1%}"])
rows.append([NAME['protected_energy'], '= protected head', '= protected head', 'exact (see Section 3)'])
s += [table(rows, [5*cm, 3.3*cm, 3*cm, 5.5*cm], good=[(1, 3), (1, 4), (1, 5)], bad=[(3, 1), (3, 2)])]
rule = acc[acc.method.isin(['pca', 'sparse', 'protected', 'protected_int'])].groupby(['dataset', 'k', 'method', 'm'])[['ari', 'r_phi']].mean().reset_index()
okm = rule[(rule.ari >= .8) & (rule.r_phi <= 1.1)].groupby(['dataset', 'k', 'method']).m.min().unstack()
counts = {m_: int(okm[m_].notna().sum()) for m_ in okm.columns}
s += [P(f"<b>Manuscript RQ1 rule</b> (smallest m with ARI >= 0.8 and r_phi <= 1.1), out of {len(okm)} dataset/k cells: "
        + ', '.join(f"{NAME[m_]} {counts[m_]}" for m_ in ['protected', 'protected_int', 'pca', 'sparse']) + '. Protected meets it at m = 16 everywhere.'),
      PageBreak()]

# Cost
s += [P('3. Proof cost per query (Stage 2, Groth16)', h1),
      P('Constraints per audited point are the exact slope between the two largest compiled sizes. '
        '"est." marks full-dimension Gisette, which does not compile in 8 GB even for one point; its value is '
        'the measured per-coordinate cost times d.', small), Image(str(FIG/'per_point.png'), width=17.5*cm, height=6.4*cm)]
rows = [['Dataset', 'k', 'Full', 'PCA', 'Sparse JL', 'Protected (tail)', 'Protected + energy', 'vs Full', 'vs PCA']]
for d_, k_ in cells:
    vals = {m_: per_point(d_, k_, m_, 'cost') for m_ in methods}
    o = vals['protected_energy'][0]
    rows.append([DS[d_], k_] + [f"{v:,.0f}{' est.' if e else ''}" for v, e in (vals[m_] for m_ in methods)]
                + [f"{vals['full'][0]/o:.0f}x", f"{vals['pca'][0]/o:.1f}x"])
s += [P('Cost query, constant centres: constraints per point', h2), table(rows, [2.6*cm, .7*cm, 2*cm, 1.6*cm, 1.6*cm, 2*cm, 2.1*cm, 1.5*cm, 1.4*cm])]
rows = [['Dataset', 'k', 'Full', 'PCA', 'Sparse JL', 'Protected (head only)']]
for d_, k_ in cells:
    rows.append([DS[d_], k_] + [f"{per_point(d_, k_, m_, 'assign')[0]:,.0f}{' est.' if per_point(d_, k_, m_, 'assign')[1] else ''}"
                                for m_ in ['full', 'pca', 'sparse', 'protected']])
s += [P('Assignment query, constant centres: constraints per point', h2), table(rows, [3*cm, .8*cm, 3*cm, 2.5*cm, 2.5*cm, 3.5*cm])]
pr = proved[(proved.n == proved.n.min())].groupby(['method', 'query'])[['prove_seconds', 'verify_seconds', 'proof_bytes', 'setup_seconds']].median()
rows = [['Method', 'Query', 'Prove s', 'Verify ms', 'Proof bytes', 'Setup s']]
for (m_, q), r in pr.iterrows():
    rows.append([NAME.get(m_, m_), q, f'{r.prove_seconds:.2f}', f'{1e3*r.verify_seconds:.1f}', f'{r.proof_bytes:.0f}', f'{r.setup_seconds:.1f}'])
s += [P(f'Measured Groth16 proofs at the smallest size (median over datasets), prove time ~ {1e6*sec_per_constraint:.2f} s per million constraints', h2),
      table(rows, [4.5*cm, 2*cm, 2*cm, 2*cm, 2.5*cm, 2*cm]),
      P('Stage 1 (proving once that z = f(x) for every committed point) is a one-time cost of about (k-1) d multiplications per '
        'point plus the data commitment; it is not included above and is amortised over all later queries.', small), PageBreak()]

# Cost accuracy from proofs
rows = [['Method', 'Median |cost error|', 'Max |cost error|']]
for m_ in methods:
    if m_ in cq.index:
        rows.append([NAME[m_], f"{cq.loc[m_,'median']:.2e}", f"{cq.loc[m_,'max']:.2e}"])
s += [P('4. Accuracy of the proved cost (from the proofs, integer arithmetic)', h1), table(rows, [6*cm, 4*cm, 4*cm])]

if att is not None:
    s += [P('5. Adaptive-prover attacks', h1),
          P('Cost hiding: the prover adds noise in the expected projection kernel so the true cost grows ~11x. '
            'Assignment flip: points move along the kernel component of (runner-up - assigned centre) just past the '
            'boundary. For commit-then-sample the attacker uses the kernel of an independent guess.', small)]
    g = att.groupby('method')[['known_before_commit', 'true_inflation_a10', 'detected_inflation_a10', 'flip_possible', 'flip_success', 'honest_assignment_error']].mean()
    rows = [['Method', 'Known before commit', 'True cost x', 'Proved cost x', 'Flip possible', 'Flip success', 'Honest error']]
    order = ['pca_public', 'pisot', 'sparse_prepublished', 'sparse_commit', 'protected_prepublished', 'protected_commit']
    for m_ in order:
        if m_ in g.index:
            r = g.loc[m_]
            rows.append([m_.replace('_', ' '), 'yes' if r.known_before_commit else 'no', f'{r.true_inflation_a10:.1f}',
                         f'{r.detected_inflation_a10:.2f}', f'{r.flip_possible:.0%}', f'{r.flip_success:.0%}', f'{r.honest_assignment_error:.1%}'])
    s += [table(rows, [3.8*cm, 2.3*cm, 2*cm, 2.2*cm, 2.1*cm, 2*cm, 2*cm]),
          P('Sparse JL commit-then-sample flip "success" equals its ordinary approximation error at the boundary, '
            'not an exploited kernel.', small)]

s += [P('6. Limitations', h1), P(
    '- Native and integer measurements on fixed public datasets; proofs on small n because of the 8 GB host; '
    'per-point costs are exact slopes, full-dimension Gisette is estimated.<br/>'
    '- Stage 1 is not yet benchmarked as a proof.<br/>'
    '- Exact assignment comes from protecting the public centre span (a design property, not an empirical surprise).<br/>'
    '- Attack study uses one natural oblivious strategy for commit-then-sample; it is evidence, not a proof of security.')]
OUT.parent.mkdir(parents=True, exist_ok=True)
SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.7*cm, rightMargin=1.7*cm, topMargin=1.5*cm, bottomMargin=1.5*cm,
                  title='Protected CommitJL results').build(s)
print('wrote', OUT)
