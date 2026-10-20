"""Export evidence for the reusable-statistics manuscript without replacing old runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
LABELS = {'learned_energy': 'Learned head + energy', 'pca_energy': 'PCA + energy',
          'span_energy': 'Span + energy', 'cached': 'Cached exact', 'affine': 'Exact affine',
          'full': 'Full (extrapolated)'}
ORDER = ['cached', 'affine', 'full', 'span_energy', 'pca_energy', 'learned_energy']


def table(caption, label, columns, rows, note, alignment):
    return '\n'.join([
        r'\begin{table*}[tp]', r'\caption{' + caption + r'}\label{' + label + '}',
        r'\centering\small', r'\begin{adjustbox}{max width=\textwidth}',
        r'\begin{tabular}{@{}' + alignment + '@{}}', r'\toprule',
        ' & '.join(columns) + r' \\', r'\midrule',
        *[' & '.join(map(str, row)) + r' \\' for row in rows], r'\bottomrule',
        r'\end{tabular}', r'\end{adjustbox}',
        r'\par\smallskip\begin{minipage}{0.98\textwidth}\footnotesize ' + note + r'\end{minipage}',
        r'\end{table*}', ''])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--snapshot', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    dest = a.out
    dest.mkdir(parents=True, exist_ok=False)
    inputs = set()

    def read(path):
        inputs.add(path)
        return pd.read_csv(path)

    checks = json.loads((a.snapshot/'checks.json').read_text())
    assert all(x['matches'] for x in checks['source_checks'])
    costs = read(a.snapshot/'cumulative_costs.csv')
    quality = read(a.snapshot/'proof_quality.csv')
    runs = ['reuse_mnist_n64', 'reuse_cifar_n64', 'reuse_mnist_n256']
    costs = costs[costs.run.isin(runs)]
    quality = quality[quality.run.isin(runs)]
    assert len(costs) == 180 and len(quality) == 15
    costs.to_csv(dest/'cumulative_costs.csv', index=False)
    quality.to_csv(dest/'proof_quality.csv', index=False)
    rows, claims = [], {}
    for run in runs:
        co = costs[costs.run == run]
        assert any(x['run'] == run and x['complete'] for x in checks['workloads'])
        corpus = {'reuse_mnist_n64': 'MNIST ($N=64$)',
                  'reuse_cifar_n64': 'CIFAR/ResNet-50 ($N=64$)',
                  'reuse_mnist_n256': 'MNIST ($N=256$)'}[run]
        for method in ORDER:
            g = co[co.method == method].set_index('T')
            rows.append([corpus, LABELS[method], *[f'{g.loc[t, "total_seconds"]:,.1f}' for t in [1, 2, 10]]])
        pivot = co.pivot(index='T', columns='method', values='total_seconds')
        crossover = pivot.index[pivot.learned_energy < pivot.cached].min()
        assert crossover == 2
        claims[run] = dict(crossover_vs_cached=int(crossover),
                          cached_over_learned_T10=float(pivot.loc[10, 'cached']/pivot.loc[10, 'learned_energy']))
    (dest/'cumulative.tex').write_text(table(
        'Cumulative cost across successive center sets, query batch $b=64$, $k=5$ (seconds).', 'tab:reuse-cost',
        ['Catalog', 'Method', '$L=1$', '$L=2$', '$L=10$'], rows,
        'Totals add measured native preparation, compilation, setup, witness generation and median proving times. '
        'All compressed/cached/affine catalog proofs were generated; Full catalog proving is extrapolated from one measured batch per set. '
        'The deployed set is first. Verification, fitting, warmups, I/O and public new-center preparation are excluded.', 'llrrr'), encoding='utf-8')

    rows = []
    for run in runs:
        corpus = {'reuse_mnist_n64': 'MNIST ($N=64$)',
                  'reuse_cifar_n64': 'CIFAR/ResNet-50 ($N=64$)',
                  'reuse_mnist_n256': 'MNIST ($N=256$)'}[run]
        for method in ['cached', 'affine', 'span_energy', 'pca_energy', 'learned_energy']:
            r = quality[(quality.run == run) & (quality.method == method)].iloc[0]
            rows.append([corpus, LABELS[method], f'{100*r.agreement:.2f}',
                         f'{100*r.min_agreement:.2f}', f'{100*r.mean_abs_cost_error:.3f}'])
    (dest/'quality.tex').write_text(table(
        'Answer fidelity on the nine new center sets in each proof workload.', 'tab:reuse-quality',
        ['Catalog', 'Method', r'Mean agree. (\%)', r'Min. agree. (\%)', r'Mean $|e_\Psi|$ (\%)'], rows,
        'Agreement is against floating-point original-space Full on the same catalog rows. '
        'Means weight the nine new center sets equally; minimum is the worst set, not a confidence bound. '
        'Cached and affine are exact on quantized inputs, so their floating-reference cost errors can be nonzero.', 'llrrr'), encoding='utf-8')

    matched = read(REPO/'code/results/new_center_reuse_20261018/summary/matched_comparisons.csv')
    selected = matched[(matched.ours == 'ours_cross_q1') & (matched.other == 'pca_energy') & matched.budget.isin([16,64,256])]
    assert len(selected) == 3
    selected.to_csv(dest/'native_jl_matched.csv', index=False)
    rows = [[int(r.budget), int(r.cells), f'{100*r.agreement_ours:.2f}', f'{100*r.agreement_other:.2f}',
             f'{100*r.abs_psi_ours:.3f}', f'{100*r.abs_psi_other:.3f}'] for r in selected.itertuples()]
    (dest/'native_jl.tex').write_text(table(
        'Native new-center comparison on matched cells: head/energy/JL versus PCA/energy.', 'tab:reuse-jl',
        ['Budget', 'Cells', r'JL agree. (\%)', r'PCA/E agree. (\%)', r'JL $|e_\Psi|$ (\%)', r'PCA/E $|e_\Psi|$ (\%)'], rows,
        'The tail estimates the residual cross term with one signed bucket per column. '
        'Maps are averaged within a corpus/$k$/start/scenario cell before aggregation. '
        'Eligibility differs across budgets; within each row both methods use identical cells. Native float64 computation, not proofs.', 'rrrrrr'), encoding='utf-8')

    shift = read(a.snapshot/'shift_owner_paired.csv')
    selected = shift[(shift.budget == 16) & shift.method.isin(['learned_energy','ours_cross','jl'])]
    selected.to_csv(dest/'shift_owner.csv', index=False)
    rows = [[r.setting, {'learned_energy':'Learned head + energy', 'ours_cross':'Head/energy/JL ($q=4$)', 'jl':'Plain JL ($q=4$)'}[r.method],
             int(r.cells), f'{100*r.agreement:.2f}', f'{100*r.pca_agreement:.2f}'] for r in selected.itertuples()]
    (dest/'shift_owner.tex').write_text(table(
        'Native budget-16 agreement under non-IID owners and class distribution shift.', 'tab:reuse-shift',
        ['Setting', 'Method', 'Cells', r'Agree. (\%)', r'Matched PCA/E (\%)'], rows,
        'Five corpora, $k=5,10$, three starts. Random maps are averaged within each matched cell. '
        'Owners: residual/PCA fitting uses owner 0, while deployed public centers use pooled training rows. '
        'Shift: fit on old classes and audit new-only/mixed cases. These are native simulations.', 'llrrr'), encoding='utf-8')

    shape_rows = []
    for run in runs:
        frame = read(a.snapshot/'results/paper_runs_20261019'/run/'workload.csv')
        for method in ['learned_energy','pca_energy','span_energy','cached','affine']:
            g = frame[frame.method == method]
            st, qu = g[g.kind == 'stage1'].iloc[0], g[g.kind == 'query'].iloc[0]
            shape_rows.append(dict(run=run, method=method, stage1_constraints=int(st.constraints),
                                   query_constraints=int(qu.constraints),
                                   peak_rss_bytes=float(g.peak_rss.max())))
    pd.DataFrame(shape_rows).to_csv(dest/'proof_shapes.csv', index=False)

    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9, 'pdf.fonttype':42})
    fig, axes = plt.subplots(1,3,figsize=(7.2,3.4))
    style = {'cached':('#bd6328','-','s'), 'affine':('#777777','-','x'),
             'full':('#aaaaaa','--',''), 'span_energy':('#92a62a','-.',''),
             'pca_energy':('#7a55a2',':','o'), 'learned_energy':('#007c83','-','')}
    for ax,run,title in zip(axes,runs,['MNIST, $N=64$', 'CIFAR/ResNet-50, $N=64$', 'MNIST, $N=256$']):
        for method in ORDER:
            g = costs[(costs.run == run) & (costs.method == method)].sort_values('T')
            color,linestyle,marker = style[method]
            ax.plot(g['T'],g.total_seconds,color=color,ls=linestyle,marker=marker,ms=3,
                    lw=2 if method=='learned_energy' else 1.2,label=LABELS[method])
        ax.axvline(2,color='#bbbbbb',lw=.7,zorder=0)
        ax.set(title=title,xlabel='Center sets audited, $L$',xticks=[1,2,4,6,8,10],xlim=(.8,10.2),ylim=(0,None))
        ax.title.set_fontsize(9)
        ax.tick_params(labelsize=8)
        ax.grid(axis='y',alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Cumulative cost (s)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.01),fontsize=8)
    fig.tight_layout(rect=(0,.18,1,1))
    fig.savefig(dest/'cumulative_cost.pdf',bbox_inches='tight')
    fig.savefig(dest/'cumulative_cost.png',dpi=180,bbox_inches='tight')
    plt.close(fig)

    # Five-start fixed-center campaign. Random maps are averaged before cells;
    # the head-only ablation is repeated across widths by the legacy driver.
    # It is excluded here, rather than treated as independent repetitions.
    fidelity_paths = [a.snapshot/f'results/paper_runs_20261019/fidelity_5starts/worker{i}/rows.csv'
                      for i in range(3)]
    fidelity_frames = [read(p) for p in fidelity_paths]
    expected_datasets = [['gisette', 'arcene'], ['mnist', 'fashion_mnist'],
                         ['gaussian_easy', 'gaussian_hard', 'dbscan_margin', '20ng_tfidf_nonzero']]
    for frame, datasets in zip(fidelity_frames, expected_datasets):
        expected = {(ds,k,s) for ds in datasets for k in (5,10) for s in range(5)}
        assert set(frame[['dataset','k','start']].itertuples(index=False, name=None)) == expected
        non_ablation = frame[frame.method != 'head_only']
        assert not non_ablation.duplicated(['dataset','k','start','method','m','map']).any()
    fidelity = pd.concat(fidelity_frames, ignore_index=True)
    fidelity['abs_cost'] = fidelity.psi_rel_error.abs()
    f16 = fidelity[fidelity.m == 16]
    assert np.isfinite(f16[['agreement','ari','abs_cost']].to_numpy()).all()
    f_cells = f16.groupby(['dataset','k','start','method'], as_index=False)[['agreement','ari','abs_cost']].mean()
    assert len(f_cells) == 320
    assert (f_cells.groupby(['dataset','k','start']).size() == 4).all()
    for (_,_,_,method), group in f16.groupby(['dataset','k','start','method']):
        assert set(group['map']) == ({-1} if method == 'pca' else set(range(10)))
    f_cells.to_csv(dest/'fixed_five_start_cells.csv', index=False)
    f_summary = f_cells.groupby(['dataset','method'], as_index=False)[['agreement','ari','abs_cost']].mean()
    f_summary.to_csv(dest/'fixed_five_start_summary.csv', index=False)
    dataset_names = {'gaussian_easy':'Gaussian-easy', 'gaussian_hard':'Gaussian-hard',
                     'dbscan_margin':'Margin synthetic', 'mnist':'MNIST',
                     'fashion_mnist':'Fashion-MNIST', 'gisette':'Gisette',
                     'arcene':'Arcene', '20ng_tfidf_nonzero':'20 Newsgroups'}
    rows = []
    for ds in dataset_names:
        g = f_summary[f_summary.dataset == ds].set_index('method')
        rows.append([dataset_names[ds], *[f'{100*g.loc[m,"agreement"]:.3f}'
                      for m in ['protected','protected_int','pca','sparse']],
                     f'{100*g.loc["protected_int","abs_cost"]:.3f}'])
    all_methods = f_cells.groupby('method')[['agreement','ari','abs_cost']].mean()
    rows.append(['Mean', *[f'{100*all_methods.loc[m,"agreement"]:.3f}'
                          for m in ['protected','protected_int','pca','sparse']],
                 f'{100*all_methods.loc["protected_int","abs_cost"]:.3f}'])
    (dest/'fixed_five_starts.tex').write_text(table(
        'Five-start fixed-center fidelity at total width 16 on the capped eight-corpus campaign.',
        'tab:fixed-five-starts',
        ['Corpus', r'Head/JL float (\%)', r'Head/JL int. (\%)', r'PCA (\%)',
         r'Sparse JL (\%)', r'Int. $|e_\Psi|$ (\%)'], rows,
        'Assignment agreement is against floating original-space Full. There are 80 corpus/$k$/start cells '
        '(eight corpora, $k=5,10$, five starts); ten random maps are averaged within each cell. '
        'Head/JL preserves the deployed-center head and uses a sparse residual tail, not the energy statistic of the main reuse method. '
        'PCA here is plain PCA, not PCA with energy. The final column is mean absolute cost error of the integer Head/JL method. '
        'These native runs do not measure proof performance.', 'lrrrrr'), encoding='utf-8')
    claims['fixed_five_starts'] = dict(cells=80, raw_rows=len(fidelity), width=16,
        protected_float_agreement=float(all_methods.loc['protected','agreement']),
        protected_integer_agreement=float(all_methods.loc['protected_int','agreement']),
        pca_agreement=float(all_methods.loc['pca','agreement']),
        sparse_jl_agreement=float(all_methods.loc['sparse','agreement']))

    inputs.update([a.snapshot/'manifest.json', a.snapshot/'checks.json'])
    provenance = dict(snapshot=str(a.snapshot), observed_utc=checks['observed_utc'], claims=claims,
        raw_verification='Recorded in-process proof verification; no independent replay of saved reuse proofs in this export.',
        inputs=[dict(path=str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p),
                     sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(inputs)])
    (dest/'manifest.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps(dict(out=str(dest),claims=claims,proof_shapes=shape_rows),indent=2))


if __name__ == '__main__':
    main()
