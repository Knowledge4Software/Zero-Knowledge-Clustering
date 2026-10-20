"""Export RQ1 v3 tables and per-dataset figures without rerunning experiments.

python -m analysis.rq1_export --runs results/rq1/runs/paper_balanced_v3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def export(runs, output=None):
    from experiments.common import campaign_lock
    with campaign_lock(runs):
        return _export(runs,output)


def _export(runs, output=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from experiments.rq1_complete import summarize

    runs = Path(runs)
    output = Path(output) if output else runs / 'analysis'
    output.mkdir(parents=True, exist_ok=True)
    folders = sorted(p.parent for p in runs.glob('*/manifest.json'))
    if not folders:
        raise ValueError('No dataset manifests: point --runs at a campaign root')
    summaries, inventories, raw_files, table_rows, run_settings = [], [], [], [], {}
    for folder in folders:
        manifest = json.loads((folder / 'manifest.json').read_text())
        cfg = manifest['config']
        run_settings[folder.name] = cfg
        summarize(folder / 'rows.jsonl', cfg)
        summary_path = folder / 'summary.csv'
        if not summary_path.exists():
            inventories.append(dict(dataset=folder.name, status='no_completed_rows'))
            continue
        summary = pd.read_csv(summary_path)
        raw = pd.read_parquet(folder / 'rows.parquet')
        summary['dataset'] = folder.name
        summaries.append(summary)
        raw_files.append(dict(dataset=folder.name, file=str((folder / 'rows.parquet').resolve())))
        for _, entry in summary.iterrows():
            inventories.append(dict(dataset=folder.name, method=entry.method, k=int(entry.k),
                                    m=int(entry.m), q=int(entry.q), rows=int(entry.rows),
                                    complete=bool(entry.complete), eligible=bool(entry.eligible)))
        plan_path = folder / 'grid_inventory.json'
        if plan_path.exists():
            present = {(row['method'], row['k'], row['m'], row['q']) for row in inventories if row['dataset'] == folder.name}
            for plan in json.loads(plan_path.read_text()):
                key = (plan['method'], plan['k'], plan['m'], plan['q'])
                if key not in present:
                    inventories.append(dict(dataset=folder.name, **plan, rows=0, complete=False,
                                            eligible=False, status='excluded' if plan['exclusion'] else 'not_completed'))
        for k, group in summary.groupby('k'):
            selected_for_plots = []
            fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
            for (method, q), curve in group[group.method != 'full'].groupby(['method', 'q']):
                curve = curve.sort_values('m')
                label = f'{method} q={int(q)}' if method == 'sparse_jl' else method
                for ax, metric in zip(axes, ['r_phi', 'ari', 'delta_q95']):
                    if metric + '_mean' not in curve:
                        continue
                    valid = curve[curve.complete & curve[metric + '_mean'].notna()]
                    if valid.empty:
                        continue
                    line, = ax.plot(valid.m, valid[metric + '_mean'], marker='o', label=label)
                    ax.vlines(valid.m, valid[metric + '_ci_low'], valid[metric + '_ci_high'],
                              color=line.get_color(), alpha=.6, linewidth=1)
                    ax.fill_between(valid.m.to_numpy(), valid[metric + '_ci_low'].to_numpy(),
                                    valid[metric + '_ci_high'].to_numpy(), alpha=.12)
                    eligible = valid[valid.eligible]
                    if len(eligible):
                        selected = eligible.iloc[0]
                        ax.scatter([selected.m], [selected[metric + '_mean']], marker='*', s=100)
            for ax, ylabel, threshold in zip(axes, ['Original-space objective ratio', 'ARI vs full', 'Q95 absolute distortion'],
                                             [cfg['tau_phi'], cfg['tau_ari'], cfg['eps']]):
                ax.axhline(threshold, color='gray', linestyle='--', linewidth=1)
                ax.set_xscale('log', base=2)
                ax.set_xlabel('Projected dimension m')
                ax.set_ylabel(ylabel)
                ax.grid(alpha=.2)
            axes[0].axhline(1., color='black', linestyle=':', label='full-space control')
            axes[1].axhline(1., color='black', linestyle=':')
            axes[2].axhline(0., color='black', linestyle=':')
            axes[2].set_yscale('symlog', linthresh=cfg['eps'])
            axes[2].set_ylim(bottom=0)
            axes[0].legend(fontsize=6)
            fig.suptitle(f'{folder.name}: n={int(group.n.iloc[0])}, d={int(group.d.iloc[0])}, k={int(k)}; fixed centers')
            fig.tight_layout()
            for suffix in ['pdf', 'png']:
                fig.savefig(output / f'{folder.name}_k{int(k)}_fidelity.{suffix}', dpi=160, bbox_inches='tight')
            plt.close(fig)

            for method, candidates in group.groupby('method'):
                complete = candidates[candidates.complete]
                if complete.empty:
                    continue
                eligible = complete[complete.eligible].sort_values(['m', 'q'])
                # Failure display is declared: largest tested m, smallest q.
                chosen = eligible.iloc[0] if len(eligible) else complete.sort_values(['m', 'q'], ascending=[False, True]).iloc[0]
                def interval(metric):
                    value = chosen.get(metric + '_mean', float('nan'))
                    if not np.isfinite(value):
                        return '--'
                    return f"{value:.3f} [{chosen[metric + '_ci_low']:.3f}, {chosen[metric + '_ci_high']:.3f}]"
                status = 'control' if method == 'full' else ('eligible' if len(eligible) else 'no eligible point')
                if status == 'eligible' and int(chosen.m) == int(chosen.d):
                    status = 'eligible, no dimension reduction'
                if method == 'pisot_surrogate':
                    status = 'unverified surrogate'
                selected_for_plots.append(chosen)
                table_rows.append(dict(dataset=folder.name, n=int(chosen.n), d=int(chosen.d), k=int(k),
                    method=method, m=int(chosen.m), q=int(chosen.q), r_phi=interval('r_phi'),
                    ari=interval('ari'), distortion_q95=interval('delta_q95'),
                    exceedance=interval('delta_exceedance'),
                    certified_coverage='n/a' if method == 'pca' else interval('qualified_coverage'), status=status))

            fig, ax = plt.subplots(figsize=(7, 4))
            for chosen in selected_for_plots:
                if chosen.method == 'full':
                    continue
                subset = raw[(raw.k == k) & (raw.method == chosen.method) &
                             (raw.m == chosen.m) & (raw.q == chosen.q)]
                if 'pointwise_artifact' not in subset:
                    continue
                deltas = []
                for artifact in subset.pointwise_artifact.dropna().unique():
                    with np.load(folder / artifact) as values:
                        delta = values['pair_delta']
                        deltas.append(np.abs(delta[np.isfinite(delta)]))
                if not deltas:
                    continue
                values = np.sort(np.concatenate(deltas))
                ax.plot(np.maximum(values, 1e-8), np.arange(1, len(values) + 1) / len(values),
                        label=f'{chosen.method} m={int(chosen.m)} q={int(chosen.q)}')
            ax.set_xscale('log')
            ax.set_xlabel('Absolute sampled squared-distance distortion')
            ax.set_ylabel('Empirical cumulative fraction')
            ax.axvline(cfg['eps'], color='gray', linestyle='--')
            ax.legend(fontsize=6)
            ax.set_title(f'{folder.name}, k={int(k)}: selected or declared fallback points')
            fig.tight_layout()
            for suffix in ['pdf', 'png']:
                fig.savefig(output / f'{folder.name}_k{int(k)}_distortion_ecdf.{suffix}', dpi=160)
            plt.close(fig)

            sparse = group[(group.method == 'sparse_jl') & group.complete]
            if not sparse.empty:
                grid = sparse.pivot(index='q', columns='m', values='ari_mean').sort_index()
                fig, ax = plt.subplots(figsize=(6, 3.5))
                image = ax.imshow(grid.to_numpy(), aspect='auto', vmin=0, vmax=1, cmap='viridis')
                ax.set_xticks(range(len(grid.columns)), labels=grid.columns)
                ax.set_yticks(range(len(grid.index)), labels=grid.index)
                ax.set_xlabel('m')
                ax.set_ylabel('q')
                ax.set_title(f'{folder.name}, k={int(k)}: sparse JL mean ARI')
                fig.colorbar(image, ax=ax)
                fig.tight_layout()
                for suffix in ['pdf', 'png']:
                    fig.savefig(output / f'{folder.name}_k{int(k)}_sparsity_heatmap.{suffix}', dpi=160)
                plt.close(fig)

    combined = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    combined.to_csv(output / 'all_configurations.csv', index=False)
    if not combined.empty:
        combined.to_parquet(output / 'all_configurations.parquet', index=False)
        combined[combined.eligible].sort_values(['dataset', 'k', 'method', 'm', 'q']).groupby(
            ['dataset', 'k', 'method']).head(1).to_csv(output / 'operating_points.csv', index=False)
    pd.DataFrame(inventories).to_csv(output / 'completion_inventory.csv', index=False)
    tables = pd.DataFrame(table_rows)
    tables.to_csv(output / 'fidelity_table.csv', index=False)
    if not tables.empty:
        (output / 'fidelity_table.tex').write_text(tables.to_latex(index=False, escape=True), encoding='utf-8')
    (output / 'raw_files.json').write_text(json.dumps(raw_files, indent=2), encoding='utf-8')
    (output / 'run_settings.json').write_text(json.dumps(run_settings, indent=2), encoding='utf-8')
    # A user can load every raw file while preserving its dataset identity.
    (output / 'README.md').write_text(
        '# RQ1 export\n\nEach figure keeps dataset and k separate; q is a distinct curve. '
        'Ribbons are conditional pointwise crossed-bootstrap CIs. Stars mark the first '
        'eligible point per curve. The Q95 epsilon line is a diagnostic, not an operating-point rule. '
        'Certified coverage means the observed margin AND pointwise embedding event; '
        'it is not unconditional theorem coverage. PCA table coverage is n/a to match the draft. '
        'Pisot-surrogate rows are exploratory, not a reproduction of the cited baseline.\n\n'
        'all_configurations.csv/parquet supports alternative aggregate plots. '
        'raw_files.json points to seed/start-level measurements. Per-dataset artifacts contain '
        'X, labels, centers, pair indices and signed distortions for histogram, ECDF, '
        'margin and pointwise analyses. No global concatenation of different shapes is performed.\n',
        encoding='utf-8')
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', type=Path, required=True)
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    print(export(args.runs, args.out))


if __name__ == '__main__':
    main()
