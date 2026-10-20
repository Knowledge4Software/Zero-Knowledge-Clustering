"""Summaries for experiments.new_center_reuse.

Maps are averaged within each (dataset, k, start, scenario, method, budget, q) cell
first; cells are then weighted equally. 'same' is the deployed centre set (sanity
check) and is excluded from the new-centre summaries.

Descriptive method means can have different coverage (e.g. PCA's training-rank
limit). Use matched_comparisons.csv for paired comparisons on common cells.

python -m analysis.new_center_reuse_report <run_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

CELL = ['dataset', 'k', 'start', 'scenario', 'family', 'method', 'budget', 'q']
METRICS = ['agreement', 'ari', 'r_phi', 'abs_psi', 'residual_fraction', 'shift']
PAIR_KEY = ['dataset', 'k', 'start', 'scenario', 'budget']


def matched_comparisons(table):
    """Compare methods only on identical corpus/start/scenario/budget cells."""
    rows = []
    for ours in sorted(x for x in table.label.unique() if x.startswith('ours_cross')):
        left = table[table.label == ours]
        for other in sorted(set(table.label) - {ours}):
            right = table[table.label == other]
            both = left.merge(right, on=PAIR_KEY, suffixes=('_ours', '_other'),
                              validate='one_to_one')
            for budget, pair in both.groupby('budget'):
                row = dict(ours=ours, other=other, budget=budget, cells=len(pair))
                for metric in ['agreement', 'ari', 'r_phi', 'abs_psi']:
                    row[metric + '_ours'] = pair[metric + '_ours'].mean()
                    row[metric + '_other'] = pair[metric + '_other'].mean()
                    row[metric + '_delta'] = (pair[metric + '_ours'] - pair[metric + '_other']).mean()
                rows.append(row)
    return pd.DataFrame(rows)


def load(run):
    df = pd.concat(pd.read_csv(p) for p in sorted((Path(run)/'jobs').glob('*.csv')))
    df['abs_psi'] = df['psi_rel_error'].abs()
    df['label'] = df['method'] + df['q'].map(lambda q: f'_q{q}' if q else '')
    return df.groupby(CELL + ['label'], as_index=False)[METRICS].mean()


def main(run):
    run = Path(run)
    cells = load(run)
    new = cells[cells.family != 'same']
    out = run/'summary'
    out.mkdir(exist_ok=True)

    # Budget-free no-JL ablation is repeated at every budget for comparison.
    nocross = new[new.method == 'ours_nocross']
    budgets = sorted(new.loc[new.method != 'ours_nocross', 'budget'].unique())
    rep = pd.concat(nocross.assign(budget=b) for b in budgets)
    table = pd.concat([new[new.method != 'ours_nocross'], rep])

    table.groupby(['budget', 'label']).size().rename('cells').to_csv(out/'coverage_by_method.csv')
    matched_comparisons(table).to_csv(out/'matched_comparisons.csv', index=False)

    overall = table.groupby(['budget', 'label'])[['agreement', 'ari', 'r_phi', 'abs_psi']].mean().unstack('label')
    overall.to_csv(out/'overall_by_budget.csv')
    fam = table.groupby(['family', 'budget', 'label'])[['agreement', 'r_phi', 'abs_psi']].mean()
    fam.to_csv(out/'by_family.csv')
    ds = table.groupby(['dataset', 'budget', 'label'])[['agreement', 'r_phi', 'abs_psi']].mean()
    ds.to_csv(out/'by_dataset.csv')
    desc = new[new.label == 'ours_nocross'].groupby('family')[['residual_fraction', 'shift']].mean()
    desc.to_csv(out/'scenario_descriptors.csv')

    # Paired wins of the proposed estimator against each comparator (same cell, same budget).
    piv = table.pivot_table(index=PAIR_KEY, columns='label', values='agreement')
    wins = {}
    for ours in [c for c in piv.columns if c.startswith('ours_cross')]:
        for other in piv.columns:
            if other == ours:
                continue
            both = piv[[ours, other]].dropna()
            if len(both):
                wins[(ours, other)] = dict(cells=len(both), win=float((both[ours] > both[other]).mean()),
                                          tie=float((both[ours] == both[other]).mean()))
    pd.DataFrame(wins).T.to_csv(out/'paired_wins.csv')

    sanity = cells[(cells.family == 'same') & cells.method.isin(['ours_cross', 'ours_nocross'])]
    with pd.option_context('display.width', 250, 'display.max_columns', 40, 'display.precision', 4):
        print('same-centre sanity: min agreement', sanity.agreement.min(), 'max |psi|', sanity.abs_psi.max())
        print('Method coverage may differ: use matched_comparisons.csv for paired comparisons; '
              'overall/family/dataset means are descriptive.')
        print('\nAGREEMENT (new centres) by budget\n', overall['agreement'])
        print('\nr_phi by budget\n', overall['r_phi'])
        print('\n|cost claim error| by budget\n', overall['abs_psi'])
        print('\nscenario descriptors\n', desc)
        print('\npaired wins (agreement)\n', pd.DataFrame(wins).T)


if __name__ == '__main__':
    main(sys.argv[1])
