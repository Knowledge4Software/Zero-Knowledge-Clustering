"""Matched comparison of hybrid learned-head + JL-tail variants against PCA+energy.

Maps are averaged within each cell; comparisons use only (dataset, k, start,
scenario, budget) cells where both methods exist.

python -m analysis.new_center_hybrid_report <run_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

KEY = ['dataset', 'k', 'start', 'scenario', 'budget']


def label(r):
    if r.method == 'hybrid':
        return f'hybrid_f{r.split}_{r.map_kind}_{r.estimator}'
    if r.method == 'learned_energy':
        return 'learned_energy_f1.0'
    return r.method


def main(run):
    run = Path(run)
    df = pd.concat(pd.read_csv(p) for p in sorted((run/'jobs').glob('*.csv')))
    df['label'] = df.apply(label, axis=1)
    df['abs_psi'] = df.psi_rel_error.abs()
    cells = df.groupby(KEY + ['family', 'label'], as_index=False)[['agreement', 'ari', 'r_phi', 'abs_psi']].mean()
    base = cells[cells.label == 'pca_energy']
    rows = []
    for lab, grp in cells[cells.label != 'pca_energy'].groupby('label'):
        both = grp.merge(base, on=KEY, suffixes=('', '_pca'))
        for budget, b in both.groupby('budget'):
            rows.append(dict(label=lab, budget=budget, cells=len(b),
                             agree=b.agreement.mean(), agree_pca=b.agreement_pca.mean(),
                             delta=(b.agreement - b.agreement_pca).mean(),
                             win=(b.agreement > b.agreement_pca).mean(),
                             r_phi=b.r_phi.mean(), r_phi_pca=b.r_phi_pca.mean(),
                             abs_psi=b.abs_psi.mean(), abs_psi_pca=b.abs_psi_pca.mean()))
    res = pd.DataFrame(rows)
    out = run/'summary'
    out.mkdir(exist_ok=True)
    res.to_csv(out/'vs_pca_energy.csv', index=False)
    fam = cells.groupby(['family', 'budget', 'label']).agreement.mean().unstack('label')
    fam.to_csv(out/'by_family.csv')
    best = res.sort_values('delta', ascending=False).groupby('budget').head(6)
    with pd.option_context('display.width', 250, 'display.max_rows', 200, 'display.precision', 4):
        print('Top variants per budget vs PCA+energy (matched agreement):\n', best.sort_values(['budget', 'delta'], ascending=[True, False]))
        print('\nNo-JL same-basis ablation:\n', res[res.label == 'learned_energy_f1.0'])


if __name__ == '__main__':
    main(sys.argv[1])
