"""Expose rank-63 audit outcomes by true update sign, without diagnostics.

Counts update requests once per dataset/fit/age/set, not once per representation.
All signs and interval-containment checks use Python integers.
"""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'code/validation/outputs/review_update_decisions_20261021T184553Z/native_capture_20261022T0530Z/native/cells.csv'
OUT = ROOT / 'paper_results/review_controls/referee_20261023'


def main():
    data = pd.read_csv(SOURCE, dtype=str)
    rows = data[(data['rank'] == '63') & (data.kind == 'lloyd_update')].copy()
    rows['truth'] = rows.true_change.map(lambda v: 'worsens' if int(v) > 0 else 'improves' if int(v) < 0 else 'equal')
    rows['conclusive'] = rows.paired_decision != 'inconclusive'
    for row in rows.to_dict('records'):
        truth = int(row['true_change']) * int(row['cost_denominator'])
        assert int(row['paired_lower_scaled']) <= truth <= int(row['paired_upper_scaled'])
        assert row['paired_decision'] == 'inconclusive' or row['paired_decision'] == row['truth']
    keys = ['dataset', 'k', 'fit_seed', 'age', 'set']
    assert rows.groupby(keys).size().eq(2).all()
    assert rows.groupby(keys).truth.nunique().eq(1).all()
    unique = rows.drop_duplicates(keys)
    regressions = unique[unique.truth == 'worsens']
    assert len(regressions) == 24
    assert regressions.groupby('dataset').size().to_dict() == {'fashion_mnist': 14, 'mnist': 10}
    assert not rows[rows.truth == 'worsens'].conclusive.any()
    OUT.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUT / 'actual_updates_by_sign.csv', index=False)
    groups = rows.groupby(['dataset', 'age', 'method', 'truth'], as_index=False).agg(
        requests=('set', 'size'), resolved=('conclusive', 'sum'))
    groups.to_csv(OUT / 'decision_breakdown.csv', index=False)
    text = [r'\begin{table*}[t]', r'\centering\small',
            r'\caption{Rank-63 native decisions stratified by the true held-out cost change. Only actual Lloyd updates are counted; null and reverse diagnostics are excluded. Each request is evaluated by both methods.}\label{tab:natural-regressions}',
            r'\begin{tabular}{llrrrr}', r'\toprule',
            r'Dataset & Age & Improving & Resolved H/P & Worsening & Resolved H/P \\', r'\midrule']
    labels = {'mnist': 'MNIST', 'fashion_mnist': 'Fashion-MNIST', '20ng_tfidf': '20NG TF--IDF'}
    for dataset, label in labels.items():
        for age in ['1', '8']:
            sub = rows[(rows.dataset == dataset) & (rows.age == age)]
            counts = []
            for sign in ['improves', 'worsens']:
                h = sub[(sub.method == 'learned') & (sub.truth == sign)]
                p = sub[(sub.method == 'pca') & (sub.truth == sign)]
                counts.extend([str(len(h)), f'{int(h.conclusive.sum())}/{int(p.conclusive.sum())}'])
            text.append(' & '.join([label, 'Early' if age == '1' else 'Mature'] + counts) + r' \\')
    text += [r'\bottomrule\end{tabular}', r'\end{table*}']
    (OUT / 'natural_regressions.tex').write_text('\n'.join(text) + '\n')
    report = dict(source=SOURCE.relative_to(ROOT).as_posix(),
                  source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  unique_actual_requests=len(unique), improving=int((unique.truth == 'improves').sum()),
                  worsening=len(regressions), certified_natural_worsenings_head=0,
                  certified_natural_worsenings_pca=0, all_regressions_mature=bool((regressions.age == '8').all()),
                  interpretation='Inconclusive is neither approval nor an incorrect sign; all observed natural regressions need exact fallback for a final sign decision.',
                  security_scope='Integer consistency checks, not a full cryptographic security audit.')
    (OUT / 'summary.json').write_text(json.dumps(report, indent=2))
    severity = []
    for row in rows[rows.truth == 'worsens'].to_dict('records'):
        delta = int(row['true_change'])
        baseline = int(row['true_cost']) - delta
        width = (int(row['paired_upper_scaled'])-int(row['paired_lower_scaled'])) / int(row['cost_denominator'])
        severity.append(dict(dataset=row['dataset'], k=int(row['k']), fit_seed=int(row['fit_seed']),
            age=int(row['age']), set=int(row['set']), method=row['method'],
            true_regression_percent=100*delta/baseline,
            interval_width_percent=100*width/baseline, width_over_true_change=width/delta))
    severity = pd.DataFrame(severity)
    severity.to_csv(OUT/'regression_severity.csv', index=False)
    diagnostic = severity.groupby('method')[['true_regression_percent', 'interval_width_percent',
        'width_over_true_change']].agg(['min', 'median', 'max'])
    diagnostic.to_csv(OUT/'regression_severity_summary.csv')
    report['severity_summary'] = {method: {
        metric: {stat: float(diagnostic.loc[method, (metric, stat)]) for stat in ['min', 'median', 'max']}
        for metric in ['true_regression_percent', 'interval_width_percent', 'width_over_true_change']}
        for method in diagnostic.index}
    report['severity_caution'] = 'Full interval width divided by actual change is descriptive; it is not a proven required contraction factor for a modified bound.'
    (OUT / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
