"""Validate captured metadata and summarize measured reuse workloads."""
import hashlib
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent/'outputs'/'server_review_20261019'
CODE = Path(__file__).resolve().parents[1]


def main():
    global ROOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=ROOT)
    args = ap.parse_args()
    ROOT = args.root
    if (ROOT/'checks.json').exists():
        raise FileExistsError('Analysis already exists; use a fresh capture to preserve earlier evidence')
    manifest = json.loads((ROOT/'manifest.json').read_text())
    for item in manifest['files']:
        assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest() == item['sha256']
    source_checks = []
    for item in manifest['files']:
        if item['path'].startswith('source/'):
            name = item['path'][7:]
            local = CODE/name
            source_checks.append(dict(path=name, matches=local.exists() and
                local.read_bytes().replace(b'\r\n', b'\n') == (ROOT/item['path']).read_bytes().replace(b'\r\n', b'\n')))
    cumulative, quality, validation = [], [], []
    for run in sorted((ROOT/'results'/'paper_runs_20261019').iterdir()):
        if not run.is_dir() or not (run/'workload.csv').exists():
            continue
        frame = pd.read_csv(run/'workload.csv')
        complete = (run/'done.json').exists()
        assert not frame.duplicated(['method', 'set', 'kind']).any()
        count = 0
        for p in run.glob('cases/*/*.json'):
            if p.stem != 'stage1' and not (p.stem.startswith(('query_', 'reuse_')) and p.stem.split('_')[-1].isdigit()):
                continue
            obj = json.loads(p.read_text())
            if 'phase' not in obj:
                continue
            assert obj['complete'] and obj['status'] == 'complete', str(p)
            assert obj['verified'] == len(obj['samples']), str(p)
            for sm in obj['samples']:
                assert len(sm['prove_seconds']) == len(sm['verify_seconds'])
                assert all(v > 0 and np.isfinite(v) for v in sm['prove_seconds'] + sm['verify_seconds'])
            count += obj['verified']
        # Independently reconstruct each exported timing from the saved phase samples.
        for row in frame.itertuples():
            if row.method in ('learned_energy', 'pca_energy', 'span_energy'):
                case = run/'cases'/f'{row.dataset}_n{row.n}_{row.method}'
            else:
                case = run/'cases'/f'{row.dataset}_n{row.n}_{row.method}_set{row.set:02d}'
            if row.kind == 'stage1':
                obj = json.loads((case/'stage1.json').read_text())
                samples = obj['samples']
            else:
                pattern = 'reuse_*.json' if row.method in ('learned_energy', 'pca_energy', 'span_energy') else 'query_*.json'
                paths = [p for p in case.glob(pattern) if 'progress' not in p.name]
                assert len(paths) == 1, str(case)
                obj = json.loads(paths[0].read_text())
                samples = [s for s in obj['samples'] if s.get('centre_set', 0) == row.set] if pattern.startswith('reuse') else obj['samples']
            expected = sum(np.median(s['prove_seconds']) + s['witness_seconds'] for s in samples)
            if row.method == 'full':
                expected *= row.n/row.full_batch
            assert np.isclose(expected, row.prove, rtol=1e-10), (str(case), expected, row.prove)
            assert obj['constraints'] == row.constraints
        validation.append(dict(run=run.name, complete=complete, rows=len(frame), recorded_verified_samples=count))
        if not complete:
            continue
        protocol = json.loads((run/'protocol.json').read_text())
        sets = len(protocol['sets'])
        for method, group in frame.groupby('method'):
            queries = group[group.kind == 'query']
            assert len(queries) == sets
            assert set(queries['set']) == set(range(sets))
            for t in range(1, sets+1):
                used = group[group['set'] < t]
                cumulative.append(dict(run=run.name, method=method, T=t,
                    total_seconds=used[['compile', 'setup', 'prove']].sum().sum()+used.get('native', pd.Series(dtype=float)).sum(),
                    witness_and_prove_seconds=used.prove.sum(),
                    reconstructed=method == 'full'))
            if method != 'full':
                new = queries[queries['set'] > 0]
                quality.append(dict(run=run.name, method=method,
                    new_sets=len(new), agreement=new.agreement.mean(), min_agreement=new.agreement.min(),
                    mean_abs_cost_error=new.cost_rel_error.abs().mean(),
                    mean_query_seconds=queries.prove.mean()))
    pd.DataFrame(cumulative).to_csv(ROOT/'cumulative_costs.csv', index=False)
    pd.DataFrame(quality).to_csv(ROOT/'proof_quality.csv', index=False)
    # Average random maps before comparing on common cells, keeping shift/owner settings apart.
    run = ROOT/'results'/'shift_owner_reuse_20261019'
    frames = [pd.read_csv(p) for p in sorted((run/'jobs').glob('*.csv'))]
    df = pd.concat(frames, ignore_index=True)
    assert len(frames) == 60
    assert np.isfinite(df[['agreement', 'ari', 'r_phi', 'psi_rel_error']].to_numpy()).all()
    df['abs_cost'] = df.psi_rel_error.abs()
    keys = ['dataset', 'setting', 'k', 'start', 'scenario', 'budget']
    cells = df.groupby(keys+['method'], as_index=False)[['agreement', 'abs_cost']].mean()
    paired = []
    for method in ['learned_energy', 'ours_cross', 'jl', 'pca_energy_oracle']:
        both = cells[cells.method == method].merge(cells[cells.method == 'pca_energy'], on=keys, suffixes=('', '_pca'))
        for (setting, budget), group in both.groupby(['setting', 'budget']):
            paired.append(dict(method=method, setting=setting, budget=budget, cells=len(group),
                               agreement=group.agreement.mean(), pca_agreement=group.agreement_pca.mean(),
                               delta=(group.agreement-group.agreement_pca).mean(),
                               mean_abs_cost_error=group.abs_cost.mean(), pca_abs_cost_error=group.abs_cost_pca.mean()))
    pd.DataFrame(paired).to_csv(ROOT/'shift_owner_paired.csv', index=False)
    summary = dict(observed_utc=manifest['observed_utc'], hashes_checked=len(manifest['files']),
                   source_checks=source_checks, workloads=validation, shift_owner_jobs=len(frames), shift_owner_rows=len(df))
    (ROOT/'checks.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print('\nNew-center proof quality:\n'+pd.DataFrame(quality).to_string(index=False))
    costs = pd.DataFrame(cumulative)
    print('\nCumulative cost T=1,2,10:')
    print(costs[costs['T'].isin([1, 2, 10])].to_string(index=False))
    print('\nShift/owner budget 16:\n'+pd.DataFrame(paired).query('budget == 16').to_string(index=False))


if __name__ == '__main__':
    main()
