"""Small local exploratory ablation on two pre-existing regression-bearing fits.

This is numerical feasibility, not proof timings or independent confirmation.
Uses all four updates at early and mature ages in both selected fits.
"""
from pathlib import Path
import argparse
import json
import platform
import time
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from data.loaders import load_dataset
from experiments.review_update_decisions import lloyd_snapshots, update_trace, array_sha
from commitjl.protected_jl import centroid_basis
from commitjl.enriched_jl import enrich_centroid_basis
from commitjl.block_certificates import block_changes

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT/'code/validation/outputs/review_update_decisions_20261021T184553Z/native_capture_20261022T0530Z/native'


def sign(lo, hi):
    return 'worsens' if lo > 0 else 'improves' if hi < 0 else 'equal' if lo == hi == 0 else 'inconclusive'


def exact_rows(rows, centers):
    # Check the int64 accumulation limit before using the fast path.
    limit = int(np.abs(rows).max())+int(np.abs(centers).max())
    assert rows.shape[1]*limit**2 < 2**63
    return ((rows[:, None, :]-centers[None, :, :])**2).sum(2).min(1).astype(object)


def refinement(lower, upper, exact, order):
    lo, hi = sum(map(int, lower)), sum(map(int, upper))
    path = [(lo, hi)]
    for i in order:
        lo += int(exact[i])-int(lower[i])
        hi += int(exact[i])-int(upper[i])
        assert lo <= hi
        path.append((lo, hi))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    fits = json.loads((NATIVE/'fit_records.json').read_text())
    frozen = pd.read_csv(NATIVE/'cells.csv', dtype=str)
    records, refinements, pins = [], [], []
    start = time.monotonic()
    protocol = dict(fits=[dict(dataset='mnist', k=5, seed=11), dict(dataset='fashion_mnist', k=3, seed=11)],
        ages=[1, 8], rank=63, blocks=[1, 4, 16, 49], n=1024,
        selection='Exploratory: first-seed regression-bearing fit per image dataset; all four actual updates at both ages retained.',
        refinement_orders=['row_order', 'private_width_descending'],
        fixed_budgets=[0, .01, .05, .10, .25, .50, .75, 1.0],
        scope='Native integer ablation only. No authenticated block/refinement circuit, privacy guarantee or proof-speed claim.',
        python=platform.python_version(), threads=2)
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    with threadpool_limits(limits=2):
        for dataset, k in [('mnist', 5), ('fashion_mnist', 3)]:
            seed = 11
            source = load_dataset(dataset, n_max=5000, seed=0)
            x = np.asarray(source['X'], float)
            x = x[np.linalg.norm(x, axis=1) > 0]
            indices = np.random.default_rng(741+seed).permutation(len(x))
            cut = int(.6*len(x)); tr = x[indices[:cut]]
            scale = 4095/max(float(np.abs(tr).max()), 1e-12)
            training = np.rint(tr*scale).astype(np.int64)
            audit = np.clip(np.rint(x[indices[cut:cut+1024]]*scale), -4096, 4096).astype(np.int64)
            snapshots = lloyd_snapshots(training, k, seed)
            for age in [1, 8]:
                meta = next(f for f in fits if (f['dataset'], f['k'], f['fit_seed'], f['age']) == (dataset, k, seed, age))
                assert array_sha(training) == meta['training_sha256'], 'training identity differs'
                assert array_sha(audit) == meta['audit_sha256'], 'audit identity differs'
                sets, _ = update_trace(snapshots, age)
                old = np.asarray(sets[0], dtype=np.int64)
                old_exact = exact_rows(audit, old)
                for method in ['learned', 'pca']:
                    if method == 'learned':
                        basis = enrich_centroid_basis(training.astype(float), old,
                            extra=63-centroid_basis(old).shape[1], random_state=seed)[0].T
                    else:
                        basis = PCA(63, svd_solver='randomized', random_state=seed).fit(training.astype(float)).components_
                    basis = np.rint(basis*16384).astype(np.int64)
                    f = frozen[(frozen.dataset == dataset) & (frozen.k == str(k)) &
                        (frozen.fit_seed == str(seed)) & (frozen.age == str(age)) &
                        (frozen.method == method) & (frozen['rank'] == '63')]
                    basis_match = array_sha(basis) == f.iloc[0].basis_sha256
                    pin = dict(dataset=dataset, k=k, seed=seed, age=age, method=method,
                        training_match=True, audit_match=True, basis_match=basis_match,
                        actual_basis_sha256=array_sha(basis), frozen_basis_sha256=f.iloc[0].basis_sha256)
                    pins.append(pin)
                    model = dict(basis=basis.tolist(), basis_scale=16384, centers=old.tolist(), input_bits=12)
                    print(json.dumps(dict(stage='fit', **pin)), flush=True)
                    for s in range(1, 5):
                        now = np.asarray(sets[s], dtype=np.int64)
                        exact = exact_rows(audit, now)-old_exact
                        baseline = f[f['set'] == str(s)].iloc[0]
                        assert sum(map(int, exact)) == int(baseline.true_change), 'true update changed'
                        bounds = block_changes(model, audit, old, now)
                        for blocks, bound in bounds.items():
                            lo, hi, den = bound['lower'], bound['upper'], bound['denominator']
                            truth = exact*den
                            assert all(l <= v <= u for l, v, u in zip(lo, truth, hi)), 'invalid row interval'
                            if blocks == 1 and basis_match:
                                assert sum(map(int, lo)) == int(baseline.paired_lower_scaled)
                                assert sum(map(int, hi)) == int(baseline.paired_upper_scaled)
                            if blocks != 1:
                                assert all(l >= o for l, o in zip(lo, bounds[1]['lower']))
                                assert all(u <= o for u, o in zip(hi, bounds[1]['upper']))
                            decision = sign(sum(map(int, lo)), sum(map(int, hi)))
                            actual = sign(sum(map(int, exact)), sum(map(int, exact)))
                            assert decision in ['inconclusive', actual]
                            key = dict(dataset=dataset, k=k, seed=seed, age=age, method=method,
                                set=s, blocks=blocks, basis_matches_frozen=basis_match, truth=actual)
                            records.append(dict(**key, decision=decision,
                                width=(sum(map(int, hi))-sum(map(int, lo)))/den,
                                global_width=(sum(map(int, bounds[1]['upper']))-sum(map(int, bounds[1]['lower'])))/den,
                                true_change=sum(map(int, exact))))
                            widths = [int(u)-int(l) for l, u in zip(lo, hi)]
                            for order_name, order in [('row_order', list(range(len(audit)))),
                                    ('private_width_descending', sorted(range(len(audit)), key=lambda i: (-widths[i], i)))]:
                                path = refinement(lo, hi, truth, order)
                                assert path[-1] == (sum(map(int, truth)),)*2
                                first = next(i for i, (a, b) in enumerate(path) if sign(a, b) == actual)
                                for budget in protocol['fixed_budgets']:
                                    count = int(np.ceil(budget*len(audit)))
                                    a, b = path[count]
                                    outcome = sign(a, b)
                                    assert outcome in ['inconclusive', actual]
                                    refinements.append(dict(**key, order=order_name, budget=budget,
                                        exact_rows=count, decision=outcome, first_exact_rows_needed=first,
                                        first_exact_fraction_needed=first/len(audit)))
                        pd.DataFrame(records).to_csv(args.out/'bounds.csv', index=False)
                        pd.DataFrame(refinements).to_csv(args.out/'refinement.csv', index=False)
                        (args.out/'pins.json').write_text(json.dumps(pins, indent=2))
                        print(json.dumps(dict(stage='update', dataset=dataset, age=age, method=method, set=s,
                            true_change=sum(map(int, exact)), elapsed=time.monotonic()-start)), flush=True)
    frame = pd.DataFrame(records)
    groups = frame.groupby(['age', 'method', 'blocks', 'truth'], as_index=False).agg(
        requests=('set', 'size'), resolved=('decision', lambda d: int((d != 'inconclusive').sum())))
    groups.to_csv(args.out/'groups.csv', index=False)
    unique_ref = pd.DataFrame(refinements).drop_duplicates(['dataset', 'age', 'method', 'set', 'blocks', 'order'])
    unique_ref.groupby(['age', 'method', 'blocks', 'order', 'truth']).first_exact_fraction_needed.agg(
        ['count', 'min', 'median', 'max']).to_csv(args.out/'refinement_summary.csv')
    summary = dict(complete=True, elapsed_seconds=time.monotonic()-start, unique_update_requests=16,
        method_block_cells=len(records), input_pins=pins, row_interval_checks='PASS',
        false_decisions=0, all_refined_endpoints_exact=True,
        all_basis_hashes_match=all(p['basis_match'] for p in pins),
        evidence_scope=protocol['scope'], selection=protocol['selection'])
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
