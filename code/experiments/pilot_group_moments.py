"""Exploratory native group-moment pilot on the preceding two-fit ablation.

No authenticated moment binding, circuit implementation or proof-speed claim.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from data.loaders import load_dataset
from experiments.review_update_decisions import lloyd_snapshots, update_trace, array_sha
from experiments.pilot_certificate_refinement import exact_rows, sign, NATIVE
from commitjl.protected_jl import centroid_basis
from commitjl.enriched_jl import enrich_centroid_basis
from commitjl.reuse_certificates import certificate
from commitjl.group_moment_reference import prepare_groups, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--extended', action='store_true')
    parser.add_argument('--new-confirmation', action='store_true')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    fits = json.loads((NATIVE/'fit_records.json').read_text())
    frozen = pd.read_csv(NATIVE/'cells.csv', dtype=str)
    records, pins = [], []
    start = time.monotonic()
    workloads = [('mnist', 5, 11), ('fashion_mnist', 3, 11)]
    if args.extended:
        workloads = [('mnist', 5, 11), ('mnist', 5, 37), ('mnist', 10, 37), ('mnist', 10, 41),
            ('fashion_mnist', 3, 11), ('fashion_mnist', 3, 37), ('fashion_mnist', 5, 11), ('fashion_mnist', 5, 41)]
        workloads += [(d, k, seed) for seed in [67, 79] for d, k in [('mnist', 5), ('fashion_mnist', 3)]]
    if args.new_confirmation:
        workloads=[(d,k,seed) for d in ['mnist','fashion_mnist'] for k in [3,5] for seed in [167,193]]
    confirmation_seeds=[167,193] if args.new_confirmation else [67,79]
    protocol = dict(selection='Eight exploratory historical regression-bearing fits plus four confirmation fits at fixed seeds67/79; all updates at both ages retained.' if args.extended else 'Same exploratory two fits as prior block/refinement pilot, no new outcome search.',
        fits=workloads, confirmation_seeds=confirmation_seeds if args.extended or args.new_confirmation else [], ages=[1, 8], n=1024, rank=63,
        grouping='Fixed approximate deployed labels, chosen once before future centers; no claim that group labels are exact.',
        stable='Both integer assignment certificates valid and both labels equal fixed group.',
        result='Exact cost change from group coordinate sums, corrected on every nonstable row.',
        scope='Native correctness and uncertain-row count only. Group moments/membership are not yet cryptographically authenticated.',
        threads=args.threads)
    if args.new_confirmation:protocol['selection']='Fresh fixed seeds167/193 on both image datasets, k3/5, both ages, both representations and all actual updates. No outcome-based seed selection.'
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    with threadpool_limits(limits=args.threads):
        for dataset, k, seed in workloads:
            loaded = load_dataset(dataset, n_max=5000, seed=0)
            x = np.asarray(loaded['X'], float); x = x[np.linalg.norm(x, axis=1) > 0]
            indices = np.random.default_rng(741+seed).permutation(len(x)); cut = int(.6*len(x))
            tr = x[indices[:cut]]; scale = 4095/max(float(np.abs(tr).max()), 1e-12)
            training = np.rint(tr*scale).astype(np.int64)
            audit = np.clip(np.rint(x[indices[cut:cut+1024]]*scale), -4096, 4096).astype(np.int64)
            trace = lloyd_snapshots(training, k, seed)
            for age in [1, 8]:
                meta = next((f for f in fits if (f['dataset'], f['k'], f['fit_seed'], f['age']) == (dataset, k, seed, age)), None)
                if meta is not None:
                    assert array_sha(training) == meta['training_sha256']
                    assert array_sha(audit) == meta['audit_sha256']
                sets, _ = update_trace(trace, age); old = np.asarray(sets[0], dtype=np.int64)
                old_exact = exact_rows(audit, old)
                for method in ['learned', 'pca']:
                    if method == 'learned':
                        fitted = enrich_centroid_basis(training.astype(float), old,
                            extra=63-centroid_basis(old).shape[1], random_state=seed)[0].T
                    else:
                        fitted = PCA(63, svd_solver='randomized', random_state=seed).fit(training.astype(float)).components_
                    basis = np.rint(fitted*16384).astype(np.int64)
                    selected = frozen[(frozen.dataset == dataset) & (frozen.k == str(k)) &
                        (frozen.fit_seed == str(seed)) & (frozen.age == str(age)) &
                        (frozen.method == method) & (frozen['rank'] == '63')]
                    match = array_sha(basis) == selected.iloc[0].basis_sha256 if len(selected) else None
                    model = dict(basis=basis.tolist(), basis_scale=16384, centers=old.tolist(), input_bits=12)
                    old_cert = certificate(model, audit, old)
                    grouped = prepare_groups(audit, old_cert['labels'], k)
                    # Persist actual local bases and group labels, not just RNG seeds.
                    name = f'{dataset}_k{k}_seed{seed}_age{age}_{method}'
                    np.savez_compressed(args.out/(name+'.npz'), basis=basis, groups=np.array(grouped['groups']),
                        centers=np.array(sets, dtype=np.int64), counts=np.array(grouped['counts']),
                        sums=np.array(grouped['sums'], dtype=np.int64))
                    pins.append(dict(fit=name, training_sha256=array_sha(training), audit_sha256=array_sha(audit),
                        basis_sha256=array_sha(basis), basis_matches_frozen=match))
                    for s in range(1, 5):
                        new = np.asarray(sets[s], dtype=np.int64)
                        new_cert = certificate(model, audit, new)
                        result = evaluate(audit, grouped, old, new, old_cert, new_cert)
                        delta = sum(map(int, exact_rows(audit, new)-old_exact))
                        assert result['exact_change'] == delta
                        if len(selected):
                            assert delta == int(selected[selected['set'] == str(s)].iloc[0].true_change)
                        uncertain = set(result['uncertain_indices'])
                        # Independent full-distance check of every row treated as stable.
                        old_labels = ((audit[:, None, :]-old[None, :, :])**2).sum(2).argmin(1)
                        new_labels = ((audit[:, None, :]-new[None, :, :])**2).sum(2).argmin(1)
                        assert all(old_labels[i] == new_labels[i] == grouped['groups'][i]
                                   for i in range(len(audit)) if i not in uncertain)
                        records.append(dict(dataset=dataset, k=k, seed=seed, age=age, method=method, set=s,
                            cohort='confirmation' if seed in confirmation_seeds else 'historical_development',
                            truth=sign(delta, delta), true_change=delta, exact_change=result['exact_change'],
                            exact_rows=result['exact_rows'], exact_fraction=result['exact_fraction'],
                            group_moment_scalars=k*(audit.shape[1]+1), membership_labels=len(audit),
                            old_certified=sum(old_cert['certified_flags']), new_certified=sum(new_cert['certified_flags']),
                            basis_matches_frozen=match))
                        pd.DataFrame(records).to_csv(args.out/'results.csv', index=False)
                        print(json.dumps(records[-1]), flush=True)
    (args.out/'pins.json').write_text(json.dumps(pins, indent=2))
    data = pd.DataFrame(records)
    data.groupby(['age', 'method', 'truth']).exact_fraction.agg(['count', 'min', 'median', 'max']).to_csv(args.out/'groups.csv')
    summary = dict(complete=True, elapsed_seconds=time.monotonic()-start,
        unique_requests=len(records)//2,
        unique_regressions=int(((data.method == 'learned') & (data.truth == 'worsens')).sum()), method_request_checks=len(records),
        all_exact_changes_match=True, all_stable_assignments_checked=True,
        evidence_scope=protocol['scope'], grouping=protocol['grouping'])
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
