"""Freeze an advantage study and run its synthetic numerical control only.

This does not generate proofs or claim a runtime advantage. Timing projections
below use old N32 receipts and are explicitly conditional planning estimates.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import time

import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from commitjl.enriched_jl import enrich_centroid_basis
from commitjl.protected_jl import centroid_basis
from commitjl.relative_certificates import relative_certificate
from commitjl.group_change_bounds import exact_old_correction_bounds
from commitjl.group_moment_reference import prepare_groups
from commitjl.jl_certificate_pilots import hint_order, refinement_work
from experiments.rank79_independent_replication import extend_basis

REPO = Path(__file__).resolve().parents[2]
OLD = REPO / 'code/validation/outputs/published_baselines_20261027'
DEFAULT = REPO / 'code/validation/outputs/advantage_design_20261028'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def plan():
    return dict(
        client_date='2026-10-28',
        status='design_locked_before_new_synthetic_results; real_proof_study_not_run',
        hypotheses=[
            'H1: at a stated N and distinct-update count Q, total certified decision cost is below exact controls after paying all setup and fallback.',
            'H2: preserving center differences helps versus equal-rank PCA when high variance is orthogonal to decision directions.',
        ],
        primary_task='Return a verified sign of the exact quantized original-space cost change. Inconclusive certificates require newly generated exact fallback.',
        separate_task='Approximate individual assignment agreement; not ground-truth classification accuracy. If exact labels are required, charge their correction separately.',
        native_grid=dict(datasets=['fashion_mnist', 'text_minilm384'],
                         fit_seeds=[521, 523, 541], catalog_sizes=[32, 128, 512, 1024],
                         distinct_update_prefixes=[1, 4, 16, 64]),
        real_data_protocol=dict(
            split='Pin raw-source versions and row IDs for train/incoming/audit before evaluation; use unused audit rows where available. Otherwise label reuse exploratory.',
            reference='Training-only Lloyd step8; freeze old centers and bases for an epoch.',
            proposals='Split 512 disjoint incoming-training rows into16 fixed batches; rates .125,.25,.5,1. in batch-major order. Do not use audit data to propose centers.',
            duplicate_rule='Cache only exact public center duplicates equally for all methods; preserve all64 request positions and report actual distinct count.',
            reference_updates='A deployed new reference starts a new epoch. Reauthenticate groups/moments and charge that work; do not assume permanent reuse.',
        ),
        algorithms=['current centroid certificate', 'same certificate with PCA basis',
                    'exact Full Groth16', 'published ZKML primitive adapter Halo2-KZG'],
        frozen_profile=dict(prediction_rank=63, certificate_rank=79, basis_bits=14,
                            max_exact_corrections_per_update=16,
                            refinement_order='interval width, index ties'),
        backend_control='Use the existing Groth16 Full for an internal same-backend comparison. Treat Halo2 adapter as a system comparison; commitment constructions differ.',
        proof_schedule=dict(
            feasibility='Ascending N32,128,512,1024 with4 fixed updates. Capacity selects the largest common feasible N, never certificate outcomes or speed ratios.',
            main='At that N evaluate all fixed Q prefixes; use3 paired timing blocks, with fresh cold bundles and rotated method order. Retain every seed and failed attempt.',
            prefix_reuse='Within a block charge each unique generated proof once, compute actual cumulative prefixes from per-bank receipts; no repeated candidates counted as new proofs.',
            resource_limits=dict(os='Linux/WSL', workers=8, serial=True, rss_cap_mib=3584,
                                 min_available_ram_mib=650, max_phase_seconds=2700,
                                 initial_total_proof_hours=12),
            censoring='Stop at budget/resource limit, preserve receipts, and report incomplete or infeasible. Do not claim a speedup from missing comparator times.',
        ),
        accounting=dict(
            cold='Method-specific basis fit, owner authentication, circuit setup/parameter generation, old-reference evidence, queries, public replay, selected corrections, fresh fallback, and unsuccessful runtime attempts.',
            warm='Reuse setup, parameters and approved old evidence for EVERY method. Report query-only and cold cumulative totals separately.',
            fairness='Same quantization, tie rule, rows, ordered center banks, CPU affinity, thread cap and memory limit. Separate compilation and diagnostic feasibility work.',
            decision_only_control='For a later stronger claim, an exact sign-only circuit is also needed; computing public exact costs is a stronger output contract.',
        ),
        success_criteria=dict(
            correctness='All independently checked intervals contain the exact integer change; zero incorrect certified decisions.',
            performance='Report paired cold/warm ratios at every measured prefix, fallback frequency, proof bytes and memory. A2x target is a practical target, not a promised result.',
            confirmation='A positive synthetic control alone is not real-data evidence. Require the effect across all3 prespecified fits on at least one real dataset and3 timing blocks before a regime-specific speed claim.',
            precision='Report individual paired runs and dispersion; repeated timings on one fit are not independent dataset replications.',
            negative='If steady query cost including corrections/fallback exceeds exact cost, more reuse alone cannot rescue asymptotic amortization in that workload.',
        ),
        synthetic_control=dict(d=128, k=5, training_rows=256, audit_rows=64,
                               seeds=[521, 523, 541], nuisance_sigmas=[8, 32, 128],
                               signal_amplitude=30, signal_sigma=2,
                               old_center_amplitude=32,
                               new_amplitudes=[28, 30, 34, 36],
                               scope='Native-only mechanism control; deliberately constructed, no proof-runtime or real-population claim.'),
        paper_edits=False,
    )


def existing_timing_diagnostic():
    rows = []
    for name, count in [('fashion_n32', 3), ('text_n32', 6)]:
        target = OLD / 'matched' / name
        resources = read(target / 'ordered_head_base/status.json')['resources']
        initial = sum(resources[key]['wall_seconds'] for key in
                      ['group_compact_prepare', 'group_keys', 'group_rows'])
        # Intentionally optimistic for the certificate: exclude its verification,
        # refinements, fallback, and process gaps. Do not present as service time.
        marginal = resources['group_query']['wall_seconds'] / count
        rust = [read(target / f'zkml_private/bank_{i:03}/result.json') for i in range(count + 1)]
        exact_query = float(np.mean([entry['total_seconds'] for entry in rust[1:]]))
        setup = read(target / 'zkml_private/setup.json')
        exact_initial = sum(setup[key] for key in ['owner_preparation_seconds', 'parameter_generation_seconds', 'setup_seconds']) + rust[0]['total_seconds']
        gap = exact_query - marginal
        qstar = max(1, math.floor((initial - exact_initial) / gap) + 1) if gap > 0 else None
        checks = read(target / 'ordered_head_refined/checked_comparison.json')
        rows.append(dict(case=name, observed_requests=count,
                         certificate_initial_component_seconds=initial,
                         certificate_optimistic_query_seconds=marginal,
                         exact_initial_component_seconds=exact_initial,
                         exact_mean_query_seconds=exact_query,
                         optimistic_linear_crossover_updates=qstar,
                         omitted_correction_seconds=checks['refinement_active_seconds'],
                         omitted_fallback_seconds=checks['fallback_active_seconds'],
                         qualification='Conditional planning model, not newly measured performance; ignores certificate verification, all corrections/fallback and workload-dependent changes.'))
    return rows


def synthetic_control(spec):
    records = []
    for seed in spec['seeds']:
        for noise in spec['nuisance_sigmas']:
            rng = np.random.default_rng(seed)
            d, k = spec['d'], spec['k']

            def sample(n):
                labels = rng.integers(k, size=n)
                points = rng.normal(0, noise, size=(n, d))
                points[:, :k] = rng.normal(0, spec['signal_sigma'], size=(n, k))
                points[np.arange(n), labels] += spec['signal_amplitude']
                return np.rint(points).astype(np.int64)

            training, audit = sample(spec['training_rows']), sample(spec['audit_rows'])
            old = np.zeros((k, d), dtype=np.int64)
            old[:, :k] = np.eye(k, dtype=np.int64) * spec['old_center_amplitude']
            distances = lambda c: ((audit[:, None, :] - c[None, :, :]) ** 2).sum(2)
            old_dist = distances(old)
            groups = old_dist.argmin(1)
            prepared = prepare_groups(audit, groups.tolist(), k)
            old_cost = int(old_dist.min(1).sum())
            for method in ['ordered_head', 'pca']:
                if method == 'ordered_head':
                    u, _ = enrich_centroid_basis(training, old, extra=63-centroid_basis(old).shape[1],
                                                random_state=seed, preserve_pca_order=True)
                    basis = u.T
                else:
                    basis = PCA(63, svd_solver='randomized', random_state=seed).fit(training).components_
                basis = np.rint(basis * 16384).astype(np.int64)
                aux, _ = extend_basis(basis, training, old)
                model = dict(centers=old.tolist(), basis=basis.tolist(), basis_scale=16384,
                             n=len(audit), d=d, k=k, method='learned_energy')
                for amplitude in spec['new_amplitudes']:
                    new = old.copy()
                    new[:, :k] = np.eye(k, dtype=np.int64) * amplitude
                    exact = distances(new)
                    truth = int(exact.min(1).sum()) - old_cost
                    assignment = relative_certificate(model, audit, new)
                    predicted = np.asarray(assignment['labels'])
                    flags = np.asarray(assignment['certified_flags'], dtype=bool)
                    assert not np.any(flags & (predicted != exact.argmin(1)))
                    bounds = exact_old_correction_bounds(dict(model, basis=aux.tolist()), audit, prepared, old, new)
                    den = bounds['cost_denominator']
                    corrections = [int(value)*den for value in exact.min(1)-exact[np.arange(len(audit)), groups]]
                    result = refinement_work(bounds, corrections, hint_order(bounds, None, 'width'), 16)
                    assert result['lower'] <= truth*den <= result['upper']
                    decision = 'improves' if truth < 0 else 'worsens' if truth > 0 else 'equal'
                    assert result['decision'] in ['inconclusive', decision]
                    records.append(dict(seed=seed, noise_sigma=noise, method=method,
                                        amplitude=amplitude, n=len(audit), true_change=truth,
                                        decision=result['decision'], exact_rows=result['exact_rows'],
                                        agreement_percent=100*float(np.mean(predicted == exact.argmin(1))),
                                        certified_labels_percent=100*float(flags.mean())))
    return records


def main(root):
    root.mkdir(parents=True, exist_ok=False)
    papers = [REPO/'manuscript.tex', REPO/'references.bib'] + sorted((REPO/'paper_results').rglob('*.tex'))
    before = {str(p.relative_to(REPO)): digest(p) for p in papers}
    spec = plan()
    save(root/'plan.json', spec)  # Freeze before looking at the new control.
    pin = dict(plan_sha256=digest(root/'plan.json'), script_sha256=digest(Path(__file__)))
    save(root/'pin.json', pin)
    started = time.monotonic()
    with threadpool_limits(limits=1):
        records = synthetic_control(spec['synthetic_control'])
    assert digest(root/'plan.json') == pin['plan_sha256']
    assert all(digest(REPO/name) == value for name, value in before.items())
    aggregates = []
    for noise in spec['synthetic_control']['nuisance_sigmas']:
        for method in ['ordered_head', 'pca']:
            selected = [r for r in records if r['noise_sigma'] == noise and r['method'] == method]
            aggregates.append(dict(noise_sigma=noise, method=method, updates=len(selected),
                                   conclusive=sum(r['decision'] != 'inconclusive' for r in selected),
                                   mean_assignment_agreement_percent=float(np.mean([r['agreement_percent'] for r in selected])),
                                   mean_certified_labels_percent=float(np.mean([r['certified_labels_percent'] for r in selected]))))
    result = dict(complete=True, native_only=True, new_proofs=0,
                  real_proof_study_status='not_run', synthetic_records=records,
                  synthetic_summary=aggregates, existing_N32_planning_diagnostic=existing_timing_diagnostic(),
                  all_intervals_contain_truth=True, false_certified_labels=0,
                  paper_unchanged=True, elapsed_seconds=time.monotonic()-started, **pin)
    save(root/'native_control.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'synthetic_records'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT)
    args = parser.parse_args()
    main(args.root.resolve())
