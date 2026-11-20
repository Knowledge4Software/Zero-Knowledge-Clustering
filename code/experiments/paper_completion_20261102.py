"""Frozen-method ablations and controls. Native work is never proof timing."""
from pathlib import Path
import argparse
import copy
import shutil
import time

import numpy as np
from threadpoolctl import threadpool_limits

from commitjl.cost_aware import exact_distances
from commitjl.final_method_20261102 import fit_bases
from commitjl.group_change_bounds import advantage_bounds, assigned_change
from commitjl.group_moment_reference import prepare_groups
from commitjl.jl_certificate_pilots import hint_order, refinement_work
from commitjl.pilot_io import array_sha
from commitjl.published_audit_baselines import (CoresetAudit, ProjectedAudit,
    point_decision, random_projection)
from commitjl.reuse_certificates import integer_stats
from experiments.pca_advantage_20261102 import read, save, sha, model_for, endpoint_order, source


def diagnostic_arrays(original, dataset, n, q):
    provenance = read(original/'provenance.json')
    with np.load(original/'arrays.npz') as z:
        a = {k: z[k].copy() for k in z.files}
    if q < 1 or q > len(a['centers']):
        raise ValueError('Diagnostic needs an available nonempty request prefix')
    a['centers'] = a['centers'][:q]
    if n != 128:
        train, audit, _, _, _ = source(dataset, 907)
        assert array_sha(train[:512]) == provenance['training_sha256']
        a['rows'] = audit[128:128+n]
    if len(a['rows']) != n:
        raise ValueError('Requested catalog size unavailable')
    return a


def scaled_profiles(profiles, dataset, n):
    profiles = copy.deepcopy(profiles)
    if n != 128:
        ratio = n/128
        for p in profiles[dataset].values():
            p['source_measured_shape'] = p['shape'].copy()
            p['shape'][0] = n
            p['costs']['full_seconds'] *= ratio
            p['costs']['full_batch_seconds'] *= ratio
            p['keys_rows_seconds'] *= ratio
            p['prepare_seconds'] *= ratio
            p['full_initialization_seconds'] *= ratio
            p['calibration_scope'] = 'Explicit extrapolation for fixed-configuration scaling, not measured-shape calibration.'
    return profiles


def ablate(rows, old, centers, basis, scale, family, rank, caps=(0, 4, 8, 16), orders=('endpoint', 'width')):
    model = model_for(basis[:rank], old, len(rows), scale)
    old_distance = exact_distances(rows, old)
    groups = old_distance.argmin(1)
    prepared = prepare_groups(rows, groups.tolist(), len(old))
    h, e = integer_stats(model, rows)
    den = 16384**6
    result = []
    for request, new in enumerate(centers):
        lo, hi = advantage_bounds(model, h, e, new, groups)
        assigned = assigned_change(prepared, old, new)*den
        bounds = dict(row_lower=lo, row_upper=hi, assigned_scaled=assigned,
            change_lower_scaled=assigned+sum(map(int, lo)),
            change_upper_scaled=assigned+sum(map(int, hi)), cost_denominator=den)
        distance = exact_distances(rows, new)
        exact = [(int(d.min())-int(d[g]))*den for d, g in zip(distance, groups)]
        truth = (sum(map(int, distance.min(1)))-sum(map(int, old_distance.min(1))))*den
        assert assigned+sum(exact) == truth
        assigned_decision = 'improves' if assigned < 0 else 'inconclusive'
        assert assigned_decision == 'inconclusive' or truth < 0
        for ordering in orders:
            indices = endpoint_order(bounds, exact) if ordering == 'endpoint' else hint_order(bounds, None, 'width')
            for cap in caps:
                work = refinement_work(bounds, exact, indices, cap)
                target = point_decision(truth)
                assert work['lower'] <= truth <= work['upper']
                assert work['decision'] in ['inconclusive', target]
                result.append(dict(family=family, rank=rank, request=request, n=len(rows),
                    ordering=ordering, cap=cap, corrections=work['exact_rows'],
                    decision=work['decision'], conclusive=work['decision'] != 'inconclusive',
                    truth=str(truth), lower=str(work['lower']), upper=str(work['upper']),
                    denominator=str(den), assigned_only_decision=assigned_decision))
    return result


def controls(rows, old, centers, seed):
    banks = [old, *centers]
    distances = [exact_distances(rows, b) for b in banks]
    costs = [sum(map(int, d.min(1))) for d in distances]
    truth = [c-costs[0] for c in costs[1:]]
    records = []
    for kind in ['sign_rp', 'sparse_rp']:
        for rank in [16, 79]:
            audit = ProjectedAudit(rows, random_projection(kind, rank, rows.shape[1], seed))
            values = [audit.query(b) for b in banks]
            for j, delta in enumerate(truth):
                estimated = values[j+1][0]-values[0][0]
                records.append(dict(method=kind, rank=rank, request=j, true_change=delta,
                    estimated_change=estimated, point_sign_correct=point_decision(estimated)==point_decision(delta),
                    assignment_agreement=float(np.mean(values[j+1][1]==distances[j+1].argmin(1))),
                    cryptographic_certificate=False))
    for lightweight in [True, False]:
        for size in [16, 32]:
            audit = CoresetAudit(rows, size, seed, lightweight)
            values = [audit.query(b) for b in banks]
            for j, delta in enumerate(truth):
                estimate = values[j+1]-values[0]
                records.append(dict(method=('lightweight' if lightweight else 'uniform')+'_coreset',
                    sample_draws=size, unique_sampled_rows=len(audit.indices), request=j,
                    true_change=delta, estimated_change=estimate,
                    point_sign_correct=point_decision(estimate)==point_decision(delta),
                    cryptographic_certificate=False))
    return records


def native_final(case, out):
    choices = read(case/'selection.json')['choices']
    provenance = read(case/'provenance.json')
    with np.load(case/'arrays.npz') as z:
        a = {k: z[k].copy() for k in z.files}
    rows, old, centers = [a[k] for k in ['rows', 'old', 'centers']]
    records = []
    for family in ['centroid', 'pca']:
        for rank in [16, 79]:
            records += ablate(rows, old, centers, a[family], provenance['scale'], family, rank)
    stress = []
    for episode, reference in enumerate([old, centers[0]]):
        # A reference change REBUILDS exact groups; old authenticated state is not reused.
        candidate = np.array([np.clip(np.rint(reference+rate*(centers[-1]-reference)), -4095, 4095)
                              for rate in [0., 1/1024, .125, 1., 2., 8.]], dtype=np.int64)
        for method, family in [('ordered_head', 'centroid'), ('pca', 'pca')]:
            rank = choices[method]['rank']
            stress += [dict(r, episode=episode, reference_groups_rebuilt=True)
                       for r in ablate(rows, reference, candidate, a[family], provenance['scale'],
                                       family, rank, caps=(0, 16), orders=('endpoint',))]
    scales = []
    if provenance['seed'] == 907:
        train, audit, _, scale, _ = source(provenance['dataset'], 907)
        assert array_sha(train[:512]) == provenance['training_sha256']
        assert array_sha(audit[128:256]) == provenance['test_rows_sha256']
        for n in [32, 128, 512, 896]:
            for method, family in [('ordered_head', 'centroid'), ('pca', 'pca')]:
                scales += ablate(audit[128:128+n], old, centers, a[family], scale, family,
                                  choices[method]['rank'], caps=(16,), orders=('endpoint',))
    save(out/'result.json', dict(complete=True, source_case=str(case),
         arrays_sha256=sha(case/'arrays.npz'), selection_sha256=sha(case/'selection.json'),
         native_only=True, ablations=records, operating_limits=stress, scale=scales,
         numerical_baselines=controls(rows, old, centers, provenance['seed']),
         all_intervals_contain_truth=True, no_false_conclusive_signs=True,
         scope='Post hoc component/limit diagnostics on fixed test inputs; selections unchanged. '
               'Near-ties, equal candidates and extrapolated updates are retained. '
               'Reference-change results rebuild groups and do not claim free authenticated reuse. '
               'No classification-label accuracy or new proof-time claim.'))


def native_historical(fitted, out):
    pin = read(fitted/'pin.json')
    assert sha(fitted/'arrays.npz') == pin['arrays_sha256']
    with np.load(fitted/'arrays.npz') as a:
        past, rows, centers = [a[k].copy() for k in ['training', 'audit', 'centers']]
    if min(past.shape) < 79:
        save(out/'result.json', dict(complete=False, excluded=True,
             reason='Frozen rank79 basis cannot be fitted from fewer than79 past rows/features.',
             available_past_rows=len(past), source_pin_sha256=sha(fitted/'pin.json'),
             scope='Historical native controls remain available; no silently modified final method.'))
        return
    bases, _ = fit_bases(past, centers[0], pin.get('seed', pin.get('fit_seed', int(fitted.name.split('seed')[-1]))))
    records = []
    for n in [32, 128]:
        if len(rows) < n:
            continue
        for family, basis in bases.items():
            for rank in [16, 79]:
                records += ablate(rows[:n], centers[0], centers[1:], basis, pin['scale'], family,
                                  rank, caps=(0, 16), orders=('endpoint',))
    save(out/'result.json', dict(complete=True, native_only=True,
         source_pin_sha256=sha(fitted/'pin.json'), records=records,
         scope='Fixed-rank final-basis diagnostics on previously inspected historical fits; '
               'not final operational rank selection or proof-service timing. '
               'All native controls from the original all-source cohort are retained separately.'))


def tamper(case, out, binary):
    from validation.verify_cost_aware import verify
    out.mkdir(parents=True, exist_ok=False)
    tests = []
    for method in ['ordered_head', 'pca']:
        source_out = case/'warm'/(case.name+'_'+method+'_final')
        target = out/method
        shutil.copytree(source_out/'routed/public', target/'public')
        shutil.copy2(source_out/'owner_pin.json', target/'pin.json')
        shutil.copy2(source_out/'approved.json', target/'task.json')
        def replay():
            return verify(target/'public', target/'pin.json', target/'task.json', binary)
        replay()
        routes = target/'public/routes.json'
        task = target/'task.json'
        changes = [
            ('missing_route', routes, lambda v: v['certificate'].pop()),
            ('duplicate_route', routes, lambda v: v['certificate'].append(v['certificate'][0])),
            ('changed_reference', task, lambda v: v['comparison_base'][0].__setitem__(0, v['comparison_base'][0][0]+1)),
            ('changed_candidate', task, lambda v: v['reuse_centers'][0][0].__setitem__(0, v['reuse_centers'][0][0][0]+1)),
        ]
        for name, path, mutate in changes:
            before = read(path); changed = copy.deepcopy(before); mutate(changed); save(path, changed)
            try:
                try: replay()
                except (ValueError, KeyError, IndexError): tests.append(dict(method=method, case=name, rejected=True))
                else: raise AssertionError('Tampering accepted: '+name)
            finally: save(path, before)
        proof = next((target/'public').rglob('*.proof'))
        before = proof.read_bytes()
        try:
            proof.write_bytes(bytes([before[0]^1])+before[1:])
            try: replay()
            except ValueError: tests.append(dict(method=method, case='proof_bytes', rejected=True))
            else: raise AssertionError('Changed proof accepted')
        finally: proof.write_bytes(before)
        replay()
    save(out/'result.json', dict(complete=True, tests=tests,
         original_evidence_unchanged=True, valid_replay_before_and_after=True,
         scope='Real final public-bundle tamper checks, not a formal security audit.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--task', choices=['native', 'historical', 'tamper'], required=True)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--binary', type=Path)
    args = p.parse_args()
    with threadpool_limits(limits=1):
        if args.task == 'native': native_final(args.source, args.out)
        elif args.task == 'historical': native_historical(args.source, args.out)
        else: tamper(args.source, args.out, args.binary)
