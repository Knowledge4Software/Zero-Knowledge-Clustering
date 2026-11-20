"""Training-only basis development, validation selection, and retained test results.

This is a new research cohort. Numerical work is not a proof-time benchmark.
The existing server cohorts and cryptographic acceptance relations are unchanged.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits

from commitjl.cost_aware import Planner, exact_distances
from commitjl.enriched_jl import enrich_centroid_basis
from commitjl.group_change_bounds import advantage_bounds, assigned_change
from commitjl.group_moment_reference import prepare_groups
from commitjl.jl_certificate_pilots import hint_order, refinement_work
from commitjl.pilot_io import array_sha, frozen_image_sample
from commitjl.protected_jl import centroid_basis
from commitjl.reuse_certificates import integer_stats
from experiments.rank79_independent_replication import trace

REPO = Path(__file__).resolve().parents[2]
DATASETS = ['fashion', 'har', 'text', 'mnist']
SEEDS = [811, 821, 829]
SCALE = 16384
FAMILIES = ['pca', 'centroid', 'history25', 'history75']


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.new')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def endpoint_order(bounds, exact):
    """Prover-private planning; acceptance still requires verified corrections.

    For nonzero change and equally costly corrections, sorting endpoint gains
    minimizes the corrections needed for the relevant strict sign endpoint.
    Equality retains width order. All original-row native work must be charged.
    """
    if len(exact) != len(bounds['row_lower']):
        raise ValueError('incomplete exact correction vector')
    for lo, value, hi in zip(bounds['row_lower'], exact, bounds['row_upper']):
        if not int(lo) <= int(value) <= int(hi):
            raise ValueError('exact correction outside the authenticated envelope')
    change = int(bounds['assigned_scaled']) + sum(map(int, exact))
    uncertain = [i for i, (lo, hi) in enumerate(zip(bounds['row_lower'], bounds['row_upper']))
                 if int(lo) < int(hi)]
    if not change:
        return hint_order(bounds, None, 'width')
    gains = [int(value)-int(lo) if change > 0 else int(hi)-int(value)
             for lo, value, hi in zip(bounds['row_lower'], exact, bounds['row_upper'])]
    return sorted(uncertain, key=lambda i: (-gains[i], i))


class EndpointPlanner(Planner):
    """The same cost/routing policy, with exact private endpoint-gain ordering."""
    def request(self, centers, request_id):
        started = time.perf_counter()
        bounds = self.bounds(centers)
        lo, hi = bounds['change_lower_scaled'], bounds['change_upper_scaled']
        if lo > 0 or hi < 0 or lo == hi == 0:
            order = []
            work = refinement_work(bounds, [0]*len(self.x), [], 0)
        else:
            distance = exact_distances(self.x, centers)
            exact = [(int(row.min())-int(row[g]))*self.denominator
                     for row, g in zip(distance, self.groups)]
            order = endpoint_order(bounds, exact)[:self.cap]
            work = refinement_work(bounds, exact, order, self.cap)
        estimate = self.costs.certificate_seconds + work['exact_rows']*self.costs.correction_seconds
        if work['exact_rows']:
            estimate += self.costs.refinement_replay_seconds
        conclusive = work['decision'] != 'inconclusive'
        use = conclusive and estimate < self.costs.full_seconds*(1-self.costs.margin)
        return dict(request_id=request_id, path='certificate' if use else 'full',
                    reason='predicted_conclusive_and_cheaper' if use else
                    'predicted_inconclusive' if not conclusive else 'estimated_proof_cost',
                    refinement_indices=order[:work['exact_rows']] if use else [],
                    predicted_corrections=work['exact_rows'], predicted_decision=work['decision'],
                    estimated_certificate_seconds=estimate,
                    estimated_full_seconds=self.costs.full_seconds,
                    native_seconds=time.perf_counter()-started)


def proposal_banks(rows, old, count=8):
    """Chronological training minibatches, all fixed rates, public deduplication."""
    banks, recipe = [], []
    for batch_id, batch in enumerate(np.array_split(rows, max(1, len(rows)//32))):
        groups = exact_distances(batch, old).argmin(1)
        proposal = np.asarray([batch[groups == g].mean(0) if np.any(groups == g) else old[g]
                               for g in range(len(old))])
        for rate in [.125, .25, .5, 1.]:
            new = np.rint((1-rate)*old+rate*proposal).astype(np.int64)
            if np.array_equal(new, old) or any(np.array_equal(new, c) for c in banks):
                continue
            banks.append(new)
            recipe.append(dict(batch=batch_id, rate=rate))
            if len(banks) == count:
                return np.asarray(banks), recipe
    raise ValueError('Insufficient distinct proposals; retain this fit failure')


def history_directions(past, old):
    """Past rows only: residual changes in relative center positions."""
    differences = []
    for batch in np.array_split(past, max(1, len(past)//32)):
        groups = exact_distances(batch, old).argmin(1)
        new = np.asarray([batch[groups == g].mean(0) if np.any(groups == g) else old[g]
                          for g in range(len(old))])
        differences.extend((new[1:]-new[0])-(old[1:]-old[0]))
    return np.asarray(differences)


def canonical_extension(directions, protected):
    directions = directions.copy()
    for _ in range(2):
        directions -= protected@(protected.T@directions)
    q, r = np.linalg.qr(directions, mode='reduced')
    if np.any(np.abs(np.diag(r)) < 1e-10):
        raise ValueError('Dependent residual directions; no arbitrary padding')
    for j in range(q.shape[1]):
        if q[np.argmax(np.abs(q[:, j])), j] < 0:
            q[:, j] *= -1
    return q


def fit_families(past, old, seed, max_rank=79):
    if max_rank >= min(past.shape):
        raise ValueError('Insufficient training size or feature dimension')
    output, seconds = {}, {}
    tick = time.perf_counter()
    output['pca'] = PCA(max_rank, svd_solver='randomized', iterated_power=4,
                        random_state=seed).fit(past.astype(float)).components_
    seconds['pca'] = time.perf_counter()-tick
    tick = time.perf_counter()
    protected = centroid_basis(old)
    output['centroid'] = enrich_centroid_basis(
        past.astype(float), old, extra=max_rank-protected.shape[1],
        random_state=seed, iterated_power=4, preserve_pca_order=True)[0].T
    seconds['centroid'] = time.perf_counter()-tick
    tick = time.perf_counter()
    historical = history_directions(past, old)
    historical -= (historical@protected)@protected.T
    residual = past.astype(float)-old[0]
    residual -= (residual@protected)@protected.T
    residual -= residual.mean(0)
    history_seconds = time.perf_counter()-tick
    seconds['shared_history_preparation'] = history_seconds
    residual /= max(float(np.linalg.norm(residual)), 1e-30)
    historical /= max(float(np.linalg.norm(historical)), 1e-30)
    for name, weight in [('history25', .25), ('history75', .75)]:
        tick = time.perf_counter()
        stack = np.concatenate((np.sqrt(1-weight)*residual,
                                np.sqrt(weight)*historical))
        _, _, directions = randomized_svd(stack, n_components=max_rank-protected.shape[1],
                                          n_iter=4, random_state=seed)
        extra = canonical_extension(directions.T, protected)
        output[name] = np.column_stack((protected, extra)).T
        seconds[name] = time.perf_counter()-tick
    for name, basis in output.items():
        if basis.shape != (max_rank, past.shape[1]) or not np.allclose(
                basis@basis.T, np.eye(max_rank), atol=1e-9, rtol=1e-9):
            raise ValueError('Unexpected basis shape or orthogonality: '+name)
        output[name] = np.rint(basis*SCALE).astype(np.int64)
    return output, seconds


def source(dataset, seed):
    if dataset not in DATASETS:
        raise ValueError('Unscheduled dataset: '+dataset)
    if dataset in ['fashion', 'mnist']:
        raw = np.asarray(frozen_image_sample(REPO, 'fashion_mnist' if dataset == 'fashion' else 'mnist'))
        ids = np.flatnonzero(np.linalg.norm(raw, axis=1) > 0)
        ids = ids[np.random.default_rng(741+seed).permutation(len(ids))]
        training, audit = raw[ids[:1024]], raw[ids[1024:2048]]
        pin = dict(source_sha256=array_sha(raw), training_ids=ids[:1024].tolist(),
                   audit_ids=ids[1024:2048].tolist(), disjoint_rows=True)
        k = 5
    else:
        filename = 'har561.npz' if dataset == 'har' else 'text_minilm384.npz'
        path = REPO/'code/validation/outputs/referee_two_reviews_20261027/data'/filename
        with np.load(path) as data:
            ti = np.random.default_rng(seed).permutation(len(data['training']))[:1024]
            ai = np.random.default_rng(seed+1).permutation(len(data['audit']))[:1024]
            training, audit = data['training'][ti], data['audit'][ai]
            pin = dict(source_sha256=sha(path), training_ids=ti.tolist(), audit_ids=ai.tolist(),
                       official_source_partitions=True)
            if dataset == 'har':
                a = np.unique(data['training_subject'][ti])
                b = np.unique(data['audit_subject'][ai])
                assert not np.intersect1d(a, b).size
                pin.update(training_subjects=a.tolist(), audit_subjects=b.tolist())
        k = 6 if dataset == 'har' else 5
    if training.shape[0] != 1024 or audit.shape[0] != 1024:
        raise ValueError('Unexpected source partition size')
    scale = 4095/float(np.abs(training[:512]).max())
    training, audit = [np.clip(np.rint(a*scale), -4095, 4095).astype(np.int64)
                       for a in [training, audit]]
    return training, audit, k, scale, pin


def model_for(basis, old, n, scale):
    return dict(version='bound-audit-v1', method='learned_energy', n=n, d=basis.shape[1],
                k=len(old), input_bits=12, basis_scale=SCALE, input_scale=str(scale),
                centers=old.tolist(), basis=basis.tolist(), tail_width=0,
                tail_rows=[], tail_signs=[], seed='pca-advantage-20261102')


def evaluate(basis, old, rows, candidates, scale, family, rank):
    model = model_for(basis[:rank], old, len(rows), scale)
    old_distances = exact_distances(rows, old)
    groups = old_distances.argmin(1)
    prepared = prepare_groups(rows, groups.tolist(), len(old))
    heads, energy = integer_stats(model, rows)
    denominator = SCALE**6
    records = []
    for request, new in enumerate(candidates):
        started = time.perf_counter()
        lower, upper = advantage_bounds(model, heads, energy, new, groups)
        assigned = assigned_change(prepared, old, new)*denominator
        bounds = dict(row_lower=lower, row_upper=upper, assigned_scaled=assigned,
                      change_lower_scaled=assigned+sum(map(int, lower)),
                      change_upper_scaled=assigned+sum(map(int, upper)),
                      cost_denominator=denominator)
        distance = exact_distances(rows, new)
        exact = [(int(row.min())-int(row[g]))*denominator for row, g in zip(distance, groups)]
        truth = assigned+sum(exact)
        order = endpoint_order(bounds, exact)
        work = refinement_work(bounds, exact, order, 16)
        width = refinement_work(bounds, exact, hint_order(bounds, None, 'width'), 16)
        target = 'worsens' if truth > 0 else 'improves' if truth < 0 else 'equal'
        for result in [work, width]:
            assert result['lower'] <= truth <= result['upper']
            assert result['decision'] in ['inconclusive', target]
        records.append(dict(family=family, rank=rank, request=request,
                            conclusive=work['decision'] != 'inconclusive',
                            corrections=work['exact_rows'], decision=work['decision'],
                            lower=str(work['lower']), upper=str(work['upper']),
                            truth=str(truth), denominator=str(denominator),
                            base_width=str(bounds['change_upper_scaled']-bounds['change_lower_scaled']),
                            width_order_corrections=width['exact_rows'],
                            width_order_conclusive=width['decision'] != 'inconclusive',
                            native_seconds=time.perf_counter()-started))
    return records


def selection_score(rows, n, d, k):
    """Declared arithmetic-work proxy, NOT a proof-time or constraint estimate.

    Same formula for all representations; it charges authentication, exact old
    membership, every correction attempt, and complete Full fallback work.
    The target is the measured eight-request mixture. Real timings decide wins.
    """
    rank = rows[0]['rank']
    setup = n*d*(rank+k)
    response = sum(k*k*rank+k*d+r['corrections']*k*d+
                   (0 if r['conclusive'] else n*k*d) for r in rows)
    return int(setup+response)


def select_validation(records, n, d, k):
    candidates = []
    for family, rank in sorted({(r['family'], r['rank']) for r in records}):
        rows = [r for r in records if (r['family'], r['rank']) == (family, rank)]
        candidates.append(dict(family=family, rank=rank,
                               score=selection_score(rows, n, d, k),
                               conclusive=sum(r['conclusive'] for r in rows),
                               corrections=sum(r['corrections'] for r in rows)))
    key = lambda r: (r['score'], -r['conclusive'], r['rank'], r['family'])
    ours = min([r for r in candidates if r['family'] != 'pca'], key=key)
    pca = min([r for r in candidates if r['family'] == 'pca'], key=key)
    return dict(ours=ours, pca=pca, all_candidates=candidates,
                selection='Training/validation only, fixed arithmetic proxy; no test outcome selection')


def fit_case(root, dataset, seed):
    root = Path(root)
    case = root/'native'/f'{dataset}_seed{seed}'
    if (case/'complete.json').exists():
        return read(case/'complete.json')
    if (case/'selection.json').exists():
        raise RuntimeError('Partial fit retained; explicit separately pinned recovery required')
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        training, audit, k, scale, source_pin = source(dataset, seed)
        past = training[:512]
        old = trace(past, k, seed)[8]
        validation_centers, vr = proposal_banks(training[512:768], old)
        test_centers, tr = proposal_banks(training[768:1024], old)
        bases, fit_seconds = fit_families(past, old, seed)
        ranks = sorted({k-1, 8, 16, 32, 79})
        case.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(case/'arrays.npz', past=past, validation=audit[:128],
                            test=audit[128:256], old=old, validation_centers=validation_centers,
                            test_centers=test_centers, **bases)
        save(case/'inputs_pin.json', dict(dataset=dataset, seed=seed, k=k, scale=scale,
             arrays_sha256=sha(case/'arrays.npz'), training_sha256=array_sha(past),
             validation_sha256=array_sha(audit[:128]), test_sha256=array_sha(audit[128:256]),
             validation_update_source_training_indices=[512, 768],
             test_update_source_training_indices=[768, 1024],
             validation_catalog_indices=[0, 128], test_catalog_indices=[128, 256],
             validation_recipe=vr, test_recipe=tr, fit_seconds=fit_seconds, source=source_pin))
        validation = []
        validation_seconds = {}
        for family, basis in bases.items():
            tick = time.perf_counter()
            for rank in ranks:
                validation += evaluate(basis, old, audit[:128], validation_centers,
                                       scale, family, rank)
            validation_seconds[family] = time.perf_counter()-tick
        save(case/'validation.json', validation)
        selection = select_validation(validation, 128, past.shape[1], k)
        selection['selection_preparation_seconds'] = {
            'ours': sum(fit_seconds[f]+validation_seconds[f] for f in FAMILIES if f != 'pca')
                    + fit_seconds['shared_history_preparation'],
            'pca': fit_seconds['pca']+validation_seconds['pca']}
        selection['frozen_unix'] = time.time()
        selection['arrays_sha256'] = sha(case/'arrays.npz')
        save(case/'selection.json', selection)  # Freeze BEFORE any test evaluation.
        selection_sha256 = sha(case/'selection.json')
        test = []
        for family, basis in bases.items():
            for rank in ranks:
                test += evaluate(basis, old, audit[128:256], test_centers,
                                 scale, family, rank)
        assert sha(case/'selection.json') == selection_sha256
        save(case/'test.json', test)
        result = dict(complete=True, dataset=dataset, seed=seed,
                      selection_sha256=selection_sha256,
                      arrays_sha256=sha(case/'arrays.npz'),
                      validation_records=len(validation), test_records=len(test),
                      all_intervals_contain_truth=True, wrong_conclusive=0,
                      native_only=True, elapsed_seconds=time.perf_counter()-started)
        save(case/'complete.json', result)
        print(json.dumps(result), flush=True)
        return result


def screen(root):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    root = Path(root)
    results, failures = [], []
    with ProcessPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(fit_case, root, d, s): (d, s) for d in DATASETS for s in SEEDS}
        for future in as_completed(futures):
            d, s = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                failures.append(dict(dataset=d, seed=s, error=str(error)))
            save(root/'screen_progress.json', dict(completed=results, failures=failures,
                 expected_fits=len(DATASETS)*len(SEEDS), updated_unix=time.time()))
    summary = dict(complete=not failures, fits=len(results), failures=failures,
                   datasets=DATASETS, seeds=SEEDS, native_only=True)
    save(root/'screen_complete.json', summary)
    if failures:
        raise RuntimeError('Screen has retained failed fits')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    screen(args.root)
