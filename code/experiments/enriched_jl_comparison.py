"""Bounded offline screen: +12 residual directions vs dedicated-rank PCA fits.

Run from code/: python -m experiments.enriched_jl_comparison --out results/...
No 512-component PCA prefix is used. Old campaigns and their source files are
not mutated. Native float64 research only; not a proof benchmark.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import psutil
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from commitjl.enriched_jl import enrich_centroid_basis
from commitjl.protected_jl import centroid_basis
from commitjl.reusable_native import NativeProjection, PreparedDataset, array_hash, file_hash, stable_squared_distances
from commitjl.sparse_gen import sparse_matrix_gen_csr
from experiments.laptop_comparison import atomic_json, save_arrays
from experiments.paper_compression import assignment, bootstrap_crossed, split_indices, stable_topk, timed
from experiments.protected_jl_study import Archive

CODE = Path(__file__).resolve().parents[1]
DATASETS = ['mnist', 'fashion_mnist', 'gisette', '20ng_tfidf_nonzero']
PCA_METHODS = ['pca_full', 'pca_randomized', 'pca_randomized_fast']
HYBRIDS = ['protected_jl', 'protected_plus12']
METRICS = ['fixed_ari', 'fresh_ari', 'validation_fresh_ari', 'test_fresh_ari',
           'knn_recall', 'pair_q95', 'original_partition_sse_ratio',
           'cold_compression_seconds', 'construction_and_compression_seconds',
           'compression_seconds', 'fresh_seconds', 'cold_fresh_seconds',
           'query_seconds', 'cold_query_seconds', 'cold_assignment_seconds']


def pca_spec(method, m, seed):
    if method not in PCA_METHODS:
        raise ValueError('Unknown PCA method')
    options = dict(n_components=int(m), svd_solver='full' if method == 'pca_full' else 'randomized',
                   random_state=90125+max(seed, 0))
    if method == 'pca_randomized_fast':
        options.update(iterated_power=2, n_oversamples=10)
    return options


def sparse_model(d, m, q, seed, centers, basis):
    tail = m-basis.shape[1]
    if tail < q:
        raise ValueError('Total output budget leaves too few JL rows')
    _, a = sparse_matrix_gen_csr(seed, tail, d, q)
    a = a.astype(float); a.data /= np.sqrt(q)
    return NativeProjection('hybrid', d, basis=basis, sparse_map=a, origin=centers[0])


def model_path(root, key):
    return root/'models'/(key+'.npz')


def cached_fit(root, key, function, repetitions):
    """Cache deterministic specification; charge the entire measured fit per use."""
    path = model_path(root, key); meta_path = path.with_suffix('.json')
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if file_hash(path) != meta['sha256']:
            raise ValueError('Corrupted fit checkpoint: '+key)
        with np.load(path, allow_pickle=False) as arrays:
            return {k: arrays[k] for k in arrays.files}, meta
    (arrays, description), measured = timed(function, repetitions, warmup=True)
    save_arrays(path, **arrays)
    meta = dict(description=description, timing=measured, sha256=file_hash(path))
    atomic_json(meta_path, meta)
    return arrays, meta


def init_indices(n, start):
    return np.random.default_rng(900+start).choice(n, 5, replace=False)


def cluster(points, initial, start):
    return KMeans(n_clusters=5, init=points[initial].copy(), n_init=1,
                  max_iter=300, tol=1e-4, algorithm='lloyd', random_state=start).fit(points)


def partition_sse(x, labels, norm_sum):
    weights = (labels[:, None] == np.arange(5)).astype(float)
    counts = weights.sum(axis=0)
    sums = weights.T@x
    keep = counts > 0
    return max(0., float(norm_sum-np.sum(np.sum(sums[keep]**2, axis=1)/counts[keep])))


def checkpoint_records(out):
    return [json.loads(p.read_text()) for p in sorted((out/'rows').glob('*.json'))]


def summarize(out, bootstrap=1000):
    rows = checkpoint_records(out)
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'rows.csv', index=False)
    summaries = []
    keys = ['dataset', 'method', 'm']
    for values, group in frame.groupby(keys):
        record = dict(zip(keys, values), cases=len(group))
        for metric in METRICS:
            grid = group.pivot(index='seed', columns='start', values=metric).to_numpy()
            if np.isfinite(grid).all():
                mean, lo, hi = bootstrap_crossed(grid, draws=bootstrap)
                record.update({metric+'_mean': mean, metric+'_ci_low': lo, metric+'_ci_high': hi})
        record['any_iteration_cap'] = bool(group[['training_hit_cap', 'fresh_hit_cap', 'heldout_hit_cap']].any().any())
        summaries.append(record)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out/'summary.csv', index=False)
    comparisons = []
    for (dataset, m), ours in frame[frame.method.eq('protected_plus12')].groupby(['dataset', 'm']):
        for competitor in PCA_METHODS+['protected_jl']:
            base = frame[(frame.dataset == dataset) & (frame.m == m) & frame.method.eq(competitor)]
            if not len(base):
                continue
            if competitor == 'pca_full':
                merged = ours.merge(base.drop(columns='seed'), on=['dataset', 'm', 'start'], suffixes=('_ours', '_base'), validate='many_to_one')
            else:
                merged = ours.merge(base, on=['dataset', 'm', 'start', 'seed'], suffixes=('_ours', '_base'), validate='one_to_one')
            entry = dict(dataset=dataset, m=int(m), competitor=competitor, cases=len(merged))
            for metric in ['fresh_ari', 'test_fresh_ari', 'validation_fresh_ari', 'knn_recall', 'pair_q95']:
                merged[metric+'_difference'] = merged[metric+'_ours']-merged[metric+'_base']
            for label, metric in [('cold_fresh_speedup', 'cold_fresh_seconds'),
                                  ('cold_compression_speedup', 'cold_compression_seconds'),
                                  ('cold_query_speedup', 'cold_query_seconds'),
                                  ('compression_speedup', 'compression_seconds'),
                                  ('query_speedup', 'query_seconds')]:
                merged[label] = merged[metric+'_base']/merged[metric+'_ours']
            names = [m+'_difference' for m in ['fresh_ari', 'test_fresh_ari', 'validation_fresh_ari', 'knn_recall', 'pair_q95']]
            names += ['cold_fresh_speedup', 'cold_compression_speedup', 'cold_query_speedup', 'compression_speedup', 'query_speedup']
            for metric in names:
                grid = merged.pivot(index='seed', columns='start', values=metric).to_numpy()
                if np.isfinite(grid).all():
                    mean, lo, hi = bootstrap_crossed(grid, draws=bootstrap)
                    entry.update({metric+'_mean': mean, metric+'_ci_low': lo, metric+'_ci_high': hi})
            no_cap = not merged[[c for c in merged if c.endswith('_hit_cap_ours') or c.endswith('_hit_cap_base')]].any().any()
            entry['observed_speed_and_fresh_ari_win'] = bool(no_cap and entry.get('fresh_ari_difference_mean', -1) > 0 and entry.get('cold_fresh_speedup_mean', 0) > 1)
            entry['ci_speed_and_fresh_ari_win'] = bool(no_cap and entry.get('fresh_ari_difference_ci_low', -1) > 0 and entry.get('cold_fresh_speedup_ci_low', 0) > 1)
            entry['ci_speed_and_test_ari_win'] = bool(no_cap and entry.get('test_fresh_ari_difference_ci_low', -1) > 0 and entry.get('cold_fresh_speedup_ci_low', 0) > 1)
            comparisons.append(entry)
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(out/'comparisons.csv', index=False)
    def markdown_table(headers, values):
        return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                         ['| '+' | '.join(str(v) for v in row)+' |' for row in values])
    lines = ['# Extra 12 protected directions: dedicated-dimension PCA comparison', '',
        'Native exploratory screen. Every PCA fit uses its own target m; no 512-component prefix. '+
        'Centroid/residual-basis learning, construction and whole-corpus conversion are charged to cold totals. '+
        'Totals are sums of measured stage medians, with warmup excluded; common input I/O is excluded.', '',
        'Fresh ARI compares new clustering with the corresponding Full run. Fixed-center ARI is a separate endpoint. '+
        'Fresh clustering on validation/test folds is also recorded independently. No hyperparameters are selected from test results.', '',
        'All methods use the same float64 corpus, four threads, point-index KMeans initialization, max_iter=300, '+
        'distance queries and neighbor query IDs. Existing saved subsets are used without further row/feature truncation in a non-smoke run.', '',
        '## All observed method means', '']
    lines.append(markdown_table(['Dataset', 'Method', 'm', 'Fixed ARI', 'Fresh ARI', 'Test fresh ARI', 'Pair Q95', 'Cold compression s', 'Cold fresh s'],
        [[r.dataset, r.method, r.m, f'{r.fixed_ari_mean:.4f}', f'{r.fresh_ari_mean:.4f}', f'{r.test_fresh_ari_mean:.4f}',
          f'{r.pair_q95_mean:.4f}', f'{r.cold_compression_seconds_mean:.4f}', f'{r.cold_fresh_seconds_mean:.4f}'] for r in summary.itertuples()]))
    lines += ['', '## Paired +12 minus competitor', '', 'Positive ARI differences favor +12. Speedup >1 favors +12. The speed includes the whole setup and a fresh KMeans run.', '']
    if len(comparison):
        lines.append(markdown_table(['Dataset', 'm', 'Competitor', 'Fresh ARI change', 'Test ARI change', 'Cold fresh speedup', 'Both lower CI bounds > threshold'],
            [[r.dataset, r.m, r.competitor, f'{r.fresh_ari_difference_mean:+.4f}', f'{r.test_fresh_ari_difference_mean:+.4f}',
              f'{r.cold_fresh_speedup_mean:.3f}', r.ci_speed_and_fresh_ari_win] for r in comparison.itertuples()]))
    lines += ['', '## Limits', '',
        'Intervals resample draw and start axes conditional on fixed corpora. Pointwise exploratory intervals have no multiple-testing correction. '+
        'A screening win is not a confirmatory publication claim. Full inputs, labels, split/query indices, model parameters and timing samples are retained or referenced by hash. '+
        'The extra residual directions depend on private training points: this suite establishes no cryptographic/public-basis privacy guarantee. '+
        'A convergence-cap case cannot count as a win. Source/run manifests state whether the screen is complete.', '']
    (out/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def run(args):
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    for folder in ['rows', 'datasets']:
        (out/folder).mkdir(exist_ok=True)
    configuration = dict(datasets=args.datasets, m=args.m, seeds=args.seeds, starts=args.starts,
        seed_offset=args.seed_offset, start_offset=args.start_offset,
        timing_repetitions=args.timing_repetitions, fit_repetitions=args.fit_repetitions,
        queries=args.queries, threads=args.threads, q=1, extra_directions=12,
        smoke=args.smoke, max_seconds=args.max_seconds, split_seed=481516,
        pca_methods=PCA_METHODS, version='enriched-dedicated-pca-v2')
    sources = [Path(__file__), CODE/'commitjl/enriched_jl.py', CODE/'commitjl/reusable_native.py',
               CODE/'commitjl/protected_jl.py', CODE/'commitjl/sparse_gen.py', CODE/'commitjl/hashing.py',
               CODE/'commitjl/encoding.py', CODE/'experiments/paper_compression.py',
               CODE/'experiments/laptop_comparison.py', CODE/'experiments/protected_jl_study.py']
    identity = dict(configuration=configuration, source_hashes={str(p.relative_to(CODE)):file_hash(p) for p in sources})
    manifest_path = out/'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous['identity'] != identity:
            raise ValueError('Resume requires unchanged source/configuration; choose a new folder')
    elif any((out/'rows').iterdir()):
        raise ValueError('Missing checkpoint identity')
    manifest = dict(identity=identity, complete=False, expected_cases=len(args.datasets)*args.starts*(1+len(args.m)*(1+4*args.seeds)),
        environment=dict(python=sys.version, sklearn=sklearn.__version__, numpy=np.__version__, platform=platform.platform(),
                         total_ram_gib=psutil.virtual_memory().total/2**30, initial_available_ram_gib=psutil.virtual_memory().available/2**30),
        scope='Exploratory native float64 screen; extra 12 directions from training residual; every PCA fitted independently at target m',
        metrics_scope=dict(fresh_ari='Whole preserved corpus, relative to Full with matched point-index initialization',
            test_fresh_ari='Fresh clustering only on disjoint held-out test rows, relative to Full on those rows',
            fixed_ari='Held-out test assignment to original training centers',
            cold_fresh='Training + map construction + whole conversion + fresh clustering; all necessary setup charged'),
        source_io_excluded=True, warmup_excluded=True, uses_largest_rank_pca_prefix=False)
    atomic_json(manifest_path, manifest)
    started = time.monotonic(); process = psutil.Process()
    def budget():
        if time.monotonic()-started > args.max_seconds:
            raise TimeoutError('Time budget reached; complete cases retained')
        if psutil.virtual_memory().available < 384*2**20:
            raise MemoryError('Available RAM below 384 MiB; checkpoints retained')
        if process.memory_info().rss > 1.6*2**30:
            raise MemoryError('Worker exceeded 1.6 GiB RSS budget')
    archive = None if args.smoke else Archive()
    try:
        with threadpool_limits(limits=args.threads):
            for dataset in args.datasets:
                budget(); root = out/'datasets'/dataset; root.mkdir(exist_ok=True)
                if args.smoke:
                    rng = np.random.default_rng(20261015)
                    y = np.arange(180)%5; centers = rng.normal(size=(5, 64))
                    x = rng.normal(size=(180, 64))*.5+centers[y]
                    original_hash = array_hash(x); source = 'synthetic correctness fixture, not corpus results'
                    source_rows = np.arange(len(x))
                    removed_zero_rows = 0
                else:
                    raw_name = '20ng_tfidf' if dataset.endswith('_nonzero') else dataset
                    source = f'results/paper_server_20261006/rq1/{raw_name}/artifacts/dataset.npz'
                    saved = archive.load(source); original = saved['X']; y = saved['y']
                    original_hash = array_hash(original)
                    old = json.loads((CODE/f'results/server_paper_20261006/rq1/{raw_name}/manifest.json').read_text())
                    if original_hash != old['dataset']['sha256']:
                        raise ValueError('Input archive disagrees with original dataset manifest')
                    source_rows = np.flatnonzero(np.any(original != 0, axis=1)) if dataset.endswith('_nonzero') else np.arange(len(original))
                    removed_zero_rows = len(original)-len(source_rows)
                    x = np.ascontiguousarray(original[source_rows], dtype=np.float64)
                    y = y[source_rows] if len(y) else y
                    del original, saved
                x.setflags(write=False); n, d = x.shape
                folds = split_indices(n); train = x[folds['train']]
                if any(m >= min(d, len(train)) for m in args.m):
                    raise ValueError('Requested dimension exceeds centered PCA training rank')
                test_x = x[folds['test']]; validation_x = x[folds['validation']]
                rng = np.random.default_rng(20261101)
                pair = rng.integers(len(test_x), size=(2000, 2)); pair = pair[pair[:, 0] != pair[:, 1]]
                pair_reference = np.sum((test_x[pair[:, 0]]-test_x[pair[:, 1]])**2, axis=1)
                norm_sum = float(np.einsum('ij,ij->', x, x))
                queries = []
                for _ in range(args.queries):
                    take = rng.choice(n, min(512, n//2)+min(64, n-n//2), replace=False)
                    queries.append((take[:min(512, n//2)], take[min(512, n//2):]))
                query_ids = np.sort(np.random.default_rng(188).choice(n, min(128, n), replace=False))
                save_arrays(root/'indices.npz', source_indices=source_rows, **folds, pair_indices=pair,
                    pair_reference=pair_reference, query_ids=query_ids,
                    query_left=np.stack([a for a,b in queries]), query_right=np.stack([b for a,b in queries]))
                atomic_json(root/'dataset.json', dict(n=n, d=d, source=source, original_sha256=original_hash,
                    used_sha256=array_hash(x), training_rows=len(train), validation_rows=len(validation_x), test_rows=len(test_x),
                    removed_zero_rows=removed_zero_rows,
                    zero_rows=int(np.count_nonzero(~np.any(x != 0, axis=1))), smoke=args.smoke))
                print(json.dumps(dict(event='dataset', dataset=dataset, n=n, d=d, available_ram_gib=psutil.virtual_memory().available/2**30)), flush=True)
                for start in range(args.start_offset, args.start_offset+args.starts):
                    budget()
                    training_iterations = []
                    def train_centers():
                        fitted = KMeans(n_clusters=5, n_init=1, random_state=start, max_iter=300,
                                         tol=1e-4, algorithm='lloyd').fit(train)
                        training_iterations.append(int(fitted.n_iter_))
                        return dict(centers=fitted.cluster_centers_), dict(iterations=max(training_iterations),
                            all_iterations=training_iterations.copy())
                    ca, cm = cached_fit(root, f'centers_s{start}', train_centers, args.fit_repetitions)
                    centers = ca['centers']; center_fit = cm['timing']['median']
                    def basis_fit():
                        return dict(basis=centroid_basis(centers)), dict(extra=0)
                    ua, um = cached_fit(root, f'basis_s{start}', basis_fit, args.fit_repetitions)
                    def enriched_fit():
                        u, description = enrich_centroid_basis(train, centers, 12, random_state=1701+start)
                        return dict(basis=u), description
                    ea, em = cached_fit(root, f'enriched_s{start}', enriched_fit, args.fit_repetitions)
                    bases = dict(protected_jl=ua['basis'], protected_plus12=ea['basis'])
                    fit_costs = dict(protected_jl=center_fit+um['timing']['median'],
                                     protected_plus12=center_fit+em['timing']['median'])
                    specs = [('full', d, -1)]
                    for m in args.m:
                        specs += [('pca_full', m, -1)]
                        for method in PCA_METHODS[1:]+HYBRIDS:
                            specs += [(method, m, seed) for seed in range(args.seed_offset, args.seed_offset+args.seeds)]
                    # Full must exist before dependent quality checks; randomize all other execution order.
                    order = np.random.default_rng(7722+start).permutation(len(specs)-1)+1
                    specs = [specs[0]]+[specs[i] for i in order]
                    full_references = {}
                    initial = init_indices(n, start)
                    for method, m, seed in specs:
                        budget(); case = f'{dataset}_s{start}_{method}_m{m}_seed{seed}'
                        target = out/'rows'/(case+'.json'); arr_path = out/'artifacts'/(case+'.npz')
                        if target.exists():
                            record = json.loads(target.read_text())
                            if file_hash(arr_path) != record['artifact_sha256']:
                                raise ValueError('Corrupt completed case')
                            if method == 'full':
                                with np.load(arr_path, allow_pickle=False) as saved:
                                    labels = saved['fresh_labels']
                                    full_references = dict(labels=labels.copy(), neighbors=saved['neighbors'].copy(),
                                        fold_labels={name:saved[name+'_fresh_labels'].copy() for name in ['validation','test']},
                                        sse=partition_sse(x, labels, norm_sum))
                            continue
                        fit_seconds = 0.; fit_spec = None
                        if method == 'full':
                            make = lambda: NativeProjection('full', d)
                        elif method in PCA_METHODS:
                            fit_spec = pca_spec(method, m, seed)
                            def pca_fit():
                                p = PCA(**fit_spec).fit(train)
                                assert p.components_.shape == (m, d)
                                return dict(components=p.components_, mean=p.mean_), dict(options=fit_spec, fitted_n_components=int(p.n_components_))
                            pa, pm = cached_fit(root, f'{method}_m{m}_seed{seed}', pca_fit, args.fit_repetitions)
                            if pm['description']['fitted_n_components'] != m:
                                raise ValueError('PCA must be trained at the requested target dimension')
                            fit_seconds = pm['timing']['median']
                            make = lambda: NativeProjection('pca', d, components=pa['components'], origin=pa['mean'])
                        else:
                            basis = bases[method]; fit_seconds = fit_costs[method]
                            payload = dict(version='enriched-screen-v1', dataset=array_hash(x), basis=array_hash(basis),
                                           m=m, seed=seed, start=start, q=1)
                            map_seed = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).digest()
                            make = lambda: sparse_model(d, m, 1, map_seed, centers, basis)
                        model, construction = timed(make, args.timing_repetitions, warmup=True)
                        if model.m != m:
                            raise ValueError('Wrong total dimension budget')
                        z, conversion = timed(lambda:model.transform(x), args.timing_repetitions, warmup=True)
                        norms, preparation = timed(lambda:np.einsum('ij,ij->i', z, z), args.timing_repetitions, warmup=True)
                        projected_c, cp = timed(lambda:model.transform(centers), args.timing_repetitions, warmup=True)
                        cn, cprep = timed(lambda:np.einsum('ij,ij->i', projected_c, projected_c), args.timing_repetitions, warmup=True)
                        _, at = timed(lambda:assignment(z, projected_c, norms, cn), args.timing_repetitions, warmup=True)
                        projected_fixed = stable_squared_distances(z[folds['test']], projected_c).argmin(axis=1)
                        true_fixed = stable_squared_distances(test_x, centers).argmin(axis=1)
                        fixed_ari = float(adjusted_rand_score(true_fixed, projected_fixed))
                        if method in HYBRIDS and not np.array_equal(projected_fixed, true_fixed):
                            raise ValueError('Protected-center assignment preservation failed')
                        fresh_iterations = []
                        def fit_fresh():
                            fitted = cluster(z, initial, start)
                            fresh_iterations.append(int(fitted.n_iter_))
                            return fitted
                        fitted, km = timed(fit_fresh, args.timing_repetitions, warmup=True)
                        fold_models = {name:cluster(z[ids], init_indices(len(ids), start), start)
                                       for name, ids in folds.items() if name != 'train'}
                        state = PreparedDataset(model, z, norms, array_hash(x))
                        def run_queries():
                            total = 0.
                            for left, right in queries:
                                total += float(state.squared_distances(left, right).sum())
                            return total
                        checksum, qt = timed(run_queries, args.timing_repetitions, warmup=True)
                        distances = state.squared_distances(query_ids, np.arange(n))
                        distances[np.arange(len(query_ids)), query_ids] = np.inf
                        neighbors = stable_topk(distances, 5)
                        del distances
                        projected_pair = np.sum((z[folds['test'][pair[:, 0]]]-z[folds['test'][pair[:, 1]]])**2, axis=1)
                        keep = pair_reference > 0
                        pair_q95 = float(np.quantile(np.abs(projected_pair[keep]/pair_reference[keep]-1), .95))
                        zero_violations = int(np.count_nonzero(projected_pair[~keep] > 1e-12))
                        if zero_violations:
                            raise ValueError('Zero-distance preservation failed')
                        if method == 'full':
                            full_references = dict(labels=fitted.labels_.copy(), neighbors=neighbors.copy(),
                                fold_labels={name:f.labels_.copy() for name,f in fold_models.items()},
                                sse=partition_sse(x, fitted.labels_, norm_sum))
                        if not full_references:
                            raise ValueError('Full reference missing')
                        sse = partition_sse(x, fitted.labels_, norm_sum)
                        compression_seconds = conversion['median'] if method != 'full' else 0.
                        construct_seconds = construction['median'] if method != 'full' else 0.
                        cold_compression = fit_seconds+construct_seconds+compression_seconds
                        timings = dict(construction=construction, compression=conversion, preparation=preparation,
                            center_projection=cp, center_preparation=cprep, assignment=at, fresh=km, queries=qt)
                        record = dict(dataset=dataset, method=method, m=m, seed=seed, start=start, n=n, d=d,
                            extra_directions=12 if method == 'protected_plus12' else 0,
                            protected_rank=model.basis.shape[1] if method in HYBRIDS else 0,
                            residual_m=m-model.basis.shape[1] if method in HYBRIDS else 0,
                            pca_fitted_m=m if method in PCA_METHODS else None,
                            model_fingerprint=model.fingerprint, pca_spec=fit_spec,
                            fixed_ari=fixed_ari, fresh_ari=float(adjusted_rand_score(full_references['labels'], fitted.labels_)),
                            validation_fresh_ari=float(adjusted_rand_score(full_references['fold_labels']['validation'], fold_models['validation'].labels_)),
                            test_fresh_ari=float(adjusted_rand_score(full_references['fold_labels']['test'], fold_models['test'].labels_)),
                            ground_truth_fresh_ari=float(adjusted_rand_score(y, fitted.labels_)) if len(y) else None,
                            knn_recall=float(np.mean([len(set(a)&set(b))/5 for a,b in zip(neighbors, full_references['neighbors'])])),
                            pair_q95=pair_q95, pair_zero_violations=zero_violations,
                            original_partition_sse_ratio=sse/full_references['sse'] if full_references['sse'] > 0 else None,
                            training_hit_cap=bool(method in HYBRIDS and cm['description']['iterations'] >= 300),
                            fresh_hit_cap=bool(max(fresh_iterations) >= 300), heldout_hit_cap=bool(any(f.n_iter_>=300 for f in fold_models.values())),
                            fresh_iterations_all=fresh_iterations,
                            fresh_iterations=int(fitted.n_iter_), model_fit_seconds=fit_seconds,
                            construction_seconds=construct_seconds, compression_seconds=compression_seconds,
                            construction_and_compression_seconds=construct_seconds+compression_seconds,
                            cold_compression_seconds=cold_compression,
                            fresh_seconds=km['median'], cold_fresh_seconds=cold_compression+km['median'],
                            query_seconds=qt['median'], cold_query_seconds=cold_compression+preparation['median']+qt['median'],
                            cold_assignment_seconds=cold_compression+cp['median']+preparation['median']+cprep['median']+at['median']+
                                (center_fit if method not in HYBRIDS else 0),
                            query_checksum=float(checksum), timings=timings)
                        arrays = dict(fresh_labels=fitted.labels_, fresh_centers=fitted.cluster_centers_,
                            fixed_test_labels=projected_fixed, validation_fresh_labels=fold_models['validation'].labels_,
                            test_fresh_labels=fold_models['test'].labels_, neighbors=neighbors,
                            projected_pair_squared_distances=projected_pair, initialization_indices=initial,
                            **{'model_'+name:value for name,value in model.parameters().items()})
                        save_arrays(arr_path, **arrays); record['artifact_sha256'] = file_hash(arr_path)
                        atomic_json(target, record)
                        count = len(list((out/'rows').glob('*.json')))
                        print(json.dumps(dict(event='case', case=case, done=count, expected=manifest['expected_cases'],
                            fresh_ari=record['fresh_ari'], test_ari=record['test_fresh_ari'], cold_fresh=record['cold_fresh_seconds'],
                            elapsed=time.monotonic()-started)), flush=True)
                        if method in PCA_METHODS:
                            del pa
                        del model, z, state, fitted, fold_models, arrays
                        gc.collect()
                    summarize(out, bootstrap=500)
                del x, train, test_x, validation_x
                gc.collect()
        records = checkpoint_records(out)
        if len(records) != manifest['expected_cases']:
            raise ValueError('Case count mismatch')
        pca_records = [r for r in records if r['method'] in PCA_METHODS]
        assert all(r['pca_fitted_m'] == r['m'] for r in pca_records)
        manifest.update(complete=True, completed_cases=len(records), elapsed_seconds=time.monotonic()-started,
            dedicated_rank_pca_check=True, fixed_assignment_violations=0, zero_distance_violations=0)
        summarize(out)
        atomic_json(manifest_path, manifest)
        print(json.dumps(dict(event='complete', cases=len(records), seconds=manifest['elapsed_seconds'], report=str(out/'report.md'))), flush=True)
    except BaseException as error:
        manifest.update(complete=False, completed_cases=len(checkpoint_records(out)),
                        stop_reason=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started)
        atomic_json(manifest_path, manifest)
        summarize(out, bootstrap=500)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--m', nargs='+', type=int, default=[128, 256])
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--starts', type=int, default=3)
    parser.add_argument('--seed-offset', type=int, default=0,
                        help='First draw ID; use new IDs for an independent-draw replication')
    parser.add_argument('--start-offset', type=int, default=0,
                        help='First center/clustering start ID; use new IDs for replication')
    parser.add_argument('--timing-repetitions', type=int, default=3)
    parser.add_argument('--fit-repetitions', type=int, default=3)
    parser.add_argument('--queries', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--max-seconds', type=float, default=1800)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.smoke:
        args.datasets=['mnist']; args.m=[24,32]; args.seeds=2; args.starts=2
        args.timing_repetitions=1; args.fit_repetitions=1; args.queries=2
    if min(args.seeds, args.starts, args.threads, args.timing_repetitions, args.fit_repetitions, args.queries, *args.m) < 1:
        parser.error('All counts must be positive')
    if min(args.seed_offset, args.start_offset) < 0:
        parser.error('Seed and start offsets must be nonnegative')
    run(args)


if __name__ == '__main__':
    main()
