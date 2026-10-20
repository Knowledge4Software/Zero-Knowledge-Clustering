"""Fair amortization screen: prepare once, then run different distance queries.

Existing learned centroids/PCA are reused; training and archive I/O are excluded.
Native only: no claim about ZK proof speed or independent-test generalization.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from commitjl.reusable_native import NativeProjection, PreparedDataset, array_hash
from commitjl.sparse_gen import sparse_matrix_gen_csr
from experiments.laptop_comparison import archived, atomic_json, save_arrays

CODE = Path(__file__).resolve().parents[1]


def quality(reference, values, norms):
    positive = reference > 0
    error = np.abs(values[positive]/reference[positive]-1)
    tolerance = 64*np.finfo(float).eps*max(1., float(norms.max()))
    zero_values = values[~positive]
    return dict(pair_q95=float(np.quantile(error, .95)) if len(error) else 0.,
                pair_cost_ratio=float(values.sum()/reference.sum()),
                zero_reference_pairs=int(np.count_nonzero(~positive)),
                zero_reference_violations=int(np.count_nonzero(zero_values > tolerance)),
                zero_roundoff_max=float(np.max(zero_values, initial=0)), zero_roundoff_tolerance=tolerance)


def report(out, frame, validation):
    lines = ['# Reusing a compressed representation', '',
             'Native fixed-input distance queries, using the existing learned models. All methods prepare their representation/norms once, then receive the same sequence of different query index sets. Original results are unchanged.', '',
             '**Timing scope:** model validation and protected A@U preparation, whole-subset projection, input hashing, and point-norm preparation are included once. Existing PCA fitting and centroid training, archive reading, cache disk I/O, quality evaluation, and artifact export are excluded. This is an exploratory deployment amortization screen using already learned models, not training-inclusive end-to-end timing or a ZK benchmark.', '',
             'Every method uses the same float64 BLAS distance formula, four threads, and three timing passes with shuffled method order. Representation-only caching and complete distance-matrix caching are tested separately. Complete distance caching uses n² × 8 bytes for every method, so subsequent queries mostly perform identical table lookups.', '',
             '| Dataset | Cache policy | Method | d → m | Startup ms | Total at 1 query ms | Total at final query count ms | Speedup vs Full at final count | First observed faster checkpoint |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for (dataset, policy, method), g in frame.groupby(['dataset', 'policy', 'method']):
        first, last = g.sort_values('queries').iloc[[0,-1]].to_dict('records')
        faster = g[g.speedup_vs_full > 1].queries.min() if method != 'full' else None
        marker = '-' if faster is None or pd.isna(faster) else str(int(faster))
        lines.append(f"| {dataset} | {policy} | {method} | {int(last['d'])} → {int(last['m'])} | {last['startup_seconds']*1000:.2f} | {first['total_seconds']*1000:.2f} | {last['total_seconds']*1000:.2f} | {last['speedup_vs_full']:.3f} | {marker} |")
    lines += ['', '## Geometry on the first fresh query', '',
              '| Dataset | Method | Pair squared-distance Q95 error | Sum-distance ratio vs Full |',
              '|---|---|---:|---:|']
    for item in validation['quality']:
        lines.append(f"| {item['dataset']} | {item['method']} | {item['pair_q95']:.4f} | {item['pair_cost_ratio']:.4f} |")
    lines += ['', '## Reading the results', '',
              '- A speedup above one means lower cumulative time than Full at that query count, with the listed one-time startup included. First observed checkpoint is measured, not a universal crossover estimate.',
              '- The mathematical model is T(Q) = T_prepare + sum(T_query). Reusing Z removes repeated projection; it does not ensure that compression always wins.',
              '- PCA also reuses its learned model, projected data and norms. Full reuses its input data and norms, and receives the same optional complete-distance cache.',
              '- These queries use changing point/landmark subsets, not repeats of an identical answer. If the task and input are identical, all methods can cache the answer itself.',
              '- Cached models and Z are bound to input/model hashes. New points must be transformed; new projection parameters require a new cache. A new cryptographic challenge/proof instance cannot reuse a projection from another challenge.',
              '- The whole original paper subset is projected once. Each query computes a bounded point-by-landmark distance matrix; timing includes gathering the selected rows.',
              '- Native BLAS subtraction can leave tiny positive distances for identical vectors. Zero-distance checks use the recorded conservative roundoff tolerance, and self-distances are explicitly zero.',
              '- Cached representations reduce stored coordinate bytes, not automatically total RAM: this study retains original X for reference checks. Projection fitting quality and held-out generalization need their own confirmation experiment.', '',
              f"Complete: {validation['complete']}; queries per sequence: {validation['configuration']['queries']}; repetitions: {validation['configuration']['repetitions']}; wall seconds: {validation['wall_seconds']:.1f}.", '']
    (out/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    datasets = frame.dataset.unique()
    fig, axes = plt.subplots(2, len(datasets), figsize=(5*len(datasets), 8), squeeze=False)
    for column, dataset in enumerate(datasets):
        for index, policy in enumerate(('representation', 'all_distances')):
            ax = axes[index,column]
            for method, rows in frame[(frame.dataset == dataset)&(frame.policy == policy)].groupby('method'):
                ax.plot(rows.queries, rows.total_seconds*1000, marker='o', label=method)
            ax.set_title(f'{dataset}: {policy}'); ax.set_xlabel('Number of different queries')
            ax.set_ylabel('Cumulative ms including startup'); ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(out/'amortization.png', dpi=160); plt.close(fig)


def run(args):
    out = args.out.resolve()
    if out.exists():
        raise ValueError('Use a fresh output directory')
    out.mkdir(parents=True)
    began = time.perf_counter()
    validation = dict(complete=False, configuration={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      kernel_sha256=hashlib.sha256((CODE/'commitjl/reusable_native.py').read_bytes()).hexdigest(),
                      inputs={}, cache_replays=[], quality=[])
    (out/'sources').mkdir()
    for path in (Path(__file__), CODE/'commitjl/reusable_native.py'):
        (out/'sources'/path.name).write_bytes(path.read_bytes())
    all_records, all_samples = [], []
    for dataset in args.datasets:
        archive, prefix, data, pca = archived(dataset)
        x = data['X']; x.setflags(write=False)
        source_hash = array_hash(x)
        # Frozen research configuration: use the previously selected k5, start3,
        # fresh-map100 row at total m256 for these three datasets.
        row_path = args.parent/'jobs'/f'geometry_{dataset}'/'rows/k5_s3_m256_r4_q1_seed100.json'
        parent = json.loads(row_path.read_text())
        if source_hash != parent['dataset_sha256']:
            raise ValueError('Parent dataset mismatch')
        with np.load(args.parent/'jobs'/f'geometry_{dataset}'/'centers_k5_start3.npz', allow_pickle=False) as saved:
            u, origin = saved['basis'], saved['centroids'][0]
        model_times = {}
        models = {}
        begin = time.perf_counter(); models['full'] = NativeProjection('full', x.shape[1]); model_times['full'] = time.perf_counter()-begin
        begin = time.perf_counter(); models['pca'] = NativeProjection('pca', x.shape[1], components=pca[:256]); model_times['pca'] = time.perf_counter()-begin
        begin = time.perf_counter()
        _, a = sparse_matrix_gen_csr(bytes.fromhex(parent['map_seed']), parent['residual_m'], parent['d'], parent['q'])
        a = a.astype(float); a.data /= np.sqrt(parent['q'])
        models['hybrid'] = NativeProjection('hybrid', x.shape[1], basis=u, sparse_map=a, origin=origin)
        model_times['hybrid'] = time.perf_counter()-begin
        states, startup = {}, {}
        for method in np.random.default_rng(884).permutation(list(models)):
            begin = time.perf_counter(); states[method] = models[method].prepare(x); elapsed = time.perf_counter()-begin
            startup[method] = model_times[method]+elapsed
        validation['inputs'][dataset] = dict(shape=list(x.shape), sha256=source_hash,
            parent=str(row_path), startup=startup, model_preparation_seconds=model_times,
            representation_bytes={method:state.z.nbytes for method,state in states.items()},
            full_distance_cache_bytes=len(x)**2*8)
        rng = np.random.default_rng(112233)
        left_n = min(args.points, len(x)-args.landmarks)
        right_n = min(args.landmarks, len(x)-left_n)
        if min(left_n, right_n) < 1:
            raise ValueError('Insufficient rows for query sizes')
        queries = [rng.choice(len(x), left_n+right_n, replace=False) for _ in range(args.queries)]
        query_i = np.stack([q[:left_n] for q in queries]); query_j = np.stack([q[left_n:] for q in queries])
        save_arrays(out/'arrays'/f'{dataset}_queries.npz', left_indices=query_i, right_indices=query_j)
        for method in states:
            states[method].squared_distances(query_i[0], query_j[0])
        for policy in ('representation', 'all_distances'):
            setup = dict(startup)
            if policy == 'all_distances':
                if len(x)**2*8 > args.distance_budget_mib*1024**2:
                    raise MemoryError('Complete distance cache exceeds configured budget')
                for method, state in states.items():
                    begin = time.perf_counter(); state.precompute_distances(args.distance_budget_mib*1024**2)
                    setup[method] += time.perf_counter()-begin
            samples = {method:[] for method in states}
            order_rng = np.random.default_rng(2323)
            for repetition in range(args.repetitions):
                measured = {method:[] for method in states}
                for query in range(args.queries):
                    answers = {}
                    for method in order_rng.permutation(list(states)):
                        begin = time.perf_counter()
                        answers[method] = states[method].squared_distances(query_i[query], query_j[query])
                        seconds = time.perf_counter()-begin
                        measured[method].append(seconds)
                        all_samples.append(dict(dataset=dataset, policy=policy, method=method,
                                                repetition=repetition, query=query, seconds=seconds))
                    if repetition == 0 and policy == 'representation':
                        save_arrays(out/'arrays'/f'{dataset}_query{query:03d}.npz', **answers)
                        if query == 0:
                            for method, state in states.items():
                                quality_row = dict(dataset=dataset, method=method, **quality(answers['full'], answers[method], state.norms))
                                if quality_row['zero_reference_violations']:
                                    raise ValueError('Zero-distance geometry violation beyond roundoff')
                                validation['quality'].append(quality_row)
                    if repetition == 0 and policy == 'all_distances':
                        with np.load(out/'arrays'/f'{dataset}_query{query:03d}.npz', allow_pickle=False) as saved:
                            for method in states:
                                tolerance = 64*np.finfo(float).eps*max(1., float(states[method].norms.max()))
                                np.testing.assert_allclose(answers[method], saved[method], rtol=1e-10, atol=tolerance)
                for method in states:
                    samples[method].append(np.cumsum(measured[method]))
            checkpoints = sorted(set([1]+[q for q in (2,4,8,16,32,64) if q <= args.queries]+[args.queries]))
            for method in states:
                for count in checkpoints:
                    totals = [setup[method]+float(sequence[count-1]) for sequence in samples[method]]
                    all_records.append(dict(dataset=dataset, policy=policy, method=method,
                        n=len(x), d=x.shape[1], m=states[method].z.shape[1], queries=count,
                        left_points=left_n, right_landmarks=right_n, pairs_per_query=left_n*right_n,
                        startup_seconds=setup[method], total_seconds=statistics.median(totals),
                        total_seconds_min=min(totals), total_seconds_max=max(totals),
                        total_samples=json.dumps(totals)))
            for state in states.values(): state.distance_cache = None
        # Persist compressed representations; Full already has its exact archived
        # X locally. Cache writing/loading is recorded separately from compute.
        for method in ('hybrid', 'pca'):
            state = states[method]
            cache = out/'cache'/f'{dataset}_{method}'
            begin = time.perf_counter(); state.save(cache); save_seconds = time.perf_counter()-begin
            begin = time.perf_counter()
            restored = PreparedDataset.load(cache, expected_source_sha256=source_hash, expected_model_sha256=models[method].fingerprint)
            load_seconds = time.perf_counter()-begin
            np.testing.assert_array_equal(restored.z, state.z)
            np.testing.assert_allclose(restored.squared_distances(query_i[0], query_j[0]), state.squared_distances(query_i[0], query_j[0]), rtol=1e-10, atol=1e-8)
            validation['cache_replays'].append(dict(dataset=dataset, method=method, replay_ok=True,
                                                    save_seconds=save_seconds, load_seconds=load_seconds))
            del restored
        pd.DataFrame(all_samples).to_csv(out/'query_timings.csv', index=False)
        atomic_json(out/'validation.json', validation)
        print(json.dumps(dict(dataset=dataset, complete=True, seconds=time.perf_counter()-began)), flush=True)
        del states, models, data, x, pca, archive
        gc.collect()
    frame = pd.DataFrame(all_records)
    for (_, _, count), group in frame.groupby(['dataset', 'policy', 'queries']):
        full = float(group[group.method == 'full'].total_seconds.iloc[0])
        frame.loc[group.index, 'speedup_vs_full'] = full/group.total_seconds
    frame.to_csv(out/'cumulative_timings.csv', index=False)
    validation.update(complete=True, wall_seconds=time.perf_counter()-began,
                      timing_rows=len(frame), cache_replay_count=len(validation['cache_replays']))
    report(out, frame, validation)
    atomic_json(out/'validation.json', validation)
    print(json.dumps(dict(complete=True, seconds=validation['wall_seconds'], cache_replays=validation['cache_replay_count'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--parent', type=Path, default=Path('results/laptop_comparison_20261011'))
    parser.add_argument('--datasets', nargs='+', choices=['gisette', '20ng_tfidf', 'mnist'], default=['gisette', '20ng_tfidf', 'mnist'])
    parser.add_argument('--queries', type=int, default=32)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--points', type=int, default=1024)
    parser.add_argument('--landmarks', type=int, default=128)
    parser.add_argument('--distance-budget-mib', type=int, default=200)
    args = parser.parse_args()
    if min(args.queries, args.repetitions, args.points, args.landmarks, args.distance_budget_mib) < 1 or args.queries > 64:
        parser.error('Positive bounded settings required; at most 64 queries')
    with threadpool_limits(limits=4): run(args)


if __name__ == '__main__':
    main()
