"""Resumable fixed-center RQ1: canonical sparse JL and native baselines.

Run from code/: python -m experiments.rq1_complete --profile
               python -m experiments.rq1_complete --run
This is native fidelity, not a ZK proof or projected KMeans training campaign.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[1]
METHODS = ['sparse_jl', 'gaussian', 'rademacher', 'achlioptas', 'coordinate', 'pca', 'pisot_surrogate']
DETERMINISTIC = {'full', 'pca'}
DATASETS = {
    'gaussian_easy': ('gaussian_mixture', dict(n=2000, d=256, k=5, gamma=3., seed=12345)),
    'gaussian_hard': ('gaussian_mixture', dict(n=2000, d=512, k=8, gamma=2., imbalance=2., seed=12345)),
    'dbscan_margin': ('dbscan_margin', dict(n=1000, d=128, rho=1., eps=.15, seed=12345)),
    'mnist': ('mnist', dict(n_max=5000, seed=12345)),
    'fashion_mnist': ('fashion_mnist', dict(n_max=5000, seed=12345)),
    'gisette': ('gisette', dict(n_max=3000, seed=12345)),
    'arcene': ('arcene', dict(n_max=100, seed=12345)),
    '20ng_tfidf': ('20ng_tfidf', dict(n_max=3000, max_features=2000, seed=12345)),
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')
    os.replace(tmp, path)


def append_row(path, row):
    # One writer per dataset. A truncated final line is discarded on resume.
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()


def read_rows(path):
    if not Path(path).exists():
        return []
    raw = Path(path).read_bytes()
    if raw and not raw.endswith(b'\n'):
        raw = raw[:raw.rfind(b'\n') + 1]
        Path(path).write_bytes(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def row_key(row):
    return (row['method'], row['k'], row['m'], row['q'],
            row['projection_seed'], row['clustering_start'])


def save_npz(path, **arrays):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, path)


def bootstrap_crossed(values, draws=2000, seed=9876):
    """Independent resampling of crossed projection-seed and start axes.

    Conditional on one fixed dataset/subset; no cross-dataset generalization.
    """
    import numpy as np
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('bootstrap requires a complete finite seed x start matrix')
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for b in range(draws):
        i = rng.integers(len(values), size=len(values))
        j = rng.integers(values.shape[1], size=values.shape[1])
        samples[b] = values[np.ix_(i, j)].mean()
    lo, hi = np.quantile(samples, [.025, .975])
    return float(values.mean()), float(lo), float(hi)


def summarize(path, cfg):
    import pandas as pd
    rows = read_rows(path)
    if not rows:
        return
    df = pd.DataFrame(rows)
    summaries = []
    for keys, group in df.groupby(['method', 'k', 'm', 'q']):
        method, k, m, q = keys
        expected = cfg['starts'] * (1 if method in DETERMINISTIC else cfg['seeds'])
        expected_seeds = {-1} if method in DETERMINISTIC else set(range(cfg['seeds']))
        axes_complete = set(group.projection_seed) == expected_seeds and set(group.clustering_start) == set(range(cfg['starts']))
        entry = dict(method=method, k=int(k), m=int(m), q=int(q), rows=len(group),
                     complete=len(group) == expected and axes_complete, n=int(group.n.iloc[0]), d=int(group.d.iloc[0]),
                     projection_matrices=int(group.projection_seed.nunique()),
                     clustering_starts=int(group.clustering_start.nunique()), reduced_dimension=int(m) < int(group.d.iloc[0]))
        for metric in ('ari', 'nmi', 'r_phi', 'fixed_center_cost_ratio', 'delta_median',
                       'delta_q95', 'delta_q99', 'delta_max', 'delta_exceedance',
                       'agreement', 'margin_coverage', 'qualified_coverage',
                       'uncertified_stability', 'ari_ground_truth', 'nmi_ground_truth'):
            if metric not in group or group[metric].isna().any():
                continue
            matrix = group.pivot(index='projection_seed', columns='clustering_start', values=metric)
            if entry['complete'] and not matrix.isna().any().any():
                mean, lo, hi = bootstrap_crossed(matrix.to_numpy(), cfg['bootstrap'])
                entry.update({metric + '_mean': mean, metric + '_ci_low': lo, metric + '_ci_high': hi})
        entry['eligible'] = bool(entry['complete'] and method not in {'full', 'pisot_surrogate'}
            and (method in DETERMINISTIC or cfg['seeds'] >= 2) and cfg['starts'] >= 2
            and entry.get('ari_ci_low', -1) >= cfg['tau_ari']
            and entry.get('r_phi_ci_high', 100) <= cfg['tau_phi'])
        summaries.append(entry)
    summary = pd.DataFrame(summaries)
    summary.to_csv(Path(path).with_name('summary.csv'), index=False)
    selected = summary[summary.eligible].sort_values(['k', 'method', 'm', 'q']).groupby(['k', 'method']).head(1)
    selected.to_csv(Path(path).with_name('selected_operating_points.csv'), index=False)
    df.to_parquet(Path(path).with_suffix('.parquet'), index=False)


def worker(name, cfg, directory):
    import numpy as np
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits
    from data.loaders import load_dataset
    from commitjl.sparse_gen import sparse_matrix_gen_csr
    from commitjl.clustering import _squared_distances
    from commitjl.metrics import sample_pair_set, delta_uv, distortion_summary, clustering_agreement
    from experiments.pipeline import _centroids_from_labels
    from experiments.pipeline import build_projection
    from sklearn.decomposition import PCA

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    loader, kwargs = DATASETS[name]
    kwargs = dict(kwargs)
    if cfg['scope'] == 'full' and 'n_max' in kwargs:
        kwargs['n_max'] = None
    data = load_dataset(loader, **kwargs)
    if data is None:
        raise RuntimeError(f'{name} unavailable; no silent skipping')
    X = np.asarray(data['X'], dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError('nonfinite dataset')
    n, d = X.shape
    manifest = dict(config=cfg, dataset=data['meta'], loader_kwargs=kwargs,
                    subset='all loader rows' if kwargs.get('n_max') is None else 'fixed random subset',
                    workload='fixed_centroid_assignment', normalization='method-specific; sparse A/sqrt(q)',
                    pisot_status='unverified repository surrogate; not cited author artifact',
                    pair_seed=20261005, source_sha256=cfg['source_sha256'])
    manifest_path = directory / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Existing run differs in config/source/data; choose a new output folder')
    write_json(manifest_path, manifest)
    planned = []
    for k in cfg['k']:
        planned.append(dict(method='full', k=k, m=d, q=0, expected_rows=cfg['starts'], exclusion=None))
        for method in cfg.get('methods', ['sparse_jl']):
            for m in cfg['m']:
                for q in (cfg['q'] if method == 'sparse_jl' else [0]):
                    reason = ('m exceeds original d' if m > d else
                              'PCA m exceeds centered sample rank' if method == 'pca' and m > min(n - 1, d) else
                              'q exceeds m' if q > m else None)
                    expected = cfg['starts'] * (1 if method in DETERMINISTIC else cfg['seeds'])
                    planned.append(dict(method=method, k=k, m=m, q=q, expected_rows=expected, exclusion=reason))
    write_json(directory / 'grid_inventory.json', planned)
    path = directory / 'rows.jsonl'
    done = {row_key(row) for row in read_rows(path)}
    pairs = sample_pair_set(n, cfg['pairs'], 20261005)
    pair_array = np.asarray(pairs, dtype=np.int64)
    save_npz(directory / 'artifacts/dataset.npz', X=X,
             y=np.asarray(data.get('y')) if data.get('y') is not None else np.array([]), pairs=pair_array)
    baseline_seconds = projection_seconds = evaluation_seconds = artifact_seconds = pca_seconds = 0.
    method_seconds = {}
    t_compute = time.perf_counter()
    with threadpool_limits(limits=cfg['threads']):
        pca_components = None
        if 'pca' in cfg.get('methods', ['sparse_jl']):
            tpca = time.perf_counter()
            rank = min(max(cfg['m']), n - 1, d)
            pca = PCA(n_components=rank, svd_solver='randomized', random_state=20261005)
            pca.fit(X)
            pca_components = pca.components_
            save_npz(directory / 'artifacts/pca.npz', components=pca_components,
                     mean=pca.mean_, explained_variance=pca.explained_variance_)
            pca_seconds = time.perf_counter() - tpca
        for k in cfg['k']:
            if k > n:
                raise ValueError(f'k={k} exceeds n={n}')
            baselines = []
            for start in range(cfg['starts']):
                tb = time.perf_counter()
                km = KMeans(n_clusters=k, n_init=1, random_state=start, algorithm='lloyd').fit(X)
                orig_sq = _squared_distances(X, km.cluster_centers_)
                labels = orig_sq.argmin(axis=1)
                phi = float(orig_sq.min(axis=1).sum())
                if phi <= 0:
                    raise ValueError('zero baseline objective; objective ratio undefined')
                sorted_sq = np.sort(orig_sq, axis=1)
                margin = sorted_sq[:, 1] > ((1 + cfg['eps']) / (1 - cfg['eps'])) * sorted_sq[:, 0]
                baselines.append((km.cluster_centers_, labels, phi, orig_sq, margin))
                save_npz(directory / f'artifacts/full_k{k}_start{start}.npz',
                         centroids=km.cluster_centers_, labels=labels,
                         squared_distances=orig_sq, margin=margin)
                baseline_seconds += time.perf_counter() - tb
                row = dict(method='full', k=k, m=d, q=0, projection_seed=-1, clustering_start=start,
                           n=n, d=d, ari=1., nmi=1., agreement=1., r_phi=1., delta_q95=0.,
                           delta_median=0., delta_q99=0., delta_max=0., delta_exceedance=0.,
                           fixed_center_cost_ratio=1., margin_coverage=float(margin.mean()),
                           qualified_coverage=1., uncertified_stability=1., qualified_violations=0,
                           identical_pairs=int(np.isnan(delta_uv(X, X, pairs)).sum()), dataset_sha256=data['meta']['sha256'])
                if data.get('y') is not None:
                    truth = clustering_agreement(data['y'], labels)
                    row.update(ari_ground_truth=truth['ari'], nmi_ground_truth=truth['nmi'])
                if row_key(row) not in done:
                    append_row(path, row)
                    done.add(row_key(row))
            configurations = [(method, m, q, seed)
                for method in cfg.get('methods', ['sparse_jl']) for m in cfg['m']
                if m <= d and (method != 'pca' or m <= min(n - 1, d))
                for q in (cfg['q'] if method == 'sparse_jl' else [0]) if q <= m
                for seed in ([-1] if method in DETERMINISTIC else range(cfg['seeds']))]
            for method, m, q, seed in configurations:
                timings = method_seconds.setdefault(method, dict(projection=0., evaluation=0., artifact=0.))
                artifact = directory / f'artifacts/{method}_k{k}_m{m}_q{q}_seed{seed}.npz'
                if artifact.exists() and all((method, k, m, q, seed, start) in done for start in range(cfg['starts'])):
                    continue
                tp = time.perf_counter()
                if method == 'sparse_jl':
                    seed_bytes = hashlib.sha256(b'CommitJL-RQ1-native-v2' + seed.to_bytes(8, 'little')).digest()
                    records, P = sparse_matrix_gen_csr(seed_bytes, m, d, q)
                    P = P.astype(np.float64)
                    P.data /= np.sqrt(q)
                    del records
                elif method == 'pca':
                    P = pca_components[:m]
                else:
                    projected_method = 'pisot' if method == 'pisot_surrogate' else method
                    P = build_projection(projected_method, X, m, max(q, 1), seed, 2**16)['real']
                Z = np.asarray((P @ X.T).T)
                projection_artifact = directory / f'artifacts/maps/{method}_m{m}_q{q}_seed{seed}.npz'
                if not projection_artifact.exists():
                    save_npz(projection_artifact, Z=Z)
                distortion = delta_uv(X, Z, pairs)
                ds = distortion_summary(distortion, cfg['eps'])
                project_elapsed = time.perf_counter() - tp
                projection_seconds += project_elapsed
                timings['projection'] += project_elapsed
                pending_rows = []
                point_labels, point_errors, original_costs, projected_costs = [], [], [], []
                for start, (C, labels, phi, orig_sq, margin) in enumerate(baselines):
                    key = (method, k, m, q, seed, start)
                    te = time.perf_counter()
                    Cp = np.asarray((P @ C.T).T)
                    sq = _squared_distances(Z, Cp)
                    assigned = sq.argmin(axis=1)
                    centers = _centroids_from_labels(X, assigned, k)
                    assigned_phi = float(_squared_distances(X, centers).min(axis=1).sum())
                    agrees = clustering_agreement(labels, assigned)
                    relative = np.divide(sq, orig_sq, out=np.ones_like(sq), where=orig_sq > 0) - 1
                    relative[(orig_sq == 0) & (sq != 0)] = np.inf
                    event = np.all((np.abs(relative) <= cfg['eps']) & ((orig_sq > 0) | (sq == 0)), axis=1)
                    qualified = margin & event
                    stable = labels == assigned
                    point_labels.append(assigned)
                    point_errors.append(np.abs(relative).max(axis=1))
                    original_costs.append(orig_sq[np.arange(n), assigned])
                    projected_costs.append(sq[np.arange(n), assigned])
                    row = dict(method=method, k=k, m=m, q=q, projection_seed=seed,
                        clustering_start=start, n=n, d=d, **agrees, r_phi=assigned_phi / phi,
                        fixed_center_cost_ratio=float(orig_sq[np.arange(n), assigned].sum() / phi),
                        delta_median=ds['median'], delta_q95=ds['q95'], delta_q99=ds['q99'], delta_max=ds['max'],
                        delta_exceedance=ds['exceedance_rate'], identical_pairs=int(np.isnan(distortion).sum()),
                        margin_coverage=float(margin.mean()), qualified_coverage=float(qualified.mean()),
                        qualified_violations=int(np.sum(qualified & ~stable)),
                        margin_count=int(margin.sum()), qualified_count=int(qualified.sum()),
                        stable_count=int(stable.sum()),
                        uncertified_stability=float(stable[~qualified].mean()) if (~qualified).any() else None,
                        global_embedding_event=bool(event.all()),
                        dataset_sha256=data['meta']['sha256'],
                        projection_artifact=projection_artifact.relative_to(directory).as_posix(),
                        pointwise_artifact=artifact.relative_to(directory).as_posix(),
                        baseline_artifact=f'artifacts/full_k{k}_start{start}.npz',
                        seed_construction='SHA256 domain-separated canonical SparseMatrixGen' if method == 'sparse_jl' else 'repository projection implementation',
                        original_objective=phi, projected_solution_objective=assigned_phi)
                    row['projection_stage_seconds_shared'] = project_elapsed
                    if data.get('y') is not None:
                        truth = clustering_agreement(data['y'], assigned)
                        row.update(ari_ground_truth=truth['ari'], nmi_ground_truth=truth['nmi'])
                    if not np.isfinite([row['r_phi'], row['ari'], row['delta_q95']]).all():
                        raise ValueError('undefined fidelity metric; inspect duplicates/pair set')
                    if key not in done:
                        pending_rows.append(row)
                    elapsed = time.perf_counter() - te
                    row['evaluation_seconds'] = elapsed
                    evaluation_seconds += elapsed
                    timings['evaluation'] += elapsed
                ta = time.perf_counter()
                save_npz(artifact, labels=np.asarray(point_labels, dtype=np.int32),
                         max_point_centroid_distortion=np.asarray(point_errors),
                         assigned_original_sq=np.asarray(original_costs),
                         assigned_projected_sq=np.asarray(projected_costs), pair_delta=distortion)
                for row in pending_rows:
                    append_row(path, row)
                    done.add(row_key(row))
                artifact_elapsed = time.perf_counter() - ta
                artifact_seconds += artifact_elapsed
                timings['artifact'] += artifact_elapsed
                print(f'{name}: k={k} {method} m={m} q={q} seed={seed}', flush=True)
                del P, Z
            del baselines
            gc.collect()
    compute_seconds = time.perf_counter() - t_compute
    summarize(path, cfg)
    write_json(directory / 'completion.json', dict(n=n, d=d, rows=len(done),
        total_seconds=time.perf_counter() - t0, compute_seconds=compute_seconds,
        baseline_seconds=baseline_seconds, projection_seconds=projection_seconds,
        evaluation_seconds=evaluation_seconds, artifact_seconds=artifact_seconds, pca_seconds=pca_seconds,
        method_seconds=method_seconds))
    print(f'{name}: {len(done)} rows, {time.perf_counter()-t0:.1f}s', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--profile', action='store_true', help='all selected datasets, 2 projection seeds x 2 starts')
    ap.add_argument('--datasets', nargs='+', choices=list(DATASETS), default=list(DATASETS))
    ap.add_argument('--scope', choices=['paper', 'full'], default='paper')
    ap.add_argument('--methods', nargs='+', choices=METHODS, default=[m for m in METHODS if m != 'pisot_surrogate'])
    ap.add_argument('--out', type=Path)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--starts', type=int, default=5)
    ap.add_argument('--k', nargs='+', type=int, default=[5, 10])
    ap.add_argument('--m', nargs='+', type=int, default=[16, 32, 64, 128, 256, 512, 1024])
    ap.add_argument('--q', nargs='+', type=int, default=[1, 2, 4])
    ap.add_argument('--pairs', type=int, default=2000)
    ap.add_argument('--bootstrap', type=int, default=2000)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--jobs', type=int, default=4, help='simultaneous dataset processes; server: 4 jobs x 4 threads')
    ap.add_argument('--max-rss-gib', type=float, default=8.)
    ap.add_argument('--dataset-timeout-hours', type=float, default=48.)
    ap.add_argument('--worker', choices=list(DATASETS), help=argparse.SUPPRESS)
    ap.add_argument('--worker-config', type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        worker(args.worker, json.loads(args.worker_config.read_text()), args.out)
        return
    args.datasets = list(dict.fromkeys(args.datasets))
    if min(args.seeds, args.starts, args.pairs, args.bootstrap, args.threads, args.jobs, *args.k, *args.m, *args.q) < 1:
        ap.error('counts/dimensions must be positive')
    if args.max_rss_gib <= 0 or args.dataset_timeout_hours <= 0:
        ap.error('resource limits must be positive')
    if min(args.k) < 2:
        ap.error('certificate diagnostics require k >= 2')
    cfg = dict(scope=args.scope, seeds=2 if args.profile else args.seeds,
        starts=2 if args.profile else args.starts, k=sorted(set(args.k)),
        m=sorted(set(args.m)), q=sorted(set(args.q)), pairs=args.pairs,
        bootstrap=args.bootstrap, threads=args.threads, methods=list(dict.fromkeys(args.methods)),
        pca_solver='randomized', version='rq1_all_methods_v3', eps=.15, tau_ari=.80, tau_phi=1.10)
    from experiments.common import source_fingerprint
    cfg['source_sha256'] = source_fingerprint()
    out = args.out or CODE / 'results/rq1/runs' / ('balanced_profile_v3' if args.profile else 'paper_balanced_v3')
    print(json.dumps(dict(datasets=args.datasets, config=cfg, output=str(out)), indent=2))
    if not (args.run or args.profile):
        print('Plan only. Add --profile for measured resource estimates or --run for the full campaign.')
        return
    import psutil
    out.mkdir(parents=True, exist_ok=True)
    from experiments.common import campaign_lock
    lock_handle = campaign_lock(out)
    config_path = out / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise ValueError('Output config changed; select a new --out directory')
    write_json(config_path, cfg)
    measurements = []
    import platform
    import importlib.metadata
    import threading
    active_processes = {}
    process_lock = threading.Lock()
    stop_event = threading.Event()
    write_json(out / 'environment.json', dict(python=sys.version, platform=platform.platform(),
        cpu_logical=psutil.cpu_count(), cpu_physical=psutil.cpu_count(logical=False),
        system_ram_gib=psutil.virtual_memory().total / 2**30,
        jobs=args.jobs, threads=args.threads, rss_limit_per_job_gib=args.max_rss_gib,
        versions={name: importlib.metadata.version(name) for name in
                  ['numpy', 'scipy', 'scikit-learn', 'pandas', 'pyarrow']}))
    def measure_dataset(name):
        if stop_event.is_set():
            return dict(dataset=name, status='cancelled', peak_rss_gib=0., wall_seconds=0.)
        directory = out / name
        directory.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONUNBUFFERED='1')
        for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[variable] = str(args.threads)
        started = time.perf_counter()
        peak = 0
        status = 'ok'
        with (directory / 'run.log').open('a', encoding='utf-8') as log:
            proc = subprocess.Popen([sys.executable, '-m', 'experiments.rq1_complete', '--worker', name,
                '--worker-config', str(config_path.resolve()), '--out', str(directory.resolve())],
                cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT)
            with process_lock:
                active_processes[name] = proc
            try:
                process = psutil.Process(proc.pid)
                while proc.poll() is None:
                    try:
                        rss = sum(p.memory_info().rss for p in [process] + process.children(recursive=True))
                        peak = max(peak, rss)
                    except psutil.Error:
                        pass
                    if peak > args.max_rss_gib * 2**30:
                        status = 'memory_limit'
                    if time.perf_counter() - started > args.dataset_timeout_hours * 3600:
                        status = 'timeout'
                    if stop_event.is_set():
                        status = 'interrupted'
                    if status != 'ok':
                        for child in process.children(recursive=True):
                            try:
                                child.kill()
                            except psutil.Error:
                                pass
                        proc.kill()
                        break
                    time.sleep(.1)
            except KeyboardInterrupt:
                proc.kill()
                proc.wait()
                raise
            proc.wait()
            with process_lock:
                active_processes.pop(name, None)
        if proc.returncode and status == 'ok':
            status = 'failed'
        measurement = dict(dataset=name, status=status, peak_rss_gib=peak / 2**30,
                           wall_seconds=time.perf_counter() - started, returncode=proc.returncode)
        if status == 'ok':
            completion = json.loads((directory / 'completion.json').read_text())
            measurement.update(completion)
            if args.profile:
                # Fresh-run compute-stage extrapolation; loading/bootstrap added once.
                b = args.starts / cfg['starts']
                s = args.seeds / cfg['seeds']
                estimate = completion['baseline_seconds'] * b + completion.get('pca_seconds', 0)
                for method, timings in completion['method_seconds'].items():
                    seed_factor = 1 if method in DETERMINISTIC else s
                    estimate += timings['projection'] * seed_factor
                    estimate += (timings['evaluation'] + timings['artifact']) * b * seed_factor
                estimate += max(0, completion['total_seconds'] - completion['compute_seconds'])
                measurement['estimated_full_seconds'] = estimate
                measurement['estimate_caveat'] = 'same subset/grid/threads; extrapolation, not measured full runtime'
        return measurement
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(measure_dataset, name) for name in args.datasets]
        try:
            for future in as_completed(futures):
                measurement = future.result()
                measurements.append(measurement)
                write_json(out / 'resources.json', measurements)
                print(json.dumps(measurement), flush=True)
        except (KeyboardInterrupt, Exception):
            stop_event.set()
            for future in futures:
                future.cancel()
            with process_lock:
                for proc in active_processes.values():
                    try:
                        process = psutil.Process(proc.pid)
                        for child in process.children(recursive=True):
                            child.kill()
                        proc.kill()
                    except (OSError, psutil.Error):
                        pass
            raise
    if any(row['status'] != 'ok' for row in measurements):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
