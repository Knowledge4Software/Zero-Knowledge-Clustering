"""Plan/run the isolated, offline compression experiment suite for the paper."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import psutil
from campaigns.laptop_comparison import atomic_json, kill_tree
from experiments.paper_artifacts import job_complete

CODE = Path(__file__).resolve().parents[1]
DATASETS = ['gaussian_easy', 'gaussian_hard', 'dbscan_margin', 'mnist', 'fashion_mnist', 'gisette', 'arcene', '20ng_tfidf']


def configuration(profile):
    cfg = dict(version='paper-compression-v2', smoke=False, profile=profile, threads=4,
        seeds=10, starts=5, k=[5,10], m=[16,32,64,128,256,512], q=[1,2], timing_m=[64,128,256],
        timing_repetitions=5, fit_repetitions=3, fit_warmup=True,
        split_seed=481516, pair_seed=20261101, pairs=2000, bootstrap=2000,
        query_points=512, landmarks=64, queries=32, query_artifact_samples=2, dbscan_points=256,
        kmeans_max_iter=300, eps=.15, randomized_pca=True, pca_fit_seed=90125,
        thresholds=dict(ari=.99, r_phi=1.10, distortion=.20, projected_cost_error=.10))
    if profile == 'balanced': cfg.update(seeds=5, starts=3, bootstrap=1000)
    if profile == 'pilot': cfg.update(seeds=1, starts=1, k=[5], m=[64,128,256], q=[1],
        timing_m=[128], timing_repetitions=3, fit_repetitions=1, fit_warmup=False, bootstrap=100, queries=2)
    if profile == 'smoke': cfg.update(smoke=True, seeds=2, starts=2, k=[5], m=[8,16], q=[1],
        timing_m=[8], timing_repetitions=1, fit_repetitions=1, fit_warmup=False, bootstrap=100,
        queries=2, pairs=200, query_points=16, landmarks=8, dbscan_points=32, kmeans_max_iter=30)
    return cfg


def identity(config, datasets):
    paths = [CODE/'campaigns/paper_compression.py', CODE/'experiments/paper_compression.py',
             CODE/'analysis/paper_compression_report.py', CODE/'commitjl/reusable_native.py',
             CODE/'analysis/replay_paper_compression.py',
             CODE/'experiments/laptop_comparison.py', CODE/'campaigns/laptop_comparison.py',
             CODE/'experiments/paper_artifacts.py',
             CODE/'commitjl/protected_jl.py', CODE/'commitjl/sparse_gen.py',
             CODE/'commitjl/certificates.py', CODE/'commitjl/metrics.py',
             CODE/'experiments/protected_jl_study.py', CODE/'experiments/rq1_complete.py']
    return dict(configuration=config, datasets=datasets,
        sources={str(p.relative_to(CODE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        archive_index_sha256=hashlib.sha256((CODE/'results/server_retirement_20261007/compact_arrays/array_index.json').read_bytes()).hexdigest())


def execute(job, folder, deadline, cap_gib):
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder/'job.json', job)
    env = dict(os.environ)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[key] = str(job['config']['threads'])
    start = time.monotonic(); peak = 0; status = 'running'
    with (folder/'worker.log').open('a', encoding='utf-8') as log:
        proc = subprocess.Popen([sys.executable, '-u', '-m', 'experiments.paper_compression',
            '--job', str((folder/'job.json').resolve()), '--out', str(folder.resolve())],
            cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            while proc.poll() is None:
                if time.monotonic() >= deadline or time.monotonic()-start > job['max_seconds']:
                    status = 'time_limit'; kill_tree(proc); break
                try:
                    parent = psutil.Process(proc.pid)
                    rss = 0
                    for p in [parent]+parent.children(recursive=True):
                        try: rss += p.memory_info().rss
                        except psutil.Error: pass
                    peak = max(peak, rss)
                    if rss > cap_gib*2**30 or psutil.virtual_memory().available < 512*2**20:
                        status = 'memory_limit'; kill_tree(proc); break
                except psutil.Error: pass
                time.sleep(.1)
        except KeyboardInterrupt:
            status = 'interrupted'; kill_tree(proc)
    if status == 'running': status = 'ok' if proc.returncode == 0 else 'error'
    record = dict(status=status, seconds=time.monotonic()-start, peak_tree_rss_gib=peak/2**30,
                  returncode=proc.returncode, cap_gib=cap_gib)
    atomic_json(folder/'execution.json', record)
    return record


@contextmanager
def exclusive_run(root):
    """OS lock survives stale lock files and is released if the process exits."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'.campaign.lock').open('a+b') as stream:
        if stream.tell()==0:stream.write(b'0');stream.flush()
        stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:raise RuntimeError('This output directory already has an active runner; it was not interrupted.') from exc
        try:yield
        finally:
            stream.seek(0)
            if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream,fcntl.LOCK_UN)


def main_unlocked():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--profile', choices=['paper','balanced','pilot','smoke'], default='paper')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--run', action='store_true', help='otherwise write the reviewable plan only')
    parser.add_argument('--max-minutes', type=float, default=480)
    parser.add_argument('--max-rss-gib', type=float, default=3)
    args = parser.parse_args()
    if args.max_minutes <= 0 or args.max_rss_gib < 1: parser.error('Positive duration and at least 1 GiB cap required')
    root = args.out.resolve(); root.mkdir(parents=True, exist_ok=True)
    cfg = configuration(args.profile); frozen = identity(cfg, args.datasets)
    path = root/'manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest['identity'] != frozen: parser.error('Source/config/archive changed: use a fresh output directory')
    else:
        import numpy, scipy, sklearn
        manifest = dict(identity=frozen, complete=False, attempts=[],
            environment=dict(python=sys.version, platform=platform.platform(), processor=platform.processor(),
                logical_cpus=os.cpu_count(), physical_cpus=psutil.cpu_count(logical=False),
                total_ram_gib=psutil.virtual_memory().total/2**30, numpy=numpy.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__),
            scope='New native fixed-centroid/compression/downstream study; ZK evidence remains separately scoped')
        for name in frozen['sources']:
            target = root/'sources'/name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes((CODE/name).read_bytes())
    jobs = [dict(dataset=name, variant=variant, config=cfg, max_seconds=4*3600)
        for name in args.datasets for variant in (['primary','nonzero'] if name == '20ng_tfidf' else ['primary'])]
    atomic_json(root/'plan.json', dict(jobs=jobs, phases=['train_models','validation_screen','freeze_selection','test_screen_and_workloads'],
        timing_boundaries=['centroid_training','pca_training','basis_fit','map_construction','whole_data_compression',
            'centroid_compression','numeric_preparation','fixed_center_assignment','distance_queries','knn','kmeans','dbscan_pair_build','dbscan_fit']))
    atomic_json(path, manifest)
    if not args.run:
        print(json.dumps(dict(plan_written=True, jobs=len(jobs), profile=args.profile, out=str(root)))); return
    deadline = time.monotonic()+args.max_minutes*60
    attempt = dict(start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), limit_minutes=args.max_minutes, jobs=[])
    manifest['attempts'].append(attempt)
    digest_cache={}
    for job in jobs:
        folder = root/'jobs'/f"{job['dataset']}_{job['variant']}"
        if job_complete(folder,digest_cache): continue
        if time.monotonic() >= deadline: break
        available = psutil.virtual_memory().available/2**30
        if available < 1.5:
            print('Insufficient free RAM: saved checkpoint; resume this command after freeing memory.', flush=True); break
        cap = min(args.max_rss_gib, max(1., available-.75))
        record = execute(job, folder, deadline, cap)
        attempt['jobs'].append(dict(dataset=job['dataset'], variant=job['variant'], **record))
        atomic_json(path, manifest)
        print(json.dumps(dict(dataset=job['dataset'], variant=job['variant'], **record)), flush=True)
        if record['status'] != 'ok': break
    manifest['complete'] = all(job_complete(root/'jobs'/f"{j['dataset']}_{j['variant']}",digest_cache) for j in jobs)
    atomic_json(path, manifest)
    from analysis.paper_compression_report import build_report
    build_report(root)
    print(json.dumps(dict(complete=manifest['complete'], out=str(root))), flush=True)
    if any(j['status'] == 'error' for j in attempt['jobs']): raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--out',type=Path)
    options,_=parser.parse_known_args()
    if options.out is None:main_unlocked();return
    try:
        with exclusive_run(options.out.resolve()):main_unlocked()
    except RuntimeError as exc:parser.error(str(exc))


if __name__ == '__main__': main()
