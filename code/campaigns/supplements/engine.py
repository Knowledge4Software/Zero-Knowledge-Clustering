"""Add measured RQ3-RQ5 experiments without changing completed campaign rows.

This launcher is outside the source tree covered by source_fingerprint(). Its
own SHA256 is included in each output context to make resume provenance safe.
No proof-system cost is inferred from native measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from commitjl.clustering import _squared_distances, kmeans_paired
from commitjl.certificates import centroid_margin_certificate
from commitjl.protocol import Prover, Verifier, run_protocol
from commitjl.quantize import quantize, scale_for_precision
from commitjl.randomness import TaskLog
from commitjl.sparse_gen import sparse_matrix_gen
from commitjl.zk_bridge import field_hash_sparse_list
from experiments.common import (ResultWriter, load_prereg, progress,
                                release_process_memory, row_hash)
from experiments.pipeline import (PipelineCache, load_row_dataset,
                                  normalize_for_quantization, run_native)
from experiments.rq4_certificates import run_centroid_cert
from experiments.rq5_precision import measure_precision_row
from experiments.statistics import write_native_summary

CODE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = CODE_ROOT / "results" / "paper_supplement_v1"
SCRIPT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SEEDS = range(10)
BASE = {"precision_bits": 16, "B": 4096, "backend": "groth16",
        "projection": "sparse_jl"}


def rq3_rows():
    datasets = [
        {"loader": "gaussian_mixture", "kwargs": {"n": 500, "d": 256, "k": 5,
            "gamma": 3.0, "imbalance": 1.0, "seed": 12345}},
        {"loader": "gaussian_mixture", "kwargs": {"n": 1000, "d": 1024, "k": 10,
            "gamma": 2.5, "imbalance": 1.0, "seed": 12345}},
        {"loader": "mnist", "kwargs": {"n_max": 500}},
    ]
    return [{**BASE, "dataset": ds, "m": 32, "q": 2, "k": 5, "seed": seed}
            for ds in datasets for seed in SEEDS]


def measure_rq3_pair(row):
    """Time both protocol orderings on the same X and public centroids.

    The pre-published map is an independent deterministic draw of identical
    shape, generated/authenticated before its transaction. That setup time is
    reported separately. It is a native timing baseline, not a ZK proof.
    """
    data = load_row_dataset(row)
    X = np.asarray(data["X"], dtype=np.float64)
    n, d = X.shape
    m, q, k, seed = (int(row[key]) for key in ("m", "q", "k", "seed"))
    S = scale_for_precision(int(row["precision_bits"]))
    Xq = quantize(normalize_for_quantization(X, S, int(row["B"])), S, int(row["B"]))
    _, centroids, _ = kmeans_paired(Xq.astype(np.float64), k, seeds=[seed])[0]
    Cq = np.rint(centroids).astype(np.int64)

    setup_start = time.perf_counter()
    public_seed = hashlib.sha256(("prepublished:" + row_hash(row)).encode()).digest()
    records, A_public = sparse_matrix_gen(public_seed, m, d, q)
    C_A_public = field_hash_sparse_list(records)
    prepublished_setup_s = time.perf_counter() - setup_start

    def prepublished():
        prover = Prover(Xq, S)
        verifier = Verifier(TaskLog(Q=1), m, q, Cq)
        _, commit_s = prover.commit_data()
        _, theta_s = verifier.record_theta()
        _, _, _, cluster_s = prover.project_and_cluster(A_public, Cq)
        return commit_s, theta_s, cluster_s

    def postcommit():
        result = run_protocol(Xq, Cq, m, q, S,
                              task_log=TaskLog(Q=1), challenge_fn=lambda: os.urandom(32))
        return dict(result["transcript"]["stage_times_s"])

    # Alternate order to reduce drift from thermal throttling and cache state.
    if seed % 2:
        post = postcommit()
        pre_commit, pre_theta, pre_cluster = prepublished()
        order = "post_first"
    else:
        pre_commit, pre_theta, pre_cluster = prepublished()
        post = postcommit()
        order = "pre_first"
    pre_total = pre_commit + pre_theta + pre_cluster
    post_total = float(sum(post.values()))
    return {**row, "dataset_name": data["meta"]["name"], "dataset_sha256": data["meta"].get("sha256", ""),
            "n": n, "d": d, "comparison": "paired_postcommit_vs_prepublished_native",
            "postcommit_randomness_source": "local_nonce",
            "pair_order": order, "prepublished_seed_hex": public_seed.hex(),
            "prepublished_c_a": str(C_A_public),
            "prepublished_setup_s_excluded": prepublished_setup_s,
            "prepublished_commit_s": pre_commit, "prepublished_theta_s": pre_theta,
            "prepublished_project_cluster_s": pre_cluster,
            "prepublished_transaction_s": pre_total,
            "postcommit_commit_s": post["commit_X"],
            "postcommit_theta_s": post["record_theta"],
            "postcommit_challenge_s": post["issue_nonce"],
            "postcommit_expand_auth_s": post["expand_sparse_and_CA"],
            "postcommit_project_cluster_s": post["project_and_cluster"],
            "postcommit_transaction_s": post_total,
            "incremental_postcommit_s": post_total - pre_total,
            "incremental_after_commit_s": (
                post["issue_nonce"] + post["expand_sparse_and_CA"] + post["project_and_cluster"]
                - pre_cluster),
            "evaluation_mode": "native_timing_only_no_zk_proof"}


def rq4_rows():
    datasets = [
        {"loader": "mnist", "kwargs": {"n_max": 1000}},
        {"loader": "fashion_mnist", "kwargs": {"n_max": 1000}},
        {"loader": "gisette", "kwargs": {"n_max": 1000}},
        {"loader": "arcene", "kwargs": {"n_max": 100}},
        {"loader": "20ng_tfidf", "kwargs": {"n_max": 1000, "max_features": 2000}},
    ]
    return [{**BASE, "dataset": ds, "m": m, "q": 2, "k": 5, "seed": seed}
            for ds in datasets for m in (32, 64) for seed in SEEDS]


def rq5_gamma_rows():
    return [{**BASE, "dataset": {"loader": "gaussian_mixture", "kwargs": {
                "n": 1000, "d": 256, "k": 5, "gamma": gamma,
                "imbalance": 1.0, "seed": 12345}},
             "m": 32, "q": 2, "k": 5, "seed": seed,
             "sweep_factor": "gamma", "sweep_value": gamma}
            for gamma in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0)
            for seed in SEEDS]


def _inject_zero_gamma_dataset(cache, row):
    """Supply the exact zero-separation limit of the existing generator.

    The core loader requires gamma > 0, so this supplement constructs the
    gamma=0 case without altering the loader and invalidating prior results.
    """
    spec = row["dataset"]["kwargs"]
    n, d, k, seed = (int(spec[key]) for key in ("n", "d", "k", "seed"))
    rng = np.random.default_rng(seed)
    rng.normal(size=(k, d))  # same center draw consumed by the core generator
    counts = np.full(k, n // k, dtype=int)
    counts[-1] += n - int(counts.sum())
    X = np.concatenate([rng.normal(size=(int(count), d)) for count in counts], axis=0)
    y = np.concatenate([np.full(int(count), j) for j, count in enumerate(counts)])
    permutation = rng.permutation(n)
    X, y = X[permutation], y[permutation]
    digest = hashlib.sha256(memoryview(np.ascontiguousarray(X)).cast("B")).hexdigest()
    cache.datasets[cache.dataset_key(row)] = {
        "X": X, "y": y, "centers": np.zeros((k, d)),
        "meta": {"name": "gaussian_mixture", "n": n, "d": d, "k": k,
                 "gamma": 0.0, "imbalance": 1.0, "seed": seed,
                 "source": "synthetic_zero_separation_supplement", "sha256": digest},
    }


def centroid_embedding_event(row, prereg, cache):
    """Check the realized JL premise on every point-to-centroid pair."""
    data, dataset_key = cache.dataset(row)
    X = np.asarray(data["X"], dtype=np.float64)
    k, seed, m, q = (int(row[key]) for key in ("k", "seed", "m", "q"))
    (_, C, _), _ = cache.kmeans_run(dataset_key, X, k, seed)
    S = scale_for_precision(int(row["precision_bits"]))
    P, _ = cache.projection(dataset_key, "sparse_jl", X, m, q, seed, S)
    Z, Cz = X @ P["real"].T, C @ P["real"].T
    original = _squared_distances(X, C)
    projected = _squared_distances(Z, Cz)
    positive = original > 1e-24
    relative = np.zeros_like(original)
    relative[positive] = np.abs(projected[positive] / original[positive] - 1.0)
    relative[~positive] = np.where(projected[~positive] <= 1e-24, 0.0, np.inf)
    eps = float(prereg.get("eps", 0.15))
    pointwise_event = np.all(relative <= eps, axis=1)
    certified = centroid_margin_certificate(X, C, eps)
    certified_with_event = certified & pointwise_event
    assignments_changed = np.argmin(original, axis=1) != np.argmin(projected, axis=1)
    return {"embedding_event_holds": bool(np.all(relative <= eps)),
            "n_point_centroid_distortion_violations": int(np.count_nonzero(relative > eps)),
            "max_point_centroid_distortion": float(np.max(relative)),
            "n_pointwise_embedding_event": int(np.count_nonzero(pointwise_event)),
            "n_certified_with_pointwise_event": int(np.count_nonzero(certified_with_event)),
            "n_certified_pointwise_event_violations": int(np.count_nonzero(certified_with_event & assignments_changed))}


def rq5_precision_rows():
    datasets = [
        {"loader": "gaussian_mixture", "kwargs": {"n": 1000, "d": 256, "k": 5,
            "gamma": 3.0, "imbalance": 1.0, "seed": 12345}},
        {"loader": "dbscan_margin", "kwargs": {"n": 1000, "d": 128,
            "rho": 1.0, "eps": 0.15, "seed": 12345}},
        {"loader": "mnist", "kwargs": {"n_max": 1000}},
    ]
    return [{**BASE, "dataset": ds, "m": 32, "q": 2, "k": 5, "seed": seed,
             "precision_bits": bits, "normalization_reference_bits": 24}
            for ds in datasets for bits in (8, 12, 16, 20, 24) for seed in SEEDS]


def _rq3_worker(row, workers=1):
    try:
        return {**measure_rq3_pair(row), "_hash": row_hash(row), "_status": "ok",
                "parallel_workers": workers,
                "timing_semantics": "per_job_elapsed_under_cpu_contention"}
    except Exception as exc:
        return {**row, "_hash": row_hash(row), "_status": "failed", "_error": repr(exc)}
    finally:
        release_process_memory()


def run_stage(stage, out_dir, prereg, workers, limit=None):
    builders = {"RQ3": (rq3_rows, "rq3_paired.parquet"),
                "RQ4": (rq4_rows, "rq4_real_certificates.parquet"),
                "RQ5Gamma": (rq5_gamma_rows, "rq5_gamma.parquet"),
                "RQ5Precision": (rq5_precision_rows, "rq5_precision_24_reference.parquet")}
    rows_fn, filename = builders[stage]
    rows = rows_fn()
    if limit is not None:
        rows = rows[:limit]
    output = out_dir / filename
    context = {**prereg, "supplement_script_sha256": SCRIPT_SHA256, "stage": stage,
               "workers": workers if stage == "RQ3" else 1}
    writer = ResultWriter(output, context=context)
    pending = [row for row in rows if not writer.should_skip(row_hash(row))]
    print(f"{stage}: {len(rows)} planned, {len(pending)} remaining; out={output}", flush=True)
    failures = 0
    cache = PipelineCache() if stage in {"RQ4", "RQ5Gamma"} else None
    try:
        if stage == "RQ3":
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_rq3_worker, row, workers): row for row in pending}
                for future in progress(as_completed(futures), total=len(futures), desc=stage):
                    row = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {**row, "_hash": row_hash(row), "_status": "failed", "_error": repr(exc)}
                    writer.write(result)
                    failures += result["_status"] != "ok"
        else:
            for row in progress(pending, desc=stage):
                try:
                    if stage == "RQ4":
                        result = run_centroid_cert(row, prereg, cache=cache)
                        result.update({"dataset_name": row["dataset"]["loader"],
                                       "evaluation_mode": "native_real_data_centroid_certificate"})
                        result.update(centroid_embedding_event(row, prereg, cache))
                    elif stage == "RQ5Gamma":
                        if float(row["sweep_value"]) == 0.0:
                            _inject_zero_gamma_dataset(cache, row)
                        result = run_native(row, prereg, cache=cache)
                        cert = run_centroid_cert(row, prereg, cache=cache)
                        result.update({"certificate_coverage": cert["coverage"],
                                       "certificate_violation_rate": cert["violation_rate"],
                                       "certificate_n_certified": cert["n_certified"]})
                        result.update(centroid_embedding_event(row, prereg, cache))
                    else:
                        result = measure_precision_row(row, prereg)
                    writer.write({**result, "_hash": row_hash(row), "_status": "ok"})
                except Exception as exc:
                    writer.write({**row, "_hash": row_hash(row), "_status": "failed", "_error": repr(exc)})
                    failures += 1
                finally:
                    release_process_memory()
    finally:
        writer.close()
    print(f"{stage}: {len(rows) - failures} ok, {failures} failed", flush=True)
    if failures:
        raise RuntimeError(f"{stage} had {failures} failed rows; see {output}")
    if limit is None and stage == "RQ5Precision":
        write_native_summary(output,
            ("dataset_name", "dataset_sha256", "n", "d", "k", "m", "q", "precision_bits"),
            metrics=("ari", "delta_q95", "r_phi", "ari_quantized_full", "mean_abs_quantization_error"))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("All", "RQ3", "RQ4", "RQ5Gamma", "RQ5Precision"), default="All")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None, help="development smoke-test rows")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    prereg = load_prereg()
    stages = ("RQ3", "RQ4", "RQ5Gamma", "RQ5Precision") if args.stage == "All" else (args.stage,)
    for stage in stages:
        run_stage(stage, args.out_dir, prereg, args.workers, args.limit)


if __name__ == "__main__":
    main()
