"""Run the unchanged RQ3 protocol benchmark in separate CPU processes.

Only the parent writes checkpoints. This launcher lives outside the source
fingerprint used by the paper results, so completed RQ1 data stays valid.
Parallel timings describe per-job latency under the recorded worker count.
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from experiments.common import (
    ResultWriter, capture_env, generate_grid, load_config, load_prereg,
    progress, release_process_memory, row_hash,
)
from experiments.rq3_overhead import SEED_SOURCES, run_one


def jobs_from_config(config):
    unique = {}
    for original in generate_grid(config):
        row = {key: value for key, value in original.items()
               if key not in {"projection", "variant"}}
        unique[row_hash(row)] = {**row, "projection": "sparse_jl"}
    return [(row, source) for row in unique.values() for source in SEED_SOURCES]


def measure_job(row, source, prereg, workers):
    """Return only the compact result row; never transfer large arrays."""
    h = row_hash({**row, "seed_source": source})
    try:
        result = run_one(row, prereg, source)
        transcript = result["transcript"]
        times = transcript["stage_times_s"]
        sizes = transcript["stage_bytes"]
        return {
            **row, "_hash": h, "_status": "ok", "seed_source": source,
            "attempts": result["_attempts"],
            "randomness_mode": "simulated_beacon_trace" if source == "beacon" else "local_" + source,
            "randomness_display_label": "Simulated local trace" if source == "beacon" else source.replace("_", "-"),
            "randomness_is_external_public_beacon": False if source == "beacon" else None,
            "rho_hex": result["rho"].hex(),
            "projection_seed_hex": result["seed"].hex(),
            **{f"time_{key}": value for key, value in times.items()},
            **{f"bytes_{key}": value for key, value in sizes.items()},
            "total_time_s": float(sum(times.values())),
            "total_bytes": int(sum(sizes.values())),
            "execution_mode": "parallel_processes",
            "parallel_workers": workers,
            "timing_semantics": "per_job_elapsed_under_cpu_contention",
        }
    except Exception as exc:
        return {**row, "_hash": h, "_status": "failed", "_error": repr(exc),
                "seed_source": source, "execution_mode": "parallel_processes",
                "parallel_workers": workers}
    finally:
        release_process_memory()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/paper_rq3.yaml")
    parser.add_argument("--prereg", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None, help="rows for a development smoke run")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")

    # Set these before spawned workers import NumPy or scikit-learn on Windows.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    os.environ["GOMAXPROCS"] = "4"

    prereg = load_prereg(args.prereg)
    jobs = jobs_from_config(load_config(args.config))
    if args.limit is not None:
        jobs = jobs[:args.limit]
    context = {**prereg, "rq3_parallel_workers": args.workers,
               "rq3_timing_semantics": "per_job_elapsed_under_cpu_contention"}
    writer = ResultWriter(args.out, context=context)
    pending = [(row, source) for row, source in jobs
               if not writer.should_skip(row_hash({**row, "seed_source": source}))]
    print(f"RQ3 overhead: {len(jobs)} planned, {len(jobs) - len(pending)} checkpointed, "
          f"{len(pending)} remaining, {args.workers} processes", flush=True)
    failures = 0
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(measure_job, row, source, prereg, args.workers): (row, source)
                       for row, source in pending}
            for future in progress(as_completed(futures), total=len(futures), desc="RQ3 overhead parallel"):
                row, source = futures[future]
                try:
                    output = future.result()
                except Exception as exc:
                    output = {**row, "_hash": row_hash({**row, "seed_source": source}),
                              "_status": "failed", "_error": repr(exc),
                              "seed_source": source, "execution_mode": "parallel_processes",
                              "parallel_workers": args.workers}
                writer.write(output)
                failures += output["_status"] != "ok"
    finally:
        writer.close()
    print(f"RQ3 overhead parallel: {len(jobs) - failures} ok, {failures} failed; "
          f"out={args.out}; git={capture_env().get('git_hash')}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
