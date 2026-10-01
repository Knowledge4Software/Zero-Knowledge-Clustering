#!/usr/bin/env python
"""RQ3b -- grinding adversary: an adversary who (illegitimately) attempts Q
candidate commitments against independently sampled challenge seeds,
measuring the empirical rate at which at least one attempt exceeds the
per-attempt distortion tolerance eps (a "favorable" projection for the
adversary), against the union-bound prediction Q * delta_geom, where
delta_geom is the single-attempt exceedance probability (Prop. 1 / Thm 1
composition budget). TaskLog's Q-budget enforcement (randomness.py) is what
the protocol uses to cap Q in practice; this script quantifies what an
attacker gains per unit of Q if that cap were violated.

Usage: python -m experiments.rq3_grinding --config experiments/configs/smoke.yaml
"""
from __future__ import annotations

import sys
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from experiments.common import (common_arg_parser, load_config, load_prereg, generate_grid,
                                 row_hash, apply_shard, ResultWriter, default_out_path, capture_env,
                                 release_process_memory)
from experiments.common import progress
from experiments.pipeline import load_row_dataset, build_projection
from commitjl.metrics import sample_pair_set, delta_uv
from commitjl.quantize import scale_for_precision


def _trial_exceeds(args):
    X, m, q, projection, S, eps, pairs, trial_seed = args
    P = build_projection(projection, X, m, q, trial_seed, S)
    Z = X if P["real"] is None else X @ P["real"].T
    delta = delta_uv(X, Z, pairs)
    finite = delta[np.isfinite(delta)]
    if not finite.size:
        raise ValueError("grinding requires at least one distinct non-coincident pair")
    return bool(np.max(np.abs(finite)) > eps)


def estimate_delta_geom(X, m, q, projection, precision_bits, eps, n_trials, seed, workers=None, pair_count=2000):
    """Single-attempt exceedance probability: over n_trials independently
    seeded projections, fraction of trials whose max |delta_uv| over a fixed
    pair set exceeds eps."""
    rng = np.random.default_rng(seed)
    S = scale_for_precision(precision_bits)
    n = X.shape[0]
    if n_trials < 1 or (workers is not None and workers < 1):
        raise ValueError("n_trials and workers must be positive")
    pairs = sample_pair_set(n, pair_count, seed)
    trial_seeds = [int(s.generate_state(1, dtype=np.uint64)[0]) for s in np.random.SeedSequence(seed).spawn(n_trials)]
    jobs = [(X, m, q, projection, S, eps, pairs, trial_seed) for trial_seed in trial_seeds]
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=workers or min(4, os.cpu_count() or 1)) as pool:
        exceed_flags = list(pool.map(_trial_exceeds, jobs))
    return float(np.mean(exceed_flags)), exceed_flags


def main():
    ap = common_arg_parser("RQ3b grinding adversary")
    ap.add_argument("--Q-list", default="1,2,4,8,16", help="comma list of adversary attempt budgets Q")
    ap.add_argument("--n-trials", type=int, default=64, help="Monte Carlo trials for delta_geom estimate")
    ap.add_argument("--workers", type=int, default=None, help="parallel projection workers")
    args = ap.parse_args()
    cfg = load_config(args.config)
    prereg = load_prereg(args.prereg)
    eps = float(prereg.get("eps", 0.15))
    rows = generate_grid(cfg)
    for r in rows:
        r.setdefault("projection", "sparse_jl")
    rows = apply_shard(rows, args.shard)
    if args.limit:
        rows = rows[: args.limit]
    Q_list = [int(x) for x in args.Q_list.split(",")]
    if args.n_trials < 1 or any(Q < 1 or Q > args.n_trials for Q in Q_list):
        ap.error("each Q must be positive and no larger than --n-trials")

    out_path = args.out or default_out_path("rq3_grinding", args.shard)
    writer = ResultWriter(out_path, context=prereg)
    env = capture_env()
    n_ok, n_fail = 0, 0

    for row in progress(rows, desc="RQ3 grinding"):
        seed = int(row.get("seed", 0))
        m, q = int(row.get("m", 16)), int(row.get("q", 1))
        try:
            data = load_row_dataset(row)
            X = np.asarray(data["X"], dtype=np.float64)
            precision_bits = int(row.get("precision_bits", 16))
            delta_geom, flags = estimate_delta_geom(
                X, m, q, row["projection"], precision_bits, eps, args.n_trials, seed, args.workers,
                int(prereg.get("pair_set_size", 2000)))
            flags = np.array(flags, dtype=bool)
            for Q in Q_list:
                h = row_hash({**row, "Q": Q, "n_trials": args.n_trials})
                if writer.should_skip(h):
                    continue
                n_chunks = len(flags) // Q
                used_flags = flags[:n_chunks * Q]
                emp = used_flags.reshape(n_chunks, Q).any(axis=1).mean()
                bound = min(1.0, Q * float(used_flags.mean()))
                out = dict(row)
                out.update({"_hash": h, "_status": "ok", "Q": Q, "delta_geom": delta_geom,
                            "empirical_success_rate": float(emp), "union_bound": bound,
                            "bound_holds": bool(emp <= bound + 1e-9),
                            "bound_kind": "empirical_plugin_not_theoretical_guarantee",
                            "n_trials": args.n_trials, "n_Q_batches": n_chunks,
                            "attempts_used": int(used_flags.size),
                            "attempts_discarded": int(len(flags) - len(used_flags))})
                writer.write(out)
                n_ok += 1
        except Exception as e:
            writer.write({"_hash": row_hash(row), "_status": "failed", "_error": str(e), **row})
            n_fail += 1
        finally:
            release_process_memory()

    print(f"rq3_grinding: {n_ok} ok, {n_fail} failed, out={out_path}")
    print(f"env: git={env.get('git_hash')}")

    writer.close()
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
