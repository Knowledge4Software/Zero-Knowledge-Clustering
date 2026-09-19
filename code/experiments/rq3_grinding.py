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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from experiments.common import (common_arg_parser, load_config, load_prereg, generate_grid,
                                 row_hash, apply_shard, ResultWriter, default_out_path, capture_env)
from experiments.pipeline import load_row_dataset, build_projection
from commitjl.metrics import sample_pair_set, delta_uv
from commitjl.quantize import scale_for_precision


def estimate_delta_geom(X, m, q, projection, precision_bits, eps, n_trials, seed):
    """Single-attempt exceedance probability: over n_trials independently
    seeded projections, fraction of trials whose max |delta_uv| over a fixed
    pair set exceeds eps."""
    rng = np.random.default_rng(seed)
    S = scale_for_precision(precision_bits)
    n = X.shape[0]
    pairs = sample_pair_set(n, min(200, n * (n - 1) // 2 or 1), seed)
    exceed_flags = []
    for t in range(n_trials):
        trial_seed = int(rng.integers(0, 2**31 - 1))
        P = build_projection(projection, X, m, q, trial_seed, S)
        Z = X @ P["real"].T
        delta = delta_uv(X, Z, pairs)
        exceed_flags.append(bool(np.nanmax(np.abs(delta)) > eps))
    return float(np.mean(exceed_flags)), exceed_flags


def main():
    ap = common_arg_parser("RQ3b grinding adversary")
    ap.add_argument("--Q-list", default="1,2,4,8,16", help="comma list of adversary attempt budgets Q")
    ap.add_argument("--n-trials", type=int, default=64, help="Monte Carlo trials for delta_geom estimate")
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

    out_path = args.out or default_out_path("rq3_grinding")
    writer = ResultWriter(out_path)
    env = capture_env()
    n_ok, n_fail = 0, 0

    for row in rows:
        seed = int(row.get("seed", 0))
        m, q = int(row.get("m", 16)), int(row.get("q", 1))
        try:
            data = load_row_dataset(row)
            X = np.asarray(data["X"], dtype=np.float64)
            precision_bits = int(row.get("precision_bits", 16))
            delta_geom, flags = estimate_delta_geom(
                X, m, q, row["projection"], precision_bits, eps, args.n_trials, seed)
            flags = np.array(flags, dtype=bool)
            for Q in Q_list:
                h = row_hash({**row, "Q": Q})
                if writer.should_skip(h):
                    continue
                n_chunks = max(1, len(flags) // Q)
                emp = np.mean([flags[i * Q:(i + 1) * Q].any() for i in range(n_chunks)]) if n_chunks else float("nan")
                bound = min(1.0, Q * delta_geom)
                out = dict(row)
                out.update({"_hash": h, "_status": "ok", "Q": Q, "delta_geom": delta_geom,
                            "empirical_success_rate": float(emp), "union_bound": bound,
                            "bound_holds": bool(emp <= bound + 1e-9)})
                writer.write(out)
                n_ok += 1
        except Exception as e:
            writer.write({"_hash": row_hash(row), "_status": "failed", "_error": str(e), **row})
            n_fail += 1

    print(f"rq3_grinding: {n_ok} ok, {n_fail} failed, out={out_path}")
    print(f"env: git={env.get('git_hash')}")


if __name__ == "__main__":
    main()
