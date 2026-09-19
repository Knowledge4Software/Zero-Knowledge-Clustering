#!/usr/bin/env python
"""RQ5 -- scaling: sweeps n/d/m/k and records native timings + symbolic
circuit-cost scaling, fitting log-log slopes (stats.loglog_slope) for Table
sensitivity-a and the Fig. scaling-dashboard.

Usage: python -m experiments.rq5_scaling --config experiments/configs/smoke.yaml
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from experiments.common import (common_arg_parser, load_config, load_prereg, generate_grid,
                                 row_hash, apply_shard, ResultWriter, default_out_path, capture_env)
from experiments.pipeline import load_row_dataset, build_projection
from commitjl.clustering import kmeans_paired, nearest_centroid_assign, psi_hat
from commitjl.cost_model import total_circuit_cost
from commitjl.quantize import scale_for_precision


def main():
    ap = common_arg_parser("RQ5 scaling")
    args = ap.parse_args()
    cfg = load_config(args.config)
    prereg = load_prereg(args.prereg)
    rows = generate_grid(cfg)
    for r in rows:
        r.setdefault("projection", "sparse_jl")
    rows = apply_shard(rows, args.shard)
    if args.limit:
        rows = rows[: args.limit]

    out_path = args.out or default_out_path("rq5_scaling")
    writer = ResultWriter(out_path)
    env = capture_env()
    n_ok, n_fail = 0, 0

    for row in rows:
        h = row_hash(row)
        if writer.should_skip(h):
            continue
        try:
            data = load_row_dataset(row)
            X = np.asarray(data["X"], dtype=np.float64)
            n, d = X.shape
            k = int(row.get("k", 5))
            seed = int(row.get("seed", 0))
            m = int(row.get("m", 16))
            q = int(row.get("q", 1))
            S = scale_for_precision(int(row.get("precision_bits", 16)))

            t0 = time.perf_counter()
            labels, centroids, _ = kmeans_paired(X, k, seeds=[seed])[0]
            t_kmeans = time.perf_counter() - t0

            t0 = time.perf_counter()
            P = build_projection(row["projection"], X, m, q, seed, S)
            t_proj_build = time.perf_counter() - t0

            t0 = time.perf_counter()
            Z = X @ P["real"].T
            Cz = centroids @ P["real"].T
            assign = nearest_centroid_assign(Z, Cz)
            _ = psi_hat(Z, Cz, assign)
            t_assign = time.perf_counter() - t0

            cost = total_circuit_cost(n, d, m, k, min(q, d))

            out = dict(row)
            out.update({"_hash": h, "_status": "ok", "n": n, "d": d, "k": k, "m": m, "q": q,
                        "t_kmeans_s": t_kmeans, "t_proj_build_s": t_proj_build,
                        "t_assign_s": t_assign, "t_total_s": t_kmeans + t_proj_build + t_assign,
                        "symbolic_total_constraints": cost["total_constraints"]})
            writer.write(out)
            n_ok += 1
        except Exception as e:
            writer.write({"_hash": h, "_status": "failed", "_error": str(e), **row})
            n_fail += 1

    print(f"rq5_scaling: {n_ok} ok, {n_fail} failed, out={out_path}")
    print(f"env: git={env.get('git_hash')}")


if __name__ == "__main__":
    main()
