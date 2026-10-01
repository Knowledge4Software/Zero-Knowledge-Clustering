#!/usr/bin/env python
"""RQ5 -- scaling: sweeps n/d/m/k and records measured native timings and
quality metrics, fitting log-log slopes (stats.loglog_slope) for Table
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
                                 row_hash, apply_shard, ResultWriter, default_out_path, capture_env,
                                 release_process_memory)
from experiments.common import progress
from experiments.pipeline import PipelineCache, run_native
from commitjl.clustering import kmeans_paired, nearest_centroid_assign, psi_hat
from commitjl.quantize import scale_for_precision
from experiments.statistics import write_native_summary


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

    out_path = args.out or default_out_path("rq5_scaling", args.shard)
    writer = ResultWriter(out_path, context=prereg)
    env = capture_env()
    n_ok, n_fail = 0, 0
    cache = PipelineCache()

    for row in progress(rows, desc="RQ5 scaling"):
        h = row_hash(row)
        if writer.should_skip(h):
            continue
        try:
            measured = run_native(row, prereg, cache=cache)
            out = dict(measured)
            out.update({"_hash": h, "_status": "ok"})
            writer.write(out)
            n_ok += 1
        except Exception as e:
            writer.write({"_hash": h, "_status": "failed", "_error": str(e), **row})
            n_fail += 1
        finally:
            release_process_memory()

    print(f"rq5_scaling: {n_ok} ok, {n_fail} failed, out={out_path}")
    print(f"env: git={env.get('git_hash')}")

    writer.close()
    if not n_fail and not args.shard:
        write_native_summary(out_path,
            ("sweep_factor", "sweep_value", "dataset_name", "n", "d", "k", "m", "q", "precision_bits"))
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
