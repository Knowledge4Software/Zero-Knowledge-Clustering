#!/usr/bin/env python
"""RQ1 -- fidelity: native (no ZK proof) grid over datasets/projections/m/q,
producing Table fidelity-a rows (ARI/NMI/agreement, r_Phi, delta_uv
distortion summary) and the data behind Fig. fidelity-vs-m.

Usage: python -m experiments.rq1_fidelity --config experiments/configs/smoke.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import (common_arg_parser, load_config, load_prereg, generate_grid,
                                 row_hash, apply_shard, ResultWriter, default_out_path, capture_env,
                                 release_process_memory)
from experiments.common import progress
from experiments.pipeline import PipelineCache, run_native
from experiments.statistics import write_native_summary


def main():
    ap = common_arg_parser("RQ1 fidelity native grid")
    args = ap.parse_args()
    cfg = load_config(args.config)
    prereg = load_prereg(args.prereg)
    rows = generate_grid(cfg)
    # rq1 needs a `projection` axis; default to sparse_jl if the config omits it (e.g. core.yaml)
    for r in rows:
        r.setdefault("projection", "sparse_jl")
    # q changes only SparseJL. The requested m does not change the identity
    # baseline. Canonicalizing these inactive axes prevents duplicated paper
    # rows and avoids giving those baselines extra statistical weight.
    canonical = {}
    for row in rows:
        row = dict(row)
        if row["projection"] != "sparse_jl":
            row["q"] = 1
        if row["projection"] == "identity":
            row["m"] = 1  # run_native reports the effective dimension d
        canonical[row_hash(row)] = row
    rows = list(canonical.values())
    rows = apply_shard(rows, args.shard)
    if args.limit:
        rows = rows[: args.limit]

    out_path = args.out or default_out_path("rq1_fidelity", args.shard)
    writer = ResultWriter(out_path, context=prereg)
    env = capture_env()

    n_ok, n_fail = 0, 0
    cache = PipelineCache()
    for row in progress(rows, desc="RQ1 fidelity"):
        h = row_hash(row)
        try:
            data, _ = cache.dataset(row)
            effective_row = dict(row)
            effective_row["m"] = (data["X"].shape[1] if row["projection"] == "identity"
                                  else min(int(row["m"]), data["X"].shape[1]))
            h = row_hash(effective_row)
            if writer.should_skip(h):
                continue
            result = run_native(effective_row, prereg, cache=cache)
            result["_hash"] = h
            result["_status"] = "ok"
            writer.write(result)
            n_ok += 1
        except Exception as e:
            writer.write({"_hash": h, "_status": "failed", "_error": str(e), **row})
            n_fail += 1
        finally:
            release_process_memory()

    print(f"rq1_fidelity: {n_ok} ok, {n_fail} failed, out={out_path}")
    print(f"env: git={env.get('git_hash')}")

    writer.close()
    if not n_fail and not args.shard:
        write_native_summary(out_path,
            ("dataset_name", "dataset_sha256", "projection", "n", "d", "k", "m", "q", "precision_bits"))
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
