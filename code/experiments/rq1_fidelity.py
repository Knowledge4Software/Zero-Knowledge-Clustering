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
                                 row_hash, apply_shard, ResultWriter, default_out_path, capture_env)
from experiments.pipeline import run_native


def main():
    ap = common_arg_parser("RQ1 fidelity native grid")
    args = ap.parse_args()
    cfg = load_config(args.config)
    prereg = load_prereg(args.prereg)
    rows = generate_grid(cfg)
    # rq1 needs a `projection` axis; default to sparse_jl if the config omits it (e.g. core.yaml)
    for r in rows:
        r.setdefault("projection", "sparse_jl")
    rows = apply_shard(rows, args.shard)
    if args.limit:
        rows = rows[: args.limit]

    out_path = args.out or default_out_path("rq1_fidelity")
    writer = ResultWriter(out_path)
    env = capture_env()

    n_ok, n_fail = 0, 0
    for row in rows:
        h = row_hash(row)
        if writer.should_skip(h):
            continue
        try:
            result = run_native(row, prereg)
            result["_hash"] = h
            result["_status"] = "ok"
            writer.write(result)
            n_ok += 1
        except Exception as e:
            writer.write({"_hash": h, "_status": "failed", "_error": str(e), **row})
            n_fail += 1

    print(f"rq1_fidelity: {n_ok} ok, {n_fail} failed, out={out_path}")
    print(f"env: git={env.get('git_hash')}")


if __name__ == "__main__":
    main()
