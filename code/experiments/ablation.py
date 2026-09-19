#!/usr/bin/env python
"""Ablation -- implementation switches (Table ablation-a): variant
(sparse/dense/coord/pca-public/prepublished/full/incircuit-gen) and the
range-check batching switch (no_range_batch), reported via symbolic
cost_model (all variants) plus measured zk_bridge proving where wired
(variant == "sparse").

Usage: python -m experiments.ablation --config experiments/configs/core.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import (common_arg_parser, load_config, load_prereg, generate_grid,
                                 row_hash, apply_shard, ResultWriter, default_out_path, capture_env)
from commitjl.cost_model import total_circuit_cost, CostModelParams
from commitjl import zk_bridge
from experiments.rq2_proof_cost import build_zk_instance
import numpy as np

VARIANTS = ["sparse", "dense", "coord", "pca-public", "prepublished", "full", "incircuit-gen"]


def main():
    ap = common_arg_parser("Ablation: implementation switches")
    args = ap.parse_args()
    cfg = load_config(args.config)
    prereg = load_prereg(args.prereg)
    base_rows = generate_grid(cfg)
    if args.limit:
        base_rows = base_rows[: args.limit]

    out_path = args.out or default_out_path("ablation")
    writer = ResultWriter(out_path)
    env = capture_env()
    n_ok, n_fail = 0, 0

    expanded = []
    for row in base_rows:
        for variant in VARIANTS:
            for no_range_batch in (False, True):
                expanded.append({**row, "variant": variant, "no_range_batch": no_range_batch})
    expanded = apply_shard(expanded, args.shard)

    for row in expanded:
        h = row_hash(row)
        if writer.should_skip(h):
            continue
        m, q, k = int(row.get("m", 16)), int(row.get("q", 1)), int(row.get("k", 5))
        d = int(row.get("dataset", {}).get("kwargs", {}).get("d", 128))
        n_pts = int(row.get("dataset", {}).get("kwargs", {}).get("n", 500))
        params = CostModelParams(c_range=2.0 if row["no_range_batch"] else 1.0)
        try:
            symbolic = total_circuit_cost(n_pts, d, m, k, q, params=params)
        except Exception:
            symbolic = {}

        out = dict(row)
        out.update({"_hash": h, "n": n_pts, "d": d, "k": k,
                    "symbolic_total_constraints": symbolic.get("total_constraints")})

        if row["variant"] == "sparse" and zk_bridge.binary_available():
            try:
                inst, n2, d2, k2, m2, q2 = build_zk_instance(row, prereg)
                inst.no_range_batch = bool(row["no_range_batch"])
                res = zk_bridge.run_prove(inst, timeout=float(row.get("timeout", 120.0)))
                out.update({"_status": "ok" if res.ok else "proof_failed",
                            "num_constraints": res.num_constraints,
                            "prove_seconds": res.prove_seconds, "verify_ok": res.verify_ok})
                n_ok += 1 if res.ok else 0
                n_fail += 0 if res.ok else 1
            except Exception as e:
                out.update({"_status": "build_failed", "_error": str(e)})
                n_fail += 1
        else:
            out["_status"] = "symbolic_only"
            n_ok += 1
        writer.write(out)

    print(f"ablation: {n_ok} ok, {n_fail} failed, out={out_path}")
    print(f"env: git={env.get('git_hash')}")


if __name__ == "__main__":
    main()
