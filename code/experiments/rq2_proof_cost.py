#!/usr/bin/env python
"""RQ2 -- proof cost: builds a ZK instance per grid row (core.yaml/smoke.yaml)
and invokes zk_bridge.run_prove: 1 warm-up + `n_reps_proof` reps, recording
every failure. Also attaches the symbolic cost_model prediction for
cross-checking (Table zkperf-a "measured" vs "symbolic" columns).

Only the `sparse` circuit variant is wired end-to-end through zk_bridge's
build_sparse_instance (P3); other named variants (dense/coord/pca-public/
prepublished/full) are recorded with symbolic cost_model rows only, with
`_status = "symbolic_only"`, per the plan's "cost model rows still produced
symbolically" fallback for anything not fully implemented in P3's bridge.

Usage: python -m experiments.rq2_proof_cost --config experiments/configs/smoke.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from experiments.common import (common_arg_parser, load_config, load_prereg, generate_grid,
                                 row_hash, apply_shard, ResultWriter, default_out_path, capture_env)
from experiments.pipeline import load_row_dataset, build_projection, _centroids_from_labels, normalize_for_quantization
from commitjl.clustering import kmeans_paired
from commitjl.cost_model import total_circuit_cost, full_vs_sparse_reduction
from commitjl.quantize import scale_for_precision
from commitjl import zk_bridge


def build_zk_instance(row, prereg):
    variant = row.get("variant", "sparse")
    if variant != "sparse":
        raise NotImplementedError(f"variant {variant!r} not wired through zk_bridge in this build")
    data = load_row_dataset(row)
    X = np.asarray(data["X"], dtype=np.float64)
    n, d = X.shape
    k = int(row.get("k", 5))
    m = int(row.get("m", 16))
    q = int(row.get("q", 1))
    seed = int(row.get("seed", 0))
    precision_bits = int(row.get("precision_bits", 16))
    S = scale_for_precision(precision_bits)
    B = int(row.get("B", 4096))
    from commitjl.quantize import quantize
    X = normalize_for_quantization(X, S, B)
    Xq = quantize(X, S, B)
    labels, centroids, _ = kmeans_paired(Xq.astype(np.float64), k, seeds=[seed])[0]
    Cq = np.round(centroids).astype(np.int64)
    seed_bytes = int(seed).to_bytes(8, "little") + b"commitjl-zk"
    inst = zk_bridge.build_sparse_instance(Xq, Cq, seed_bytes, q, m,
                                            backend=row.get("backend", "groth16"))
    return inst, n, d, k, m, q


def main():
    ap = common_arg_parser("RQ2 proof cost")
    args = ap.parse_args()
    cfg = load_config(args.config)
    prereg = load_prereg(args.prereg)
    rows = generate_grid(cfg)
    for r in rows:
        r.setdefault("variant", "sparse")
    rows = apply_shard(rows, args.shard)
    if args.limit:
        rows = rows[: args.limit]

    out_path = args.out or default_out_path("rq2_proof_cost")
    writer = ResultWriter(out_path)
    env = capture_env()
    n_reps = int(cfg.get("fixed", {}).get("n_reps_proof", prereg.get("n_reps_proof", 10)))
    n_warmup = int(cfg.get("fixed", {}).get("n_warmup_proof", prereg.get("n_warmup_proof", 1)))

    n_ok, n_fail, n_sym = 0, 0, 0
    for row in rows:
        h = row_hash(row)
        if writer.should_skip(h):
            continue
        m, q, k = int(row.get("m", 16)), int(row.get("q", 1)), int(row.get("k", 5))
        d = int(row.get("dataset", {}).get("kwargs", {}).get("d", 128))
        n_pts = int(row.get("dataset", {}).get("kwargs", {}).get("n", 500))
        try:
            symbolic = total_circuit_cost(n_pts, d, m, k, q)
            reduction = full_vs_sparse_reduction(n_pts, d, m, k, q)
        except Exception:
            symbolic, reduction = {}, {}

        if row.get("variant", "sparse") != "sparse" or not zk_bridge.binary_available():
            writer.write({"_hash": h, "_status": "symbolic_only", **row,
                          "symbolic_total_constraints": symbolic.get("total_constraints"),
                          "reduction_pct": reduction.get("reduction_pct")})
            n_sym += 1
            continue
        try:
            inst, n_pts, d, k, m, q = build_zk_instance(row, prereg)
        except Exception as e:
            writer.write({"_hash": h, "_status": "build_failed", "_error": str(e), **row})
            n_fail += 1
            continue

        reps_results = []
        for rep in range(n_warmup + n_reps):
            res = zk_bridge.run_prove(inst, timeout=float(row.get("timeout", 120.0)))
            if rep >= n_warmup:
                reps_results.append(res)

        oks = [r for r in reps_results if r.ok]
        base = dict(row)
        base.update({
            "_hash": h,
            "_status": "ok" if oks else "proof_failed",
            "n": n_pts, "d": d, "k": k, "m": m, "q": q,
            "n_reps": len(reps_results), "n_success": len(oks),
            "n_failure": len(reps_results) - len(oks),
            "num_constraints": oks[0].num_constraints if oks else 0,
            "prove_seconds_mean": float(np.mean([r.prove_seconds for r in oks])) if oks else float("nan"),
            "verify_seconds_mean": float(np.mean([r.verify_seconds for r in oks])) if oks else float("nan"),
            "witness_seconds_mean": float(np.mean([r.witness_seconds for r in oks])) if oks else float("nan"),
            "setup_seconds_mean": float(np.mean([r.setup_seconds for r in oks])) if oks else float("nan"),
            "compile_seconds_mean": float(np.mean([r.compile_seconds for r in oks])) if oks else float("nan"),
            "proof_bytes": oks[0].proof_bytes if oks else 0,
            "verify_ok_rate": float(np.mean([r.verify_ok for r in oks])) if oks else 0.0,
            "peak_rss_bytes_psutil_max": max([r.peak_rss_bytes_psutil for r in reps_results], default=0),
            "symbolic_total_constraints": symbolic.get("total_constraints"),
            "reduction_pct": reduction.get("reduction_pct"),
        })
        writer.write(base)
        if oks:
            n_ok += 1
        else:
            n_fail += 1

    print(f"rq2_proof_cost: {n_ok} ok, {n_fail} failed, {n_sym} symbolic_only, out={out_path}")
    print(f"env: git={env.get('git_hash')}")


if __name__ == "__main__":
    main()
