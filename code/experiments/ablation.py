#!/usr/bin/env python
"""Ablation -- measured implementation switches (Table ablation-a).

Implemented circuits are proved with gnark. The incircuit-gen variant remains
an explicit native-only baseline because its generator circuit is not implemented.

Usage: python -m experiments.ablation --config experiments/configs/core.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import (common_arg_parser, load_config, load_prereg, generate_grid,
                                 row_hash, apply_shard, ResultWriter, default_out_path, capture_env,
                                 release_process_memory)
from experiments.common import progress
from commitjl import zk_bridge
from experiments.rq2_proof_cost import build_zk_instance
from experiments.pipeline import PipelineCache, run_native

VARIANTS = ["sparse", "dense", "coord", "pca-public", "prepublished", "full", "incircuit-gen"]


def main():
    ap = common_arg_parser("Ablation: implementation switches")
    args = ap.parse_args()
    cfg = load_config(args.config)
    prereg = load_prereg(args.prereg)
    base_rows = generate_grid(cfg)
    # The ablation runner owns the variant axis. Configurations shared with
    # RQ2 may already contain it, so remove and deduplicate that axis here.
    deduplicated = {}
    for row in base_rows:
        base = {k: v for k, v in row.items() if k not in {"variant", "no_range_batch"}}
        deduplicated[row_hash(base)] = base
    base_rows = list(deduplicated.values())
    if args.limit:
        base_rows = base_rows[: args.limit]

    out_path = args.out or default_out_path("ablation", args.shard)
    writer = ResultWriter(out_path, context=prereg)
    env = capture_env()
    n_ok, n_fail = 0, 0
    cache = PipelineCache()

    expanded = []
    for row in base_rows:
        for variant in VARIANTS:
            for no_range_batch in (False, True):
                expanded.append({**row, "variant": variant, "no_range_batch": no_range_batch})
    expanded = apply_shard(expanded, args.shard)

    for row in progress(expanded, desc="Ablation"):
        h = row_hash(row)
        if writer.should_skip(h):
            continue
        projection_for_variant = {
            "sparse": "sparse_jl", "dense": "rademacher", "coord": "coordinate",
            "pca-public": "pca", "prepublished": "sparse_jl", "full": "identity",
            "incircuit-gen": "sparse_jl",
        }
        try:
            out = dict(row)
            out.update({"_hash": h, "variant": row["variant"],
                        "no_range_batch": row["no_range_batch"]})

            if row["variant"] != "incircuit-gen":
                inst, n2, d2, k2, m2, q2 = build_zk_instance(row, prereg)
                inst.no_range_batch = bool(row["no_range_batch"])
                repetitions = int(row.get("n_reps_proof", prereg.get("n_reps_proof", 10)))
                warmup = int(row.get("n_warmup_proof", prereg.get("n_warmup_proof", 1)))
                res = zk_bridge.run_prove(inst, timeout=float(row.get("timeout", 120.0)) * (repetitions + warmup),
                                          repetitions=repetitions, warmup=warmup)
                out.update(inst.benchmark_metrics)
                out.update({"_status": "ok" if res.ok else "proof_failed",
                            "_error": res.error, "n_reps": repetitions,
                            "n_success": res.successful_proofs,
                            "prove_seconds_samples": res.raw.get("prove_seconds_samples", []),
                            "verify_seconds_samples": res.raw.get("verify_seconds_samples", []),
                            "execution_mode": "zk_proof",
                            "num_constraints": res.num_constraints,
                            "compile_seconds": res.compile_seconds,
                            "setup_seconds": res.setup_seconds,
                            "witness_seconds": res.witness_seconds,
                            "prove_seconds": res.prove_seconds,
                            "verify_seconds": res.verify_seconds,
                            "proof_bytes": res.proof_bytes,
                            "peak_rss_bytes": res.peak_rss_bytes_psutil,
                            "verify_ok": res.verify_ok})
                n_ok += 1 if res.ok else 0
                n_fail += 0 if res.ok else 1
            else:
                native_row = {**row, "projection": projection_for_variant[row["variant"]]}
                out.update(run_native(native_row, prereg, cache=cache))
                out.update({"_status": "native_only", "execution_mode": "native_only",
                            "circuit_available": False,
                            "implementation_status": "not_implemented_in_circuit",
                            "reporting_label": "Evaluated as native-only baseline",
                            "result_scope": "native_quality_only_no_proof_metrics"})
                n_ok += 1
        except Exception as e:
            out = {"_hash": h, "_status": "failed", "_error": str(e), **row}
            n_fail += 1
        writer.write(out)
        release_process_memory()

    print(f"ablation: {n_ok} ok, {n_fail} failed, out={out_path}")
    print(f"env: git={env.get('git_hash')}")

    writer.close()
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
