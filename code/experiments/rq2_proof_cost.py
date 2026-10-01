#!/usr/bin/env python
"""RQ2 -- measured proof cost for every implemented circuit variant.

Usage: python -m experiments.rq2_proof_cost --config experiments/configs/smoke.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from experiments.common import (common_arg_parser, load_config, load_prereg, generate_grid,
                                 row_hash, apply_shard, ResultWriter, default_out_path, capture_env,
                                 release_process_memory)
from experiments.common import progress
from experiments.pipeline import (PipelineCache, load_row_dataset, build_projection,
                                  normalize_for_quantization, run_native)
from commitjl.clustering import kmeans_paired, nearest_centroid_assign, psi_hat
from commitjl.quantize import scale_for_precision
from commitjl import zk_bridge
from commitjl.protocol import run_protocol
from commitjl.commit import commit_matrix
from commitjl.encoding import signed_to_field
from commitjl.hashing import BN254_FR, mimc_hash
from commitjl.metrics import sample_pair_set, delta_uv, distortion_summary, clustering_agreement
from commitjl.clustering import phi_X
from experiments.pipeline import _centroids_from_labels


def build_zk_instance(row, prereg):
    variant = row.get("variant", "sparse")
    supported = {"sparse", "dense", "coord", "pca-public", "pisot", "prepublished", "full"}
    if variant not in supported:
        raise NotImplementedError(f"variant {variant!r} has no implemented circuit")
    data = load_row_dataset(row)
    expected_sha = row.get("selection_dataset_sha256")
    if expected_sha and data["meta"].get("sha256") != expected_sha:
        raise ValueError("selected RQ1 dataset hash differs from current proof dataset")
    X = np.asarray(data["X"], dtype=np.float64)
    n, d = X.shape
    k = int(row.get("k", data.get("meta", {}).get("k") or 5))
    m = d if variant == "full" else min(int(row.get("m", 16)), d)
    q = int(row.get("q", 1))
    seed = int(row.get("seed", 0))
    precision_bits = int(row.get("precision_bits", 16))
    S = scale_for_precision(precision_bits)
    B = int(row.get("B", 4096))
    if not 1 <= B <= 4096:
        raise ValueError("B must be in [1, 4096] for the current circuit")
    from commitjl.quantize import quantize
    X = normalize_for_quantization(X, S, B)
    Xq = quantize(X, S, B)
    labels, centroids, _ = kmeans_paired(Xq.astype(np.float64), k, seeds=[seed])[0]
    Cq = np.round(centroids).astype(np.int64)
    if variant == "sparse":
        sparse_range_bits = max(int(row.get("range_bits", 48)), (d * B).bit_length() + 1)
        protocol_result = run_protocol(Xq, Cq, m=m, q=q, scale=S)
        A = protocol_result["A_dense"]
        inst = zk_bridge.build_sparse_instance_from_transcript(
            Xq, Cq, seed=protocol_result["seed"], rho=protocol_result["rho"],
            records=protocol_result["records"], A_dense=protocol_result["A_dense"],
            assignments=protocol_result["assignments"], psi_hat=protocol_result["Psi_hat"],
            c_x=protocol_result["C_X"], r_x=protocol_result["r_X"],
            c_theta=protocol_result["C_theta"], c_a=protocol_result["C_A"],
            q=q, m=m, backend=row.get("backend", "groth16"),
            range_bits=sparse_range_bits,
        )
    else:
        projection_names = {"dense": "rademacher", "coord": "coordinate", "pca-public": "pca",
                            "pisot": "pisot", "prepublished": "sparse_jl", "full": "identity"}
        P = build_projection(projection_names[variant], Xq.astype(float), m, q, seed, S)
        A = np.eye(d, dtype=np.int64) if P["int"] is None else np.asarray(P["int"], dtype=np.int64)
        m = A.shape[0]
        max_coefficient = int(np.max(np.abs(A))) if A.size else 1
        safe_range_bits = max(
            int(row.get("range_bits", 48)),
            (d * B * max(1, max_coefficient)).bit_length() + 1,
        )
        Z, Cz = (A @ Xq.T).T, (A @ Cq.T).T
        assignments = nearest_centroid_assign(Z, Cz)
        projected_cost = int(psi_hat(Z, Cz, assignments))
        merkle = commit_matrix(Xq)
        c_theta = zk_bridge.field_hash_theta(Cq)
        encoded_A = [signed_to_field(int(v), BN254_FR) for v in A.ravel(order="C")]
        c_a = mimc_hash(encoded_A) if variant in {"dense", "coord"} else 0
        centroids_field = np.array([[signed_to_field(int(v), BN254_FR) for v in r] for r in Cq], dtype=object)
        x_field = np.array([[signed_to_field(int(v), BN254_FR) for v in r] for r in Xq], dtype=object)
        inst = zk_bridge.ZKInstance(
            n=n, d=d, m=m, k=k, q=q, variant=variant, backend=row.get("backend", "groth16"),
            c_x=merkle.root, c_theta=c_theta, c_a=c_a, psi_hat=projected_cost,
            centroids=centroids_field, X=x_field, r_x=merkle.salt,
            assign=[int(a) for a in assignments],
            dense_A=A if variant in {"dense", "coord"} else None,
            constant_A=A if variant in {"pca-public", "pisot", "prepublished", "full"} else None,
            range_bits=safe_range_bits,
        )
    # Quality measurements must use precisely the quantized witness and matrix
    # authenticated by the proof, including the transcript-derived sparse seed.
    factor = 1.0
    if variant in {"sparse", "prepublished"}:
        factor = 1.0 / np.sqrt(q)
    elif variant == "dense":
        factor = 1.0 / np.sqrt(m)
    elif variant in {"pca-public", "pisot"}:
        factor = 1.0 / S
    projected = Xq.astype(float) @ (A.astype(float) * factor).T
    assigned = np.asarray(inst.assign, dtype=np.int64)
    baseline = nearest_centroid_assign(Xq, Cq)
    agreement = clustering_agreement(baseline, assigned)
    pairs = sample_pair_set(n, int(prereg.get("pair_set_size", 500)), seed)
    distortion = distortion_summary(delta_uv(Xq, projected, pairs), float(prereg.get("eps", 0.15)))
    full_cost = phi_X(Xq, Cq)
    assigned_cost = phi_X(Xq, _centroids_from_labels(Xq, assigned, k))
    inst.benchmark_metrics = {
        **row, "n": n, "d": d, "k": k, "m": m, "q": q,
        "requested_m": int(row.get("m", m)), "evaluation_mode": "quantized_proof_witness",
        "dataset_name": data["meta"]["name"], "dataset_sha256": data["meta"].get("sha256", ""),
        "ari": agreement["ari"], "nmi": agreement["nmi"], "agreement": agreement["agreement"],
        "phi_full": full_cost, "phi_proj": assigned_cost,
        "r_phi": assigned_cost / full_cost if full_cost else float("nan"),
        "psi_hat": float(inst.psi_hat), "psi_hat_exact": str(inst.psi_hat),
        "projection_normalization": float(factor),
        "delta_median": distortion["median"], "delta_q95": distortion["q95"],
        "delta_q99": distortion["q99"], "delta_max": distortion["max"],
        "delta_exceedance": distortion["exceedance_rate"],
        "c_x": str(inst.c_x), "c_theta": str(inst.c_theta), "c_a": str(inst.c_a),
        "projection_seed_hex": inst.projection_seed.hex() if inst.projection_seed else None,
        "rho_hex": inst.challenge_rho.hex() if inst.challenge_rho else None,
        "normalization_policy": "adaptive_to_B_and_S",
        "input_commitment_elements": int(n * d),
        "projected_distance_coordinate_ops": int(n * k * m),
        "commitment_scaling_dimension": "d",
        "distance_scaling_dimension": "m",
        "cost_scope_note": "Input commitment/authentication scales with original d; projected distance operations scale with reduced m.",
    }
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

    out_path = args.out or default_out_path("rq2_proof_cost", args.shard)
    writer = ResultWriter(out_path, context=prereg)
    env = capture_env()
    n_reps = int(cfg.get("fixed", {}).get("n_reps_proof", prereg.get("n_reps_proof", 10)))
    n_warmup = int(cfg.get("fixed", {}).get("n_warmup_proof", prereg.get("n_warmup_proof", 1)))

    n_ok, n_fail = 0, 0
    cache = PipelineCache()
    for row in progress(rows, desc="RQ2 proof cost"):
        h = row_hash(row)
        if writer.should_skip(h):
            continue
        m, q, k = int(row.get("m", 16)), int(row.get("q", 1)), int(row.get("k", 5))
        d = int(row.get("dataset", {}).get("kwargs", {}).get("d", 128))
        n_pts = int(row.get("dataset", {}).get("kwargs", {}).get("n", 500))
        if not zk_bridge.binary_available():
            writer.write({"_hash": h, "_status": "unavailable", **row,
                          "_error": "Go proof runner is unavailable"})
            n_fail += 1
            continue
        try:
            inst, n_pts, d, k, m, q = build_zk_instance(row, prereg)
        except Exception as e:
            writer.write({"_hash": h, "_status": "build_failed", "_error": str(e), **row})
            n_fail += 1
            continue

        print(f"RQ2 shape n={n_pts}, d={d}, m={m}, k={k}, q={q}: "
              f"input commitment covers {n_pts*d} coordinates (scales with d); "
              f"projected distances cover {n_pts*k*m} coordinates (scales with m).")
        res = zk_bridge.run_prove(
            inst, timeout=float(row.get("timeout", 120.0)) * (n_warmup + n_reps),
            repetitions=n_reps, warmup=n_warmup,
        )
        oks = [res] if res.ok else []
        base = dict(inst.benchmark_metrics)
        base.update({
            "_hash": h,
            "_status": "ok" if oks else "proof_failed",
            "n": n_pts, "d": d, "k": k, "m": m, "q": q,
            "n_reps": n_reps, "n_success": res.successful_proofs,
            "n_failure": n_reps - res.successful_proofs,
            "num_constraints": oks[0].num_constraints if oks else 0,
            "prove_seconds_mean": float(np.mean([r.prove_seconds for r in oks])) if oks else float("nan"),
            "verify_seconds_mean": float(np.mean([r.verify_seconds for r in oks])) if oks else float("nan"),
            "witness_seconds_mean": float(np.mean([r.witness_seconds for r in oks])) if oks else float("nan"),
            "setup_seconds_mean": float(np.mean([r.setup_seconds for r in oks])) if oks else float("nan"),
            "compile_seconds_mean": float(np.mean([r.compile_seconds for r in oks])) if oks else float("nan"),
            "proof_bytes": oks[0].proof_bytes if oks else 0,
            "verify_ok_rate": float(np.mean([r.verify_ok for r in oks])) if oks else 0.0,
            "peak_rss_bytes_psutil_max": res.peak_rss_bytes_psutil,
            "_error": res.error, "failure_kind": res.failure_kind,
            "setup_reuse_scope": "within_row_repetitions",
            "benchmark_scope_label": "Commitment overhead scales with d; distance operations scale with m",
            "prove_seconds_samples": res.raw.get("prove_seconds_samples", []),
            "verify_seconds_samples": res.raw.get("verify_seconds_samples", []),
        })
        writer.write(base)
        if oks:
            n_ok += 1
        else:
            n_fail += 1
        release_process_memory()

    print(f"rq2_proof_cost: {n_ok} ok, {n_fail} failed, out={out_path}")
    print(f"env: git={env.get('git_hash')}")

    writer.close()
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
