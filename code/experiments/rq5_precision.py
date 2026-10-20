"""One-factor quantization precision sweep with fixed physical input scale.

SparseJL column sparsity q is held fixed. Precision is precision_bits.
The current circuit checks the fixed integer bound B=4096, so changing S
alone does not change the implemented range gadget's constraint count.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from commitjl.clustering import kmeans_paired, nearest_centroid_assign, phi_X
from commitjl.cost_model import total_circuit_cost
from commitjl.metrics import clustering_agreement, delta_uv, distortion_summary, sample_pair_set
from commitjl.quantize import quantize, scale_for_precision
from experiments.common import (ResultWriter, apply_shard, capture_env, common_arg_parser,
                                default_out_path, generate_grid, load_config, load_prereg,
                                progress, release_process_memory, row_hash)
from experiments.pipeline import (build_projection, load_row_dataset,
                                  normalize_for_quantization, _centroids_from_labels)
from experiments.statistics import write_native_summary


def measure_precision_row(row, prereg, *, data=None, cache=None, artifact_path=None):
    if row.get("projection", "sparse_jl") != "sparse_jl":
        raise ValueError("precision sweep supports SparseJL only")
    data = data or load_row_dataset(row)
    X = np.asarray(data["X"], dtype=np.float64)
    n, d = X.shape
    bits = int(row["precision_bits"])
    max_bits = int(row.get("normalization_reference_bits", 20))
    if bits > max_bits:
        raise ValueError("precision_bits exceeds fixed normalization reference")
    B = int(row.get("B", 4096))
    if not 1 <= B <= 4096:
        raise ValueError("B must remain within the circuit input bound")
    m = min(int(row["m"]), d)
    q = int(row["q"])
    k = int(row["k"])
    seed = int(row["seed"])
    if not 1 <= q <= m:
        raise ValueError("SparseJL q must be in [1,m]")

    # Every precision sees the same physical data. Normalizing separately
    # at each S would erase the precision effect through adaptive rescaling.
    cache = cache if cache is not None else {}
    namespace=(data['meta'].get('sha256'),k,bits,max_bits,B,m,q)
    prepared_key=('prepared',namespace)
    S = scale_for_precision(bits)
    if prepared_key not in cache:
        physical = normalize_for_quantization(X, scale_for_precision(max_bits), B)
        Xq = quantize(physical, S, B)
        cache[prepared_key]=(physical,Xq,Xq.astype(np.float64)/S)
    physical,Xq,decoded=cache[prepared_key]
    start = int(row.get("clustering_start", seed))
    baseline_key=('baseline',namespace,start)
    if baseline_key not in cache:
        cache[baseline_key]=(kmeans_paired(physical,k,seeds=[start])[0],
                             kmeans_paired(decoded,k,seeds=[start])[0])
    (full_labels,_,full_cost),(quant_labels,quant_centroids,_)=cache[baseline_key]
    map_key=('map',namespace,seed)
    if map_key not in cache:
        P=build_projection('sparse_jl',decoded,m,q,seed,S)['real']
        cache[map_key]=(P,decoded@P.T)
    P,Z=cache[map_key]
    C = quant_centroids @ P.T
    assigned = nearest_centroid_assign(Z, C)
    agreement = clustering_agreement(full_labels, assigned)
    quant_agreement = clustering_agreement(full_labels, quant_labels)
    assigned_cost = phi_X(physical, _centroids_from_labels(physical, assigned, k))
    pairs = sample_pair_set(n, min(int(prereg.get("pair_set_size", 2000)), n * (n - 1) // 2), 20261005)
    delta=delta_uv(physical,Z,pairs)
    distortion = distortion_summary(delta, float(prereg.get("eps", .15)))
    if artifact_path is not None:
        from experiments.rq1_complete import save_npz
        artifact_path=Path(artifact_path)
        prepared_path=artifact_path.parent/'physical_quantized.npz'
        if not prepared_path.exists():
            save_npz(prepared_path,physical=physical,Xq=Xq,decoded=decoded,pairs=np.asarray(pairs))
        projection_path=artifact_path.parent/f'projection_seed{seed}.npz'
        if not projection_path.exists():
            save_npz(projection_path,P=P,Z=Z,pair_delta=delta)
        save_npz(artifact_path,full_labels=full_labels,quantized_labels=quant_labels,
                 projected_labels=assigned,quantized_centroids=quant_centroids)
    estimate = total_circuit_cost(n, d, m, k, q)
    return {
        **row, "n": n, "d": d, "m": m, "q": q, "k": k, "seed": seed,
        "dataset_name": data["meta"]["name"], "dataset_sha256": data["meta"].get("sha256", ""),
        "ari": agreement["ari"], "nmi": agreement["nmi"],
        "ari_quantized_full": quant_agreement["ari"],
        "phi_full": full_cost, "phi_proj": assigned_cost,
        "r_phi": assigned_cost / full_cost if full_cost > 0 else float("nan"),
        "delta_q95": distortion["q95"], "delta_exceedance": distortion["exceedance_rate"],
        "mean_abs_quantization_error": float(np.mean(np.abs(decoded - physical))),
        "nonzero_quantized_fraction": float(np.mean(Xq != 0)),
        "estimated_constraints": int(estimate["total_constraints"]),
        "constraint_count_kind": "symbolic_unit_estimate_not_measured_gnark_constraints",
        "constraint_precision_dependence": "none_at_fixed_B_4096_in_current_circuit",
        "normalization_policy": "single_fixed_physical_scale_at_reference_bits",
        "input_bound_B": B, "quantization_scale_S": S,
        "n_input_commitment_elements": n * d,
        "n_projected_distance_coordinates": n * k * m,
        "evaluation_mode": "native_quantized_fixed_centroid_assignment",
    }


def main():
    ap = common_arg_parser("RQ5 targeted quantization precision sweep")
    args = ap.parse_args()
    cfg = load_config(args.config)
    prereg = load_prereg(args.prereg)
    rows = apply_shard(generate_grid(cfg), args.shard)
    if args.limit:
        rows = rows[:args.limit]
    out_path = args.out or default_out_path("rq5_precision", args.shard)
    writer = ResultWriter(out_path, context=prereg)
    failures = 0
    for row in progress(rows, desc="Precision sweep"):
        h = row_hash(row)
        if writer.should_skip(h):
            continue
        try:
            result = measure_precision_row(row, prereg)
            writer.write({**result, "_hash": h, "_status": "ok"})
        except Exception as exc:
            writer.write({**row, "_hash": h, "_status": "failed", "_error": str(exc)})
            failures += 1
        finally:
            release_process_memory()
    writer.close()
    if not failures and not args.shard:
        write_native_summary(out_path,
            ("dataset_name", "dataset_sha256", "n", "d", "k", "m", "q", "precision_bits"),
            metrics=("ari", "delta_q95", "r_phi", "ari_quantized_full", "mean_abs_quantization_error"))
    print(f"rq5_precision: {len(rows)-failures} ok, {failures} failed, out={out_path}")
    print(f"env: git={capture_env().get('git_hash')}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
