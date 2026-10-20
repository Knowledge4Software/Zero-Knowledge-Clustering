#!/usr/bin/env python
"""RQ4 -- certificates: centroid-margin and DBSCAN-ambiguity-band coverage /
violation-rate / uncertified-stability (Table cert-a rows 1-4), plus a
phase-diagram grid over (m, gamma) for Fig. phase-diagram.

Usage: python -m experiments.rq4_certificates --config experiments/configs/smoke.yaml
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
from experiments.pipeline import load_row_dataset, build_projection, _centroids_from_labels, PipelineCache
from commitjl.clustering import kmeans_paired, nearest_centroid_assign, dbscan_deterministic, _squared_distances
from commitjl.certificates import (centroid_margin_certificate, dbscan_pair_certificate,
                                    coverage_violation_stability, cost_bounds,
                                    pair_squared_distances)
from commitjl.clustering import phi_X
from commitjl.quantize import scale_for_precision


def run_centroid_cert(row, prereg, cache=None, *, artifact_path=None):
    cache = cache or PipelineCache()
    data, dataset_key = cache.dataset(row)
    X = np.asarray(data["X"], dtype=np.float64)
    k = int(row.get("k", 5))
    seed = int(row.get("seed", 0))
    m = int(row.get("m", 16))
    q = int(row.get("q", 1))
    eps = float(prereg.get("eps", 0.15))
    (labels, centroids, _), _ = cache.kmeans_run(dataset_key, X, k, int(row.get('clustering_start', seed)))
    labels = nearest_centroid_assign(X, centroids)

    certified = centroid_margin_certificate(X, centroids, eps)

    S = scale_for_precision(int(row.get("precision_bits", 16)))
    P, _ = cache.projection(dataset_key, row.get("projection", "sparse_jl"), X, m, q, seed, S)
    Z = X if P["real"] is None else X @ P["real"].T
    Cz = centroids if P["real"] is None else centroids @ P["real"].T
    assign_proj = nearest_centroid_assign(Z, Cz)
    observed_stable = (assign_proj == labels)

    original_sq = _squared_distances(X, centroids)
    projected_sq = _squared_distances(Z, Cz)
    event = np.all(np.abs(projected_sq-original_sq) <= eps*original_sq + 1e-12, axis=1)
    qualified = certified & event
    n_qualified = int(qualified.sum())
    n_violations = int(np.sum(qualified & ~observed_stable))
    if artifact_path is not None:
        from experiments.rq1_complete import save_npz
        save_npz(artifact_path, labels=labels, projected_labels=assign_proj, centroids=centroids,
                 original_sq=original_sq, projected_sq=projected_sq, margin=certified, embedding_event=event)

    cov = coverage_violation_stability(certified, observed_stable)
    phi_full = phi_X(X, centroids)
    # build_projection returns the normalized real map P=A/sqrt(q), so its
    # squared cost already equals ||A(x-c)||^2/q. Do not divide by q again.
    phi_projected_normalized = phi_X(Z, Cz)
    bounds = cost_bounds(phi_full, phi_projected_normalized, eps)

    out = dict(row)
    out.update({"cert_type": "centroid_margin", "eps": eps,
                "n": len(X), "d": X.shape[1], "m": Z.shape[1],
                "requested_m": m, "dataset_sha256": data["meta"].get("sha256", ""),
                "coverage": cov.coverage, "violation_rate": cov.violation_rate,
                "uncertified_stability": cov.uncertified_stability,
                "n_certified": cov.n_certified, "n_total": cov.n_total,
                "n_qualified": n_qualified, "n_violations": n_violations,
                "qualified_coverage": n_qualified/len(X),
                "conditional_violation_rate": n_violations/n_qualified if n_qualified else None,
                "global_embedding_event": bool(event.all()),
                "cost_bound_holds": bounds["holds"], "cost_bound_lower": bounds["lower"],
                "cost_bound_upper": bounds["upper"], "cost_bound_observed": bounds["observed"]})
    return out


def run_dbscan_cert(row, prereg, cache=None):
    cache = cache or PipelineCache()
    data, dataset_key = cache.dataset(row)
    X = np.asarray(data["X"], dtype=np.float64)
    eps = float(prereg.get("eps", 0.15))
    rho = float(row.get("rho", 1.0))
    m, q, seed = int(row.get("m", 16)), int(row.get("q", 1)), int(row.get("seed", 0))

    # Cap the number of pairs evaluated to keep peak memory bounded on large
    # n (dbscan_pair_certificate additionally chunks the computation so it
    # never materializes the full (n_choose_2, d) array at once). 2,000,000
    # pairs at d~1000 is already only tens of MB per chunk; the cap mainly
    # guards against very large n where n_choose_2 itself is huge.
    max_pairs = int(row.get("max_pairs", 2_000_000))
    pair_count = len(X) * (len(X) - 1) // 2
    certificate_key = (dataset_key, rho, eps, max_pairs,
                       seed if pair_count > max_pairs else None)
    if certificate_key in cache.certificates:
        pc = cache.certificates[certificate_key]
    else:
        pc = dbscan_pair_certificate(X, rho, eps, max_pairs=max_pairs, seed=seed)
        cache.certificates[certificate_key] = pc
    S = scale_for_precision(int(row.get("precision_bits", 16)))
    P, _ = cache.projection(dataset_key, row.get("projection", "sparse_jl"), X, m, q, seed, S)
    Z = X if P["real"] is None else X @ P["real"].T
    proj_sqd = pair_squared_distances(Z, pc["i"], pc["j"])
    orig_neighbor = pc["orig_sqdist"] <= rho ** 2
    proj_neighbor = proj_sqd <= rho ** 2
    observed_stable = orig_neighbor == proj_neighbor

    cov = coverage_violation_stability(pc["certified"], observed_stable)
    out = dict(row)
    out.update({"cert_type": "dbscan_band", "eps": eps, "rho": rho,
                "n": len(X), "d": X.shape[1], "m": Z.shape[1],
                "requested_m": m, "dataset_sha256": data["meta"].get("sha256", ""),
                "coverage": cov.coverage, "violation_rate": cov.violation_rate,
                "uncertified_stability": cov.uncertified_stability,
                "n_certified": cov.n_certified, "n_total": cov.n_total,
                "pairs_evaluated": pc["pairs_evaluated"], "pairs_total": pc["pairs_total"],
                "pairs_subsampled": pc["subsampled"]})
    return out


def main():
    ap = common_arg_parser("RQ4 certificates")
    ap.add_argument("--phase-diagram", action="store_true",
                     help="also sweep (m, gamma) for the phase-diagram figure")
    args = ap.parse_args()
    cfg = load_config(args.config)
    prereg = load_prereg(args.prereg)
    rows = generate_grid(cfg)
    for r in rows:
        r.setdefault("projection", "sparse_jl")
        r.setdefault("rho", 1.0)
    rows = apply_shard(rows, args.shard)
    if args.limit:
        rows = rows[: args.limit]

    out_path = args.out or default_out_path("rq4_certificates", args.shard)
    writer = ResultWriter(out_path, context=prereg)
    env = capture_env()
    n_ok, n_fail = 0, 0
    cache = PipelineCache()

    certificate_jobs = [(row, fn, tag) for row in rows for fn, tag in
                        ((run_centroid_cert, "centroid"), (run_dbscan_cert, "dbscan"))]
    for row, fn, tag in progress(certificate_jobs, desc="RQ4 certificates"):
        h = row_hash({**row, "cert_fn": tag})
        if writer.should_skip(h):
            continue
        try:
            out = fn(row, prereg, cache=cache)
            out["_hash"] = h
            out["_status"] = "ok"
            writer.write(out)
            n_ok += 1
        except Exception as e:
            writer.write({"_hash": h, "_status": "failed", "_error": str(e), "cert_fn": tag, **row})
            n_fail += 1
        finally:
            release_process_memory()

    if args.phase_diagram:
        base_row = rows[0] if rows else {"dataset": {"loader": "gaussian_mixture",
                                          "kwargs": {"n": 500, "d": 128, "k": 5}}, "seed": 0, "q": 1}
        m_list = [8, 16, 32, 64, 128]
        gamma_list = [1.0, 1.5, 2.0, 3.0, 4.0]
        phase_jobs = [(m, gamma) for m in m_list for gamma in gamma_list]
        for m, gamma in progress(phase_jobs, desc="RQ4 phase diagram"):
            source_kwargs = base_row.get("dataset", {}).get("kwargs", {})
            ds = {"loader": "gaussian_mixture", "kwargs": {
                "n": int(source_kwargs.get("n", 500)),
                "d": int(source_kwargs.get("d", 128)),
                "k": int(base_row.get("k", source_kwargs.get("k", 5))),
                "gamma": gamma,
                "imbalance": float(source_kwargs.get("imbalance", 1.0)),
                "seed": int(source_kwargs.get("seed", 12345)),
            }}
            phase_row = {**base_row, "dataset": ds, "m": m}
            h = row_hash({**phase_row, "cert_fn": "phase"})
            if writer.should_skip(h):
                continue
            try:
                out = run_centroid_cert(phase_row, prereg, cache=cache)
                out["_hash"] = h
                out["_status"] = "ok"
                out["cert_fn"] = "phase"
                out["gamma"] = gamma
                writer.write(out)
                n_ok += 1
            except Exception as e:
                writer.write({"_hash": h, "_status": "failed", "_error": str(e),
                              "cert_fn": "phase", "m": m, "gamma": gamma})
                n_fail += 1
            finally:
                release_process_memory()

    print(f"rq4_certificates: {n_ok} ok, {n_fail} failed, out={out_path}")
    print(f"env: git={env.get('git_hash')}")

    writer.close()
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
