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
                                 row_hash, apply_shard, ResultWriter, default_out_path, capture_env)
from experiments.pipeline import load_row_dataset, build_projection, _centroids_from_labels
from commitjl.clustering import kmeans_paired, nearest_centroid_assign, dbscan_deterministic
from commitjl.certificates import (centroid_margin_certificate, dbscan_pair_certificate,
                                    coverage_violation_stability, cost_bounds)
from commitjl.clustering import phi_X
from commitjl.quantize import scale_for_precision


def run_centroid_cert(row, prereg):
    data = load_row_dataset(row)
    X = np.asarray(data["X"], dtype=np.float64)
    k = int(row.get("k", 5))
    seed = int(row.get("seed", 0))
    m = int(row.get("m", 16))
    q = int(row.get("q", 1))
    eps = float(prereg.get("eps", 0.15))
    labels, centroids, _ = kmeans_paired(X, k, seeds=[seed])[0]

    certified = centroid_margin_certificate(X, centroids, eps)

    S = scale_for_precision(int(row.get("precision_bits", 16)))
    P = build_projection(row.get("projection", "sparse_jl"), X, m, q, seed, S)
    Z = X @ P["real"].T
    Cz = centroids @ P["real"].T
    assign_proj = nearest_centroid_assign(Z, Cz)
    observed_stable = (assign_proj == labels)

    cov = coverage_violation_stability(certified, observed_stable)
    phi_full = phi_X(X, centroids)
    phi_proj_over_q = float(np.sum(np.min(
        np.sum((Z[:, None, :] - Cz[None, :, :]) ** 2, axis=2), axis=1))) / max(q, 1)
    bounds = cost_bounds(phi_full, phi_proj_over_q, eps)

    out = dict(row)
    out.update({"cert_type": "centroid_margin", "eps": eps,
                "coverage": cov.coverage, "violation_rate": cov.violation_rate,
                "uncertified_stability": cov.uncertified_stability,
                "n_certified": cov.n_certified, "n_total": cov.n_total,
                "cost_bound_holds": bounds["holds"], "cost_bound_lower": bounds["lower"],
                "cost_bound_upper": bounds["upper"], "cost_bound_observed": bounds["observed"]})
    return out


def run_dbscan_cert(row, prereg):
    data = load_row_dataset({**row, "dataset": {"loader": "dbscan_margin",
                             "kwargs": {"n": row.get("dataset", {}).get("kwargs", {}).get("n", 300),
                                        "d": row.get("dataset", {}).get("kwargs", {}).get("d", 64),
                                        "rho": row.get("rho", 1.0), "eps": prereg.get("eps", 0.15)}}})
    X = np.asarray(data["X"], dtype=np.float64)
    eps = float(prereg.get("eps", 0.15))
    rho = float(row.get("rho", 1.0))
    m, q, seed = int(row.get("m", 16)), int(row.get("q", 1)), int(row.get("seed", 0))

    pc = dbscan_pair_certificate(X, rho, eps)
    S = scale_for_precision(int(row.get("precision_bits", 16)))
    P = build_projection(row.get("projection", "sparse_jl"), X, m, q, seed, S)
    Z = X @ P["real"].T
    diffs = Z[pc["i"]] - Z[pc["j"]]
    proj_sqd = np.sum(diffs ** 2, axis=1)
    orig_neighbor = pc["orig_sqdist"] <= rho ** 2
    proj_neighbor = proj_sqd <= rho ** 2
    observed_stable = orig_neighbor == proj_neighbor

    cov = coverage_violation_stability(pc["certified"], observed_stable)
    out = dict(row)
    out.update({"cert_type": "dbscan_band", "eps": eps, "rho": rho,
                "coverage": cov.coverage, "violation_rate": cov.violation_rate,
                "uncertified_stability": cov.uncertified_stability,
                "n_certified": cov.n_certified, "n_total": cov.n_total})
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

    out_path = args.out or default_out_path("rq4_certificates")
    writer = ResultWriter(out_path)
    env = capture_env()
    n_ok, n_fail = 0, 0

    for row in rows:
        for fn, tag in ((run_centroid_cert, "centroid"), (run_dbscan_cert, "dbscan")):
            h = row_hash({**row, "cert_fn": tag})
            if writer.should_skip(h):
                continue
            try:
                out = fn(row, prereg)
                out["_hash"] = h
                out["_status"] = "ok"
                writer.write(out)
                n_ok += 1
            except Exception as e:
                writer.write({"_hash": h, "_status": "failed", "_error": str(e), "cert_fn": tag, **row})
                n_fail += 1

    if args.phase_diagram:
        base_row = rows[0] if rows else {"dataset": {"loader": "gaussian_mixture",
                                          "kwargs": {"n": 500, "d": 128, "k": 5}}, "seed": 0, "q": 1}
        m_list = [8, 16, 32, 64, 128]
        gamma_list = [1.0, 1.5, 2.0, 3.0, 4.0]
        for m in m_list:
            for gamma in gamma_list:
                ds = dict(base_row.get("dataset", {}))
                kwargs = dict(ds.get("kwargs", {}))
                kwargs["gamma"] = gamma
                ds["kwargs"] = kwargs
                phase_row = {**base_row, "dataset": ds, "m": m}
                h = row_hash({**phase_row, "cert_fn": "phase"})
                if writer.should_skip(h):
                    continue
                try:
                    out = run_centroid_cert(phase_row, prereg)
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

    print(f"rq4_certificates: {n_ok} ok, {n_fail} failed, out={out_path}")
    print(f"env: git={env.get('git_hash')}")


if __name__ == "__main__":
    main()
