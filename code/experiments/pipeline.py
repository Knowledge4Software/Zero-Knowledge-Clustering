"""Shared native-pipeline helper used by rq1/rq3/rq4/rq5/ablation: load a
dataset row, build a projection, run paired k-means natively in full and
projected space, and compute the manuscript's core metrics. Kept separate
from common.py (generic grid/IO plumbing) so each RQ script stays short.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.loaders import load_dataset  # noqa: E402
from commitjl import projections as proj  # noqa: E402
from commitjl.clustering import kmeans_paired, nearest_centroid_assign, psi_hat, phi_X  # noqa: E402
from commitjl.metrics import sample_pair_set, delta_uv, distortion_summary, r_phi, clustering_agreement  # noqa: E402
from commitjl.quantize import scale_for_precision  # noqa: E402


def load_row_dataset(row: Dict[str, Any]) -> Dict[str, Any]:
    ds_spec = row["dataset"]
    loader, kwargs = ds_spec["loader"], dict(ds_spec.get("kwargs", {}))
    if "seed" not in kwargs and loader in ("gaussian_mixture", "dbscan_margin"):
        kwargs["seed"] = int(row.get("seed", 0))
    out = load_dataset(loader, **kwargs)
    return out


def build_projection(name: str, X: np.ndarray, m: int, q: int, seed: int, S: int) -> Dict:
    n, d = X.shape
    m_eff = min(m, d)
    if name == "identity":
        return proj.identity_projection(d)
    if name == "gaussian":
        return proj.gaussian_projection(m_eff, d, seed, S)
    if name == "rademacher":
        return proj.rademacher_projection(m_eff, d, seed)
    if name == "achlioptas":
        return proj.achlioptas_projection(m_eff, d, seed)
    if name == "sparse_jl":
        seed_bytes = int(seed).to_bytes(8, "little") + b"commitjl-native"
        return proj.sparse_jl_projection(m_eff, d, min(q, d), seed_bytes)
    if name == "coordinate":
        return proj.coordinate_sampling_projection(m_eff, d, seed)
    if name == "pca":
        return proj.pca_projection(X, m_eff, S)
    if name == "pisot":
        return proj.pisot_beta_projection(m_eff, d, seed, S)
    raise KeyError(f"unknown projection {name!r}")


def run_native(row: Dict[str, Any], prereg: Dict[str, Any]) -> Dict[str, Any]:
    """Runs one native fidelity/cost pipeline row; returns a flat dict of
    metrics ready to append to a parquet result file."""
    data = load_row_dataset(row)
    X = np.asarray(data["X"], dtype=np.float64)
    n, d = X.shape
    k = int(row.get("k", 5))
    seed = int(row.get("seed", 0))
    m = int(row.get("m", min(64, d)))
    q = int(row.get("q", 1))
    projection_name = row.get("projection", "sparse_jl")
    precision_bits = int(row.get("precision_bits", 16))
    S = scale_for_precision(precision_bits)

    full_runs = kmeans_paired(X, k, seeds=[seed])
    labels_full, centroids_full, phi_full = full_runs[0]

    P = build_projection(projection_name, X, m, q, seed, S)
    real_P = P["real"]
    Z = X @ real_P.T if real_P is not None else X
    Cz = centroids_full @ real_P.T if real_P is not None else centroids_full
    assign_proj = nearest_centroid_assign(Z, Cz)
    psi_proj = float(psi_hat(Z, Cz, assign_proj))

    phi_proj_in_full_space = phi_X(X, _centroids_from_labels(X, assign_proj, k))
    r_phi_val = r_phi(phi_proj_in_full_space, phi_full)

    pairs = sample_pair_set(n, min(int(prereg.get("pair_set_size", 500)), n * (n - 1) // 2 or 1), seed)
    delta = delta_uv(X, Z, pairs)
    dist_summary = distortion_summary(delta, float(prereg.get("eps", 0.15)))
    agree = clustering_agreement(labels_full, assign_proj)

    out = dict(row)
    out.update({
        "n": n, "d": d, "k": k, "m": int(real_P.shape[0]) if real_P is not None else d,
        "q": q, "seed": seed, "projection": projection_name,
        "phi_full": float(phi_full), "phi_proj": float(phi_proj_in_full_space),
        "r_phi": float(r_phi_val), "psi_hat": psi_proj,
        "ari": agree["ari"], "nmi": agree["nmi"], "agreement": agree["agreement"],
        "delta_median": dist_summary["median"], "delta_q95": dist_summary["q95"],
        "delta_q99": dist_summary["q99"], "delta_max": dist_summary["max"],
        "delta_exceedance": dist_summary["exceedance_rate"],
        "nnz_density": P["nnz"]["density"],
        "dataset_name": data["meta"]["name"], "dataset_sha256": data["meta"].get("sha256", ""),
        "fidelity_ok": bool(agree["ari"] >= float(prereg.get("tau_ARI", 0.8)) and
                             abs(r_phi_val - 1.0) <= float(prereg.get("tau_Phi", 0.1))),
    })
    return out


def normalize_for_quantization(X: np.ndarray, S: int, B: int, headroom: float = 0.9) -> np.ndarray:
    """Rescales X (in place of the raw dataset units) so that round(x*S) fits
    within [-B, B] with `headroom` margin, i.e. a generic host-side
    normalization consistent with quantize.py's documented bound B on
    |x_t|. Real datasets/synthetic generators are not pre-scaled to any
    particular unit, so callers that need to quantize (rq2/rq3/rq4 ZK and
    protocol paths) apply this first."""
    max_abs = float(np.max(np.abs(X))) if X.size else 1.0
    if max_abs <= 0:
        return X
    limit = (B * headroom) / S
    scale = min(1.0, limit / max_abs)
    return X * scale


def _centroids_from_labels(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    d = X.shape[1]
    C = np.zeros((k, d), dtype=np.float64)
    for j in range(k):
        mask = labels == j
        if np.any(mask):
            C[j] = X[mask].mean(axis=0)
    return C
