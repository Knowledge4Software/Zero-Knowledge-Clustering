"""Shared native-pipeline helper used by rq1/rq3/rq4/rq5/ablation: load a
dataset row, build a projection, run paired k-means natively in full and
projected space, and compute the manuscript's core metrics. Kept separate
from common.py (generic grid/IO plumbing) so each RQ script stays short.
"""
from __future__ import annotations

import sys
import json
import time
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
from commitjl.certificates import pair_squared_distances
from experiments.cache import ByteLRU


class PipelineCache:
    """Process-local cache for immutable benchmark intermediates."""
    def __init__(self, max_bytes=512 * 1024**2):
        self.datasets = ByteLRU(max_bytes // 3)
        self.kmeans = ByteLRU(max_bytes // 12)
        self.projections = ByteLRU(max_bytes // 4)
        self.pca = ByteLRU(max_bytes // 4)
        self.pairs = ByteLRU(max_bytes // 48)
        self.pair_distances = ByteLRU(max_bytes // 48)
        self.certificates = ByteLRU(max_bytes // 24)

    @staticmethod
    def dataset_key(row):
        spec = dict(row["dataset"])
        kwargs = dict(spec.get("kwargs", {}))
        if "seed" not in kwargs and spec["loader"] in ("gaussian_mixture", "dbscan_margin"):
            kwargs["seed"] = int(row.get("seed", 0))
        return json.dumps({"loader": spec["loader"], "kwargs": kwargs}, sort_keys=True, separators=(",", ":"))

    def dataset(self, row):
        key = self.dataset_key(row)
        if key not in self.datasets:
            value = load_row_dataset(row)
            self.datasets[key] = value
            return value, key
        return self.datasets[key], key

    def kmeans_run(self, dataset_key, X, k, seed):
        key = (dataset_key, int(k), int(seed))
        if key not in self.kmeans:
            t0 = time.perf_counter()
            value = kmeans_paired(X, k, seeds=[seed])[0]
            result = (value, time.perf_counter() - t0)
            self.kmeans[key] = result
            return result
        return self.kmeans[key]

    def projection(self, dataset_key, name, X, m, q, seed, S):
        q_key = q if name == "sparse_jl" else None
        m_key = X.shape[1] if name == "identity" else min(m, X.shape[1])
        key = (dataset_key, name, m_key, q_key, None if name in {"pca", "identity"} else int(seed), int(S))
        if key not in self.projections:
            t0 = time.perf_counter()
            decomposition_s = 0.0
            if name == "pca":
                if dataset_key in self.pca:
                    components, decomposition_s = self.pca[dataset_key]
                else:
                    _, _, components = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
                    decomposition_s = time.perf_counter() - t0
                    self.pca[dataset_key] = (components, decomposition_s)
                t0 = time.perf_counter()
                real = components[:min(m, len(components))].copy()
                value = {"real": real, "int": proj.quantize(real, S, 10*S), "nnz": proj._nnz_report(real)}
            else:
                value = build_projection(name, X, m, q, seed, S)
            result = (value, time.perf_counter() - t0 + decomposition_s)
            self.projections[key] = result
            return result
        return self.projections[key]

    def pair_set(self, n, count, seed):
        key = (int(n), int(count), int(seed))
        if key not in self.pairs:
            value = sample_pair_set(n, count, seed)
            self.pairs[key] = value
            return value
        return self.pairs[key]

    def original_pair_distances(self, dataset_key, X, pairs):
        key = (dataset_key, tuple(pairs))
        if key not in self.pair_distances:
            i = np.fromiter((p[0] for p in pairs), dtype=np.int64)
            j = np.fromiter((p[1] for p in pairs), dtype=np.int64)
            value = pair_squared_distances(X, i, j)
            self.pair_distances[key] = value
            return value
        return self.pair_distances[key]


def load_row_dataset(row: Dict[str, Any]) -> Dict[str, Any]:
    ds_spec = row["dataset"]
    loader, kwargs = ds_spec["loader"], dict(ds_spec.get("kwargs", {}))
    if "seed" not in kwargs and loader in ("gaussian_mixture", "dbscan_margin"):
        kwargs["seed"] = int(row.get("seed", 0))
    out = load_dataset(loader, **kwargs)
    if out is None:
        raise RuntimeError(f"dataset {loader!r} is unavailable; install its optional dependencies or remove it explicitly from the campaign")
    return out


def build_projection(name: str, X: np.ndarray, m: int, q: int, seed: int, S: int) -> Dict:
    n, d = X.shape
    if n < 1 or d < 1 or m < 1:
        raise ValueError("projection requires nonempty data and positive m")
    m_eff = min(m, d)
    if name == "identity":
        return {"real": None, "int": None, "nnz": {"total": d, "density": 1/d}}
    if name == "gaussian":
        return proj.gaussian_projection(m_eff, d, seed, S)
    if name == "rademacher":
        return proj.rademacher_projection(m_eff, d, seed)
    if name == "achlioptas":
        return proj.achlioptas_projection(m_eff, d, seed)
    if name == "sparse_jl":
        seed_bytes = int(seed).to_bytes(8, "little") + b"commitjl-native"
        return proj.sparse_jl_projection(m_eff, d, q, seed_bytes)
    if name == "coordinate":
        return proj.coordinate_sampling_projection(m_eff, d, seed)
    if name == "pca":
        return proj.pca_projection(X, m_eff, S)
    if name == "pisot":
        return proj.pisot_beta_projection(m_eff, d, seed, S)
    raise KeyError(f"unknown projection {name!r}")


def run_native(row: Dict[str, Any], prereg: Dict[str, Any], cache: PipelineCache | None = None) -> Dict[str, Any]:
    """Runs one native fidelity/cost pipeline row; returns a flat dict of
    metrics ready to append to a parquet result file."""
    cache = cache or PipelineCache()
    data, dataset_key = cache.dataset(row)
    X = np.asarray(data["X"], dtype=np.float64)
    n, d = X.shape
    k = int(row.get("k", data.get("meta", {}).get("k") or 5))
    seed = int(row.get("seed", 0))
    m = int(row.get("m", min(64, d)))
    q = int(row.get("q", 1))
    projection_name = row.get("projection", "sparse_jl")
    precision_bits = int(row.get("precision_bits", 16))
    S = scale_for_precision(precision_bits)

    (labels_full, centroids_full, phi_full), t_kmeans = cache.kmeans_run(dataset_key, X, k, seed)

    P, t_build = cache.projection(dataset_key, projection_name, X, m, q, seed, S)
    real_P = P["real"]
    t0 = time.perf_counter()
    Z = X @ real_P.T if real_P is not None else X
    Cz = centroids_full @ real_P.T if real_P is not None else centroids_full
    assign_proj = nearest_centroid_assign(Z, Cz)
    psi_proj = float(psi_hat(Z, Cz, assign_proj))
    t_assign = time.perf_counter() - t0

    phi_proj_in_full_space = phi_X(X, _centroids_from_labels(X, assign_proj, k))
    r_phi_val = r_phi(phi_proj_in_full_space, phi_full)

    pairs = cache.pair_set(n, min(int(prereg.get("pair_set_size", 500)), n * (n - 1) // 2 or 1), seed)
    orig_sq = cache.original_pair_distances(dataset_key, X, pairs)
    pair_i = np.fromiter((p[0] for p in pairs), dtype=np.int64)
    pair_j = np.fromiter((p[1] for p in pairs), dtype=np.int64)
    proj_sq = pair_squared_distances(Z, pair_i, pair_j)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(orig_sq > 0, proj_sq / np.maximum(orig_sq, 1e-300) - 1.0, np.nan)
    dist_summary = distortion_summary(delta, float(prereg.get("eps", 0.15)))
    agree = clustering_agreement(labels_full, assign_proj)
    if data.get("y") is not None:
        ground_truth = np.asarray(data["y"])
        full_truth = clustering_agreement(ground_truth, labels_full)
        projected_truth = clustering_agreement(ground_truth, assign_proj)
    else:
        full_truth = projected_truth = {"ari": float("nan"), "nmi": float("nan")}

    out = dict(row)
    out.update({
        "n": n, "d": d, "k": k, "m": int(real_P.shape[0]) if real_P is not None else d,
        "q": q, "seed": seed, "projection": projection_name,
        "phi_full": float(phi_full), "phi_proj": float(phi_proj_in_full_space),
        "r_phi": float(r_phi_val), "psi_hat": psi_proj,
        "ari": agree["ari"], "nmi": agree["nmi"], "agreement": agree["agreement"],
        "ari_ground_truth_full": full_truth["ari"],
        "ari_ground_truth_projected": projected_truth["ari"],
        "nmi_ground_truth_full": full_truth["nmi"],
        "nmi_ground_truth_projected": projected_truth["nmi"],
        "delta_median": dist_summary["median"], "delta_q95": dist_summary["q95"],
        "delta_q99": dist_summary["q99"], "delta_max": dist_summary["max"],
        "delta_exceedance": dist_summary["exceedance_rate"],
        "nnz_density": P["nnz"]["density"],
        "dataset_name": data["meta"]["name"], "dataset_sha256": data["meta"].get("sha256", ""),
        "requested_m": None if projection_name == "identity" else m,
        "parameter_active_q": projection_name == "sparse_jl",
        "evaluation_mode": "fixed_centroid_assignment",
        "t_kmeans_s": t_kmeans, "t_proj_build_s": t_build, "t_assign_s": t_assign,
        "t_total_s": t_kmeans + t_build + t_assign,
        "timing_semantics": "cached_component_timings_plus_measured_assignment",
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
    centers = []
    for j in range(k):
        mask = labels == j
        if np.any(mask):
            centers.append(X[mask].mean(axis=0))
    return np.asarray(centers, dtype=np.float64).reshape(-1, d)
