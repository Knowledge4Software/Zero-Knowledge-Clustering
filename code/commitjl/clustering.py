"""Integer-exact nearest-centroid assignment, Psi_hat, and clustering wrappers.

Manuscript reference: Sec. "Clustering objectives" (nearest-centroid
assignment with deterministic lowest-index tie-breaking; Phi_X); Proof
statement (iv)-(v) (a_i is the lowest-index minimizer of
||A(x_i - c_j)||_2^2; Psi_hat = sum_i ||A(x_i - c_{a_i})||_2^2).
"""
from __future__ import annotations

import math
from typing import List, Tuple, Union

import numpy as np
from sklearn.cluster import KMeans, DBSCAN

DEFAULT_KMEANS_SEEDS = list(range(10))  # 10 paired initialization seeds, per plan P2
_INT64_MAX = np.iinfo(np.int64).max
DISTANCE_WORKSPACE_BYTES = 32 * 1024**2


def _integer_array(a):
    return np.issubdtype(a.dtype, np.integer) or (
        a.dtype == object and all(isinstance(v, (int, np.integer)) for v in a.flat))


def _squared_distances(Z: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Return the exact/precision-preserving pairwise squared distances.

    Integer inputs use fast int64 arithmetic only when a worst-case bound
    proves that subtraction, squaring, and summation cannot overflow. Larger
    integer values fall back to Python's arbitrary-precision integers. Float
    inputs remain float64 and are never truncated to integers.
    """
    Z = np.asarray(Z)
    C = np.asarray(C)
    if Z.ndim != 2 or C.ndim != 2 or Z.shape[1] != C.shape[1]:
        raise ValueError(f"incompatible point/centroid shapes: {Z.shape} and {C.shape}")

    integer_inputs = _integer_array(Z) and _integer_array(C)
    # Slice both axes before constructing differences. This also bounds memory
    # for identity projections of wide datasets such as Arcene.
    per_pair = max(1, Z.shape[1]) * (96 if Z.dtype == object or C.dtype == object else 24)
    block = max(1, DISTANCE_WORKSPACE_BYTES // per_pair)
    if Z.shape[0] * C.shape[0] > block:
        result = np.empty((len(Z), len(C)), dtype=object if integer_inputs else np.float64)
        cb = min(max(1, len(C)), block)
        nb = max(1, block // cb)
        for i in range(0, len(Z), nb):
            for j in range(0, len(C), cb):
                result[i:i+nb, j:j+cb] = _squared_distances(Z[i:i+nb], C[j:j+cb])
        return result
    if not integer_inputs:
        diff = Z.astype(np.float64, copy=False)[:, None, :] - C.astype(np.float64, copy=False)[None, :, :]
        return np.einsum("nkd,nkd->nk", diff, diff, dtype=np.float64)

    if Z.size == 0 or C.size == 0:
        return np.empty((Z.shape[0], C.shape[0]), dtype=np.int64)

    z_min, z_max = int(np.min(Z)), int(np.max(Z))
    c_min, c_max = int(np.min(C)), int(np.max(C))
    max_abs_diff = max(abs(z_min - c_max), abs(z_max - c_min))
    d = Z.shape[1]
    safe_int64 = (max_abs_diff == 0 or max_abs_diff <= math.isqrt(_INT64_MAX // max(d, 1)))
    safe_int64 = safe_int64 and min(z_min, c_min) >= -2**63 and max(z_max, c_max) <= _INT64_MAX

    if safe_int64:
        diff = Z.astype(np.int64, copy=False)[:, None, :] - C.astype(np.int64, copy=False)[None, :, :]
        return np.einsum("nkd,nkd->nk", diff, diff, dtype=np.int64)

    # NumPy has no portable int128 dtype (notably on Windows). Object arrays
    # use Python integers and therefore preserve exact squared distances.
    diff = Z.astype(object)[:, None, :] - C.astype(object)[None, :, :]
    return np.sum(diff * diff, axis=2)


def nearest_centroid_assign(Z: np.ndarray, projected_centroids: np.ndarray) -> np.ndarray:
    """Integer-exact nearest-centroid assignment with lowest-index tie-breaking.

    Z: (n, m) projected points (integer or real).
    projected_centroids: (k, m) projected centroids, same space as Z.
    Returns: (n,) int array of assignments a_i in [0, k).
    """
    Z = np.asarray(Z)
    C = np.asarray(projected_centroids)
    sq = _squared_distances(Z, C)
    # lowest-index argmin: np.argmin already returns the first occurrence of the min
    assignments = np.argmin(sq, axis=1).astype(np.int64)
    return assignments


def psi_hat(Z: np.ndarray, projected_centroids: np.ndarray,
            assignments: np.ndarray) -> Union[int, float]:
    """Return ``sum_i ||z_i-c_{a_i}||_2^2`` without truncation or overflow.

    Integer inputs return an exact Python ``int``. Floating-point inputs
    retain float64 precision and return ``float``. Circuit-facing callers use
    quantized integer arrays, while native fidelity experiments use floats.
    """
    Z = np.asarray(Z)
    C = np.asarray(projected_centroids)
    assignments = np.asarray(assignments)
    if not _integer_array(assignments):
        raise TypeError("assignments must contain integer indices")
    assignments = assignments.astype(np.int64)
    if assignments.shape != (Z.shape[0],):
        raise ValueError(f"assignments must have shape ({Z.shape[0]},), got {assignments.shape}")
    if np.any(assignments < 0) or np.any(assignments >= C.shape[0]):
        raise ValueError("assignment index outside centroid range")

    integer_inputs = _integer_array(Z) and _integer_array(C)
    # Only evaluate the assigned center; computing the full n*k matrix here
    # duplicates the argmin work and can be substantially more expensive.
    total = 0 if integer_inputs else 0.0
    for j in np.unique(assignments):
        selected = _squared_distances(Z[assignments == j], C[j:j+1])[:, 0]
        total += sum(int(v) for v in selected) if integer_inputs else float(np.sum(selected))
    return total


def phi_X(X: np.ndarray, centroids: np.ndarray) -> float:
    """Phi_X(C) = sum_i min_j ||x_i - c_j||_2^2, in the original space."""
    X = np.asarray(X, dtype=np.float64)
    C = np.asarray(centroids, dtype=np.float64)
    sq = _squared_distances(X, C)
    return float(np.sum(np.min(sq, axis=1)))


def kmeans_paired(X: np.ndarray, k: int, seeds: List[int] = None) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Run sklearn KMeans with `seeds` (default: 10 paired init seeds).

    Returns a list of (labels, centroids, inertia) tuples, one per seed, so
    that other projections/methods can be run with the *same* paired seeds
    for a fair comparison (per plan: "clustering initialization is paired
    across methods with 10 fixed initialization seeds").
    """
    if seeds is None:
        seeds = DEFAULT_KMEANS_SEEDS
    results = []
    for s in seeds:
        km = KMeans(n_clusters=k, init="k-means++", n_init=1, random_state=s)
        labels = km.fit_predict(X)
        results.append((labels, km.cluster_centers_, float(km.inertia_)))
    return results


def dbscan_deterministic(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Deterministic DBSCAN: fixed point ordering (input order preserved,
    no shuffling) with sklearn's DBSCAN, which is itself deterministic given
    fixed input order and algorithm='auto' (no randomness in the core
    algorithm; only neighbor-query tie order depends on point index, which we
    fix by not permuting X)."""
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    return labels
