"""Integer-exact nearest-centroid assignment, Psi_hat, and clustering wrappers.

Manuscript reference: Sec. "Clustering objectives" (nearest-centroid
assignment with deterministic lowest-index tie-breaking; Phi_X); Proof
statement (iv)-(v) (a_i is the lowest-index minimizer of
||A(x_i - c_j)||_2^2; Psi_hat = sum_i ||A(x_i - c_{a_i})||_2^2).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans, DBSCAN

DEFAULT_KMEANS_SEEDS = list(range(10))  # 10 paired initialization seeds, per plan P2


def nearest_centroid_assign(Z: np.ndarray, projected_centroids: np.ndarray) -> np.ndarray:
    """Integer-exact nearest-centroid assignment with lowest-index tie-breaking.

    Z: (n, m) projected points (integer or real).
    projected_centroids: (k, m) projected centroids, same space as Z.
    Returns: (n,) int array of assignments a_i in [0, k).
    """
    Z = np.asarray(Z)
    C = np.asarray(projected_centroids)
    n = Z.shape[0]
    k = C.shape[0]
    # squared distances (n, k); use exact integer arithmetic when inputs are integer dtype
    diffs = Z[:, None, :] - C[None, :, :]
    sq = np.sum(diffs.astype(np.int64 if np.issubdtype(Z.dtype, np.integer) else np.float64) ** 2, axis=2)
    # lowest-index argmin: np.argmin already returns the first occurrence of the min
    assignments = np.argmin(sq, axis=1).astype(np.int64)
    return assignments


def psi_hat(Z: np.ndarray, projected_centroids: np.ndarray, assignments: np.ndarray) -> int:
    """Psi_hat = sum_i ||A(x_i - c_{a_i})||_2^2, computed in exact integer arithmetic
    when inputs are integer."""
    Z = np.asarray(Z)
    C = np.asarray(projected_centroids)
    assigned_c = C[assignments]
    diff = Z.astype(np.int64) - assigned_c.astype(np.int64)
    sq = np.sum(diff ** 2, axis=1)
    return int(np.sum(sq))


def phi_X(X: np.ndarray, centroids: np.ndarray) -> float:
    """Phi_X(C) = sum_i min_j ||x_i - c_j||_2^2, in the original space."""
    X = np.asarray(X, dtype=np.float64)
    C = np.asarray(centroids, dtype=np.float64)
    diffs = X[:, None, :] - C[None, :, :]
    sq = np.sum(diffs ** 2, axis=2)
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
