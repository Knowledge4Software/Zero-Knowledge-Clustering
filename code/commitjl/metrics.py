"""Metrics: delta_uv distortion, r_Phi, ARI/NMI/assignment-agreement.

Manuscript reference: Sec. "Metrics, hypotheses, and statistics".
delta_uv = ||P(u-v)||_2^2 / ||u-v||_2^2 - 1, over a preregistered pair set P
(identical pairs excluded and reported separately). Summaries: median, Q95,
Q99 of |delta_uv|, max, exceedance rate Pr(|delta_uv|>eps).
r_Phi = Phi_X(C_hat_P) / Phi_X(C_hat_full), both evaluated in original space.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from .certificates import _pair_ranks_to_indices, pair_squared_distances
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def sample_pair_set(n: int, n_pairs: int, seed: int, exclude_identical: bool = True) -> List[Tuple[int, int]]:
    """Preregistered pair-set sampler: uniformly samples n_pairs distinct
    unordered index pairs (i, j), i != j by default."""
    rng = np.random.default_rng(seed)
    if n < 0 or n_pairs < 0:
        raise ValueError("pair counts and n must be nonnegative")
    total = n * (n - 1) // 2 if exclude_identical else n * (n + 1) // 2
    ranks = np.sort(rng.choice(total, size=min(n_pairs, total), replace=False))
    i, j = _pair_ranks_to_indices(n if exclude_identical else n + 1, ranks)
    if not exclude_identical:
        j = j - 1
    return list(zip(i.tolist(), j.tolist()))


def delta_uv(U: np.ndarray, V: np.ndarray, pairs: Sequence[Tuple[int, int]],
             P: Optional[np.ndarray] = None) -> np.ndarray:
    """delta_uv for each (i,j) in pairs, comparing original-space points U[i],U[j]
    (n x d) against their projected images V[i],V[j] (n x m) directly (V is
    already the projected representation, e.g. Z or X @ P.T). If both U and V
    refer to the same array (no projection), delta_uv will be ~0 by construction.
    """
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    idx_i = np.array([p[0] for p in pairs], dtype=np.int64)
    idx_j = np.array([p[1] for p in pairs], dtype=np.int64)
    orig_sq = pair_squared_distances(U, idx_i, idx_j)
    proj_sq = pair_squared_distances(V, idx_i, idx_j)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(orig_sq > 0, proj_sq / np.maximum(orig_sq, 1e-300) - 1.0, np.nan)
    return delta


def distortion_summary(delta: np.ndarray, eps: float) -> Dict[str, float]:
    d = np.asarray(delta, dtype=np.float64)
    d = d[~np.isnan(d)]
    abs_d = np.abs(d)
    if len(abs_d) == 0:
        return {"median": float("nan"), "q95": float("nan"), "q99": float("nan"),
                "max": float("nan"), "exceedance_rate": float("nan"), "n": 0}
    return {
        "median": float(np.median(abs_d)),
        "q95": float(np.quantile(abs_d, 0.95)),
        "q99": float(np.quantile(abs_d, 0.99)),
        "max": float(np.max(abs_d)),
        "exceedance_rate": float(np.mean(abs_d > eps)),
        "n": int(len(abs_d)),
    }


def r_phi(phi_projected_solution_in_original_space: float, phi_full_solution: float) -> float:
    """r_Phi = Phi_X(C_hat_P) / Phi_X(C_hat_full)."""
    if phi_full_solution == 0:
        return float("nan")
    return phi_projected_solution_in_original_space / phi_full_solution


def clustering_agreement(labels_a: np.ndarray, labels_b: np.ndarray) -> Dict[str, float]:
    """ARI, NMI, and raw assignment agreement rate between two labelings of
    the same points (e.g. projected vs full-dimensional, or vs ground truth)."""
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    ari = adjusted_rand_score(labels_a, labels_b)
    nmi = normalized_mutual_info_score(labels_a, labels_b)
    agreement = float(np.mean(labels_a == labels_b))
    return {"ari": float(ari), "nmi": float(nmi), "agreement": agreement}
