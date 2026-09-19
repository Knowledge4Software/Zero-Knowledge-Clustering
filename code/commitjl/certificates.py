"""Certificates: centroid margin, DBSCAN ambiguity band, cost bounds,
coverage/violation-rate/uncertified-stability.

Manuscript reference:
- Eq. (centroid-margin): assignment is certified stable whenever
  d2^2/d1^2 > (1+eps)/(1-eps), where d1<d2 are the closest/runner-up
  original-space distances.
- DBSCAN ambiguity band A_eps(rho) = [rho^2/(1+eps), rho^2/(1-eps)]; pairs
  outside the band are certified (their neighbor/non-neighbor status is
  provably unchanged under the JL event).
- Fixed-center cost lemma: (1-eps)*Phi_X(C) <= q^{-1}*Phi_{AX}(A C) <= (1+eps)*Phi_X(C).
- Certificate coverage = |S|/N; violation rate = |S \\ U| / max(1,|S|);
  uncertified stability = |U \\ S| / max(1, N-|S|).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


def centroid_margin_certificate(X: np.ndarray, centroids: np.ndarray, eps: float) -> np.ndarray:
    """Returns a boolean array of length n: True if point i is certified
    stable under distortion +-eps, i.e. d2^2/d1^2 > (1+eps)/(1-eps), where
    d1 <= d2 are the two smallest original-space squared distances to
    centroids."""
    X = np.asarray(X, dtype=np.float64)
    C = np.asarray(centroids, dtype=np.float64)
    diffs = X[:, None, :] - C[None, :, :]
    sq = np.sum(diffs ** 2, axis=2)  # (n, k)
    sorted_sq = np.sort(sq, axis=1)
    d1_sq = sorted_sq[:, 0]
    d2_sq = sorted_sq[:, 1] if sq.shape[1] > 1 else np.full(sq.shape[0], np.inf)
    threshold = (1 + eps) / (1 - eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(d1_sq > 0, d2_sq / np.maximum(d1_sq, 1e-300), np.inf)
    certified = ratio > threshold
    return certified


def dbscan_ambiguity_band(rho: float, eps: float) -> Tuple[float, float]:
    """A_eps(rho) = [rho^2/(1+eps), rho^2/(1-eps)]."""
    lo = (rho ** 2) / (1 + eps)
    hi = (rho ** 2) / (1 - eps) if eps < 1 else float("inf")
    return lo, hi


def dbscan_pair_certificate(X: np.ndarray, rho: float, eps: float) -> Dict[str, np.ndarray]:
    """For all pairs (i,j), i<j: certified if squared distance lies outside
    the ambiguity band A_eps(rho). Returns dict with pair indices and
    boolean 'certified' + 'orig_sqdist' arrays (upper triangle, flattened)."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    iu = np.triu_indices(n, k=1)
    diffs = X[iu[0]] - X[iu[1]]
    sqd = np.sum(diffs ** 2, axis=1)
    lo, hi = dbscan_ambiguity_band(rho, eps)
    certified = (sqd < lo) | (sqd > hi)
    return {"i": iu[0], "j": iu[1], "orig_sqdist": sqd, "certified": certified, "band": (lo, hi)}


def cost_bounds(Phi_X_C: float, Phi_AX_AC_over_q: float, eps: float) -> Dict[str, float]:
    """Fixed-center cost lemma check: verifies
    (1-eps)*Phi_X(C) <= Phi_AX(AC)/q <= (1+eps)*Phi_X(C)
    and returns the bounds plus a boolean 'holds'."""
    lower = (1 - eps) * Phi_X_C
    upper = (1 + eps) * Phi_X_C
    holds = (lower - 1e-9 <= Phi_AX_AC_over_q) and (Phi_AX_AC_over_q <= upper + 1e-9)
    return {"lower": lower, "upper": upper, "observed": Phi_AX_AC_over_q, "holds": bool(holds)}


@dataclass
class CoverageResult:
    coverage: float
    violation_rate: float
    uncertified_stability: float
    n_certified: int
    n_total: int


def coverage_violation_stability(certified: np.ndarray, observed_stable: np.ndarray) -> CoverageResult:
    """certified: bool array, S = {i : certified[i]}.
    observed_stable: bool array, U = {i : observed_stable[i]} (assignment/
    adjacency unchanged after projection, measured empirically).

    coverage = |S|/N
    violation_rate = |S \\ U| / max(1,|S|)   (certified but NOT observed stable -> violation)
    uncertified_stability = |U \\ S| / max(1, N-|S|)
    """
    certified = np.asarray(certified, dtype=bool)
    observed_stable = np.asarray(observed_stable, dtype=bool)
    N = len(certified)
    S = certified
    U = observed_stable
    n_S = int(np.sum(S))
    violation = S & (~U)
    uncertified_stable = (~S) & U
    coverage = n_S / N if N else 0.0
    violation_rate = int(np.sum(violation)) / max(1, n_S)
    n_uncert = N - n_S
    uncertified_stability = int(np.sum(uncertified_stable)) / max(1, n_uncert)
    return CoverageResult(
        coverage=coverage,
        violation_rate=violation_rate,
        uncertified_stability=uncertified_stability,
        n_certified=n_S,
        n_total=N,
    )
