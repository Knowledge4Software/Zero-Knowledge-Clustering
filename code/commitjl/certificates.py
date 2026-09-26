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
from typing import Dict, Optional, Tuple

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


def _all_pair_indices(n: int) -> Tuple[np.ndarray, np.ndarray]:
    iu = np.triu_indices(n, k=1)
    return iu[0], iu[1]


def dbscan_pair_certificate(X: np.ndarray, rho: float, eps: float,
                             max_pairs: Optional[int] = None,
                             seed: int = 0,
                             chunk_size: int = 50_000) -> Dict[str, np.ndarray]:
    """For all pairs (i,j), i<j: certified if squared distance lies outside
    the ambiguity band A_eps(rho). Returns dict with pair indices and
    boolean 'certified' + 'orig_sqdist' arrays (upper triangle, flattened).

    Memory: the naive implementation materializes a dense
    (n_choose_2, d) array of all pairwise differences, which is O(n^2 * d)
    and blows up for even moderately large n (e.g. ~12.5 GB for MNIST-sized
    n=2000, d=784). To keep peak memory bounded we instead:

      (a) process pairs in chunks of at most `chunk_size` at a time (each
          chunk only ever materializes a (chunk_size, d) diff array), and
      (b) optionally subsample the pair set uniformly (with `seed`) down to
          `max_pairs` when the full pair count n_choose_2 exceeds it.

    Subsampling only kicks in when the full pair count exceeds `max_pairs`,
    so default behaviour (max_pairs=None, or n small enough) is unchanged
    and exactly matches the old exact/dense computation. The returned dict
    always reports how many pairs were actually evaluated and whether
    subsampling happened, via 'pairs_evaluated', 'pairs_total', and
    'subsampled', so downstream reporting can state this honestly.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    i_all, j_all = _all_pair_indices(n)
    pairs_total = i_all.shape[0]

    subsampled = False
    if max_pairs is not None and pairs_total > max_pairs:
        rng = np.random.default_rng(seed)
        sel = rng.choice(pairs_total, size=max_pairs, replace=False)
        sel.sort()
        i_sel, j_sel = i_all[sel], j_all[sel]
        subsampled = True
    else:
        i_sel, j_sel = i_all, j_all

    n_pairs = i_sel.shape[0]
    sqd = np.empty(n_pairs, dtype=np.float64)
    for start in range(0, n_pairs, chunk_size):
        end = min(start + chunk_size, n_pairs)
        diffs = X[i_sel[start:end]] - X[j_sel[start:end]]
        sqd[start:end] = np.sum(diffs ** 2, axis=1)

    lo, hi = dbscan_ambiguity_band(rho, eps)
    certified = (sqd < lo) | (sqd > hi)
    return {
        "i": i_sel, "j": j_sel, "orig_sqdist": sqd, "certified": certified, "band": (lo, hi),
        "pairs_evaluated": int(n_pairs), "pairs_total": int(pairs_total), "subsampled": subsampled,
    }


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
