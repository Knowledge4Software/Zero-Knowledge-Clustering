"""Projection baselines (Table "Projection baselines and circuit-relevant
characteristics"): identity, Gaussian JL, Rademacher JL, Achlioptas(s=3),
SparseJL(q), coordinate sampling, PCA (public, quantized), Pisot-beta.

Each `make_*` function returns a dict with keys:
  - "real": (m, d) real-valued projection matrix (or None for identity)
  - "int": (m, d) integer-quantized projection matrix
  - "nnz": non-zero count report (dict: total, per_row_mean, per_col_mean)
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .quantize import quantize
from .sparse_gen import sparse_matrix_gen


def _nnz_report(M: np.ndarray) -> Dict[str, float]:
    nz = np.count_nonzero(M)
    m, d = M.shape
    return {
        "total": int(nz),
        "per_row_mean": float(nz) / m if m else 0.0,
        "per_col_mean": float(nz) / d if d else 0.0,
        "density": float(nz) / (m * d) if m * d else 0.0,
    }


def identity_projection(d: int) -> Dict:
    """No reduction: identity map (m=d)."""
    M = np.eye(d, dtype=np.float64)
    return {"real": M, "int": M.astype(np.int64), "nnz": _nnz_report(M)}


def gaussian_projection(m: int, d: int, seed: int, S: int, B: Optional[int] = None) -> Dict:
    rng = np.random.default_rng(seed)
    real = rng.normal(loc=0.0, scale=1.0 / np.sqrt(m), size=(m, d))
    B = B if B is not None else int(10 * S)
    integer = quantize(real, S, B)
    return {"real": real, "int": integer, "nnz": _nnz_report(real)}


def rademacher_projection(m: int, d: int, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(m, d)) * 2 - 1
    real = signs.astype(np.float64) / np.sqrt(m)
    integer = signs.astype(np.int64)  # exact +-1, no quantization error
    return {"real": real, "int": integer, "nnz": _nnz_report(real)}


def achlioptas_projection(m: int, d: int, seed: int, s: int = 3) -> Dict:
    """Achlioptas sparse random projection: entries in {-1,0,+1} with
    P(+1)=P(-1)=1/(2s), P(0)=1-1/s, scaled by sqrt(s/m)."""
    rng = np.random.default_rng(seed)
    u = rng.random(size=(m, d))
    entries = np.zeros((m, d), dtype=np.int64)
    entries[u < 1.0 / (2 * s)] = 1
    entries[(u >= 1.0 / (2 * s)) & (u < 1.0 / s)] = -1
    real = entries.astype(np.float64) * np.sqrt(s / m)
    return {"real": real, "int": entries, "nnz": _nnz_report(entries)}


def sparse_jl_projection(m: int, d: int, q: int, seed_bytes: bytes) -> Dict:
    """Primary candidate: canonical SparseMatrixGen(s, m, d, q). Integer form
    is the raw {-1,0,+1} matrix A; real form is P = A/sqrt(q)."""
    records, A = sparse_matrix_gen(seed_bytes, m, d, q)
    real = A.astype(np.float64) / np.sqrt(q)
    return {"real": real, "int": A, "nnz": _nnz_report(A), "records": records}


def coordinate_sampling_projection(m: int, d: int, seed: int) -> Dict:
    """Cheap naive baseline: {0,1} selection matrix picking m of d coordinates
    uniformly without replacement."""
    rng = np.random.default_rng(seed)
    cols = rng.choice(d, size=min(m, d), replace=False)
    M = np.zeros((m, d), dtype=np.int64)
    for r, c in enumerate(cols):
        M[r, c] = 1
    real = M.astype(np.float64)
    return {"real": real, "int": M, "nnz": _nnz_report(M)}


def pca_projection(X: np.ndarray, m: int, S: int, B: Optional[int] = None) -> Dict:
    """Public, quantized PCA projection matrix (top-m principal directions).
    Data-dependent baseline; PCA training itself is not proved (per Table
    zkperf-a note 'excludes PCA training proof')."""
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD-based PCA; components are rows of Vt
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    m_eff = min(m, Vt.shape[0])
    real = Vt[:m_eff, :]
    B = B if B is not None else int(10 * S)
    integer = quantize(real, S, B)
    return {"real": real, "int": integer, "nnz": _nnz_report(real)}


def pisot_beta_projection(m: int, d: int, seed: int, S: int, B: Optional[int] = None,
                            beta: Optional[float] = None) -> Dict:
    """Deterministic Pisot-number-based projection.

    NOTE (own reproduction / explicit assumption): the manuscript cites a
    "Pisot-beta ZK projection" (dey2026pisot) only as a related-work baseline
    with a deterministic, orbit-derived, publicly reproducible coefficient
    structure and does not give this excerpt of manuscript.tex an explicit
    construction formula (Table "transforms" row for Pisot-beta literally
    says "inspect implementation" / "measure from reproduced artifact").
    This function is therefore OUR OWN reproduction/assumption of a
    plausible Pisot-beta construction, not a verified match to the cited
    paper: coefficients are generated from powers of a Pisot number beta
    (default: the plastic number, the smallest Pisot number, root of
    x^3 = x + 1) reduced into a bounded coefficient range, i.e. row r,
    column t entry is derived from the fractional part of beta^(r*d+t) via a
    deterministic seed-independent expansion (seed only selects a
    pseudo-random row/column permutation, keeping the coefficient sequence
    itself public/reproducible as the manuscript requires).
    """
    if beta is None:
        # plastic number (minimal Pisot number), root of x^3 - x - 1 = 0
        beta = 1.3247179572447460259609088544780973407344040569017333645340154
    idx = np.arange(m * d, dtype=np.float64).reshape(m, d)
    # deterministic orbit: fractional parts of beta^(k+1) for k = row-major index
    powers = beta ** (idx + 1.0)
    frac = powers - np.floor(powers)
    real = 2.0 * frac - 1.0  # map [0,1) -> [-1,1)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(d)
    real = real[:, perm]
    B = B if B is not None else int(10 * S)
    integer = quantize(real, S, B)
    return {"real": real, "int": integer, "nnz": _nnz_report(real)}
