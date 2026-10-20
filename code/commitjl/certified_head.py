"""Certified-head projection: a large learned head plus a small JL safety tail.

f(x) = [U^T(x-o), S (I-UU^T)(x-o)] with tail weight 1. U is any committed
orthonormal basis (here: centroid span + residual PCA, learned on training
rows). S is the canonical challenge-derived sparse JL map on the residual.

Why the head need not be proven to be PCA: for every pair x, y the head part
is exact and only the residual difference r = (I-UU^T)(x-y) is sketched, so
| ||f(x)-f(y)||^2 - ||x-y||^2 | = | ||S r||^2 - ||r||^2 |, which is bounded by
the JL error of S on r alone. The guarantee therefore holds for ANY
orthonormal U; a good U only makes ||r|| (and hence the absolute error)
small. A verifier needs U^T U = I, which Freivalds-style random checks give
in O(d h) work instead of proving an SVD.

Native float64 research module. The circuit does not yet implement the
orthonormality check or this head size.
"""
from __future__ import annotations

import numpy as np

from .enriched_jl import enrich_centroid_basis
from .residual_maps import ResidualProjection


def learn_certified_head(training, centers, head_rank, *, random_state=0):
    """Centroid span plus residual PCA directions, up to ``head_rank`` columns."""
    centroid_rank = np.linalg.matrix_rank(np.asarray(centers, float)[1:]-np.asarray(centers, float)[0])
    extra = max(0, int(head_rank)-int(centroid_rank))
    return enrich_centroid_basis(training, centers, extra=extra, random_state=random_state)


def freivalds_orthonormal(basis, *, rounds=8, seed=0, rtol=1e-8):
    """Probabilistic check that basis^T basis = I using only matrix-vector products.

    Each round draws a random vector v and tests basis^T (basis v) == v. A
    non-orthonormal basis passes one round with probability ~0 for Gaussian v.
    """
    u = np.asarray(basis, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for _ in range(int(rounds)):
        v = rng.standard_normal(u.shape[1])
        if not np.allclose(u.T@(u@v), v, rtol=rtol, atol=rtol*np.linalg.norm(v)):
            return False
    return True


def make_certified_head_model(d, m, seed, centers, basis):
    """Canonical q=1 sparse tail of width m - rank(basis), tail weight 1."""
    if not freivalds_orthonormal(basis):
        raise ValueError('Head basis failed the orthonormality check')
    return ResidualProjection(d, m, basis, np.asarray(centers, float)[0], 'canonical', 1, seed=seed)


def residual_distance_bound(points, basis, origin, model):
    """Per-pair absolute distortion is |‖S r‖² - ‖r‖²| for r the residual difference.

    Returns the residual energy share ‖(I-UU^T)(x-o)‖² / ‖x-o‖² per point; the
    smaller it is, the smaller the absolute distance error the tail can cause.
    """
    x = np.asarray(points, float)-origin
    head = x@basis
    total = (x**2).sum(1)
    return np.where(total > 0, 1-(head**2).sum(1)/np.where(total > 0, total, 1), 0.0)
