"""Opt-in native research extension: centroid span plus residual PCA directions.

Training points enter this basis. This module makes no ZK/privacy or novelty
claim and leaves the canonical sparse generator and existing runners unchanged.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from .protected_jl import centroid_basis


def enrich_centroid_basis(training, centers, extra=12, *, random_state=1701,
                          iterated_power=2, n_oversamples=10, preserve_pca_order=False):
    """Return [U_centers, U_extra], preserving the complete centroid span.

    PCA sees only training residuals, centered by sklearn. Its orthonormal
    directions are reprojected against U_centers and rank-revealed by SVD.
    Degenerate residuals contribute fewer than ``extra`` usable directions.
    Caller must charge this complete function, including residual construction,
    to model fitting. Output dimension elsewhere is total head + tail.
    """
    x = np.asarray(training, dtype=np.float64)
    c = np.asarray(centers, dtype=np.float64)
    if x.ndim != 2 or c.ndim != 2 or not len(x) or not len(c) or x.shape[1] != c.shape[1]:
        raise ValueError('Expected nonempty training/center arrays with matching features')
    if not np.isfinite(x).all() or not np.isfinite(c).all():
        raise ValueError('Nonfinite training data or centers')
    if not isinstance(extra, (int, np.integer)) or extra < 0:
        raise ValueError('extra must be a nonnegative integer')
    u = centroid_basis(c)
    cap = min(int(extra), max(0, len(x)-1), x.shape[1]-u.shape[1])
    metadata = dict(requested_extra=int(extra), centroid_rank=u.shape[1], added_rank=0,
                    solver='randomized', iterated_power=iterated_power,
                    n_oversamples=n_oversamples, random_state=random_state,
                    training_rows=len(x), training_features=x.shape[1])
    if not cap:
        return u, metadata
    residual = x-c[0]
    original_norm = float(np.linalg.norm(residual))
    residual -= (residual@u)@u.T
    # The covariance directions concern variation, not the residual offset.
    residual -= residual.mean(axis=0)
    residual_norm = float(np.linalg.norm(residual))
    if residual_norm <= 64*np.finfo(float).eps*max(original_norm, np.finfo(float).tiny):
        return u, metadata
    pca = PCA(n_components=cap, svd_solver='randomized', random_state=random_state,
              iterated_power=iterated_power, n_oversamples=n_oversamples).fit(residual)
    tolerance = np.finfo(float).eps*max(residual.shape)*pca.singular_values_[0]
    usable = pca.singular_values_ > tolerance
    directions = pca.components_[usable].T.copy()
    # Two passes reduce loss of orthogonality for near-degenerate inputs.
    for _ in range(2):
        directions -= u@(u.T@directions)
    if not directions.shape[1]:
        return u, metadata
    if preserve_pca_order:
        # Nonpivoted QR preserves each leading PCA subspace. SVD of nearly
        # orthonormal columns can rotate their order within tied singular values.
        extra_u, triangular = np.linalg.qr(directions, mode='reduced')
        if np.any(np.abs(np.diag(triangular)) < 1e-10):
            raise ValueError('Ordered residual PCA has dependent directions')
        metadata['basis_order'] = 'PCA order with nonpivoted QR'
    else:
        # Preserve historical full-basis results byte for byte by default.
        extra_u, singular, _ = np.linalg.svd(directions, full_matrices=False)
        keep = singular > np.finfo(float).eps*max(directions.shape)*singular[0]
        extra_u = extra_u[:, keep]
    for j in range(extra_u.shape[1]):
        pivot = np.argmax(np.abs(extra_u[:, j]))
        if extra_u[pivot, j] < 0:
            extra_u[:, j] *= -1
    result = np.column_stack((u, extra_u))
    if not np.allclose(result.T@result, np.eye(result.shape[1]), atol=1e-10, rtol=1e-10):
        raise ValueError('Enriched basis lost orthogonality')
    metadata.update(added_rank=extra_u.shape[1], residual_training_frobenius=residual_norm,
                    residual_pca_singular_values=pca.singular_values_.tolist())
    return result, metadata
