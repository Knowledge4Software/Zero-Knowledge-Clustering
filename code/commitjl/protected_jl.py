"""Centroid-informed protected coordinates plus a fresh residual Sparse JL.

Research variant, not a replacement for the original oblivious JL theorem.
The basis is deterministic from already committed/public centroids and must
be fixed before the fresh projection challenge. No private points enter it.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


def centroid_basis(centroids: np.ndarray, rank: int | None = None) -> np.ndarray:
    centers = np.asarray(centroids, dtype=np.float64)
    if centers.ndim != 2 or not min(centers.shape) or not np.isfinite(centers).all():
        raise ValueError("Expected finite nonempty centroids")
    if rank is not None and (not isinstance(rank, (int, np.integer)) or rank < 0):
        raise ValueError("rank must be a nonnegative integer or None")
    differences = centers[1:] - centers[0]
    if not len(differences):
        return np.empty((centers.shape[1], 0))
    u, singular, _ = np.linalg.svd(differences.T, full_matrices=False)
    tolerance = np.finfo(float).eps * max(differences.shape) * (singular[0] if len(singular) else 0)
    available = int(np.count_nonzero(singular > tolerance))
    # Canonicalize signs; repeated singular values can still rotate between
    # BLAS implementations. Persist the actual basis for reproducibility.
    u = u[:, :available if rank is None else min(rank, available)].copy()
    for column in range(u.shape[1]):
        pivot = np.argmax(np.abs(u[:, column]))
        if u[pivot, column] < 0:
            u[:, column] *= -1
    return u


def protected_project(points: np.ndarray, basis: np.ndarray, sparse_map: csr_matrix | None,
                      origin: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(points, dtype=np.float64)
    u = np.asarray(basis, dtype=np.float64)
    if x.ndim != 2 or u.ndim != 2 or x.shape[1] != u.shape[0]:
        raise ValueError("Incompatible point and basis dimensions")
    if not np.isfinite(x).all() or not np.isfinite(u).all():
        raise ValueError("Nonfinite projection input")
    if not np.allclose(u.T @ u, np.eye(u.shape[1]), rtol=1e-10, atol=1e-10):
        raise ValueError("The protected basis must be orthonormal")
    shifted = x if origin is None else x - np.asarray(origin, dtype=np.float64)
    head = shifted @ u
    if sparse_map is None:
        return head
    if sparse_map.shape[1] != x.shape[1]:
        raise ValueError("Incompatible Sparse JL dimensions")
    # A(x-UU^T x) = Ax - (AU)(U^T x). Never allocate n*d residuals
    # or a dense m*d combined matrix in the normal native path.
    residual = (sparse_map @ shifted.T).T - head @ (sparse_map @ u).T
    return np.concatenate((head, residual), axis=1)


def combined_matrix(basis: np.ndarray, sparse_map: csr_matrix) -> np.ndarray:
    """Small-test/reference matrix; do not materialize for large native runs."""
    u = np.asarray(basis, dtype=float)
    return np.vstack((u.T, sparse_map.toarray() - (sparse_map @ u) @ u.T))


def integer_score_assign(points: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, int]:
    """Exact affine-score reference, with Python integers and first-index ties.

    This is an exact Full control, not a compressed representation or JL map.
    """
    x, c = np.asarray(points), np.asarray(centers)
    if x.ndim != 2 or c.ndim != 2 or x.shape[1] != c.shape[1] or len(c) < 1:
        raise ValueError("Invalid point/center dimensions")
    if any(not isinstance(v, (int, np.integer)) for a in (x, c) for v in a.flat):
        raise ValueError("Integer coordinates required")
    x, c = x.astype(object), c.astype(object)
    norms = np.sum(c*c, axis=1)
    scores = 2*(x @ c.T) - norms
    labels = np.asarray(np.argmax(scores, axis=1), dtype=np.int64)
    cost = sum(int(np.sum(row*row) - scores[i, labels[i]]) for i, row in enumerate(x))
    return labels, cost


def integer_protected_project(points, basis, sparse_map, scale):
    """Exact fixed-point hybrid projection, guarded against int64 overflow."""
    x=np.asarray(points); b=np.asarray(basis)
    if x.ndim!=2 or b.ndim!=2 or b.shape[1]!=x.shape[1] or sparse_map.shape[1]!=x.shape[1] or not isinstance(scale,(int,np.integer)) or scale<1:
        raise ValueError('Invalid integer hybrid dimensions/scale')
    scale=int(scale)
    if any(not np.issubdtype(arr.dtype,np.integer) and
           (arr.dtype!=object or any(not isinstance(v,(int,np.integer)) for v in arr.flat)) for arr in (x,b)):
        raise ValueError('Integer inputs required')
    if np.any(~np.isin(sparse_map.data,[-1,0,1])):
        raise ValueError('Sparse map entries must be signed units')
    max_x=max(abs(int(np.min(x))),abs(int(np.max(x)))) if x.size else 0
    l1=[sum(abs(int(v)) for v in row) for row in b]
    bound=max([scale*max(l1,default=0),x.shape[1]*scale*scale+sum(v*v for v in l1)])*max_x
    # The bound includes every partial product/sum in the factorized path.
    dtype=object if bound>np.iinfo(np.int64).max else np.int64
    x=x.astype(dtype); b=b.astype(dtype)
    a=sparse_map.toarray().astype(dtype) if dtype==object else sparse_map.astype(np.int64)
    head=x@b.T
    residual=(a@x.T).T*(scale*scale)-head@(a@b.T).T
    return np.concatenate((head*scale,residual),axis=1)


def weighted_integer_assignment(points, centers, head_rank, q):
    from .clustering import _squared_distances
    x=np.asarray(points); c=np.asarray(centers)
    # Separate components; distance implementation already guards overflow.
    distances=q*_squared_distances(x[:,:head_rank],c[:,:head_rank]).astype(object)
    distances+=_squared_distances(x[:,head_rank:],c[:,head_rank:]).astype(object)
    labels=np.asarray(np.argmin(distances,axis=1),dtype=np.int64)
    cost=sum(int(distances[i,label]) for i,label in enumerate(labels))
    return labels,cost


def build_protected_instance(points, centers, seed, residual_m, q, rank, scale=64,
                             backend='groth16', specialize=False, scores=True):
    """Research proof instance. B is compiled into the verification relation.

    A deployment must authenticate B/centroids before the challenge and pin
    the corresponding verification key. This helper is not a new protocol.
    """
    from . import zk_bridge
    from .sparse_gen import sparse_matrix_gen_csr
    from dataclasses import replace
    if not isinstance(scale, (int, np.integer)) or not 1 <= scale <= 2**52:
        raise ValueError('Basis scale must be an integer in [1, 2**52] for float basis quantization')
    scale=int(scale)
    points, centers = np.asarray(points), np.asarray(centers)
    for arr in (np.asarray(points),np.asarray(centers)):
        if not np.issubdtype(arr.dtype,np.integer) and (arr.dtype!=object or any(not isinstance(v,(int,np.integer)) for v in arr.flat)):
            raise ValueError('Proof inputs must be quantized integers')
        if arr.size and max(abs(int(np.min(arr))),abs(int(np.max(arr))))>4096:
            raise ValueError('Proof inputs exceed the signed coordinate bound 4096')
    u=centroid_basis(centers,rank)
    b=np.rint(scale*u.T).astype(np.int64)
    if len(b)==0:
        # A zero protected rank is the ordinary Sparse JL control. Do not
        # scale its host objective while the circuit uses unscaled A*x.
        ordinary=zk_bridge.build_sparse_instance(points,centers,seed,q,residual_m,backend=backend)
        return replace(ordinary,specialize_sparse=specialize,score_distances=scores)
    # Basis is constructed before the caller-supplied fresh-map seed is used.
    _,a=sparse_matrix_gen_csr(seed,residual_m,points.shape[1],q)
    z=integer_protected_project(points,b,a,scale)
    projected_centers=integer_protected_project(centers,b,a,scale)
    labels,cost=weighted_integer_assignment(z,projected_centers,len(b),q)
    original=zk_bridge.build_sparse_instance(points,centers,seed,q,residual_m,backend=backend)
    l1=[sum(abs(int(v)) for v in row) for row in b]
    coordinate_bound=4096*max([points.shape[1]*scale*scale+sum(v*v for v in l1),
                              scale*max(l1,default=0)])
    range_bits=max(original.range_bits,coordinate_bound.bit_length()+1)
    return replace(original,assign=labels.tolist(),psi_hat=cost,protected_basis=b,basis_scale=scale,
                   specialize_sparse=specialize,score_distances=scores,range_bits=range_bits)
