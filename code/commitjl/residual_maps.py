"""Experimental residual sketches with the same protected orthonormal head.

Float64 native experiments only. New map families are not implemented in the
canonical commitment/ZK circuit. All actual random parameters are serializable.
"""
from __future__ import annotations

import hashlib
import numpy as np
from scipy.sparse import csr_matrix, vstack

from .reusable_native import array_hash, readonly
from .sparse_gen import sparse_matrix_gen_csr


VARIANTS = {
    'protected_plus12': ('canonical', 1),
    'jl_q2': ('canonical', 2),
    'jl_q4': ('canonical', 4),
    'jl_q8': ('canonical', 8),
    'block_q4': ('block', 4),
    'mixed_q1': ('mixed', 1),
    'mixed_q4': ('mixed', 4),
    'dense_gaussian': ('gaussian', 0),
}


def normalized_hadamard(points):
    """Row-wise orthogonal Sylvester transform; bounded row batches by caller."""
    answer = np.array(points, dtype=np.float64, copy=True, order='C')
    if answer.ndim != 2 or answer.shape[1] < 1 or answer.shape[1] & (answer.shape[1]-1):
        raise ValueError('Hadamard width must be a positive power of two')
    width = 1
    while width < answer.shape[1]:
        blocks = answer.reshape(len(answer), -1, 2*width)
        left = blocks[:, :, :width].copy()
        right = blocks[:, :, width:].copy()
        blocks[:, :, :width] = left + right
        blocks[:, :, width:] = left - right
        width *= 2
    answer /= np.sqrt(answer.shape[1])
    return answer


def block_matrix(seed, output_d, input_d, q):
    if not 1 <= q <= output_d:
        raise ValueError('Need at least one output row per block')
    # Unequal blocks differ by at most one row. Each column has one signed
    # entry per block, hence squared column norm one after 1/sqrt(q) scaling.
    widths = [output_d//q + int(b < output_d % q) for b in range(q)]
    blocks = []
    for block, width in enumerate(widths):
        block_seed = hashlib.sha256(seed+b'block'+block.to_bytes(4, 'little')).digest()
        _, matrix = sparse_matrix_gen_csr(block_seed, width, input_d, 1)
        blocks.append(matrix)
    result = vstack(blocks, format='csr', dtype=np.float64)
    result.data /= np.sqrt(q)
    return result


class ResidualProjection:
    """f(x) = [U^T(x-origin), S R (I-UU^T)(x-origin)].

    R is identity except for mixed variants, which use zero padding followed
    by a normalized Hadamard transform with independent random input signs.
    The dense Gaussian control has entries N(0, 1/tail). Its full cost counts.
    """
    version = 'residual-projection-v1'

    def __init__(self, d, m, basis, origin, kind, q, *, seed=None, parameters=None, tail_weight=1.0):
        self.method = 'hybrid'
        # Linear rescaling of the JL tail; 1.0 is the original map. Weights
        # below one keep a linear map but scale tail distances by tail_weight^2.
        self.tail_weight = float(tail_weight)
        if not np.isfinite(self.tail_weight) or self.tail_weight <= 0:
            raise ValueError('tail_weight must be positive and finite')
        self.d, self.m, self.kind, self.q = int(d), int(m), kind, int(q)
        if self.d < 1 or self.m < 1 or kind not in ['canonical', 'block', 'mixed', 'gaussian']:
            raise ValueError('Invalid projection specification')
        self.basis, self.origin = readonly(basis), readonly(origin)
        if self.basis.ndim != 2 or self.basis.shape[0] != self.d or self.origin.shape != (self.d,):
            raise ValueError('Invalid head/origin dimensions')
        if not np.isfinite(self.basis).all() or not np.isfinite(self.origin).all():
            raise ValueError('Nonfinite head/origin')
        rank = self.basis.shape[1]
        if not np.allclose(self.basis.T@self.basis, np.eye(rank), atol=1e-10, rtol=1e-10):
            raise ValueError('Head must be orthonormal')
        self.tail = self.m-rank
        self.input_d = (1 << (self.d-1).bit_length()) if kind == 'mixed' else self.d
        if self.tail < 1 or (kind != 'gaussian' and not 1 <= self.q <= self.tail) or (kind == 'gaussian' and self.q != 0):
            raise ValueError('Invalid tail width or column sparsity')
        self.sparse_map = self.dense_map = self.mix_signs = None
        if parameters is not None:
            if kind == 'gaussian':
                self.dense_map = readonly(parameters['dense_map'])
            else:
                self.sparse_map = csr_matrix((parameters['a_data'], parameters['a_indices'],
                    parameters['a_indptr']), shape=tuple(parameters['a_shape']), copy=True)
            if kind == 'mixed': self.mix_signs = readonly(parameters['mix_signs'])
        else:
            if not isinstance(seed, bytes): raise ValueError('A byte seed is required')
            if kind == 'gaussian':
                rng = np.random.default_rng(int.from_bytes(hashlib.sha256(seed+b'gaussian').digest(), 'little'))
                self.dense_map = readonly(rng.standard_normal((self.tail, self.d))/np.sqrt(self.tail))
            elif kind == 'block':
                self.sparse_map = block_matrix(seed, self.tail, self.input_d, self.q)
            else:
                _, a = sparse_matrix_gen_csr(seed, self.tail, self.input_d, self.q)
                self.sparse_map = a.astype(np.float64)
                self.sparse_map.data /= np.sqrt(self.q)
            if kind == 'mixed':
                rng = np.random.default_rng(int.from_bytes(hashlib.sha256(seed+b'mixing-signs').digest(), 'little'))
                self.mix_signs = readonly(2*rng.integers(2, size=self.input_d)-1)
        if self.dense_map is not None:
            if self.dense_map.shape != (self.tail, self.d) or not np.isfinite(self.dense_map).all():
                raise ValueError('Invalid dense map')
        if self.sparse_map is not None:
            a = self.sparse_map
            if a.shape != (self.tail, self.input_d) or not np.isfinite(a.data).all():
                raise ValueError('Invalid sparse map')
            a.sum_duplicates(); a.sort_indices()
            counts = np.bincount(a.indices, minlength=self.input_d)
            if not np.all(counts == self.q) or not np.allclose(np.abs(a.data), 1/np.sqrt(self.q)):
                raise ValueError('Invalid signed column sparsity/normalization')
            a.data.setflags(write=False); a.indices.setflags(write=False); a.indptr.setflags(write=False)
        if kind == 'mixed' and (self.mix_signs.shape != (self.input_d,) or not np.isin(self.mix_signs, [-1, 1]).all()):
            raise ValueError('Invalid mixing signs')
        self.au = readonly(self._sketch(self.basis.T).T)
        digest = hashlib.sha256(self.version.encode())
        for name, value in sorted(self.parameters().items()):
            digest.update(name.encode()); digest.update(array_hash(value).encode())
            digest.update(str(value.shape).encode()); digest.update(str(value.dtype).encode())
        self.fingerprint = digest.hexdigest()

    def _sketch(self, rows):
        if self.kind == 'gaussian': return rows@self.dense_map.T
        if self.kind == 'mixed':
            padded = np.zeros((len(rows), self.input_d))
            padded[:, :self.d] = rows
            padded *= self.mix_signs
            rows = normalized_hadamard(padded)
        return (self.sparse_map@rows.T).T

    def transform(self, points, batch_rows=256):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != self.d or not np.isfinite(points).all():
            raise ValueError('Invalid points')
        if not isinstance(batch_rows, (int, np.integer)) or batch_rows < 1:
            raise ValueError('Positive integer batch_rows required')
        result = np.empty((len(points), self.m)); rank = self.basis.shape[1]
        for first in range(0, len(points), batch_rows):
            shifted = points[first:first+batch_rows]-self.origin
            head = shifted@self.basis
            result[first:first+batch_rows, :rank] = head
            result[first:first+batch_rows, rank:] = self._sketch(shifted)-head@self.au.T
            if self.tail_weight != 1.0: result[first:first+batch_rows, rank:] *= self.tail_weight
        if not np.isfinite(result).all(): raise ValueError('Projection overflow')
        return result

    def parameters(self):
        result = dict(version=np.asarray(self.version), kind=np.asarray(self.kind),
            d=np.asarray(self.d), m=np.asarray(self.m), q=np.asarray(self.q),
            basis=self.basis, origin=self.origin)
        if self.dense_map is not None: result['dense_map'] = self.dense_map
        if self.sparse_map is not None:
            a = self.sparse_map
            result.update(a_data=a.data, a_indices=a.indices.astype(np.int64),
                a_indptr=a.indptr.astype(np.int64), a_shape=np.asarray(a.shape, dtype=np.int64))
        if self.mix_signs is not None: result['mix_signs'] = self.mix_signs
        # Omitted at 1.0 so original maps keep their fingerprints.
        if self.tail_weight != 1.0: result['tail_weight'] = np.asarray(self.tail_weight)
        return result

    @classmethod
    def from_parameters(cls, parameters):
        if str(parameters['version']) != cls.version: raise ValueError('Unknown map version')
        return cls(int(parameters['d']), int(parameters['m']), parameters['basis'],
            parameters['origin'], str(parameters['kind']), int(parameters['q']), parameters=parameters,
            tail_weight=float(parameters['tail_weight']) if 'tail_weight' in parameters else 1.0)


def make_residual_model(method, d, m, seed, centers, basis, tail_weight=1.0):
    kind, q = VARIANTS[method]
    return ResidualProjection(d, m, basis, centers[0], kind, q, seed=seed, tail_weight=tail_weight)


TAIL_WEIGHT_GRID = (0.25, 0.5, 1.0)


def select_tail_weight(projected_training, rank, k, *, grid=TAIL_WEIGHT_GRID, pairs=3, seed=0):
    """Label-free tail weight: maximize between-start KMeans agreement.

    ``projected_training`` is the unweighted (tail_weight=1) projection of
    training rows only. No labels, Full clustering or held-out rows are used.
    Ties go to the larger weight, i.e. closer to the original map.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    z = np.asarray(projected_training, dtype=np.float64)
    scores = {}
    for w in grid:
        zw = z.copy(); zw[:, rank:] *= w
        labels = [KMeans(k, n_init=1, random_state=seed+i).fit(zw).labels_ for i in range(pairs+1)]
        scores[w] = float(np.mean([adjusted_rand_score(labels[0], b) for b in labels[1:]]))
    best = max(sorted(grid, reverse=True), key=lambda w: scores[w])
    return best, scores


CONTINUATION_SCHEDULE = (0.25, 0.5, 1.0)


def sketch_partition_cost(z, labels):
    """k-means cost of ``labels`` measured in the committed (tail_weight=1) sketch."""
    z = np.asarray(z, dtype=np.float64); total = 0.0
    for j in np.unique(labels):
        part = z[labels == j]; total += float(((part-part.mean(0))**2).sum())
    return total


def continuation_cluster(z, rank, k, *, schedule=CONTINUATION_SCHEDULE, n_init=5, seed=0):
    """Search with a denoised view, score in the faithful sketch.

    ``z`` is the unweighted (tail_weight=1) projection; it is the only object
    that is committed or proven. The tail is temporarily scaled along a fixed
    schedule (coarse to faithful); each stage warm-starts Lloyd from the
    previous partition. Every stage's partition is a candidate, and the one
    with the lowest cost in ``z`` itself is returned. The schedule is a search
    heuristic only and needs no proof: a verifier checks the returned
    partition's cost in ``z``. Uses no labels, held-out rows or original data.
    """
    from sklearn.cluster import KMeans
    z = np.asarray(z, dtype=np.float64); labels = None; candidates = []
    for w in schedule:
        zw = z.copy(); zw[:, rank:] *= w
        if labels is None:
            model = KMeans(k, n_init=n_init, random_state=seed).fit(zw)
        else:
            init = np.array([zw[labels == j].mean(0) if np.any(labels == j) else zw[j] for j in range(k)])
            model = KMeans(k, n_init=1, init=init, random_state=seed).fit(zw)
        labels = model.labels_
        candidates.append(dict(weight=w, labels=labels, sketch_cost=sketch_partition_cost(z, labels)))
    best = min(candidates, key=lambda c: c['sketch_cost'])
    return best['labels'], best['weight'], candidates


def select_tail_weight_by_cost(z_train, x_train, z_val, x_val, rank, k, *,
                               grid=TAIL_WEIGHT_GRID, n_init=2, seed=0):
    """Label-free tail weight: minimize held-out original-space k-means cost.

    For each weight, cluster the weighted training sketch, form original-space
    centroids from those training labels, assign validation rows in the sketch
    and charge their original-space squared distance to the assigned centroid.
    ``z_*`` are unweighted (tail_weight=1) projections. No labels are used.
    Ties go to the larger weight, i.e. closer to the original map.
    """
    from sklearn.cluster import KMeans
    zt, zv = np.asarray(z_train, dtype=np.float64), np.asarray(z_val, dtype=np.float64)
    xt, xv = np.asarray(x_train, dtype=np.float64), np.asarray(x_val, dtype=np.float64)
    scores = {}
    for w in grid:
        scale = np.ones(zt.shape[1]); scale[rank:] = w
        model = KMeans(k, n_init=n_init, random_state=seed).fit(zt*scale)
        labels = model.labels_
        centers = np.array([xt[labels == j].mean(0) if np.any(labels == j) else xt.mean(0) for j in range(k)])
        assigned = model.predict(zv*scale)
        scores[w] = float(((xv-centers[assigned])**2).sum()/len(xv))
    best = min(sorted(grid, reverse=True), key=lambda w: scores[w])
    return best, scores
