"""Reusable native representations; separate from integer/ZK proof execution.

All three methods cache point norms. Protected JL also caches A@U and checks
the basis once. Persistent caches bind the input and model by content hashes.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

VERSION = 'reusable-native-v1'


def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(memoryview(a).cast('B') if a.size else b'').hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def readonly(a):
    a = np.array(a, dtype=np.float64, copy=True)
    a.setflags(write=False)
    return a


def stable_squared_distances(x, y, xn=None, yn=None):
    """BLAS distances with direct differences for cancellation-prone pairs."""
    xn = np.einsum('ij,ij->i',x,x) if xn is None else xn
    yn = np.einsum('ij,ij->i',y,y) if yn is None else yn
    scale = xn[:,None]+yn[None,:]
    answer = scale-2*(x@y.T)
    if not np.isfinite(answer).all(): raise ValueError('Distance overflow')
    i,j = np.nonzero((answer <= 64*np.finfo(float).eps*scale) & (scale > 0))
    batch = max(1,32*1024**2//max(1,3*x.shape[1]*8))
    for first in range(0,len(i),batch):
        a,b=i[first:first+batch],j[first:first+batch]
        delta=x[a]-y[b]
        answer[a,b]=np.einsum('ij,ij->i',delta,delta)
    np.maximum(answer,0,out=answer)
    if not np.isfinite(answer).all(): raise ValueError('Distance overflow')
    return answer


class NativeProjection:
    """Already specified projection; PCA fitting and centroid training are external."""
    def __setattr__(self, name, value):
        if getattr(self, '_frozen', False):
            raise AttributeError('Projection parameters are immutable; construct a new model')
        object.__setattr__(self, name, value)

    def __init__(self, method, d, *, basis=None, sparse_map=None, components=None, origin=None):
        if method not in ('full', 'pca', 'hybrid') or not isinstance(d, (int, np.integer)) or d < 1:
            raise ValueError('Invalid method or input dimension')
        self.method, self.d = method, int(d)
        self.origin = readonly(np.zeros(d) if origin is None else origin)
        if self.origin.shape != (d,) or not np.isfinite(self.origin).all():
            raise ValueError('Invalid origin')
        self.basis = self.components = self.sparse_map = self.au = None
        self.m = self.d
        if method == 'pca':
            self.components = readonly(components)
            p = self.components
            if p.ndim != 2 or p.shape[1] != d or not len(p) or not np.isfinite(p).all():
                raise ValueError('Invalid PCA components')
            if not np.allclose(p@p.T, np.eye(len(p)), rtol=1e-10, atol=1e-10):
                raise ValueError('PCA components must be orthonormal')
            self.m = len(p)
        if method == 'hybrid':
            self.basis = readonly(basis)
            u = self.basis
            if u.ndim != 2 or u.shape[0] != d or not np.isfinite(u).all():
                raise ValueError('Invalid protected basis')
            if not np.allclose(u.T@u, np.eye(u.shape[1]), rtol=1e-10, atol=1e-10):
                raise ValueError('Protected basis must be orthonormal')
            a = csr_matrix(sparse_map, dtype=np.float64, copy=True)
            if a.shape[1] != d or a.shape[0] < 1 or not np.isfinite(a.data).all():
                raise ValueError('Invalid residual map')
            a.sum_duplicates(); a.sort_indices()
            self.au = readonly(a@u)
            a.data.setflags(write=False); a.indices.setflags(write=False); a.indptr.setflags(write=False)
            self.sparse_map = a
            self.m = u.shape[1]+a.shape[0]
        # Fingerprint actual parameters, not merely the random seed.
        digest = hashlib.sha256(f'{VERSION}:{method}:{self.d}:{self.m}'.encode())
        for key, value in sorted(self.parameters().items()):
            digest.update(key.encode()); digest.update(str(value.dtype).encode())
            digest.update(str(value.shape).encode())
            if value.size:
                digest.update(memoryview(np.ascontiguousarray(value)).cast('B'))
        self.fingerprint = digest.hexdigest()
        self._frozen = True

    def parameters(self):
        values = dict(origin=self.origin)
        if self.method == 'pca':
            values['components'] = self.components
        elif self.method == 'hybrid':
            a = self.sparse_map
            values.update(basis=self.basis, a_data=a.data, a_indices=a.indices.astype(np.int64),
                          a_indptr=a.indptr.astype(np.int64), a_shape=np.asarray(a.shape, dtype=np.int64))
        return values

    def transform(self, points, batch_rows=256):
        x = np.asarray(points, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.d or not np.isfinite(x).all():
            raise ValueError('Invalid projection points')
        if not isinstance(batch_rows, (int, np.integer)) or batch_rows < 1:
            raise ValueError('batch_rows must be positive')
        if self.method == 'full':
            return x
        z = np.empty((len(x), self.m))
        for first in range(0, len(x), batch_rows):
            shifted = x[first:first+batch_rows] - self.origin
            if self.method == 'pca':
                z[first:first+batch_rows] = shifted@self.components.T
            else:
                head = shifted@self.basis
                rank = self.basis.shape[1]
                z[first:first+batch_rows, :rank] = head
                z[first:first+batch_rows, rank:] = (self.sparse_map@shifted.T).T - head@self.au.T
        if not np.isfinite(z).all():
            raise ValueError('Projection overflow')
        return z

    def prepare(self, points, batch_rows=256):
        x = np.asarray(points, dtype=np.float64)
        z = self.transform(x, batch_rows=batch_rows)
        # An immutable caller input lets Full reuse X directly. Mutable caller
        # input is copied so later edits cannot silently stale the norm cache.
        if z.flags.writeable and self.method == 'full':
            z = z.copy()
        z.setflags(write=False)
        norms = np.einsum('ij,ij->i', z, z)
        if not np.isfinite(norms).all():
            raise ValueError('Norm overflow')
        norms.setflags(write=False)
        return PreparedDataset(self, z, norms, array_hash(x))


def indices(value, n):
    a = np.asarray(value)
    if a.ndim != 1 or (a.size and a.dtype.kind not in 'iu') or np.any(a < 0) or np.any(a >= n):
        raise ValueError('Expected one-dimensional in-range integer indices')
    return a.astype(np.intp, copy=False)


@dataclass
class PreparedDataset:
    model: NativeProjection
    z: np.ndarray
    norms: np.ndarray
    source_sha256: str
    distance_cache: np.ndarray | None = None

    def squared_distances(self, left, right):
        i, j = indices(left, len(self.z)), indices(right, len(self.z))
        if self.distance_cache is not None:
            return self.distance_cache[np.ix_(i, j)]
        answer = stable_squared_distances(self.z[i],self.z[j],self.norms[i],self.norms[j])
        answer[i[:,None] == j[None,:]] = 0
        if not np.isfinite(answer).all():
            raise ValueError('Distance overflow')
        return answer

    def precompute_distances(self, max_bytes=128*1024**2):
        required = len(self.z)**2*np.dtype(np.float64).itemsize
        if required > max_bytes:
            raise MemoryError(f'Distance matrix needs {required} bytes, budget is {max_bytes}')
        # Bound workspace: one n*n output plus one row block, not several n*n arrays.
        answer = np.empty((len(self.z), len(self.z)))
        for first in range(0, len(self.z), 256):
            end = min(first+256, len(self.z))
            block = stable_squared_distances(self.z[first:end],self.z,self.norms[first:end],self.norms)
            answer[first:end] = block
        if not np.isfinite(answer).all():
            raise ValueError('Distance overflow')
        np.fill_diagonal(answer, 0)
        answer.setflags(write=False)
        self.distance_cache = answer
        return answer

    def save(self, destination):
        path = Path(destination)
        path.mkdir(parents=True, exist_ok=False)
        np.save(path/'Z.npy', self.z, allow_pickle=False)
        np.save(path/'norms.npy', self.norms, allow_pickle=False)
        np.savez(path/'model.npz', **self.model.parameters())
        metadata = dict(version=VERSION, method=self.model.method, d=self.model.d,
                        m=self.model.m, n=len(self.z), dtype='float64',
                        source_sha256=self.source_sha256, model_sha256=self.model.fingerprint,
                        numpy_version=np.__version__,
                        files={name:file_hash(path/name) for name in ('Z.npy', 'norms.npy', 'model.npz')})
        # Publish metadata last: interrupted writes are never accepted as a cache.
        temporary = path/'manifest.tmp'
        temporary.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        temporary.replace(path/'manifest.json')

    @classmethod
    def load(cls, source, *, expected_source_sha256, expected_model_sha256=None):
        path = Path(source)
        metadata = json.loads((path/'manifest.json').read_text())
        if metadata['version'] != VERSION or metadata['source_sha256'] != expected_source_sha256:
            raise ValueError('Cache version or dataset mismatch')
        if expected_model_sha256 is not None and metadata['model_sha256'] != expected_model_sha256:
            raise ValueError('Cache projection mismatch')
        for name in ('Z.npy', 'norms.npy', 'model.npz'):
            if file_hash(path/name) != metadata['files'][name]:
                raise ValueError(f'Corrupt cache file: {name}')
        with np.load(path/'model.npz', allow_pickle=False) as parameters:
            options = dict(origin=parameters['origin'])
            if metadata['method'] == 'pca':
                options['components'] = parameters['components']
            elif metadata['method'] == 'hybrid':
                options.update(basis=parameters['basis'], sparse_map=csr_matrix(
                    (parameters['a_data'], parameters['a_indices'], parameters['a_indptr']),
                    shape=tuple(parameters['a_shape'])))
            model = NativeProjection(metadata['method'], metadata['d'], **options)
        if model.fingerprint != metadata['model_sha256']:
            raise ValueError('Cache model fingerprint mismatch')
        z = np.load(path/'Z.npy', mmap_mode='r', allow_pickle=False)
        norms = np.load(path/'norms.npy', mmap_mode='r', allow_pickle=False)
        if z.shape != (metadata['n'], metadata['m']) or norms.shape != (metadata['n'],) or z.dtype != np.float64 or norms.dtype != np.float64 or not np.isfinite(z).all() or not np.isfinite(norms).all():
            raise ValueError('Invalid cached representation')
        if not np.allclose(norms, np.einsum('ij,ij->i', z, z), rtol=1e-12, atol=1e-12):
            raise ValueError('Stale cached norms')
        return cls(model, z, norms, metadata['source_sha256'])
