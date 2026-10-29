"""Paper-derived numerical controls for a fixed-bank cost-change audit.

The projection and sampling algorithms do not themselves provide cryptographic
certificates. Attaching the project's existing interval relation is explicitly
an adapter, not an implementation of a published proof system.
"""
from __future__ import annotations

import math
import numpy as np


def random_projection(kind: str, rank: int, dimension: int, seed: int) -> np.ndarray:
    """Boutsidis et al. (2010) signs or Achlioptas (2003) sparse signs.

    Return a dimension-by-rank matrix. No orthogonalization or data-dependent
    selection is performed: those would change the published distribution.
    """
    if rank < 1 or dimension < 1:
        raise ValueError('positive projection dimensions required')
    rng = np.random.default_rng(seed)
    if kind == 'sign_rp':
        return (2 * rng.integers(0, 2, (dimension, rank)) - 1) / math.sqrt(rank)
    if kind == 'sparse_rp':
        draws = rng.integers(0, 6, (dimension, rank))
        return np.where(draws == 0, -1., np.where(draws == 1, 1., 0.)) * math.sqrt(3 / rank)
    raise ValueError('unknown projection distribution')


def projection_bases(kind: str, prediction_rank: int, certificate_rank: int,
                     dimension: int, seed: int, scale: int = 16384):
    """Keep the raw prediction sketch; append independent rows for the adapter.

    The appended rows use the same coefficient scale as the prediction sketch.
    The 79-row adapter is not claimed to be an independently normalized JL79
    sketch, and its residual-aware predictor differs from raw JL prediction.
    """
    if not 0 < prediction_rank <= certificate_rank:
        raise ValueError('invalid prediction/certificate ranks')
    prediction = random_projection(kind, prediction_rank, dimension, seed).T
    extra = certificate_rank - prediction_rank
    if extra:
        # Generate at the prediction scale while using a distinct seed.
        tail = random_projection(kind, extra, dimension, seed + 100003).T
        tail *= math.sqrt(extra / prediction_rank)
        wider = np.concatenate((prediction, tail))
    else:
        wider = prediction
    return (np.rint(prediction * scale).astype(np.int64),
            np.rint(wider * scale).astype(np.int64))


def distances(rows, centers):
    """Exact int64 distances only after a conservative overflow check."""
    x, c = np.asarray(rows, dtype=np.int64), np.asarray(centers, dtype=np.int64)
    if x.ndim != 2 or c.ndim != 2 or x.shape[1] != c.shape[1] or not len(x) or len(c) < 2:
        raise ValueError('invalid catalog/center shape')
    maximum = max(abs(int(x.min())), abs(int(x.max())), abs(int(c.min())), abs(int(c.max())))
    if 4 * x.shape[1] * maximum ** 2 >= 2 ** 63:
        return np.array([[sum((int(a) - int(b)) ** 2 for a, b in zip(row, center))
                          for center in c] for row in x], dtype=object)
    return ((x[:, None] - c[None]) ** 2).sum(2)


class ProjectedAudit:
    """Raw published projection: estimate costs of supplied centers, no refit."""
    def __init__(self, rows, projection):
        self.projection = np.asarray(projection, dtype=float)
        self.rows = np.asarray(rows, dtype=float) @ self.projection

    def query(self, centers):
        c = np.asarray(centers, dtype=float) @ self.projection
        d = ((self.rows[:, None] - c[None]) ** 2).sum(2)
        return float(d.min(1).sum()), d.argmin(1)


def lightweight_probabilities(rows):
    """Bachem, Lucic, Krause (KDD 2018), Algorithm 1.

    q_i = 1/(2n) + ||x_i-mean||^2 / (2 sum_j ||x_j-mean||^2).
    The uniform zero-variance case is the continuous degenerate extension.
    """
    x = np.asarray(rows, dtype=float)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('nonempty finite matrix required')
    energies = ((x - x.mean(0)) ** 2).sum(1)
    total = float(energies.sum())
    return (np.full(len(x), 1 / len(x)) if total == 0 else
            .5 / len(x) + .5 * energies / total)


class CoresetAudit:
    """Importance sampling WITH replacement and weights 1/(m*q_i).

    Fixed sample budgets are empirical operating points. They are not claimed
    to meet an unspecified constant in the paper's uniform coreset theorem.
    No assignment prediction for unsampled rows or sign certificate is implied.
    """
    def __init__(self, rows, sample_size: int, seed: int, lightweight: bool = True):
        self.x = np.asarray(rows, dtype=np.int64)
        if sample_size < 1 or self.x.ndim != 2 or not len(self.x):
            raise ValueError('invalid coreset input')
        probabilities = (lightweight_probabilities(self.x) if lightweight else
                         np.full(len(self.x), 1 / len(self.x)))
        sampled = np.random.default_rng(seed).choice(len(self.x), sample_size,
                                                   replace=True, p=probabilities)
        ids, counts = np.unique(sampled, return_counts=True)
        self.indices = ids
        self.rows = self.x[ids]
        self.weights = counts / (sample_size * probabilities[ids])
        self.draw_count = sample_size
        self.probabilities = probabilities

    def query(self, centers):
        return float(np.dot(self.weights, distances(self.rows, centers).min(1).astype(float)))


def point_decision(change):
    return 'improves' if change < 0 else 'worsens' if change > 0 else 'equal'


def interval_decision(lower, upper):
    if lower > upper:
        raise ValueError('reversed interval')
    return ('improves' if upper < 0 else 'worsens' if lower > 0 else
            'equal' if lower == upper == 0 else 'inconclusive')
