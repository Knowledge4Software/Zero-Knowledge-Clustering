"""Exact integer reference for a center family approved before preprocessing.

This is not a proof implementation. Stored statistics require authentication
in a cryptographic baseline. Unknown future centers are explicitly rejected.
"""
from __future__ import annotations


def prepare(rows, center_sets):
    if not rows or not center_sets or any(not c for c in center_sets):
        raise ValueError('nonempty rows and center sets required')
    anchor = tuple(map(int, center_sets[0][0]))
    dim = len(anchor)
    unique = list(dict.fromkeys(tuple(map(int, c)) for centers in center_sets for c in centers))
    if any(len(c) != dim for c in unique) or any(len(x) != dim for x in rows):
        raise ValueError('dimension mismatch')
    offsets = [tuple(c[j] - anchor[j] for j in range(dim)) for c in unique]
    stats = []
    for row in rows:
        w = [int(row[j]) - anchor[j] for j in range(dim)]
        beta = sum(v*v for v in w)
        scores = [2*sum(a*b for a, b in zip(w, v))-sum(b*b for b in v) for v in offsets]
        stats.append((beta, scores))
    return dict(centers=unique, stats=stats, scalar_count_per_row=1+len(offsets),
                nonconstant_scalar_count_per_row=1+sum(any(v) for v in offsets))


def query(prepared, centers):
    approved = prepared['centers']
    requested = [tuple(map(int, c)) for c in centers]
    if not requested or any(c not in approved for c in requested):
        raise ValueError('center not in preprocessed family')
    positions = [approved.index(c) for c in requested]
    labels, costs = [], []
    for beta, scores in prepared['stats']:
        chosen = max(range(len(positions)), key=lambda j: scores[positions[j]])
        labels.append(chosen)
        costs.append(beta-scores[positions[chosen]])
    return dict(labels=labels, costs=costs, total=sum(costs))
