"""Canonical sparse matrix generator: SparseMatrixGen(s, m, d, q).

Manuscript reference: Sec. "Sparse matrix generation". For each column
t in [d]: domain-separated SHAKE256 expansion of the seed samples an ordered
q-tuple of *distinct* rows in [m] by rejection sampling without replacement,
followed by independent sign bits sigma_ell(t) in {-1,+1}. A candidate 32-bit
word u is accepted only if u < 2^32 - (2^32 mod m) (removes modulo bias);
the row is then u mod m. Duplicate rows within a column are rejected and
re-sampled. Domain tag is seed || "col" || t (t encoded as little-endian u32).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .encoding import SparseRecord
from .hashing import shake256_stream

WORD_BITS = 32
WORD_MOD = 1 << WORD_BITS


def _accept_threshold(m: int) -> int:
    return WORD_MOD - (WORD_MOD % m)


def _domain_tag(seed: bytes, t: int) -> bytes:
    return seed + b"col" + t.to_bytes(4, "little")


def _sign_domain_tag(seed: bytes, t: int) -> bytes:
    return seed + b"sgn" + t.to_bytes(4, "little")


def gen_column(seed: bytes, t: int, m: int, q: int) -> List[Tuple[int, int]]:
    """Generate the q distinct (row, sign) pairs for column t.

    Rows are drawn via rejection-sampled SHAKE256 words from a domain-
    separated stream (seed || "col" || t); signs are drawn from a
    separately domain-separated stream (seed || "sgn" || t), one bit per
    accepted row, in the order the rows were accepted.
    """
    assert 1 <= q <= m
    threshold = _accept_threshold(m)
    rows: List[int] = []
    seen = set()
    word_stream = shake256_stream(_domain_tag(seed, t))
    for w in word_stream:
        u = int.from_bytes(w, "little")
        if u >= threshold:
            continue  # reject: modulo-bias avoidance
        row = u % m
        if row in seen:
            continue  # reject: duplicate row within this column
        seen.add(row)
        rows.append(row)
        if len(rows) == q:
            break

    sign_stream = shake256_stream(_sign_domain_tag(seed, t))
    signs: List[int] = []
    for w in sign_stream:
        bit = w[0] & 1
        signs.append(1 if bit == 0 else -1)
        if len(signs) == q:
            break

    return list(zip(rows, signs))


def sparse_matrix_gen(seed: bytes, m: int, d: int, q: int):
    """Full SparseMatrixGen(s, m, d, q).

    Returns (records, A_dense) where records is the canonical column-major
    sparse list [(t, ell, row, sign), ...] and A_dense is an (m, d) numpy
    int64 array in {-1, 0, +1}.
    """
    records: List[SparseRecord] = []
    A = np.zeros((m, d), dtype=np.int64)
    for t in range(d):
        col = gen_column(seed, t, m, q)
        for ell, (row, sign) in enumerate(col):
            records.append(SparseRecord(t=t, ell=ell, row=row, sign=sign))
            A[row, t] = sign
    return records, A


def sparse_matrix_gen_fixed_length(seed: bytes, m: int, d: int, q: int, max_words_per_column: int):
    """In-circuit-compatible fixed-length variant.

    Draws exactly `max_words_per_column` candidate words per column
    regardless of how many are actually needed, so that the generator's
    per-column work is data-independent and bounded (suitable for a
    fixed-size circuit ablation, per plan P1 "fixed-length variant for
    in-circuit generation"). If fewer than q distinct valid rows are found
    within the budget, raises ValueError (caller should pick a budget with
    negligible failure probability, e.g. >= 8q for m >> q).

    Returns the same (records, A_dense) shape as sparse_matrix_gen.
    """
    threshold = _accept_threshold(m)
    records: List[SparseRecord] = []
    A = np.zeros((m, d), dtype=np.int64)
    for t in range(d):
        word_stream = shake256_stream(_domain_tag(seed, t))
        sign_stream = shake256_stream(_sign_domain_tag(seed, t))
        rows: List[int] = []
        seen = set()
        for _ in range(max_words_per_column):
            w = next(word_stream)
            u = int.from_bytes(w, "little")
            sbit = next(sign_stream)[0] & 1
            if u >= threshold:
                continue
            row = u % m
            if row in seen:
                continue
            seen.add(row)
            rows.append((row, 1 if sbit == 0 else -1))
            if len(rows) == q:
                break
        if len(rows) < q:
            raise ValueError(
                f"fixed-length budget {max_words_per_column} insufficient for column {t} "
                f"(got {len(rows)} of {q} rows); increase max_words_per_column"
            )
        for ell, (row, sign) in enumerate(rows):
            records.append(SparseRecord(t=t, ell=ell, row=row, sign=sign))
            A[row, t] = sign
    return records, A
