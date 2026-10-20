"""Protocol equivalence across adaptive refills, rejection and duplicate draws."""
import hashlib
from itertools import islice

import numpy as np
import pytest

from commitjl import sparse_gen
from commitjl.hashing import shake256_stream


def legacy_stream(data):
    shake = hashlib.shake_256(data)
    pos = 0
    while True:
        buf = shake.digest(pos + 4096)
        while pos + 4 <= len(buf):
            yield buf[pos:pos + 4]
            pos += 4


@pytest.mark.parametrize('seed', [b'', b'canonical-stream', bytes(range(256))])
def test_adaptive_refills_are_the_exact_uninterrupted_xof(seed):
    # Cross every growing refill and several of the capped 4096-byte refills.
    count = 12000
    words = list(islice(shake256_stream(seed), count))
    assert all(len(word) == 4 for word in words)
    assert b''.join(words) == hashlib.shake_256(seed).digest(4 * count)
    assert words == list(islice(legacy_stream(seed), count))


@pytest.mark.parametrize('m,q,d', [(1,1,19), (112,1,784), (240,1,5000),
                                  (17,17,23), (257,129,7), (2**31+1,8,64),
                                  (2**32,3,37)])
def test_column_records_match_previous_stream_even_under_rejection(monkeypatch, m, q, d):
    seed = b'adaptive-rejection-regression'
    actual = sparse_gen.sparse_matrix_records(seed, m, d, q)
    with monkeypatch.context() as patch:
        patch.setattr(sparse_gen, 'shake256_stream', legacy_stream)
        expected = sparse_gen.sparse_matrix_records(seed, m, d, q)
    assert actual == expected


def test_entire_protected_projection_and_fingerprint_remain_identical(monkeypatch):
    from commitjl.reusable_native import NativeProjection
    d, m, q = 47, 17, 5
    rng = np.random.default_rng(123)
    basis = np.linalg.qr(rng.normal(size=(d, 4)))[0]
    origin = rng.normal(size=d)
    points = rng.normal(size=(23, d))
    def make():
        _, matrix = sparse_gen.sparse_matrix_gen_csr(b'protected-equivalence', m, d, q)
        matrix = matrix.astype(float); matrix.data /= np.sqrt(q)
        return NativeProjection('hybrid', d, basis=basis, origin=origin, sparse_map=matrix)
    actual = make()
    with monkeypatch.context() as patch:
        patch.setattr(sparse_gen, 'shake256_stream', legacy_stream)
        expected = make()
    assert actual.fingerprint == expected.fingerprint
    np.testing.assert_array_equal(actual.transform(points), expected.transform(points))
