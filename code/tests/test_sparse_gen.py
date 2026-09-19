import numpy as np
import pytest
from scipy import stats as scipy_stats

from commitjl.sparse_gen import sparse_matrix_gen, gen_column, sparse_matrix_gen_fixed_length


def test_determinism_same_seed():
    seed = b"determinism-seed"
    records1, A1 = sparse_matrix_gen(seed, m=16, d=32, q=3)
    records2, A2 = sparse_matrix_gen(seed, m=16, d=32, q=3)
    assert records1 == records2
    assert np.array_equal(A1, A2)


def test_different_seed_differs():
    A1 = sparse_matrix_gen(b"seedA", m=16, d=32, q=3)[1]
    A2 = sparse_matrix_gen(b"seedB", m=16, d=32, q=3)[1]
    assert not np.array_equal(A1, A2)


def test_shape_and_entries():
    m, d, q = 20, 40, 4
    _, A = sparse_matrix_gen(b"shape-seed", m, d, q)
    assert A.shape == (m, d)
    assert set(np.unique(A)).issubset({-1, 0, 1})


def test_exactly_q_distinct_rows_per_column():
    m, d, q = 25, 50, 5
    _, A = sparse_matrix_gen(b"qcols-seed", m, d, q)
    for t in range(d):
        col = A[:, t]
        nz_rows = np.nonzero(col)[0]
        assert len(nz_rows) == q
        assert len(set(nz_rows.tolist())) == q  # distinct


def test_no_modulo_bias_frequency_uniform():
    """Chi-square uniformity check on rejection-sampled row indices.

    Draw many single-row columns (q=1) across many distinct columns/seeds
    and check the resulting row distribution is consistent with uniform.
    """
    m = 7  # non-power-of-two modulus to actually exercise rejection sampling
    rows_observed = []
    for t in range(4000):
        col = gen_column(b"bias-check-seed", t, m, q=1)
        rows_observed.append(col[0][0])
    counts = np.bincount(rows_observed, minlength=m)
    chi2, p = scipy_stats.chisquare(counts)
    assert p > 0.001, f"rejection-sampled rows look non-uniform: counts={counts}, p={p}"


def test_fixed_length_variant_matches_shape():
    m, d, q = 16, 8, 3
    records, A = sparse_matrix_gen_fixed_length(b"fixedlen-seed", m, d, q, max_words_per_column=64)
    assert A.shape == (m, d)
    for t in range(d):
        assert np.count_nonzero(A[:, t]) == q


def test_fixed_length_raises_on_insufficient_budget():
    with pytest.raises(ValueError):
        sparse_matrix_gen_fixed_length(b"fixedlen-fail-seed", m=4, d=2, q=4, max_words_per_column=1)
