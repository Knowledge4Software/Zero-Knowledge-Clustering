import numpy as np

from commitjl.certificates import (
    _all_pair_indices,
    _pair_ranks_to_indices,
    pair_squared_distances,
)
from experiments.pipeline import PipelineCache
from experiments.common import generate_grid


def test_pair_rank_mapping_matches_upper_triangle():
    for n in (2, 3, 8, 31):
        expected = np.triu_indices(n, 1)
        ranks = np.arange(n * (n - 1) // 2, dtype=np.int64)
        actual = _pair_ranks_to_indices(n, ranks)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        all_pairs = _all_pair_indices(n)
        np.testing.assert_array_equal(all_pairs[0], expected[0])
        np.testing.assert_array_equal(all_pairs[1], expected[1])


def test_chunked_pair_distances_match_direct_computation():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(47, 19))
    i, j = np.triu_indices(len(X), 1)
    expected = np.sum((X[i] - X[j]) ** 2, axis=1)
    actual = pair_squared_distances(X, i, j, chunk_size=13)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_pipeline_cache_ignores_q_for_pca_but_not_sparse_jl():
    cache = PipelineCache()
    X = np.arange(60, dtype=float).reshape(12, 5)
    pca1, _ = cache.projection("dataset", "pca", X, 3, 1, 9, 256)
    pca2, _ = cache.projection("dataset", "pca", X, 3, 4, 9, 256)
    sparse1, _ = cache.projection("dataset", "sparse_jl", X, 3, 1, 9, 256)
    sparse2, _ = cache.projection("dataset", "sparse_jl", X, 3, 2, 9, 256)
    assert pca1 is pca2
    assert sparse1 is not sparse2


def test_block_grid_supports_one_factor_at_a_time_designs():
    rows = generate_grid({
        "fixed": {"shared": 1},
        "blocks": [
            {"fixed": {"factor": "n"}, "grid": {"value": [100, 200], "seed": [0, 1]}},
            {"fixed": {"factor": "d"}, "grid": {"value": [32], "seed": [0, 1]}},
        ],
    })
    assert len(rows) == 6
    assert all(row["shared"] == 1 for row in rows)
    assert sum(row["factor"] == "n" for row in rows) == 4
