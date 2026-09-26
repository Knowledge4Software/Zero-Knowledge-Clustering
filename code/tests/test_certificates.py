import numpy as np

from commitjl.certificates import (
    centroid_margin_certificate, dbscan_pair_certificate, cost_bounds,
    coverage_violation_stability,
)


def test_centroid_margin_certificate_sound_on_large_margin_data():
    """Synthetic dataset with a known large margin: points clustered tightly
    around well-separated centroids should be certified stable at a small
    eps, and empirically ARE stable (trivially, since we test the certified
    flag directly here; the certificate should flag ~all points)."""
    rng = np.random.default_rng(0)
    centroids = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]])
    X = []
    for c in centroids:
        X.append(c + rng.normal(scale=0.5, size=(20, 2)))
    X = np.vstack(X)

    certified = centroid_margin_certificate(X, centroids, eps=0.1)
    assert certified.all(), "expected all large-margin points to be certified"


def test_centroid_margin_certificate_flags_ambiguous_point():
    """A point placed exactly on the boundary between two centroids should
    NOT be certified (small margin) even at modest eps."""
    centroids = np.array([[0.0, 0.0], [10.0, 0.0]])
    ambiguous_point = np.array([[5.0, 0.0]])  # equidistant -> d1 == d2
    certified = centroid_margin_certificate(ambiguous_point, centroids, eps=0.1)
    assert not certified[0], "equidistant/ambiguous point should not be certified"


def test_dbscan_ambiguity_band_flags_close_pairs():
    rho = 5.0
    eps = 0.2
    # one pair well inside neighbor radius, one pair well outside, one pair
    # placed exactly at rho (ambiguous)
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [50.0, 0.0], [0.0, 0.0], [5.0, 0.0]])
    result = dbscan_pair_certificate(X, rho=rho, eps=eps)
    # pair (0,1): distance 1 << rho -> certified
    # pair (2,3): distance 50 >> rho -> certified
    # pair (4,5): distance exactly rho -> inside ambiguity band -> NOT certified
    certified = result["certified"]
    i, j = result["i"], result["j"]
    pair_certified = {(int(a), int(b)): bool(c) for a, b, c in zip(i, j, certified)}
    assert pair_certified[(0, 1)] is True
    assert pair_certified[(2, 3)] is True
    assert pair_certified[(4, 5)] is False


def test_dbscan_pair_certificate_chunked_matches_exact_small_case():
    """Regression test for the O(n^2*d) memory blowup: chunking the pair
    computation into small blocks must give bit-for-bit the same result as
    a single dense pass on a small case."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    rho, eps = 2.0, 0.2
    exact = dbscan_pair_certificate(X, rho, eps, chunk_size=10_000)
    chunked = dbscan_pair_certificate(X, rho, eps, chunk_size=3)
    assert np.array_equal(exact["i"], chunked["i"])
    assert np.array_equal(exact["j"], chunked["j"])
    np.testing.assert_allclose(exact["orig_sqdist"], chunked["orig_sqdist"])
    assert np.array_equal(exact["certified"], chunked["certified"])
    assert chunked["subsampled"] is False
    assert chunked["pairs_evaluated"] == chunked["pairs_total"] == 40 * 39 // 2


def test_dbscan_pair_certificate_default_unchanged_for_small_n():
    """Default behaviour (no max_pairs) must not silently subsample small
    inputs -- exact result, all pairs evaluated."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 3))
    res = dbscan_pair_certificate(X, rho=1.0, eps=0.1)
    n_pairs = 20 * 19 // 2
    assert res["pairs_evaluated"] == n_pairs
    assert res["pairs_total"] == n_pairs
    assert res["subsampled"] is False


def test_dbscan_pair_certificate_subsamples_when_over_cap():
    """When n_choose_2 exceeds max_pairs, the memory-safe subsampling path
    is taken, and the returned metadata honestly reports it."""
    rng = np.random.default_rng(2)
    n, d = 200, 4  # n_choose_2 = 19900
    X = rng.normal(size=(n, d))
    max_pairs = 500
    res = dbscan_pair_certificate(X, rho=1.0, eps=0.1, max_pairs=max_pairs, seed=42)
    assert res["subsampled"] is True
    assert res["pairs_evaluated"] == max_pairs
    assert res["pairs_total"] == n * (n - 1) // 2
    assert res["i"].shape[0] == max_pairs
    assert res["j"].shape[0] == max_pairs
    # sampled pairs are still valid upper-triangle indices
    assert np.all(res["i"] < res["j"])
    assert np.all(res["j"] < n)


def test_cost_bounds_holds():
    res = cost_bounds(Phi_X_C=100.0, Phi_AX_AC_over_q=105.0, eps=0.1)
    assert res["holds"]
    res_bad = cost_bounds(Phi_X_C=100.0, Phi_AX_AC_over_q=200.0, eps=0.1)
    assert not res_bad["holds"]


def test_coverage_violation_stability_basic():
    certified = np.array([True, True, False, False])
    observed_stable = np.array([True, False, True, False])
    res = coverage_violation_stability(certified, observed_stable)
    assert res.coverage == 0.5
    # violation: certified but not stable -> index 1
    assert res.violation_rate == 0.5
    # uncertified stable: not certified but stable -> index 2
    assert res.uncertified_stability == 0.5
