import numpy as np

from commitjl import projections as proj
from commitjl.clustering import nearest_centroid_assign, psi_hat, phi_X, kmeans_paired, dbscan_deterministic
from commitjl.metrics import sample_pair_set, delta_uv, distortion_summary, clustering_agreement
from commitjl.stats import hierarchical_bootstrap, paired_bootstrap, holm_bonferroni, loglog_slope
from commitjl.cost_model import total_circuit_cost, full_vs_sparse_reduction


def test_all_projection_methods_produce_expected_shapes():
    m, d, seed = 10, 30, 0
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, d))

    out_id = proj.identity_projection(d)
    assert out_id["real"].shape == (d, d)

    out_g = proj.gaussian_projection(m, d, seed, S=256)
    assert out_g["int"].shape == (m, d)

    out_r = proj.rademacher_projection(m, d, seed)
    assert set(np.unique(out_r["int"])).issubset({-1, 1})

    out_a = proj.achlioptas_projection(m, d, seed, s=3)
    assert out_a["nnz"]["density"] <= 1.0

    out_s = proj.sparse_jl_projection(m, d, q=3, seed_bytes=b"projseed")
    for t in range(d):
        assert np.count_nonzero(out_s["int"][:, t]) == 3

    out_c = proj.coordinate_sampling_projection(m, d, seed)
    assert np.count_nonzero(out_c["int"]) == min(m, d)

    out_pca = proj.pca_projection(X, m, S=256)
    assert out_pca["int"].shape[0] <= m

    out_pisot = proj.pisot_beta_projection(m, d, seed, S=256)
    assert out_pisot["int"].shape == (m, d)


def test_nearest_centroid_lowest_index_tie_break():
    Z = np.array([[0, 0]])
    centroids = np.array([[1, 0], [-1, 0], [0, 1]])  # first two are equidistant
    a = nearest_centroid_assign(Z, centroids)
    assert a[0] == 0  # lowest index among ties


def test_psi_hat_and_phi_X_nonnegative():
    rng = np.random.default_rng(2)
    X = rng.integers(-10, 10, size=(20, 5))
    centroids = rng.integers(-10, 10, size=(3, 5))
    a = nearest_centroid_assign(X, centroids)
    psi = psi_hat(X, centroids, a)
    assert psi >= 0
    phi = phi_X(X, centroids)
    assert phi >= 0


def test_kmeans_paired_and_dbscan_run():
    rng = np.random.default_rng(3)
    X = np.vstack([rng.normal(loc=(0, 0), size=(20, 2)), rng.normal(loc=(10, 10), size=(20, 2))])
    results = kmeans_paired(X, k=2, seeds=[0, 1])
    assert len(results) == 2
    labels = dbscan_deterministic(X, eps=2.0, min_samples=3)
    assert len(labels) == X.shape[0]


def test_metrics_pipeline():
    rng = np.random.default_rng(4)
    U = rng.normal(size=(30, 10))
    pairs = sample_pair_set(30, 50, seed=0)
    delta = delta_uv(U, U, pairs)  # identity map -> delta ~ 0
    summary = distortion_summary(delta, eps=0.1)
    assert summary["max"] < 1e-8

    a = rng.integers(0, 3, size=30)
    b = a.copy()
    agree = clustering_agreement(a, b)
    assert agree["ari"] == 1.0
    assert agree["agreement"] == 1.0


def test_stats_functions():
    data = {"ds1": {0: [1.0, 2.0, 3.0], 1: [2.0, 3.0, 4.0]}, "ds2": {0: [5.0, 6.0]}}
    res = hierarchical_bootstrap(data, n_resamples=200, seed=0)
    assert res.ci_low <= res.point_estimate <= res.ci_high

    pb = paired_bootstrap([1, 2, 3], [2, 4, 6], mode="ratio", n_resamples=200, seed=0)
    assert 0 < pb.point_estimate < 1

    rejected = holm_bonferroni([0.001, 0.2, 0.03], alpha=0.05)
    assert isinstance(rejected, list) and len(rejected) == 3

    slope = loglog_slope([1, 10, 100], [1, 100, 10000])
    assert abs(slope["slope"] - 2.0) < 0.05


def test_cost_model_reduction_positive():
    res = full_vs_sparse_reduction(n=100, d=1024, m=64, k=5, q=4)
    assert res["reduction_pct"] > 0
    total = total_circuit_cost(n=100, d=1024, m=64, k=5, q=4)
    assert total["total_constraints"] > 0
