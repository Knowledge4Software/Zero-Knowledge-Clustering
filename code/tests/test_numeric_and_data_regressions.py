import numpy as np

from commitjl.clustering import nearest_centroid_assign, psi_hat
from commitjl.quantize import quantize
from data.loaders import DBSCAN_MARGIN_BENCHMARK, dbscan_margin_dataset
from experiments.pipeline import build_projection, load_row_dataset
from experiments.rq4_certificates import run_centroid_cert
from commitjl.clustering import kmeans_paired
from commitjl.quantize import scale_for_precision


def test_psi_hat_preserves_fractional_distance_and_scaled_integer_equivalent():
    points = np.array([[0.9]], dtype=np.float64)
    centroids = np.array([[0.1]], dtype=np.float64)
    assignments = np.array([0])

    assert np.isclose(psi_hat(points, centroids, assignments), 0.64)

    scale = 10
    points_q = quantize(points, scale, B=100)
    centroids_q = quantize(centroids, scale, B=100)
    assert psi_hat(points_q, centroids_q, assignments) == 64
    assert np.isclose(psi_hat(points_q, centroids_q, assignments) / scale**2, 0.64)


def test_squared_distances_are_exact_at_declared_maximum_bound():
    d = 1024
    points = np.full((1, d), 4096, dtype=np.int64)
    centroids = np.vstack([
        np.full(d, -4096, dtype=np.int64),
        np.full(d, 4095, dtype=np.int64),
    ])
    assignments = nearest_centroid_assign(points, centroids)
    assert assignments.tolist() == [1]

    far_assignments = np.array([0])
    expected = d * (2 * 4096) ** 2
    assert psi_hat(points, centroids, far_assignments) == expected


def test_squared_distances_fall_back_to_exact_arithmetic_beyond_int64():
    points = np.array([[0]], dtype=np.int64)
    centroids = np.array([[3_100_000_000], [1]], dtype=np.int64)
    assert nearest_centroid_assign(points, centroids).tolist() == [1]
    assert psi_hat(points, centroids, np.array([0])) == 3_100_000_000**2


def test_dbscan_generator_matches_frozen_benchmark_hash():
    result = dbscan_margin_dataset(**DBSCAN_MARGIN_BENCHMARK["kwargs"])
    assert result["meta"]["generator_version"] == "dbscan-margin-v1"
    assert result["meta"]["sha256"] == DBSCAN_MARGIN_BENCHMARK["sha256"]


def test_rq4_normalized_sparse_projection_is_not_divided_by_q_twice():
    row = {
        "dataset": {"loader": "gaussian_mixture", "kwargs": {
            "n": 30, "d": 8, "k": 2, "gamma": 3.0, "imbalance": 1.0,
        }},
        "projection": "sparse_jl", "m": 8, "q": 4, "k": 2,
        "seed": 0, "precision_bits": 16,
    }
    prereg = {"eps": 0.15}
    observed = run_centroid_cert(row, prereg)["cost_bound_observed"]

    X = np.asarray(load_row_dataset(row)["X"], dtype=np.float64)
    labels, centroids, _ = kmeans_paired(X, 2, seeds=[0])[0]
    projection = build_projection("sparse_jl", X, 8, 4, 0, scale_for_precision(16))["real"]
    Z = X @ projection.T
    Cz = centroids @ projection.T
    expected = float(np.sum(np.min(np.sum((Z[:, None, :] - Cz[None, :, :]) ** 2, axis=2), axis=1)))
    assert np.isclose(observed, expected)

