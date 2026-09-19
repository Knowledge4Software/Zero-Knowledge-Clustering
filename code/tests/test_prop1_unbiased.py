"""Monte Carlo check of Proposition 1 (Unbiased squared norm):
E_A[ q^{-1} ||Av||_2^2 ] = ||v||_2^2, for fixed v, sign bits iid uniform
+-1 independent of sampled rows.
"""
import numpy as np

from commitjl.sparse_gen import sparse_matrix_gen


def test_prop1_unbiasedness_monte_carlo():
    rng = np.random.default_rng(42)
    d = 20
    m, q = 30, 4
    v = rng.normal(size=d)
    v_norm_sq = float(np.sum(v ** 2))

    n_trials = 800
    estimates = []
    for i in range(n_trials):
        seed = f"prop1-trial-{i}".encode()
        _, A = sparse_matrix_gen(seed, m, d, q)
        Av = A @ v
        estimates.append(np.sum(Av ** 2) / q)

    mean_estimate = float(np.mean(estimates))
    se = float(np.std(estimates, ddof=1) / np.sqrt(n_trials))

    # within ~6 standard errors of the true value (generous tolerance to
    # avoid flakiness while still being a meaningful check)
    assert abs(mean_estimate - v_norm_sq) < 6 * se, (
        f"mean_estimate={mean_estimate}, true={v_norm_sq}, se={se}"
    )
