import numpy as np
from scipy.linalg import hadamard

from experiments.new_center_hybrid import Tail, cross_estimate, fwht, run_job


def test_fwht_matches_hadamard():
    a = np.random.default_rng(0).normal(size=(3, 16))
    np.testing.assert_allclose(fwht(a), a @ hadamard(16).T, atol=1e-9)


def test_srht_preserves_norm_in_expectation():
    v = np.random.default_rng(1).normal(size=(1, 300))
    ratios = [(Tail('srht', bytes([i])*32, 64, 300)(v)**2).sum()/(v**2).sum() for i in range(60)]
    assert abs(np.mean(ratios) - 1) < 0.1


def test_clip_respects_cauchy_schwarz():
    tx, tc = np.full((2, 3), 10.), np.full((1, 3), 10.)
    est = cross_estimate(tx, tc, np.array([1., 4.]), np.array([1.]), 'clip', np.ones(1))
    np.testing.assert_allclose(est[:, 0], [1., 2.])


def test_run_job_smoke():
    rng = np.random.default_rng(2)
    x = np.concatenate([rng.normal(loc=5*j, size=(60, 40)) for j in range(5)])
    rows = run_job('toy', x, 5, 0)
    assert {r['method'] for r in rows} >= {'pca_energy', 'hybrid', 'learned_energy'}
