"""Scientific validity checks, including adversarial approximate sign errors."""
import numpy as np
import pytest

from commitjl.published_audit_baselines import (
    CoresetAudit, ProjectedAudit, distances, interval_decision,
    lightweight_probabilities, projection_bases, random_projection,
)
from commitjl.group_change_bounds import exact_old_correction_bounds
from commitjl.group_moment_reference import prepare_groups


def test_published_sampling_probabilities_and_degenerate_case():
    x = np.array([[0., 0.], [2., 0.], [10., 0.]])
    energy = ((x - x.mean(0)) ** 2).sum(1)
    np.testing.assert_allclose(lightweight_probabilities(x), 1 / 6 + energy / (2 * energy.sum()))
    np.testing.assert_allclose(lightweight_probabilities(np.ones((5, 3))), np.full(5, .2))


def test_importance_weights_are_unbiased_by_enumeration():
    x = np.array([[0, 0], [1, 2], [8, 5]])
    c = np.array([[0, 0], [7, 7]])
    p = lightweight_probabilities(x)
    # Enumerate all one-draw outcomes rather than testing one lucky sample.
    expected = sum(p[i] * int(distances(x[i:i+1], c).min()) / p[i] for i in range(len(x)))
    assert expected == pytest.approx(int(distances(x, c).min(1).sum()))
    sample = CoresetAudit(x, 17, 41)
    assert sample.query(c) == pytest.approx(float(np.dot(sample.weights, distances(sample.rows, c).min(1))))


@pytest.mark.parametrize('kind', ['sign_rp', 'sparse_rp'])
def test_random_sketches_are_reproducible_and_keep_prefix(kind):
    r = random_projection(kind, 7, 30, 41)
    np.testing.assert_array_equal(r, random_projection(kind, 7, 30, 41))
    p, wider = projection_bases(kind, 7, 11, 30, 41, 128)
    np.testing.assert_array_equal(p, wider[:7])
    if kind == 'sign_rp':
        np.testing.assert_allclose(np.abs(r), np.full_like(r, 1 / np.sqrt(7)))
    else:
        assert set(np.round(np.unique(r), 12)) <= set(np.round([-np.sqrt(3/7), 0, np.sqrt(3/7)], 12))


def test_raw_projection_can_reverse_sign_and_is_not_a_certificate():
    x = np.array([[0, 0], [0, 2]])
    old = [[1, 0], [10, 10]]
    new = [[0, 5], [10, 10]]
    audit = ProjectedAudit(x, np.array([[1.], [0.]]))
    assert audit.query(new)[0] < audit.query(old)[0]
    assert int(distances(x, new).min(1).sum()) > int(distances(x, old).min(1).sum())


@pytest.mark.parametrize('kind', ['sign_rp', 'sparse_rp'])
def test_adapter_bounds_hold_for_nonorthogonal_published_sketches(kind):
    rng = np.random.default_rng(592)
    rows = rng.integers(-15, 16, (18, 9))
    old = rng.integers(-10, 11, (3, 9))
    _, b = projection_bases(kind, 3, 5, 9, 7, 64)
    model = dict(basis=b.tolist(), basis_scale=64, centers=old.tolist(), input_bits=6)
    od = distances(rows, old)
    groups = prepare_groups(rows, od.argmin(1), 3)
    for _ in range(12):
        new = rng.integers(-15, 16, (3, 9))
        truth = int(distances(rows, new).min(1).sum()) - int(od.min(1).sum())
        result = exact_old_correction_bounds(model, rows, groups, old, new)
        assert result['change_lower_scaled'] <= truth * 64 ** 6 <= result['change_upper_scaled']
        answer = interval_decision(result['change_lower_scaled'], result['change_upper_scaled'])
        assert answer in ['inconclusive', interval_decision(truth, truth)]


def test_distance_overflow_guard_matches_python_integers():
    x, c = [[2 ** 40, -2 ** 40]], [[0, 0], [2 ** 40, 0]]
    assert distances(x, c).tolist() == [[2 ** 81, 2 ** 80]]
