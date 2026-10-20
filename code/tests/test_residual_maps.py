import hashlib

import numpy as np
import pytest
from scipy.linalg import hadamard

from commitjl.residual_maps import (VARIANTS, ResidualProjection, make_residual_model,
    normalized_hadamard, block_matrix)
from commitjl.protected_jl import centroid_basis
from commitjl.reusable_native import NativeProjection, stable_squared_distances
from commitjl.sparse_gen import sparse_matrix_gen_csr
from experiments.residual_map_comparison import case_plan, dimension_plan, make_hybrid


def fixture():
    rng = np.random.default_rng(524)
    centers = rng.normal(size=(5, 23))
    return rng.normal(size=(31, 23)), centers, centroid_basis(centers)


@pytest.mark.parametrize('method', VARIANTS)
def test_transform_matches_explicit_linear_operator_and_center_decisions(method, tmp_path):
    x, centers, u = fixture()
    model = make_residual_model(method, 23, 20, b'case', centers, u)
    r = np.eye(23)
    if model.kind == 'mixed':
        padding = np.eye(model.input_d)[:, :23]
        r = hadamard(model.input_d)/np.sqrt(model.input_d) @ np.diag(model.mix_signs) @ padding
    a = model.dense_map if model.kind == 'gaussian' else model.sparse_map.toarray()
    operator = np.vstack((u.T, a@r@(np.eye(23)-u@u.T)))
    expected = (x-centers[0])@operator.T
    z = model.transform(x, batch_rows=7)
    np.testing.assert_allclose(z, expected, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(z[:, :u.shape[1]], (x-centers[0])@u, atol=1e-14)
    np.testing.assert_array_equal(stable_squared_distances(z, model.transform(centers)).argmin(1),
        stable_squared_distances(x, centers).argmin(1))
    path = tmp_path/'model.npz'
    np.savez(path, **model.parameters())
    with np.load(path, allow_pickle=False) as saved:
        restored = ResidualProjection.from_parameters(saved)
    assert restored.fingerprint == model.fingerprint
    np.testing.assert_allclose(restored.transform(x), z, atol=2e-13, rtol=2e-13)
    again = make_residual_model(method, 23, 20, b'case', centers, u)
    assert again.fingerprint == model.fingerprint
    other = make_residual_model(method, 23, 20, b'other', centers, u)
    assert other.fingerprint != model.fingerprint


def test_hadamard_is_orthogonal_and_not_in_place():
    x = np.random.default_rng(3).normal(size=(10, 32)); before=x.copy()
    actual = normalized_hadamard(x)
    np.testing.assert_array_equal(x, before)
    np.testing.assert_allclose(actual, x@hadamard(32)/np.sqrt(32), atol=1e-14)
    np.testing.assert_allclose(actual@actual.T, x@x.T, atol=1e-13)
    np.testing.assert_allclose(normalized_hadamard(actual), x, atol=1e-14)
    with pytest.raises(ValueError): normalized_hadamard(np.ones((2, 31)))


def test_block_sketch_has_one_entry_per_block_even_when_width_not_divisible():
    a = block_matrix(b'block', 43, 101, 4).toarray()
    offsets = [0, 11, 22, 33, 43]
    for left, right in zip(offsets, offsets[1:]):
        np.testing.assert_array_equal(np.count_nonzero(a[left:right], axis=0), np.ones(101))
    np.testing.assert_allclose(np.sum(a*a, axis=0), np.ones(101))


@pytest.mark.parametrize('q', [1, 2, 4, 8])
def test_canonical_maps_match_existing_normalized_implementation(q):
    x, centers, u = fixture()
    seed = b'canonical'
    _, a = sparse_matrix_gen_csr(seed, 16, 23, q)
    old = NativeProjection('hybrid', 23, basis=u, sparse_map=a/np.sqrt(q), origin=centers[0])
    new = ResidualProjection(23, 20, u, centers[0], 'canonical', q, seed=seed)
    np.testing.assert_allclose(new.transform(x), old.transform(x), atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(np.asarray(new.sparse_map.power(2).sum(axis=0)), np.ones((1, 23)))


def test_rank_limited_pca_does_not_silently_add_unrequested_dimensions():
    widths, cap = dimension_plan([64, 128, 256], 60, 10000)
    assert widths == [64, 128, 256] and cap == 59
    cases = list(case_plan(widths, cap, 10000, 5, 70, [70]))
    assert not any(v['method'].startswith('pca') for v in cases)
    assert {v['m'] for v in cases if v['method']!='full'} == set(widths)


def test_tiny_output_skips_maps_with_too_many_nonzeros_per_column():
    cases = list(case_plan([24, 32], 64, 64, 10, 70, [70]))
    for case in cases:
        if case['method'] in VARIANTS:
            assert case['m']-21 >= max(1, VARIANTS[case['method']][1])


def test_all_variants_share_the_head_and_change_only_the_residual():
    x, centers, u = fixture(); bases={name:u for name in VARIANTS}
    outputs = [make_hybrid(name, 23, 20, 70, 'source', 5, 70, centers, bases).transform(x)
               for name in VARIANTS]
    for output in outputs:
        np.testing.assert_array_equal(output[:, :4], outputs[0][:, :4])
    assert not np.allclose(outputs[0][:, 4:], outputs[-1][:, 4:])


def test_tail_weight_scales_only_tail_and_round_trips():
    import numpy as np
    from commitjl.residual_maps import ResidualProjection, select_tail_weight
    rng = np.random.default_rng(3)
    x = rng.standard_normal((40, 20)); basis, _ = np.linalg.qr(rng.standard_normal((20, 3)))
    base = ResidualProjection(20, 10, basis, x[0], 'canonical', 1, seed=b's')
    weighted = ResidualProjection(20, 10, basis, x[0], 'canonical', 1, seed=b's', tail_weight=0.25)
    a, b = base.transform(x), weighted.transform(x)
    assert np.allclose(a[:, :3], b[:, :3]) and np.allclose(0.25*a[:, 3:], b[:, 3:])
    assert 'tail_weight' not in base.parameters() and base.fingerprint != weighted.fingerprint
    again = ResidualProjection.from_parameters(weighted.parameters())
    assert again.fingerprint == weighted.fingerprint and np.allclose(again.transform(x), b)
    w, scores = select_tail_weight(a, 3, 2)
    assert w in scores and set(scores) == {0.25, 0.5, 1.0}


def test_cost_tail_weight_selector_returns_grid_choice():
    import numpy as np
    from commitjl.residual_maps import select_tail_weight_by_cost
    rng = np.random.default_rng(4)
    x = np.vstack([rng.normal(c, 1, (30, 6)) for c in (0, 6)])
    w, scores = select_tail_weight_by_cost(x[::2], x[::2], x[1::2], x[1::2], 2, 2)
    assert w in scores and set(scores) == {0.25, 0.5, 1.0} and all(np.isfinite(list(scores.values())))


def test_continuation_cluster_scores_candidates_in_unweighted_sketch():
    import numpy as np
    from commitjl.residual_maps import continuation_cluster, sketch_partition_cost
    rng = np.random.default_rng(5)
    z = np.vstack([rng.normal(c, 1, (40, 8)) for c in (0, 8)])
    labels, w, cands = continuation_cluster(z, 3, 2, seed=1)
    assert [c['weight'] for c in cands] == [0.25, 0.5, 1.0] and w in (0.25, 0.5, 1.0)
    assert all(np.isclose(c['sketch_cost'], sketch_partition_cost(z, c['labels'])) for c in cands)
    assert sketch_partition_cost(z, labels) == min(c['sketch_cost'] for c in cands)
