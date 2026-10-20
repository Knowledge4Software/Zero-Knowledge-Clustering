import numpy as np
import pytest
from scipy.sparse import csr_matrix

from commitjl.protected_jl import centroid_basis, combined_matrix
from commitjl.reusable_native import NativeProjection, PreparedDataset


def fixtures():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(29, 11))
    c = rng.normal(size=(4, 11))
    u = centroid_basis(c)
    a = csr_matrix(rng.choice([-1., 0., 1.], size=(5, 11)))
    model = NativeProjection('hybrid', 11, basis=u, sparse_map=a, origin=c[0])
    return x, c, u, a, model


def test_reuse_matches_independent_dense_map_and_distances():
    x, c, u, a, model = fixtures()
    state = model.prepare(x, batch_rows=7)
    dense = (x-c[0])@combined_matrix(u, a).T
    np.testing.assert_allclose(state.z, dense, atol=1e-12)
    i, j = np.array([0, 4, 8]), np.array([2, 7, 14, 25])
    expected = np.sum((dense[i,None]-dense[j][None,:])**2, axis=2)
    np.testing.assert_allclose(state.squared_distances(i, j), expected, atol=1e-12)
    state.precompute_distances(max_bytes=29*29*8)
    np.testing.assert_allclose(state.squared_distances(i, j), expected, atol=1e-12)
    with pytest.raises(MemoryError): state.precompute_distances(max_bytes=1)
    for bad in ([29], [-1], [1.5], [[1]], [True]):
        with pytest.raises(ValueError): state.squared_distances(bad, j)


def test_mutable_input_cannot_stale_full_cache():
    x = np.arange(15., dtype=float).reshape(5, 3)
    state = NativeProjection('full', 3).prepare(x)
    expected = np.sum((x[0]-x[4])**2)
    x[:] = 0
    assert state.squared_distances([0], [4])[0,0] == expected
    assert not state.z.flags.writeable
    frozen = np.ones((5, 3)); frozen.setflags(write=False)
    assert NativeProjection('full', 3).prepare(frozen).z is frozen


def test_zero_rank_empty_batches_and_overflow_are_handled(tmp_path):
    model = NativeProjection('hybrid', 3, basis=np.empty((3, 0)), sparse_map=csr_matrix([[1., -1., 0.]]))
    state = model.prepare(np.array([[1., 2., 3.], [2., 0., 3.]]))
    np.testing.assert_allclose(state.z, [[-1.], [2.]])
    assert state.squared_distances([], [0]).shape == (0, 1)
    np.testing.assert_array_equal(state.squared_distances([0, 1], [0, 1]).diagonal(), [0., 0.])
    empty = model.prepare(np.empty((0, 3)))
    empty.save(tmp_path/'empty')
    assert PreparedDataset.load(tmp_path/'empty', expected_source_sha256=empty.source_sha256).z.shape == (0, 1)
    with pytest.raises(AttributeError): model.origin = np.ones(3)
    enormous = NativeProjection('full', 1).prepare(np.array([[1e154], [-1e154]]))
    with np.errstate(over='ignore', invalid='ignore'), pytest.raises(ValueError, match='Distance overflow'):
        enormous.squared_distances([0], [1])


@pytest.mark.parametrize('method', ['full', 'pca', 'hybrid'])
def test_persistent_cache_binding_replay_and_corruption(tmp_path, method):
    x, c, u, a, hybrid = fixtures()
    model = hybrid if method == 'hybrid' else NativeProjection(method, 11, components=u.T) if method == 'pca' else NativeProjection('full', 11)
    state = model.prepare(x)
    destination = tmp_path/'cache'
    state.save(destination)
    restored = PreparedDataset.load(destination, expected_source_sha256=state.source_sha256,
                                    expected_model_sha256=model.fingerprint)
    np.testing.assert_array_equal(restored.z, state.z)
    np.testing.assert_allclose(restored.squared_distances([0, 4], [3, 7]), state.squared_distances([0, 4], [3, 7]))
    np.testing.assert_allclose(restored.model.transform(x[:3]), model.transform(x[:3]))
    with pytest.raises(ValueError, match='dataset mismatch'):
        PreparedDataset.load(destination, expected_source_sha256='different')
    with pytest.raises(ValueError, match='projection mismatch'):
        PreparedDataset.load(destination, expected_source_sha256=state.source_sha256, expected_model_sha256='different')
    with (destination/'Z.npy').open('ab') as stream: stream.write(b'corrupt')
    with pytest.raises(ValueError, match='Corrupt cache'):
        PreparedDataset.load(destination, expected_source_sha256=state.source_sha256)
