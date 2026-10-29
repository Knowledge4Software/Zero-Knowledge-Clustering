import numpy as np
import pytest
from scipy.sparse import csr_matrix

from commitjl.enriched_jl import enrich_centroid_basis
from commitjl.reusable_native import NativeProjection, stable_squared_distances


def test_extra_directions_preserve_centers_and_improve_training_residual_capture():
    rng = np.random.default_rng(192)
    c = rng.normal(size=(5, 32))
    x = rng.normal(size=(160, 32))*np.linspace(5, .1, 32)
    u, meta = enrich_centroid_basis(x, c, 12)
    assert meta['centroid_rank'] == 4 and meta['added_rank'] == 12
    assert u.shape == (32, 16)
    np.testing.assert_allclose(u.T@u, np.eye(16), atol=1e-12)
    delta = c-c[0]
    np.testing.assert_allclose(delta-(delta@u)@u.T, 0, atol=1e-12)
    variation = x-x.mean(axis=0)
    old = variation-(variation@u[:, :4])@u[:, :4].T
    new = variation-(variation@u)@u.T
    assert np.linalg.norm(new) < .65*np.linalg.norm(old)
    # Check actual output width and fixed-center decisions on UNSEEN points.
    s = csr_matrix(rng.normal(size=(8, 32)))
    model = NativeProjection('hybrid', 32, basis=u, sparse_map=s, origin=c[0])
    heldout = rng.normal(size=(90, 32))
    z, cc = model.transform(heldout), model.transform(c)
    assert z.shape == (90, 24)
    expected = np.column_stack(((heldout-c[0])@u,
        ((heldout-c[0])-((heldout-c[0])@u)@u.T)@s.T))
    np.testing.assert_allclose(z, expected, atol=1e-12)
    np.testing.assert_array_equal(stable_squared_distances(heldout, c).argmin(axis=1),
        stable_squared_distances(z, cc).argmin(axis=1))


def test_degenerate_residual_does_not_add_noise_directions():
    c = np.zeros((3, 8)); c[1, 0] = 2; c[2, 1] = 2
    x = np.zeros((50, 8)); x[:, :2] = np.random.default_rng(5).normal(size=(50, 2))
    u, meta = enrich_centroid_basis(x, c, 12)
    assert u.shape == (8, 2) and meta['added_rank'] == 0
    u2, meta2 = enrich_centroid_basis(np.ones((25, 8)), np.ones((3, 8)), 12)
    assert u2.shape == (8, 0) and meta2['added_rank'] == 0


def test_reproducible_and_rejects_invalid_inputs():
    rng = np.random.default_rng(0); x = rng.normal(size=(50, 20)); c = x[:5]
    a, ma = enrich_centroid_basis(x, c, 3)
    b, mb = enrich_centroid_basis(x, c, 3)
    np.testing.assert_array_equal(a, b)
    assert ma == mb
    for extra in [-1, 1.5]:
        with pytest.raises(ValueError): enrich_centroid_basis(x, c, extra)
    with pytest.raises(ValueError): enrich_centroid_basis(x[:, :3], c)


def test_ordered_prefix_keeps_leading_residual_pca_subspace():
    from sklearn.decomposition import PCA
    from commitjl.protected_jl import centroid_basis
    rng = np.random.default_rng(613)
    x = rng.normal(size=(100, 24))*np.geomspace(8, .01, 24)
    centers = rng.normal(size=(3, 24))
    u = centroid_basis(centers)
    residual = x-centers[0]
    residual -= (residual@u)@u.T
    residual -= residual.mean(axis=0)
    pca = PCA(10, svd_solver='randomized', random_state=7,
              iterated_power=2, n_oversamples=10).fit(residual)
    basis, _ = enrich_centroid_basis(x, centers, 10, random_state=7, preserve_pca_order=True)
    for take in [1, 3, 7]:
        target = pca.components_[:take].T
        prefix = basis[:, u.shape[1]:u.shape[1]+take]
        np.testing.assert_allclose(target@target.T, prefix@prefix.T, atol=1e-10)
    np.testing.assert_allclose(basis.T@basis, np.eye(basis.shape[1]), atol=1e-12)
