import numpy as np

from commitjl.certified_head import (freivalds_orthonormal, learn_certified_head,
                                     make_certified_head_model, residual_distance_bound)


def test_head_rank_and_orthonormality():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 40)); c = x[:4]
    u, _ = learn_certified_head(x, c, 20)
    assert u.shape == (40, 20) and freivalds_orthonormal(u)
    assert not freivalds_orthonormal(u*1.01)


def test_distortion_comes_only_from_residual():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((60, 30)); c = x[:3]
    u, _ = learn_certified_head(x, c, 12)
    model = make_certified_head_model(30, 20, b'seed', c, u)
    z = model.transform(x)
    i, j = 3, 7
    r = (x[i]-x[j]) - u@(u.T@(x[i]-x[j]))
    tail = model.transform(x[[i, j]])[:, 12:]
    assert np.isclose(((z[i]-z[j])**2).sum()-((x[i]-x[j])**2).sum(),
                      ((tail[0]-tail[1])**2).sum()-(r**2).sum())
    share = residual_distance_bound(x, u, c[0], model)
    assert share.shape == (60,) and np.all((share >= -1e-12) & (share <= 1+1e-12))


def test_full_rank_head_is_exact_isometry_on_head():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((50, 10)); c = x[:2]
    u, _ = learn_certified_head(x, c, 9)
    z = make_certified_head_model(10, 10, b's', c, u).transform(x)
    assert np.allclose(np.linalg.norm(z[:, :9], axis=1), np.linalg.norm((x-c[0])@u, axis=1))
