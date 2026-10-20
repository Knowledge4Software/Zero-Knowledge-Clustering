import numpy as np
import pytest
from scipy.sparse import csr_matrix
from commitjl.protected_jl import centroid_basis, protected_project, combined_matrix, integer_score_assign
from commitjl.protected_jl import integer_protected_project,weighted_integer_assignment,build_protected_instance
from commitjl.clustering import nearest_centroid_assign, psi_hat


def test_residual_implementation_and_full_rank_assignment():
    rng = np.random.default_rng(32)
    x = rng.normal(size=(83, 19))
    c = rng.normal(size=(5, 19))
    u = centroid_basis(c)
    a = csr_matrix(rng.choice([-1., 0., 1.], size=(7, 19)))
    z = protected_project(x, u, a, c[0])
    t = protected_project(c, u, a, c[0])
    np.testing.assert_allclose(z, (x-c[0]) @ combined_matrix(u, a).T, atol=1e-12)
    np.testing.assert_allclose(t[:, u.shape[1]:], 0, atol=1e-12)
    np.testing.assert_array_equal(nearest_centroid_assign(x, c), nearest_centroid_assign(z, t))
    # Protecting the centroid span alone is also sufficient for this task.
    np.testing.assert_array_equal(nearest_centroid_assign(x, c),
        nearest_centroid_assign(protected_project(x,u,None,c[0]), protected_project(c,u,None,c[0])))


def test_degenerate_centers_and_ties():
    c = np.array([[0.,0.],[0.,0.],[2.,0.]])
    u = centroid_basis(c)
    assert u.shape == (2,1)
    a = csr_matrix([[1.,-1.]])
    x = np.array([[0.,5.],[1.,0.],[2.,-10.]])
    np.testing.assert_array_equal(nearest_centroid_assign(protected_project(x,u,a),
        protected_project(c,u,a)), [0,0,2])


def test_exact_score_with_overflow_and_tie():
    x = np.array([[2**60, -2**60],[0,0]],dtype=object)
    c = np.array([[0,0],[0,0],[2**60,-2**60]],dtype=object)
    labels,cost = integer_score_assign(x,c)
    np.testing.assert_array_equal(labels, nearest_centroid_assign(x,c))
    assert cost == psi_hat(x,c,labels)
    assert labels.tolist() == [2,0]


def test_invalid_basis_is_rejected():
    with pytest.raises(ValueError): protected_project(np.ones((2,3)),np.ones((3,1)),None)
    with pytest.raises(ValueError): centroid_basis(np.ones((2,3)), -1)
    with pytest.raises(ValueError): centroid_basis(np.ones((1,3)), -1)


def test_integer_hybrid_independent_dense_reference_and_weights():
    x=np.array([[10,-3,7],[-4,6,8]],dtype=np.int64)
    c=np.array([[2,3,1],[8,-2,0]],dtype=np.int64)
    b=np.array([[3,-1,2]],dtype=np.int64); a=csr_matrix([[1,0,-1],[0,1,1]])
    scale=4
    dense=np.vstack((scale*b,scale*scale*a.toarray()-a.toarray()@b.T@b))
    z=integer_protected_project(x,b,a,scale)
    projected=integer_protected_project(c,b,a,scale)
    np.testing.assert_array_equal(z,x@dense.T)
    labels,cost=weighted_integer_assignment(z,projected,1,2)
    sq=np.sum((z.astype(object)[:,None,:]-projected.astype(object)[None,:,:])**2*np.array([2,1,1],dtype=object),axis=2)
    np.testing.assert_array_equal(labels,sq.argmin(axis=1))
    assert cost==sum(sq[i,j] for i,j in enumerate(labels))


def test_builder_json_flags_and_complete_objective():
    x=np.array([[3,-4,1,2],[-2,1,4,-3]],dtype=np.int64)
    c=np.array([[0,1,-1,2],[2,-1,0,1]],dtype=np.int64)
    inst=build_protected_instance(x,c,b'protected-unit',2,2,1,8,specialize=True)
    payload=inst.to_json_dict()
    assert payload['score_distances'] and payload['specialize_sparse']
    assert payload['basis_scale']==8 and len(payload['protected_basis'])==1


def test_study_objective_matches_paper_when_clusters_empty():
    from experiments.protected_jl_study import phi_from_labels
    from experiments.pipeline import _centroids_from_labels
    from commitjl.clustering import phi_X
    x=np.array([[0.,0.],[1.,2.],[4.,5.],[4.,6.]])
    labels=np.array([0,0,2,2])
    assert phi_from_labels(x,labels,4,np.sum(x*x))==pytest.approx(phi_X(x,_centroids_from_labels(x,labels,4)))


def test_zero_rank_builder_uses_consistent_sparse_objective():
    from commitjl.zk_bridge import build_sparse_instance
    x=np.array([[1,2],[3,4]],dtype=np.int64);c=np.zeros((2,2),dtype=np.int64)
    ordinary=build_sparse_instance(x,c,b'zero-rank',1,2)
    hybrid=build_protected_instance(x,c,b'zero-rank',2,1,1,64)
    assert hybrid.protected_basis is None and hybrid.basis_scale==0
    assert hybrid.assign==ordinary.assign and hybrid.psi_hat==ordinary.psi_hat
    with pytest.raises(ValueError):build_protected_instance(x.astype(float),c,b'bad',2,1,1)
    for scale in [0, -1, 1.5, 2**63]:
        with pytest.raises(ValueError):build_protected_instance(x,c,b'bad',2,1,1,scale)


def test_integer_hybrid_overflow_uses_exact_arithmetic():
    x=np.array([[2**60,-2**60]],dtype=object)
    b=np.array([[3,-2]],dtype=np.int64)
    a=csr_matrix([[1,-1]])
    dense=np.vstack((4*b,16*a.toarray()-a.toarray()@b.T@b)).astype(object)
    projected=integer_protected_project(x,b,a,np.int64(4))
    assert projected.dtype==object
    np.testing.assert_array_equal(projected,x@dense.T)
