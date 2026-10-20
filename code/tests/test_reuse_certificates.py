import numpy as np
from commitjl.reuse_certificates import certificate, exact_reference, gram_defect


def model(basis):
    return dict(basis=basis,basis_scale=8,centers=[[0]*len(basis[0]),[2]+[0]*(len(basis[0])-1)])


def test_rounded_basis_error_exists_even_with_zero_center_residual():
    m=model([[7]])
    out=certificate(m,[[1]],[[0],[2]])
    assert gram_defect(m)==15
    assert out['psi']==3361
    assert out['cost_lower_scaled']<=8**6<=out['cost_upper_scaled']
    assert not out['certified_flags'][0]  # original-space tie conservatively uncertified


def test_exact_in_span_unique_assignments_are_certified():
    m=model([[8,0,0],[0,8,0]])
    rows=[[1,0,2],[3,0,4],[-2,2,1]];centers=[[0,0,0],[4,0,0]]
    out=certificate(m,rows,centers);labels,cost=exact_reference(rows,centers)
    assert out['labels']==labels
    assert out['certified_flags']==[True,True,True]
    assert out['cost_lower_scaled']==out['cost_upper_scaled']==cost*8**6


def test_rounded_random_certificates_do_not_certify_false_assignments():
    rng=np.random.default_rng(745)
    for _ in range(40):
        m=model([[7,1,0,0],[0,7,1,0]])
        rows=rng.integers(-12,13,(8,4));centers=rng.integers(-12,13,(3,4))
        out=certificate(m,rows,centers);labels,cost=exact_reference(rows,centers)
        assert out['cost_lower_scaled']<=cost*8**6<=out['cost_upper_scaled']
        assert all(not flag or a==b for flag,a,b in zip(out['certified_flags'],out['labels'],labels))
