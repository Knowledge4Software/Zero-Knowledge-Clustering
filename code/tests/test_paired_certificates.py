import numpy as np
import pytest
from commitjl.paired_certificates import paired_change
from commitjl.reuse_certificates import exact_reference


@pytest.mark.parametrize('seed',range(12))
def test_paired_bound_contains_exact_change_with_nonorthogonal_basis(seed):
    rng=np.random.default_rng(seed)
    d=5;t=8
    rows=rng.integers(-8,9,(9,d)).tolist()
    old=rng.integers(-6,7,(3,d)).tolist();new=rng.integers(-6,7,(2,d)).tolist()
    # Deliberately not orthonormal: the exact Gram correction must handle this.
    model=dict(basis=rng.integers(-9,10,(3,d)).tolist(),basis_scale=t,centers=old,input_bits=4)
    result=paired_change(model,rows,old,new,row_outputs=True)
    for i,x in enumerate(rows):
        _,a=exact_reference([x],old);_,b=exact_reference([x],new)
        delta=(b-a)*t**6
        assert result['row_lower'][i]<=delta<=result['row_upper'][i]
    _,a=exact_reference(rows,old);_,b=exact_reference(rows,new)
    assert result['change_lower_scaled']<=(b-a)*t**6<=result['change_upper_scaled']


def test_identical_center_sets_have_exact_zero_change_despite_large_residuals():
    model=dict(basis=[[8,1,0]],basis_scale=8,centers=[[1,2,3],[-2,-3,4]])
    result=paired_change(model,[[5,8,-9],[-8,3,6]],model['centers'],model['centers'])
    assert result['change_lower_scaled']==result['change_upper_scaled']==0


def test_exact_represented_change_can_be_decided_and_reversed():
    model=dict(basis=[[8,0]],basis_scale=8,centers=[[0,0],[4,0]])
    rows=[[2,7],[3,-4]];old=model['centers'];new=[[2,0],[3,0]]
    result=paired_change(model,rows,old,new)
    assert result['decision']=='improves'
    reverse=paired_change(model,rows,new,old)
    assert reverse['decision']=='worsens'
    assert reverse['change_lower_scaled']==-result['change_upper_scaled']
