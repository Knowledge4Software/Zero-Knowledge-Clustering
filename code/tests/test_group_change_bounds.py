import numpy as np
from commitjl.group_change_bounds import correction_bounds,paired_correction_bounds,exact_old_correction_bounds,compressed_old_correction_bounds,refine
import pytest
from commitjl.group_moment_reference import prepare_groups


def oracle(rows,groups,old,new):
    assigned=[];delta=[]
    for row,g in zip(rows,groups):
        a=[sum((int(x)-int(c))**2 for x,c in zip(row,center)) for center in old]
        b=[sum((int(x)-int(c))**2 for x,c in zip(row,center)) for center in new]
        delta.append(min(b)-min(a));assigned.append(b[g]-a[g])
    return delta,[d-a for d,a in zip(delta,assigned)]


def test_group_bounds_arbitrary_groups_rounded_basis_and_full_refinement():
    rng=np.random.default_rng(2449)
    for _ in range(18):
        rows=rng.integers(-7,8,(9,4));old=rng.integers(-5,6,(3,4));new=rng.integers(-5,6,(3,4))
        groups=rng.integers(0,3,len(rows)).tolist()
        model=dict(basis=[[7,1,0,0],[0,7,1,0]],basis_scale=8,centers=old.tolist(),input_bits=4)
        prepared=prepare_groups(rows,groups,3);result=correction_bounds(model,rows,prepared,old,new)
        delta,correction=oracle(rows,groups,old,new);den=result['cost_denominator'];exact=sum(delta)*den
        scaled=[v*den for v in correction]
        assert all(int(a)<=v<=int(b) for a,v,b in zip(result['row_lower'],scaled,result['row_upper']))
        joint=paired_correction_bounds(model,rows,prepared,old,new)
        assert all(int(a)<=int(c)<=v<=int(d)<=int(b) for a,c,v,d,b in zip(result['row_lower'],joint['row_lower'],scaled,joint['row_upper'],result['row_upper']))
        previous=result
        for budget in [0,.1,.3,.6,1]:
            refined=refine(result,scaled,budget)
            assert previous['change_lower_scaled']<=refined['change_lower_scaled']<=exact<=refined['change_upper_scaled']<=previous['change_upper_scaled']
            previous=refined
        assert previous['change_lower_scaled']==previous['change_upper_scaled']==exact


def test_stable_in_span_groups_need_no_exact_rows():
    rows=np.asarray([[0,1],[1,1],[9,1],[10,1]])
    old=np.asarray([[0,0],[10,0]]);new=np.asarray([[0,2],[10,2]])
    groups=[0,0,1,1];model=dict(basis=[[8,0]],basis_scale=8,centers=old.tolist(),input_bits=4)
    result=correction_bounds(model,rows,prepare_groups(rows,groups,2),old,new)
    delta,correction=oracle(rows,groups,old,new)
    out=refine(result,[v*result['cost_denominator'] for v in correction],0)
    assert out['exact_rows']==out['uncertain_rows']==0
    assert out['change_lower_scaled']==out['change_upper_scaled']==sum(delta)*result['cost_denominator']


def test_assigned_group_change_alone_can_have_the_wrong_sign():
    rows=np.asarray([[0]]);old=np.asarray([[0],[10]]);new=np.asarray([[9],[1]])
    groups=[1];model=dict(basis=[[8]],basis_scale=8,centers=old.tolist(),input_bits=4)
    result=correction_bounds(model,rows,prepare_groups(rows,groups,2),old,new)
    assert result['assigned_scaled']==-99*result['cost_denominator']
    # The true nearest-center cost increases by1. A alone would say improves.
    assert result['change_lower_scaled']==result['change_upper_scaled']==result['cost_denominator']
    with pytest.raises(ValueError,match='not exact old nearest'):
        exact_old_correction_bounds(model,rows,prepare_groups(rows,groups,2),old,new)


def test_exact_old_profile_contains_full_and_fast_sign_is_sound():
    rng=np.random.default_rng(241024)
    for _ in range(20):
        rows=rng.integers(-7,8,(9,4));old=rng.integers(-5,6,(3,4));new=rng.integers(-5,6,(3,4))
        groups=((rows[:,None,:]-old[None,:,:])**2).sum(2).argmin(1).tolist()
        model=dict(basis=[[7,1,0,0],[0,7,1,0]],basis_scale=8,centers=old.tolist(),input_bits=4)
        bounds=exact_old_correction_bounds(model,rows,prepare_groups(rows,groups,3),old,new)
        delta,_=oracle(rows,groups,old,new);truth=sum(delta)*bounds['cost_denominator']
        assert bounds['change_lower_scaled']<=truth<=bounds['change_upper_scaled']<=bounds['assigned_scaled']
        if bounds['assigned_upper_bound_fast_path']:assert truth<0


def test_exact_old_ties_are_valid():
    rows=np.asarray([[4,1]]);old=np.asarray([[0,0],[8,0]]);new=np.asarray([[2,1],[6,1]])
    model=dict(basis=[[8,0]],basis_scale=8,centers=old.tolist(),input_bits=4)
    for group in [0,1]:
        bounds=exact_old_correction_bounds(model,rows,prepare_groups(rows,[group],2),old,new)
        assert bounds['assigned_upper_bound_fast_path']
        assert bounds['change_upper_scaled']<0


def test_compressed_group_moment_bounds_contain_exact_change():
    rng=np.random.default_rng(251025)
    for _ in range(12):
        rows=rng.integers(-7,8,(9,4));old=rng.integers(-5,6,(3,4));new=rng.integers(-5,6,(3,4));groups=((rows[:,None,:]-old[None,:,:])**2).sum(2).argmin(1).tolist()
        model=dict(basis=[[7,1,0,0],[0,7,1,0]],basis_scale=8,centers=old.tolist(),input_bits=4)
        bounds=compressed_old_correction_bounds(model,rows,prepare_groups(rows,groups,3),old,new);delta,_=oracle(rows,groups,old,new);truth=sum(delta)*bounds['cost_denominator']
        assert bounds['change_lower_scaled']<=truth<=bounds['change_upper_scaled']
        assert bounds['assigned_lower_scaled']<=bounds['assigned_scaled']<=bounds['assigned_upper_scaled']
