import numpy as np
from commitjl.old_gap_bounds import old_gap_correction_bounds
from commitjl.group_change_bounds import exact_old_correction_bounds
from commitjl.group_moment_reference import prepare_groups


def test_gap_intersections_contain_row_truth_and_strengthen_base():
    rng=np.random.default_rng(251025)
    for _ in range(24):
        x=rng.integers(-7,8,(11,4));old=rng.integers(-5,6,(3,4));new=rng.integers(-5,6,(3,4))
        od=((x[:,None,:]-old[None,:,:])**2).sum(2);nd=((x[:,None,:]-new[None,:,:])**2).sum(2);groups=od.argmin(1)
        model=dict(basis=[[7,1,0,0],[0,7,1,0]],basis_scale=8,centers=old.tolist(),input_bits=4);prepared=prepare_groups(x,groups.tolist(),3)
        base=exact_old_correction_bounds(model,x,prepared,old,new);correction=(nd.min(1)-nd[np.arange(len(x)),groups])*base['cost_denominator']
        for minimum in [False,True]:
            bounds=old_gap_correction_bounds(model,x,prepared,old,new,minimum_only=minimum)
            assert all(int(a)<=int(b)<=int(v)<=int(c)<=int(d) for a,b,v,c,d in zip(base['row_lower'],bounds['row_lower'],correction,bounds['row_upper'],base['row_upper']))


def test_identity_is_exact_even_with_rounded_basis_and_old_ties():
    x=np.asarray([[0,1],[4,1],[8,1]]);old=np.asarray([[0,0],[8,0]])
    model=dict(basis=[[7,1]],basis_scale=8,centers=old.tolist(),input_bits=4);groups=[0,1,1]
    for minimum in [False,True]:
        bounds=old_gap_correction_bounds(model,x,prepare_groups(x,groups,2),old,old,minimum_only=minimum)
        assert bounds['change_lower_scaled']==bounds['change_upper_scaled']==0
