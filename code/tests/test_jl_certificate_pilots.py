import numpy as np
from commitjl.jl_certificate_pilots import random_residual_basis,hint_context,sketch_hints,hint_order,refinement_work,aggregate_stats,compressed_aggregate_bounds
from commitjl.group_change_bounds import exact_old_correction_bounds
from commitjl.group_moment_reference import prepare_groups
from scipy.sparse import eye


def fixture(seed):
    rng=np.random.default_rng(seed);x=rng.integers(-7,8,(11,6));old=rng.integers(-5,6,(3,6));new=rng.integers(-5,6,(3,6));od=((x[:,None,:]-old[None,:,:])**2).sum(2);nd=((x[:,None,:]-new[None,:,:])**2).sum(2);groups=od.argmin(1)
    m=dict(basis=[[7,1,0,0,0,0],[0,7,1,0,0,0]],basis_scale=8,centers=old.tolist(),input_bits=4);g=prepare_groups(x,groups.tolist(),3);bounds=exact_old_correction_bounds(m,x,g,old,new);correction=(nd.min(1)-nd[np.arange(len(x)),groups])*8**6;truth=int(nd.min(1).sum()-od.min(1).sum())*8**6
    return x,old,new,groups,m,g,bounds,correction,truth,nd


def test_identity_sketch_recovers_exact_distance_differences():
    for seed in range(12):
        x,old,new,groups,m,g,bounds,correction,truth,nd=fixture(seed)
        hint,labels=sketch_hints(hint_context(m,x,new,groups),eye(6,format='csr'),q=1)
        assert np.allclose(hint,np.asarray(correction,float)/8**6,atol=1e-8)
        assert np.all(nd[np.arange(len(x)),labels]==nd.min(1))


def test_bad_hints_cannot_make_refinement_bounds_incorrect():
    for seed in range(16):
        x,old,new,groups,m,g,bounds,correction,truth,nd=fixture(seed)
        for hint in [np.full(len(x),1e10),np.full(len(x),-1e10),np.full(len(x),np.nan)]:
            order=hint_order(bounds,hint)
            for count in [0,1,4,len(x)]:
                result=refinement_work(bounds,correction,order,count)
                assert result['lower']<=truth<=result['upper']


def test_group_norm_compression_and_random_extensions_contain_full():
    for seed in range(14):
        x,old,new,groups,m,g,bounds,correction,truth,nd=fixture(seed);extended=random_residual_basis(m,2,seed)
        assert extended['basis'][:2]==m['basis']
        for group_model in [m,extended]:
            result=compressed_aggregate_bounds(bounds,group_model,aggregate_stats(group_model,g),old,new)
            assert result['lower']<=truth<=result['upper']
