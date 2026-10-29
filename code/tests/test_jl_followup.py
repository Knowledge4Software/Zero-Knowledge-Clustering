import numpy as np
from commitjl.jl_followup import ensemble_scores,covariance_jl_basis
from commitjl.jl_certificate_pilots import hint_context
from commitjl.group_change_bounds import exact_old_correction_bounds
from commitjl.group_moment_reference import prepare_groups


def test_identity_ensemble_recovers_original_space_relative_scores():
    rng=np.random.default_rng(113);x=rng.integers(-20,21,(13,24));old=rng.integers(-10,11,(3,24));new=rng.integers(-10,11,(3,24));basis=np.zeros((3,24),dtype=int);basis[:,:3]=np.eye(3,dtype=int)*16384
    model=dict(basis=basis.tolist(),basis_scale=16384,centers=old.tolist())
    context=hint_context(model,x,new,[0]*len(x));a=np.eye(24)*np.sqrt(3)
    truth=((x[:,None,:]-new[None,:,:])**2).sum(2);truth=truth-truth[:,[0]]
    np.testing.assert_allclose(ensemble_scores(context,[a,a,a]),truth,atol=1e-7)
    np.testing.assert_allclose(ensemble_scores(context,[a,a,a],gated=True),truth,atol=1e-7)


def test_covariance_extensions_keep_prefix_and_valid_integer_envelopes():
    rng=np.random.default_rng(819);training=rng.integers(-30,31,(150,32));x=rng.integers(-30,31,(17,32));old=rng.integers(-15,16,(3,32));new=rng.integers(-15,16,(3,32));basis=np.zeros((3,32),dtype=int);basis[:,:3]=np.eye(3,dtype=int)*16384
    model=dict(basis=basis.tolist(),basis_scale=16384,centers=old.tolist(),input_bits=12)
    od=((x[:,None,:]-old[None,:,:])**2).sum(2);groups=prepare_groups(x,od.argmin(1).tolist(),3);truth=int(((x[:,None,:]-new[None,:,:])**2).sum(2).min(1).sum()-od.min(1).sum())
    for power in (0,1,2):
        m=covariance_jl_basis(model,training,4,9981,power);assert m['basis'][:3]==model['basis'];assert len(m['basis'])==7
        b=exact_old_correction_bounds(m,x,groups,old,new);assert b['change_lower_scaled']<=truth*b['cost_denominator']<=b['change_upper_scaled']
