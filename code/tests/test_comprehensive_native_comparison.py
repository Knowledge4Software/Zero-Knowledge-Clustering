import numpy as np
import pytest

from experiments.comprehensive_native_comparison import dimension_plan,case_plan
from commitjl.protected_jl import centroid_basis
from commitjl.reusable_native import NativeProjection,stable_squared_distances
from commitjl.sparse_gen import sparse_matrix_gen_csr


def test_rank_limited_pca_has_an_explicit_equal_width_control():
    widths,cap=dimension_plan([32,64,128,256,512],60,10000)
    assert cap==59 and widths==[32,59,64,128,256,512]
    cases=list(case_plan(widths,cap,10000,10,30,range(30,33)))
    assert {r['m'] for r in cases if r['method'].startswith('pca_')}=={32,59}
    assert {r['m'] for r in cases if r['method']=='protected_plus12'}==set(widths)
    assert {r['seed'] for r in cases if r['method']=='protected_plus12'}=={30,31,32}


def test_uncompressed_widths_are_not_counted_as_compression_and_low_rank_is_valid():
    widths,cap=dimension_plan([32,64,128,256,512],600,128)
    assert widths==[32,64]
    cases=list(case_plan(widths,cap,128,5,30,[30],rank=2))
    assert next(r for r in cases if r['method']=='protected_only')['m']==2


def test_residual_downweighting_preserves_fixed_centers_but_changes_geometry():
    rng=np.random.default_rng(32);centers=rng.normal(size=(5,40));u=centroid_basis(centers)
    _,a=sparse_matrix_gen_csr(b'weight-check',20,40,4);a=a.astype(float);a.data/=2
    x=rng.normal(size=(50,40))
    full_labels=stable_squared_distances(x,centers).argmin(axis=1)
    outputs=[]
    for weight in [1.,.5,.25]:
        model=NativeProjection('hybrid',40,basis=u,sparse_map=a*weight,origin=centers[0])
        z=model.transform(x);outputs.append(z)
        np.testing.assert_array_equal(stable_squared_distances(z,model.transform(centers)).argmin(axis=1),full_labels)
    np.testing.assert_array_equal(outputs[0][:,:4],outputs[2][:,:4])
    np.testing.assert_allclose(outputs[2][:,4:],outputs[0][:,4:]*.25,atol=1e-14)
    assert not np.allclose(stable_squared_distances(outputs[0],outputs[0]),stable_squared_distances(outputs[2],outputs[2]))


def test_norm_correction_preserves_original_center_costs_not_just_labels():
    from commitjl.experimental_norm_correction import residual_norm_corrected_transform
    rng=np.random.default_rng(143);centers=rng.normal(size=(5,40));u=centroid_basis(centers)
    _,a=sparse_matrix_gen_csr(b'norm-correction-check',20,40,1)
    model=NativeProjection('hybrid',40,basis=u,sparse_map=a,origin=centers[0])
    x=rng.normal(size=(50,40));z=residual_norm_corrected_transform(model,x)
    c=residual_norm_corrected_transform(model,centers)
    np.testing.assert_allclose(stable_squared_distances(z,c),stable_squared_distances(x,centers),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(np.sum(z*z,axis=1),np.sum((x-centers[0])**2,axis=1),rtol=1e-12)
    np.testing.assert_array_equal(z[:,:4],model.transform(x)[:,:4])


def test_norm_correction_rejects_annihilated_positive_residual():
    from commitjl.experimental_norm_correction import residual_norm_corrected_transform
    model=NativeProjection('hybrid',4,basis=np.empty((4,0)),sparse_map=np.zeros((2,4)))
    with pytest.raises(ValueError,match='annihilated'):
        residual_norm_corrected_transform(model,np.ones((1,4)))
