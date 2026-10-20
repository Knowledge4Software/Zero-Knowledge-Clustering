"""Bridge regression checks for selecting the retained diagnostic circuits."""
import numpy as np

from commitjl.zk_bridge import ZKInstance
from experiments.rq2_proof_cost import build_zk_instance


def test_optimization_controls_are_explicit_in_witness_json():
    inst = ZKInstance(n=1,d=1,m=1,k=1,q=1,variant='sparse',backend='groth16',
        c_x=0,c_theta=0,c_a=0,psi_hat=0,centroids=np.zeros((1,1),dtype=int),
        X=np.zeros((1,1),dtype=int),r_x=0,assign=[0])
    payload = inst.to_json_dict()
    assert payload['legacy_projection'] is False
    assert payload['legacy_bounds'] is False
    inst.legacy_projection = inst.legacy_bounds = True
    payload = inst.to_json_dict()
    assert payload['legacy_projection'] is True
    assert payload['legacy_bounds'] is True


def test_proof_runner_propagates_diagnostic_controls_without_changing_relation():
    data = dict(X=np.array([[1.,2.],[-2.,1.],[3.,-1.],[-1.,-3.]]),
                meta=dict(name='optimization-fixture',sha256='fixed-data'))
    row = dict(variant='full',k=2,m=2,q=1,seed=0,precision_bits=16)
    modern,*_ = build_zk_instance(row,{},data=data)
    legacy,*_ = build_zk_instance({**row,'legacy_projection':True,'legacy_bounds':True},{},data=data)
    assert modern.legacy_bounds is False
    assert legacy.legacy_bounds is True and legacy.legacy_projection is True
    np.testing.assert_array_equal(modern.X,legacy.X)
    np.testing.assert_array_equal(modern.centroids,legacy.centroids)
    assert modern.assign == legacy.assign and modern.psi_hat == legacy.psi_hat
