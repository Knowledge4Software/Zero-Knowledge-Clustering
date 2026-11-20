"""Truth invariants, unchanged selection inputs and honest scale calibration."""
import copy
import numpy as np
import pytest

from commitjl.pilot_io import array_sha
from experiments import paper_completion_20261102 as study


def toy():
    rows=np.array([[0,0,1,0],[1,0,1,0],[7,7,0,1],[8,8,0,1]],dtype=np.int64)
    old=np.array([[0,0,0,0],[8,8,0,0]],dtype=np.int64)
    centers=np.array([old,[[1,0,1,0],[7,8,0,1]],[[30,30,0,0],[-30,-30,0,0]]])
    basis=np.eye(4,dtype=np.int64)[:2]*16384
    return rows,old,centers,basis


def test_all_ablations_contain_exact_truth_and_never_certify_wrong_sign():
    rows,old,centers,basis=toy()
    records=study.ablate(rows,old,centers,basis,1.,'toy',2)
    assert len(records)==3*2*4
    for r in records:
        lo,truth,hi=map(int,[r['lower'],r['truth'],r['upper']])
        assert lo<=truth<=hi
        assert r['decision'] in ['inconclusive',study.point_decision(truth)]
        assert r['corrections']<=r['cap']
    assert all(r['decision']=='equal' for r in records if r['request']==0)


def test_refinement_tightens_intervals_as_budget_grows():
    rows,old,centers,basis=toy()
    records=study.ablate(rows,old,centers,basis,1.,'toy',2)
    for request in range(3):
        for order in ['endpoint','width']:
            seq=[r for r in records if r['request']==request and r['ordering']==order]
            for a,b in zip(seq,seq[1:]):
                assert int(a['lower'])<=int(b['lower'])<=int(b['upper'])<=int(a['upper'])


def test_numerical_baselines_are_reproducible_and_not_labeled_proof_systems():
    rows,old,centers,_=toy()
    a=study.controls(rows,old,centers,907)
    assert a==study.controls(rows,old,centers,907)
    assert len(a)==8*3
    assert all(r['cryptographic_certificate'] is False for r in a)
    assert all(r['true_change']==0 and r['estimated_change']==0 for r in a if r['request']==0)


def test_request_prefix_preserves_validation_and_original_arrays(tmp_path):
    arrays=dict(rows=np.zeros((128,4),dtype=np.int64),past=np.zeros((512,4)),
        centers=np.arange(8*2*4).reshape(8,2,4),validation_rows=np.ones((128,4)),
        validation_centers=np.ones((8,2,4)),old=np.zeros((2,4)))
    np.savez(tmp_path/'arrays.npz',**arrays)
    study.save(tmp_path/'provenance.json',{})
    before=study.sha(tmp_path/'arrays.npz')
    a=study.diagnostic_arrays(tmp_path,'fashion',128,1)
    np.testing.assert_array_equal(a['centers'],arrays['centers'][:1])
    np.testing.assert_array_equal(a['validation_rows'],arrays['validation_rows'])
    np.testing.assert_array_equal(a['validation_centers'],arrays['validation_centers'])
    assert study.sha(tmp_path/'arrays.npz')==before
    with pytest.raises(ValueError):study.diagnostic_arrays(tmp_path,'fashion',128,9)


def test_scale_uses_disjoint_audit_prefix_and_checks_training_pin(tmp_path,monkeypatch):
    arrays=dict(rows=np.zeros((128,4)),past=np.zeros((512,4)),centers=np.zeros((8,2,4)))
    np.savez(tmp_path/'arrays.npz',**arrays)
    train=np.zeros((1024,4));audit=np.arange(1024*4).reshape(1024,4)
    study.save(tmp_path/'provenance.json',dict(training_sha256=array_sha(train[:512])))
    monkeypatch.setattr(study,'source',lambda *_:(train,audit,2,1.,{}))
    a=study.diagnostic_arrays(tmp_path,'fashion',512,1)
    np.testing.assert_array_equal(a['rows'],audit[128:640])
    study.save(tmp_path/'provenance.json',dict(training_sha256='wrong'))
    with pytest.raises(AssertionError):study.diagnostic_arrays(tmp_path,'fashion',512,1)


def test_scaled_cost_profiles_are_labeled_estimates_and_leave_original_unchanged():
    p={'fashion':{'16':dict(shape=[128,784,5,16,12,16384],
        costs=dict(full_seconds=100.,full_batch_seconds=10.),keys_rows_seconds=20.,
        prepare_seconds=2.,full_initialization_seconds=50.)}}
    before=copy.deepcopy(p);q=study.scaled_profiles(p,'fashion',512)
    assert p==before and q['fashion']['16']['shape'][0]==512
    assert q['fashion']['16']['source_measured_shape'][0]==128
    assert q['fashion']['16']['costs']['full_seconds']==400.
    assert 'extrapolation' in q['fashion']['16']['calibration_scope']


def test_small_historical_training_set_is_explicitly_excluded(tmp_path):
    source=tmp_path/'arcene_seed701';source.mkdir()
    np.savez(source/'arrays.npz',training=np.zeros((64,100)),audit=np.zeros((32,100)),centers=np.zeros((5,2,100)))
    study.save(source/'pin.json',dict(arrays_sha256=study.sha(source/'arrays.npz')))
    study.native_historical(source,tmp_path/'out')
    result=study.read(tmp_path/'out/result.json')
    assert result['excluded'] and not result['complete'] and result['available_past_rows']==64
