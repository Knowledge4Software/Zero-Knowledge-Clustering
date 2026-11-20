"""Final selection is validation-only; portable evidence survives key retirement."""
from dataclasses import asdict
import json
import sys
import tarfile

import numpy as np
import pytest

from commitjl.calibrated_routing_20261102 import BatchCosts
from commitjl.final_method_20261102 import fit_bases, validation_choice


def profiles():
    return {str(rank):dict(costs=asdict(BatchCosts(100,1 if rank==16 else 20,1)),
             prepare_seconds=1,keys_rows_seconds=5,refinement_setup_seconds=2,
             full_initialization_seconds=10) for rank in [16,79]}


def records():
    return [dict(family=f,rank=r,request=i,conclusive=True,corrections=0)
            for f in ['centroid','pca'] for r in [16,79] for i in range(8)]


def test_lower_rank_requires_perfect_validation_and_measured_savings():
    data=records()
    assert validation_choice(data,'centroid',profiles())['rank']==16
    data[0]['conclusive']=False
    assert validation_choice(data,'centroid',profiles())['rank']==79
    assert validation_choice(data,'pca',profiles())['rank']==16
    p=profiles();p['16']['costs']['certificate_seconds']=19
    assert validation_choice(records(),'centroid',p)['rank']==79


def test_missing_profile_defaults_conservatively_and_future_fields_are_ignored():
    assert validation_choice(records(),'centroid',{'79':profiles()['79']})['rank']==79
    a=validation_choice(records(),'centroid',profiles())
    altered=[dict(r,test_conclusive=False,test_time=10**10) for r in records()]
    assert validation_choice(altered,'centroid',profiles())==a
    with pytest.raises(ValueError):validation_choice(records()[:-1],'pca',profiles())


def test_final_basis_uses_training_rows_and_preserves_old_center_directions():
    rng=np.random.default_rng(123)
    past=rng.integers(-30,31,size=(128,96));old=past[[0,32,64]]
    bases,seconds=fit_bases(past,old,907)
    assert set(bases)=={'centroid','pca'} and all(v>=0 for v in seconds.values())
    for b in bases.values():
        assert b.shape==(79,96)
        np.testing.assert_allclose(b@b.T/16384**2,np.eye(79),atol=2e-4)
    b=bases['centroid'][:2].astype(float)/16384
    differences=(old[1:]-old[0]).astype(float)
    relative=np.linalg.norm(differences-differences@b.T@b)/np.linalg.norm(differences)
    assert relative<.002


@pytest.mark.skipif(sys.platform=='win32',reason='Linux proof controller uses flock')
def test_archive_retains_inputs_proofs_and_vk_but_retires_only_owned_keys(tmp_path):
    from campaigns.server_final_method_20261102 import retire_and_archive
    root=tmp_path/'study';case=root/'cases/toy';case.mkdir(parents=True)
    outside=tmp_path/'outside';outside.mkdir();(outside/'keep.pk').write_bytes(b'outside')
    (case/'outside_alias').symlink_to(outside,target_is_directory=True)
    (case/'result.json').write_text('{"complete":true}')
    (case/'input.json').write_text('{"rows":[1,2]}')
    (case/'key.pk').write_bytes(b'private-key')
    (case/'key.vk').write_bytes(b'verification-key')
    (case/'answer.proof').write_bytes(b'proof')
    retire_and_archive(root,case)
    assert not(case/'key.pk').exists() and (outside/'keep.pk').exists()
    with tarfile.open(root/'archives/toy.tar.gz') as a:
        names=a.getnames()
        assert 'toy/key.vk' in names and 'toy/answer.proof' in names and 'toy/input.json' in names
        assert not any(n.endswith('.pk') or 'outside_alias' in n for n in names)
    with pytest.raises(AssertionError):retire_and_archive(root,outside)


@pytest.mark.skipif(sys.platform=='win32',reason='Linux proof controller uses flock')
def test_fit_freezes_selection_before_test_evaluation_and_retains_validation_inputs(tmp_path,monkeypatch):
    from campaigns import server_final_method_20261102 as final
    rng=np.random.default_rng(7)
    training=rng.integers(-20,21,size=(1024,96),dtype=np.int64)
    audit=rng.integers(-20,21,size=(1024,96),dtype=np.int64)
    monkeypatch.setattr(final,'source',lambda *_:(training,audit,3,1.,{'test_source':True}))
    calls=[]
    def evaluated(basis,old,rows,candidates,scale,family,rank):
        pin=tmp_path/'cases/fashion_seed907_n128_q8/selection.json'
        assert pin.exists()==(len(calls)>=4)
        calls.append((family,rank))
        return [dict(family=family,rank=rank,request=i,conclusive=True,corrections=0) for i in range(8)]
    monkeypatch.setattr(final,'evaluate',evaluated)
    (tmp_path/'cost_profiles.json').write_text('{}')
    case,choices,_,_=final.fit_case(tmp_path,'fashion',907,{'fashion':profiles()})
    assert len(calls)==8
    assert all(c['rank']==16 for c in choices.values())
    with np.load(case/'arrays.npz') as a:
        np.testing.assert_array_equal(a['validation_rows'],audit[:128])
        np.testing.assert_array_equal(a['rows'],audit[128:256])
        np.testing.assert_array_equal(a['validation_incoming'],training[512:768])
        np.testing.assert_array_equal(a['test_incoming'],training[768:1024])
