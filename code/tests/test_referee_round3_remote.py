import pytest
from campaigns import referee_round3_remote as runner
from campaigns.local_fast_tradeoff import save,sha


def test_staging_pins_inputs_and_charges_same_host_frozen_model_contract(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'REPO',tmp_path);monkeypatch.setattr(runner,'SOURCE',tmp_path/'source')
    for method in ['ordered_head','pca']:
        parent=runner.SOURCE/method
        for name in ['head_r63/input.json','full/input.json']:save(parent/name,dict(model=dict(n=128),x=[[1,2]]))
        (parent/'head_r63/input.json').write_bytes(b'{"model":{"n":128},"x":[[1,2]]}\r\n')
        save(parent/'auxiliary_basis.json',dict(source_input_sha256=sha(parent/'head_r63/input.json'),fitting_seconds=5.5))
        save(parent/'protocol.json',dict(fitting_seconds=99.,proof_runtime_workers=8))
    target=tmp_path/'new';plan=runner.prepare(target)
    assert plan==runner.prepare(target)
    assert plan['repetitions']==3 and plan['workers']==4
    for method in ['ordered_head','pca']:
        parent=target/'staged'/method
        assert runner.read(parent/'protocol.json')['fitting_seconds']==0.
        assert runner.read(parent/'auxiliary_basis.json')['fitting_seconds']==0.
        assert runner.read(parent/'auxiliary_basis.json')['source_input_sha256']==sha(parent/'head_r63/input.json')
        assert (parent/'head_r63/input.json').read_bytes()==(runner.SOURCE/method/'head_r63/input.json').read_bytes()
        copy=tmp_path/'repetition'/method;runner.copy_staged_parent(parent,copy)
        assert (copy/'head_r63/input.json').read_bytes()==(parent/'head_r63/input.json').read_bytes()
    assert all(sha(target/name)==digest for name,digest in plan['staged_sha256'].items())
    assert not (target/'queue.json').exists()


def test_remote_queue_rejects_laptop_before_reading_inputs(tmp_path,monkeypatch):
    policy=tmp_path/'policy.json';save(policy,dict(remote_heavy_runs_allowed=False));monkeypatch.setattr(runner,'POLICY',policy)
    with pytest.raises(RuntimeError,match='laptop'):runner.execute(tmp_path,tmp_path/'binary',tmp_path/'plonk')
