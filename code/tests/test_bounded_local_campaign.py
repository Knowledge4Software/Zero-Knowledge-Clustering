import pytest
from campaigns.bounded_local_campaign import within_scope,validate_adopted_process
from campaigns import referee_round3_remote as runner
from campaigns.local_fast_tradeoff import save,sha


@pytest.mark.parametrize('n,allowed',[(128,True),(1024,True),(4096,False),(16384,False)])
def test_laptop_scope_keeps_small_cases_and_defers_large_proofs(n,allowed):
    assert within_scope(dict(n=n),dict(maximum_proof_catalog_records=1024)) is allowed


def test_scope_cannot_reenable_larger_work():
    with pytest.raises(ValueError):within_scope(dict(n=4096),dict(maximum_proof_catalog_records=4096))


def test_adoption_checks_command_and_pid_creation_time():
    class Process:
        pid=71
        def create_time(self):return 100.
        def cmdline(self):return ['python','active_worker','same-root']
    args=Process().cmdline();handover=dict(child_pid=71,child_created_unix=100.,child_command=args)
    validate_adopted_process(Process(),args,handover)
    with pytest.raises(RuntimeError):validate_adopted_process(Process(),args,dict(handover,child_created_unix=80.))
    with pytest.raises(RuntimeError):validate_adopted_process(Process(),['python','unrelated'],handover)


def test_resume_preserves_completed_phases_and_rejects_a_live_coordinator(tmp_path,monkeypatch):
    binary=tmp_path/'proof.exe';backend=tmp_path/'backend.exe';binary.write_bytes(b'proof');backend.write_bytes(b'backend')
    save(tmp_path/'plan.json',dict(workers=8))
    state=dict(complete=False,pid=91,active_phase='r1_moments',child_pid=71,completed=['r1_full'],
        attempts=[dict(phase='r1_full',wall_seconds=1000)],plan_sha256=sha(tmp_path/'plan.json'),
        binary_sha256=sha(binary),plonk_binary_sha256=sha(backend))
    save(tmp_path/'queue.json',state)
    handover=dict(cohort_root=str(tmp_path),old_coordinator_pid=91,active_phase='r1_moments',child_pid=71)
    monkeypatch.setattr(runner.psutil,'pid_exists',lambda pid:True)
    with pytest.raises(RuntimeError,match='still exists'):runner.resume_state(tmp_path,binary,backend,handover)
    assert runner.read(tmp_path/'queue.json')==state
    monkeypatch.setattr(runner.psutil,'pid_exists',lambda pid:False)
    result=runner.resume_state(tmp_path,binary,backend,handover)
    assert result['completed']==['r1_full'] and result['attempts']==state['attempts']
    assert result['child_pid']==71 and result['coordinator_handovers']==[handover]


def test_handover_requires_current_plan_pin_and_phase(tmp_path,monkeypatch):
    binary=tmp_path/'proof.exe';backend=tmp_path/'backend.exe';binary.write_bytes(b'proof');backend.write_bytes(b'backend')
    save(tmp_path/'plan.json',{})
    state=dict(complete=False,pid=91,active_phase='r1_moments',child_pid=71,
        plan_sha256=sha(tmp_path/'plan.json'),binary_sha256=sha(binary),plonk_binary_sha256=sha(backend))
    save(tmp_path/'queue.json',state);monkeypatch.setattr(runner.psutil,'pid_exists',lambda pid:False)
    handover=dict(cohort_root=str(tmp_path),old_coordinator_pid=91,active_phase='online_affine',child_pid=71)
    with pytest.raises(RuntimeError,match='phase changed'):runner.resume_state(tmp_path,binary,backend,handover)
    binary.write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='pin differs'):runner.resume_state(tmp_path,binary,backend,handover)


def test_existing_staged_input_cannot_be_overwritten(tmp_path):
    source=tmp_path/'source';target=tmp_path/'target'
    save(source/'head_r63/input.json',dict(catalog='original'))
    save(target/'head_r63/input.json',dict(catalog='different'))
    with pytest.raises(RuntimeError,match='differs'):runner.copy_staged_parent(source,target)
    assert runner.read(target/'head_r63/input.json')==dict(catalog='different')


def test_adopted_phase_requires_checked_completion_not_just_worker_exit(tmp_path):
    with pytest.raises(RuntimeError,match='no completed'):runner.adopted_phase_receipt(tmp_path,'online_affine')
    save(tmp_path/'repetition_01/online_affine/checked_comparison.json',dict(complete=True))
    assert runner.adopted_phase_receipt(tmp_path,'online_affine').exists()
    save(tmp_path/'repetition_01/moments_only/checked_comparison.json',dict(complete=True,no_wrong_decisions=False))
    with pytest.raises(RuntimeError,match='checks failed'):runner.adopted_phase_receipt(tmp_path,'r1_moments')


def test_large_cases_are_deferred_before_any_input_preparation_or_proof(tmp_path,monkeypatch):
    from campaigns import referee_extended_local as extended
    monkeypatch.setattr(extended,'POLICY',tmp_path/'policy.json');save(extended.POLICY,{})
    monkeypatch.setattr(extended,'require_execution',lambda *a,**k:None)
    save(tmp_path/'plan.json',dict(extended_proof_cases=[dict(id='large4096',n=4096),dict(id='large16384',n=16384)]))
    extended.execute_extended(tmp_path,scope=dict(maximum_proof_catalog_records=1024,reason='User laptop budget'))
    result=runner.read(tmp_path/'extended_status.json')
    assert result['authorized_scheduling_complete'] and not result['complete']
    assert [r['status'] for r in result['cases']]==['deferred_by_user_scope']*2
    assert not (tmp_path/'extended_inputs').exists() and not (tmp_path/'extended').exists()
