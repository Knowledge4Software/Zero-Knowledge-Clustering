import pytest
from campaigns.proof_runtime import proof_environment,proof_workers,launch_headroom
import campaigns.proof_runtime as runtime


def test_large_catalog_wall_budget_requires_explicit_profile_and_keeps_old_caps():
    assert runtime.phase_limit({})==4*3600
    with pytest.raises(ValueError):runtime.phase_limit(dict(n=128,phase_wall_limit_seconds=72*3600))
    with pytest.raises(ValueError):runtime.phase_limit(dict(n=16384,phase_wall_limit_seconds=72*3600))
    assert runtime.phase_limit(dict(n=16384,long_catalog_phase='local_20261026',phase_wall_limit_seconds=72*3600))==72*3600
    with pytest.raises(ValueError):runtime.phase_limit(dict(n=16384,long_catalog_phase='local_20261026',phase_wall_limit_seconds=96*3600))


def test_old_protocol_keeps_its_frozen_runtime_despite_host_overrides(monkeypatch):
    monkeypatch.setenv('GOMAXPROCS','16')
    monkeypatch.setenv('GOMEMLIMIT','8GiB')
    env=proof_environment({})
    assert env['GOMAXPROCS']=='2' and env['GOMEMLIMIT']=='1200MiB'


@pytest.mark.parametrize('workers',[True,'8',0,-1,65])
def test_invalid_parallelism_fails_instead_of_silently_changing_a_measurement(workers):
    with pytest.raises(ValueError):proof_workers(dict(proof_runtime_workers=workers))


def test_launch_attempt_never_relaxes_inflight_guards_or_repeats_after_memory_stop(monkeypatch):
    monkeypatch.delenv('COMMITJL_MEMORY_PRESSURE_POLICY',raising=False)
    monkeypatch.setenv('COMMITJL_LAUNCH_AVAILABLE_MIB','1152')
    previous=dict(peak_rss_bytes=2*2**30, attempts=[])
    assert launch_headroom(previous)==1152*2**20
    previous['attempts']=[dict(stop='RAM reserve')]
    assert launch_headroom(previous)==2*2**30+(650+256)*2**20
    assert proof_environment({})['GOMEMLIMIT']=='1200MiB'
    monkeypatch.setenv('COMMITJL_LAUNCH_AVAILABLE_MIB','650')
    with pytest.raises(ValueError):launch_headroom(previous)


def test_os_memory_policy_checks_pressure_commit_headroom_and_unchanged_process_cap(monkeypatch):
    monkeypatch.setenv('COMMITJL_MEMORY_PRESSURE_POLICY','windows_os')
    monkeypatch.setattr(runtime,'_windows_memory_pressure',lambda:(False,2*2**30))
    assert runtime.memory_stop(2*2**30) is None
    assert runtime.memory_stop(4*2**30)=='process RSS guard'
    monkeypatch.setattr(runtime,'_windows_memory_pressure',lambda:(True,2*2**30))
    assert runtime.memory_stop(0)=='RAM reserve'


def test_os_memory_retry_requires_reclaimed_headroom(monkeypatch):
    monkeypatch.setenv('COMMITJL_MEMORY_PRESSURE_POLICY','windows_os')
    if runtime.os.name != 'nt':pytest.skip('Windows scheduling policy')
    assert launch_headroom({}) == 768*2**20
    previous=dict(peak_rss_bytes=2*2**30,attempts=[dict(stop='RAM reserve')])
    assert launch_headroom(previous)==2*2**30+768*2**20
    monkeypatch.setattr(runtime,'_windows_memory_pressure',lambda:(False,512*2**20))
    assert runtime.memory_stop(0)=='RAM reserve'


def test_memory_query_failure_stops_the_worker(monkeypatch):
    monkeypatch.setenv('COMMITJL_MEMORY_PRESSURE_POLICY','windows_os')
    def unavailable():raise OSError('unavailable')
    monkeypatch.setattr(runtime,'_windows_memory_pressure',unavailable)
    assert runtime.memory_stop(0)=='RAM reserve'
