import json
from types import SimpleNamespace
import pytest
from campaigns import proof_runtime as runtime


def configure(monkeypatch, tmp_path, **overrides):
    profile = dict(hostname='new-server', workers=12, rss_cap_gib=108,
                   reserve_ram_gib=12, reserve_disk_gib=16, go_soft_memory_gib=80)
    profile.update(overrides)
    path = tmp_path/'profile.json'
    path.write_text(json.dumps(profile))
    monkeypatch.setenv('COMMITJL_SERVER_PROFILE', str(path))
    monkeypatch.setattr(runtime.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(runtime.platform, 'node', lambda: 'new-server')
    monkeypatch.setattr(runtime.os, 'sched_getaffinity', lambda _: set(range(12)), raising=False)
    runtime._read_server_profile.cache_clear()
    return profile


def test_server_profile_uses_all_allocated_workers_and_keeps_blas_bounded(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path)
    env = runtime.proof_environment(dict(proof_runtime_workers=12))
    assert env['GOMAXPROCS'] == env['RAYON_NUM_THREADS'] == '12'
    assert env['GOMEMLIMIT'] == '80GiB'
    assert env['OPENBLAS_NUM_THREADS'] == '1'
    with pytest.raises(ValueError):
        runtime.proof_environment(dict(proof_runtime_workers=8))


def test_server_profile_retains_ram_reserve_and_process_cap(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path)
    monkeypatch.setattr(runtime.psutil, 'virtual_memory', lambda: SimpleNamespace(available=100*2**30))
    assert runtime.memory_stop(8*2**30) is None
    assert runtime.memory_stop(109*2**30) == 'server process RSS guard'
    monkeypatch.setattr(runtime.psutil, 'virtual_memory', lambda: SimpleNamespace(available=11*2**30))
    assert runtime.memory_stop(0) == 'server RAM reserve'
    assert runtime.launch_headroom(dict(peak_rss_bytes=15*2**30)) == 27*2**30


@pytest.mark.parametrize('override', [dict(hostname='different-server'), dict(workers=13),
                                     dict(rss_cap_gib=120), dict(reserve_ram_gib=1)])
def test_server_profile_rejects_wrong_host_or_unsafe_allocation(monkeypatch, tmp_path, override):
    configure(monkeypatch, tmp_path, **override)
    with pytest.raises(ValueError):
        runtime.server_profile()


def test_absent_server_profile_preserves_historical_limits(monkeypatch):
    monkeypatch.delenv('COMMITJL_SERVER_PROFILE', raising=False)
    monkeypatch.delenv('COMMITJL_MEMORY_PRESSURE_POLICY', raising=False)
    assert runtime.proof_environment({})['GOMEMLIMIT'] == '1200MiB'
    assert runtime.memory_stop(4*2**30) == 'process RSS guard'
