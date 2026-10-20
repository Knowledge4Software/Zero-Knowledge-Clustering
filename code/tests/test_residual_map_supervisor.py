import json
from types import SimpleNamespace

import pytest

from campaigns import run_residual_maps as campaign


@pytest.mark.parametrize('reason, expected_attempts', [
    ('MemoryError: Free RAM below 768 MiB', 4),
    ('ValueError: corrupted saved map', 1),
])
def test_resource_retries_are_bounded_and_code_errors_do_not_loop(tmp_path, monkeypatch, reason, expected_attempts):
    (tmp_path/'job.json').write_text(json.dumps(dict(threads=4, arguments=[])))
    launched=[]
    def popen(*args, **kwargs):
        launched.append(args)
        (tmp_path/'manifest.json').write_text(json.dumps(dict(complete=False,stop_reason=reason)))
        return SimpleNamespace(pid=12345,wait=lambda:1)
    monkeypatch.setattr(campaign.subprocess,'Popen',popen)
    monkeypatch.setattr(campaign.time,'sleep',lambda _:None)
    monkeypatch.setattr(campaign.psutil,'virtual_memory',lambda:SimpleNamespace(available=4*2**30))
    monkeypatch.setattr(campaign,'ctypes',SimpleNamespace(windll=SimpleNamespace(kernel32=SimpleNamespace(SetThreadExecutionState=lambda _:1))))
    campaign.supervise(tmp_path)
    state=json.loads((tmp_path/'background_status.json').read_text())
    assert state['status']=='failed'
    assert len(launched)==len(state['attempts'])==expected_attempts


def test_explicit_stop_marker_prevents_a_restart(tmp_path, monkeypatch):
    (tmp_path/'job.json').write_text(json.dumps(dict(threads=4, arguments=[])))
    (tmp_path/'STOP_REQUESTED').touch()
    def unexpected(*args, **kwargs):raise AssertionError('Must not launch after a user stop')
    monkeypatch.setattr(campaign.subprocess,'Popen',unexpected)
    monkeypatch.setattr(campaign,'ctypes',SimpleNamespace(windll=SimpleNamespace(kernel32=SimpleNamespace(SetThreadExecutionState=lambda _:1))))
    campaign.supervise(tmp_path)
    state=json.loads((tmp_path/'background_status.json').read_text())
    assert state['status']=='stopped' and state['stage']=='user_stop'
