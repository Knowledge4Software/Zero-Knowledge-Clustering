import hashlib
import json

import numpy as np
import pytest

from commitjl import durable_io
from tools import recover_comprehensive_run as recovery


def test_manifest_backup_and_failed_serialization_preserve_last_commit(tmp_path):
    path=tmp_path/'manifest.json'
    durable_io.atomic_json(path,dict(count=1))
    durable_io.atomic_json(path,dict(count=2))
    assert json.loads(path.read_text())==dict(count=2)
    assert json.loads((tmp_path/'manifest.previous.json').read_text())==dict(count=1)
    with pytest.raises(ValueError):durable_io.atomic_json(path,dict(count=float('nan')))
    assert json.loads(path.read_text())==dict(count=2)
    assert json.loads((tmp_path/'manifest.previous.json').read_text())==dict(count=1)


def test_contents_are_flushed_before_replacement(tmp_path,monkeypatch):
    events=[];real_sync=durable_io.os.fsync;real_replace=durable_io.os.replace
    def sync(fd):
        events.append('flush');return real_sync(fd)
    def replace(source,destination):
        events.append('replace');return real_replace(source,destination)
    monkeypatch.setattr(durable_io.os,'fsync',sync);monkeypatch.setattr(durable_io.os,'replace',replace)
    durable_io.atomic_json(tmp_path/'row.json',dict(a=1))
    assert events==['flush','replace']
    events.clear();durable_io.save_arrays(tmp_path/'arrays.npz',x=np.arange(12).reshape(3,4))
    assert events==['flush','replace']
    with np.load(tmp_path/'arrays.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['x'],np.arange(12).reshape(3,4))


def test_recovery_retains_valid_case_and_quarantines_only_broken_records(tmp_path,monkeypatch):
    root=tmp_path/'run';(root/'rows').mkdir(parents=True);(root/'cases').mkdir()
    (root/'cases'/'good.bin').write_bytes(b'correct')
    (root/'cases'/'bad.bin').write_bytes(b'corrupted')
    good=dict(artifacts={'cases/good.bin':hashlib.sha256(b'correct').hexdigest()})
    bad=dict(artifacts={'cases/bad.bin':hashlib.sha256(b'expected').hexdigest()})
    (root/'rows'/'good.json').write_text(json.dumps(good))
    (root/'rows'/'bad.json').write_text(json.dumps(bad))
    (root/'rows'/'interrupted.json').write_bytes(b'\0'*12)
    (root/'manifest.json').write_bytes(b'\0'*24)
    (root/'plan.json').write_text(json.dumps(dict(expected_cases=3)))
    (root/'run.log').write_text(json.dumps(dict(elapsed=12.5))+'\n')
    identity=dict(configuration={},sources={},archive_index_sha256='fixture')
    monkeypatch.setattr(recovery,'recovered_identity',lambda *args:identity)
    monkeypatch.setattr(recovery,'active_workers',lambda *args:[])
    result=recovery.recover(root,root,repair=True)
    assert result['verified_cases']==1 and result['repaired']
    assert json.loads((root/'rows'/'good.json').read_text())==good
    assert (root/'cases'/'good.bin').read_bytes()==b'correct'
    assert not (root/'rows'/'bad.json').exists()
    assert not (root/'rows'/'interrupted.json').exists()
    recovered=json.loads((root/'manifest.json').read_text())
    assert recovered['completed_cases']==1 and not recovered['complete']
    folder=next(root.glob('recovery_*[0-9]'))
    assert (folder/'damaged_manifest.json').read_bytes()==b'\0'*24
    assert (folder/'quarantine'/'cases'/'bad.bin').read_bytes()==b'corrupted'


def test_recovery_refuses_to_touch_an_active_campaign(tmp_path,monkeypatch):
    monkeypatch.setattr(recovery,'active_workers',lambda *args:[123])
    with pytest.raises(RuntimeError,match='active'):recovery.recover(tmp_path,tmp_path,repair=True)


def test_relearning_centers_invalidates_even_undamaged_dependent_basis_cache(tmp_path):
    directory=tmp_path/'datasets'/'fixture'/'k5'/'models';directory.mkdir(parents=True)
    for name in ['basis_s32.json','enriched_s32.json']:(directory/name).write_text('{}')
    center=str((directory/'centers_s32.json').relative_to(tmp_path))
    expanded=recovery.invalidate_dependent_fits(tmp_path,{center:'damaged'})
    assert str((directory/'basis_s32.json').relative_to(tmp_path)) in expanded
    assert str((directory/'enriched_s32.json').relative_to(tmp_path)) in expanded
