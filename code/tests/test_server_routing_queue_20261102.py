"""Recovery boundaries and prospective source separation on the Linux server."""
import json
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(sys.platform == 'win32', reason='Linux queue uses flock')


def test_future_image_rows_exclude_all_prior_training_and_audit_rows(monkeypatch):
    from campaigns import server_calibrated_routing_20261102 as queue
    from commitjl.pilot_io import array_sha
    raw = np.random.default_rng(77).integers(1, 20, size=(3000, 3))
    ids = np.random.default_rng(741+811).permutation(len(raw))
    pin = dict(scale=1., source=dict(source_sha256=array_sha(raw),
               training_ids=ids[:1024].tolist(), audit_ids=ids[1024:2048].tolist()))
    monkeypatch.setattr(queue, 'frozen_image_sample', lambda *_: raw)
    rows, provenance = queue.future_rows('fashion', 811, pin)
    np.testing.assert_array_equal(rows, raw[ids[2048:2304]])
    assert not set(provenance['row_ids']) & set(pin['source']['training_ids']+pin['source']['audit_ids'])
    pin['source']['audit_ids'].append(provenance['row_ids'][0])
    with pytest.raises(AssertionError):
        queue.future_rows('fashion', 811, pin)


def test_recovery_only_supplies_import_after_exact_error_and_owner_exit(tmp_path, monkeypatch):
    from campaigns import server_all_sources_warm_recovery_20261102 as recovery
    from campaigns import server_all_sources_warm_20261101 as warm
    from campaigns import server_all_sources_20261101 as original
    old, root = tmp_path/'old', tmp_path/'recovery'
    old.mkdir(); root.mkdir()
    state = dict(status='stopped_with_preserved_evidence', error="name 'shutil' is not defined", pid=99999999)
    (old/'matched_queue.json').write_text(json.dumps(state))
    (old/'evidence_manifest.json').write_text('{"queue_complete":false}')
    manifest = tmp_path/'manifest.json';manifest.write_text('{}')
    for key, value in [('OLD', old), ('ROOT', root), ('MANIFEST', manifest), ('REPO', tmp_path)]:
        monkeypatch.setattr(recovery, key, value)
    def exited(pid):
        raise recovery.psutil.NoSuchProcess(pid)
    monkeypatch.setattr(recovery.psutil, 'Process', exited)
    monkeypatch.setattr(warm, 'shutil', None, raising=False)
    called = []
    def resume(path):
        assert warm.shutil is recovery.shutil
        called.append(path)
    monkeypatch.setattr(original, 'run', resume)
    monkeypatch.setattr(recovery, 'DEADLINE', recovery.time.time()+3600)
    recovery.run()
    assert called == [old]
    assert json.loads((root/'before_matched_queue.json').read_text()) == state
    assert json.loads((root/'finished.json').read_text())['recovery_needed']


def test_unexpected_older_error_is_preserved_without_retry(tmp_path, monkeypatch):
    from campaigns import server_all_sources_warm_recovery_20261102 as recovery
    old, root = tmp_path/'old', tmp_path/'recovery';old.mkdir();root.mkdir()
    (old/'matched_queue.json').write_text(json.dumps(dict(status='stopped_with_preserved_evidence', error='other failure')))
    monkeypatch.setattr(recovery, 'OLD', old);monkeypatch.setattr(recovery, 'ROOT', root)
    monkeypatch.setattr(recovery, 'DEADLINE', recovery.time.time()+3600)
    recovery.run()
    assert (root/'not_recovered.json').exists()
    assert not (root/'recovery.json').exists()
