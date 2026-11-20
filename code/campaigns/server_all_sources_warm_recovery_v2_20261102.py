"""Recorded recovery with explicit server binaries; frozen cohort unchanged."""
from pathlib import Path
import os
import shutil
import time

REPO = Path(__file__).resolve().parents[2]
os.environ['COMMITJL_GO_BINARY'] = str(REPO/'tmp/external_baselines_20261027/commitjl-bound-linux')
os.environ['COMMITJL_ZKML_BINARY'] = str(REPO/'tmp/external_baselines_20261027/prototype-audit-linux')

from experiments.pca_advantage_20261102 import read, save, sha
from campaigns import server_all_sources_warm_20261101 as warm_module
from campaigns import server_all_sources_20261101 as original


def run():
    root = REPO/'code/validation/outputs/server_all_sources_warm_recovery_v2_20261102'
    old = REPO/'code/validation/outputs/server_all_sources_20261101'
    if (root/'recovery.json').exists():
        raise RuntimeError('Existing recovery receipt retained')
    root.mkdir(parents=True, exist_ok=True)
    state = read(old/'matched_queue.json')
    assert state.get('error') == "name 'shutil' is not defined"
    for name in ['matched_queue.json', 'evidence_manifest.json']:
        if (old/name).exists():
            shutil.copy2(old/name, root/('before_'+name))
    pins = read(old/'plan_pin.json')
    assert sha(original.GO) == pins['go_sha256'] and sha(original.RUST) == pins['rust_sha256']
    save(root/'recovery.json', dict(started_unix=time.time(), previous_state=state,
        go_binary=str(original.GO), rust_binary=str(original.RUST),
        scope='Supply the missing standard-library import and explicit server binary paths. '
              'Resume preserved attempts with original source/input pins and admission budget.'))
    warm_module.shutil = shutil
    try:
        original.run(old)
    except BaseException as error:
        save(root/'failure.json', dict(error=str(error), finished_unix=time.time()))
        raise
    save(root/'finished.json', dict(complete=True, finished_unix=time.time()))


if __name__ == '__main__':
    run()
