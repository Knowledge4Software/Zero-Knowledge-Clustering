"""One narrowly scoped recovery for the frozen warm-export missing import.

Wait for the original owner to exit. Preserve its final receipts, supply the
standard-library import in a new process, then resume the unchanged campaign.
No original source, input, proof, verification relation or timing is edited.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys
import time

import psutil

from experiments.pca_advantage_20261102 import read, save, sha

REPO = Path(__file__).resolve().parents[2]
OLD = REPO/'code/validation/outputs/server_all_sources_20261101'
ROOT = REPO/'code/validation/outputs/server_all_sources_warm_recovery_20261102'
MANIFEST = REPO/'calibrated_routing_deployment_manifest_20261102_v2.json'
DEADLINE = 1793524934.1857634+36*3600


def run():
    if (ROOT/'launch.json').exists():
        raise RuntimeError('Recovery watcher already exists')
    save(ROOT/'launch.json', dict(pid=os.getpid(), started_unix=time.time(),
         deadline_unix=DEADLINE, exact_error="name 'shutil' is not defined"))
    while time.time() < DEADLINE:
        state = read(OLD/'matched_queue.json')
        if state.get('complete'):
            save(ROOT/'finished.json', dict(complete=True, recovery_needed=False, finished_unix=time.time()))
            return
        if state.get('status') != 'stopped_with_preserved_evidence':
            time.sleep(5)
            continue
        if state.get('error') != "name 'shutil' is not defined":
            save(ROOT/'not_recovered.json', dict(reason='Unexpected error', earlier_state=state))
            return
        try:
            p = psutil.Process(state['pid'])
            if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                time.sleep(2)
                continue
        except psutil.NoSuchProcess:
            pass
        manifest = read(MANIFEST)
        for name, expected in manifest.items():
            assert sha(REPO/name) == expected, name
        for name in ['matched_queue.json', 'evidence_manifest.json']:
            if (OLD/name).exists():
                shutil.copy2(OLD/name, ROOT/('before_'+name))
        save(ROOT/'recovery.json', dict(started_unix=time.time(),
             manifest_sha256=sha(MANIFEST), original_state=state,
             change='Supply shutil in the warm helper namespace; original source files untouched.',
             scope='Resume the partial existing warm repetition, retaining its earlier component timings.'))
        from campaigns import server_all_sources_warm_20261101 as warm_module
        from campaigns import server_all_sources_20261101 as original
        warm_module.shutil = shutil
        try:
            original.run(OLD)
        except BaseException as error:
            save(ROOT/'failure.json', dict(error=str(error), finished_unix=time.time()))
            raise
        save(ROOT/'finished.json', dict(complete=True, recovery_needed=True, finished_unix=time.time()))
        return
    save(ROOT/'not_recovered.json', dict(reason='Original study deadline', finished_unix=time.time()))


if __name__ == '__main__':
    run()
