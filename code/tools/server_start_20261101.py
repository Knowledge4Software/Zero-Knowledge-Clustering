"""Durable, bounded handoff from the active resource canary to the main queue."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time
import fcntl


def main():
    repo = Path(__file__).resolve().parents[2]
    lock = (repo/'startup.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = repo/'startup_state.json'
    def status(**kwargs):
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(updated_unix=time.time(),pid=os.getpid(),**kwargs),indent=2))
        temporary.replace(state_path)
    started = time.time()
    canary_path = repo/'code/validation/outputs/server_canary_20261101/state.json'
    while time.time()-started < 2*3600:
        canary = json.loads(canary_path.read_text()) if canary_path.exists() else {}
        if canary.get('status') == 'stopped_with_preserved_evidence':
            status(status='blocked_by_canary_failure', error=canary.get('error'))
            return
        ready = (repo/'ENVIRONMENT_READY').exists()
        status(status='waiting_for_canary_and_environment', environment_ready=ready,
               canary_phase=canary.get('active_phase'),canary_complete=canary.get('complete',False))
        if ready and canary.get('complete'):
            break
        time.sleep(10)
    else:
        status(status='prerequisites_timeout', error='No indefinite wait during the paid allocation')
        return
    manifest = json.loads((repo/'deployment_manifest.json').read_text())
    for name,digest in manifest.items():
        with (repo/name).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=digest:
                status(status='source_pin_failure',file=name)
                return
    env = dict(os.environ, PYTHONPATH=str(repo/'code'),
               COMMITJL_GO_BINARY=str(repo/'tmp/external_baselines_20261027/commitjl-bound-linux'),
               COMMITJL_ZKML_BINARY=str(repo/'tmp/external_baselines_20261027/prototype-audit-linux'),
               OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    python = repo/'.venv/bin/python'
    status(status='validating_runtime')
    with (repo/'server_tests.log').open('ab') as log:
        result = subprocess.run([str(python),'-m','pytest','code/tests/test_proof_runtime.py',
            'code/tests/test_server_runtime_20261101.py','code/tests/test_cost_aware.py',
            'code/tests/test_shared_full_reference.py','-q'],cwd=repo,env=env,
            stdout=log,stderr=subprocess.STDOUT,timeout=180)
    if result.returncode:
        status(status='tests_failed',returncode=result.returncode)
        return
    env['COMMITJL_SERVER_PROFILE']=str(repo/'server_profile.json')
    status(status='campaign_running', test_log='server_tests.log')
    with (repo/'campaign.log').open('ab') as log:
        result = subprocess.run([str(python),'-u','-m','campaigns.server_campaign_20261101',
            '--root',str(repo/'code/validation/outputs/server_campaign_20261101')],
            cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=32*3600)
    status(status='campaign_complete' if result.returncode==0 else 'campaign_stopped',
           returncode=result.returncode)


if __name__=='__main__':
    main()
