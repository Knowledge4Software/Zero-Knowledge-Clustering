"""Wait for the current Full run, then test and time cached public parameters."""
from pathlib import Path
import argparse,json,os,shutil,subprocess,sys,time
import psutil
import pandas as pd
from campaigns.local_fast_tradeoff import save,sha,run
from analysis.audit_local_fast_tradeoff import run as audit


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--binary',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    parent=a.parent.resolve();root=a.root.resolve();binary=a.binary.resolve();root.mkdir(parents=True,exist_ok=a.resume)
    initial=json.loads((parent/'status.json').read_text());launch=json.loads((parent/'launch.json').read_text())
    worker=psutil.Process(launch['pid']) if not initial.get('complete') else None
    worker_created=worker.create_time() if worker else None
    state=dict(complete=False,phase='waiting_for_full',parent=str(parent),parent_worker_pid=worker.pid if worker else launch['pid'],binary_sha256=sha(binary),queued_unix=time.time())
    if a.resume:
        old=json.loads((root/'queue_status.json').read_text())
        assert old['parent']==str(parent) and old['binary_sha256']==state['binary_sha256']
        state['previous_queue_status']=old
    save(root/'queue_status.json',state)
    for case in ['head_r39','head_r63','full']:
        (root/case).mkdir(exist_ok=a.resume)
        if a.resume:assert sha(root/case/'input.json')==sha(parent/case/'input.json')
        else:shutil.copy2(parent/case/'input.json',root/case/'input.json')
    shutil.copy2(parent/'quality.csv',root/'quality.csv')
    protocol=json.loads((parent/'protocol.json').read_text());protocol.update(ranks=[63],
        optimization='Identical circuits and public claims; derive approved public-center parameters once per center set rather than once per proof batch.',
        parent_protocol_sha256=sha(parent/'protocol.json'),comparison='Fresh cold rank63 workflow against the completed equally reusable Full baseline on identical data/centers. Prior proofs are replayed only as a separate compatibility diagnostic.')
    if not a.resume:save(root/'protocol.json',protocol)
    try:
        while True:
            parent_state=json.loads((parent/'status.json').read_text())
            assert parent_state['binary_sha256']==initial['binary_sha256']
            if parent_state.get('complete'):break
            if not worker.is_running() or worker.create_time()!=worker_created:raise RuntimeError('Parent stopped before completing Full; inspect its logs before continuing.')
            state['last_checked_unix']=time.time();save(root/'queue_status.json',state);time.sleep(15)
        env=dict(os.environ,GOMAXPROCS='2',GOMEMLIMIT='1200MiB')
        state.update(phase='smoke_and_legacy_replay',started_unix=time.time());save(root/'queue_status.json',state)
        with (root/'checks.log').open('w',encoding='utf-8') as log:
            smoke=root/'smoke/checks.json'
            smoke_done=json.loads(smoke.read_text()) if smoke.exists() else {}
            if not smoke_done.get('complete'):
                subprocess.run([sys.executable,'-m','validation.check_relative_reuse','--root',str(root/'smoke'),'--binary',str(binary)],check=True,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
            else:assert smoke_done['binary_sha256']==sha(binary)
            legacy_check=root/'legacy_replay_check.json'
            legacy_done=json.loads(legacy_check.read_text()) if legacy_check.exists() else {}
            if not legacy_done.get('complete'):
                legacy=root/'legacy_public_replay';shutil.copytree(parent/'head_r63/public_bundle',legacy)
                started=time.monotonic()
                subprocess.run([str(binary),'-phase','verify','-out',str(legacy),'-trust',str(parent/'head_r63/owner_pin.json'),'-public-task',str(parent/'head_r63/approved_public_task.json')],check=True,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
                replay=json.loads((legacy/'verify.json').read_text());assert replay['complete'] and replay['verified']==288
                save(legacy_check,dict(complete=True,verified=replay['verified'],wall_seconds=time.monotonic()-started,
                    meaning='New public builder verifies original untouched circuit proofs against original owner/task pins. Diagnostic excluded from the fresh deployment timing.'))
        state['phase']='cold_head_r63';save(root/'queue_status.json',state)
        run(root,binary,['head_r63'])
        state['phase']='independent_output_and_timing_audit';save(root/'queue_status.json',state)
        audit(parent,root,root/'checked_comparison')
        state.update(complete=True,phase='complete',finished_unix=time.time());save(root/'queue_status.json',state)
    except BaseException as error:
        state.update(phase='failed',error=str(error));save(root/'queue_status.json',state);raise


if __name__=='__main__':main()
