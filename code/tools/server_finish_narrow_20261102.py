"""Drain the superseded coordinator's current proof before terminating it."""
from pathlib import Path
import os
import signal
import time

import psutil

from experiments.pca_advantage_20261102 import save
from tools.server_pca_advantage_lease_20261102 import matches


def main():
    repo=Path(__file__).resolve().parents[2]
    receipt=repo/'final_day_narrow_transition_20261102.json'
    if receipt.exists():
        raise RuntimeError('Transition receipt exists')
    process=psutil.Process(61709)
    pin=dict(pid=process.pid,created=process.create_time(),module='campaigns.server_calibrated_routing_20261102')
    if not matches(process,pin):
        raise RuntimeError('Unexpected coordinator identity')
    watcher=61712
    state=dict(started_unix=time.time(),pid=os.getpid(),owner=pin,status='draining_current_proof',
               reason='User confirmed22hours; superseded by final-method evaluation. All partial evidence retained.')
    save(receipt,state)
    process.send_signal(signal.SIGSTOP)
    try:
        while time.time()-state['started_unix']<3600:
            live=[p for p in process.children(recursive=True)
                  if p.pid!=watcher and p.status()!=psutil.STATUS_ZOMBIE]
            if not live:
                process.send_signal(signal.SIGTERM)
                process.send_signal(signal.SIGCONT)
                process.wait(timeout=40)
                state.update(status='superseded_after_phase_drain',finished_unix=time.time())
                save(receipt,state)
                return
            time.sleep(2)
        raise RuntimeError('Drain timeout; original coordinator resumed')
    except BaseException as error:
        if matches(process,pin):
            process.send_signal(signal.SIGCONT)
        state.update(status='resumed_after_transition_error',error=str(error),finished_unix=time.time())
        save(receipt,state)
        raise


if __name__=='__main__':
    main()
