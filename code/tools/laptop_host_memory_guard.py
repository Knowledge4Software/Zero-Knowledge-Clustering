"""Read Windows OS pressure during the bounded WSL proving campaign."""
from pathlib import Path
import argparse
import datetime
import json
import sys
import time

import psutil

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'code'))
from campaigns.proof_runtime import _windows_memory_pressure
from commitjl.pilot_io import save


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();plan=json.loads((root/'plan.json').read_text())
    deadline=datetime.datetime.fromisoformat(plan['deadline_utc']).timestamp()
    state=dict(started_unix=time.time(),os_low_memory_stop=True,commit_reserve_bytes=1024*2**20,
               minimum_available_physical_bytes=None,minimum_commit_headroom_bytes=None,
               stopped_for_pressure=False)
    while time.time()<deadline and not (root/'FINISHED').exists():
        low,commit=_windows_memory_pressure();available=psutil.virtual_memory().available
        state['minimum_available_physical_bytes']=min(state['minimum_available_physical_bytes'] or available,available)
        state['minimum_commit_headroom_bytes']=min(state['minimum_commit_headroom_bytes'] or commit,commit)
        state.update(updated_unix=time.time(),available_physical_bytes=available,commit_headroom_bytes=commit,os_low_memory=low)
        if low or commit<state['commit_reserve_bytes']:
            state.update(stopped_for_pressure=True,stop_unix=time.time())
            for directory in plan['stop_directories']:
                target=Path(directory)
                if not target.is_absolute():target=REPO/target
                target.mkdir(parents=True,exist_ok=True)
                (target/'STOP').write_text('Windows OS low-memory or commit-reserve stop.\n',encoding='utf-8')
        save(root/'host_memory_guard.json',state)
        if state['stopped_for_pressure']:return
        time.sleep(5)
    state['finished_unix']=time.time();save(root/'host_memory_guard.json',state)


if __name__=='__main__':main()
