"""Keep this authorized Windows run awake, request STOP at its UTC deadline."""
from pathlib import Path
import argparse
import ctypes
import datetime
import json
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deadline',required=True)
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    deadline=datetime.datetime.fromisoformat(a.deadline).timestamp()
    root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    awake=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    if not awake:raise RuntimeError('Windows rejected scoped execution-state request')
    state=dict(started_unix=time.time(),deadline_unix=deadline,system_awake=True,
               no_persistent_power_setting_changes=True)
    (root/'awake_guard.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    try:
        while time.time()<deadline and not (root/'FINISHED').exists():time.sleep(2)
        if not (root/'FINISHED').exists():
            plan=json.loads((root/'plan.json').read_text())
            for directory in plan['stop_directories']:
                target=Path(directory)
                if not target.is_absolute():target=Path(__file__).resolve().parents[2]/target
                target.mkdir(parents=True,exist_ok=True)
                (target/'STOP').write_text('Four-hour user deadline reached.\n',encoding='utf-8')
            state['stop_reason']='four-hour deadline'
        else:state['stop_reason']='campaign finished'
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        state.update(system_awake=False,finished_unix=time.time())
        (root/'awake_guard.json').write_text(json.dumps(state,indent=2),encoding='utf-8')


if __name__=='__main__':main()
