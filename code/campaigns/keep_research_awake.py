"""Prevent idle sleep only while the authorized supervisor is active.

No persistent power setting is changed. Display sleep and explicit user sleep
or shutdown remain under Windows/user control.
"""
from pathlib import Path
import argparse,ctypes,json,time
import psutil


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--supervisor',type=int,required=True);p.add_argument('--status',type=Path,required=True);a=p.parse_args()
    process=psutil.Process(a.supervisor);created=process.create_time()
    assert any('autonomous_referee' in item for item in process.cmdline())
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.SetThreadExecutionState.argtypes=[ctypes.c_uint];kernel.SetThreadExecutionState.restype=ctypes.c_uint
    if not kernel.SetThreadExecutionState(0x80000001):raise OSError(ctypes.get_last_error(),'Idle-sleep request failed')
    try:
        while process.is_running() and process.create_time()==created:
            state=json.loads(a.status.read_text())
            if state.get('complete') or state.get('status') in ['paused','stopped']:break
            time.sleep(15)
    except psutil.NoSuchProcess:pass
    finally:kernel.SetThreadExecutionState(0x80000000)


if __name__=='__main__':main()
