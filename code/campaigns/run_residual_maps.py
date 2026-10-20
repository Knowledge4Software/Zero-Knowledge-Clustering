"""Detached residual-map campaign with checkpoints and bounded RAM recovery."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
from commitjl.durable_io import atomic_json

CODE=Path(__file__).resolve().parents[1]


def supervise(root):
    job=json.loads((root/'job.json').read_text())
    env=dict(os.environ)
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        env[name]=str(job['threads'])
    awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)) if os.name=='nt' else False
    start=time.time();state=dict(status='running',stage='experiment',supervisor_pid=os.getpid(),
        started_unix=start,idle_sleep_inhibited=awake,output=str(root),attempts=[])
    state_path=root/'background_status.json';atomic_json(state_path,state)
    try:
        for attempt in range(4):
            if (root/'STOP_REQUESTED').exists():
                state.update(status='stopped',stage='user_stop');return
            command=[sys.executable,'-u','-m','experiments.residual_map_comparison','--out',str(root)]+job['arguments']
            with (root/'run.log').open('a',encoding='utf-8') as log:
                process=subprocess.Popen(command,cwd=CODE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                state.update(worker_pid=process.pid,stage='experiment',attempt=attempt+1);atomic_json(state_path,state)
                returncode=process.wait()
            state['attempts'].append(dict(returncode=returncode,finished_unix=time.time()))
            manifest=json.loads((root/'manifest.json').read_text()) if (root/'manifest.json').exists() else {}
            if returncode==0 and manifest.get('complete'):break
            if (root/'STOP_REQUESTED').exists():
                state.update(status='stopped',stage='user_stop');return
            reason=manifest.get('stop_reason','Worker failed before publishing a manifest')
            # Retry only the explicitly identified resource guard, never code
            # errors, source mismatches, or a case that fails repeatedly.
            if not reason.startswith('MemoryError:') or attempt>=3:raise RuntimeError(reason)
            state.update(stage='waiting_for_memory',reason=reason);atomic_json(state_path,state)
            deadline=time.time()+3600;stable=0
            while time.time()<deadline:
                if (root/'STOP_REQUESTED').exists():
                    state.update(status='stopped',stage='user_stop');return
                stable=stable+1 if psutil.virtual_memory().available>=3*2**30 else 0
                if stable>=3:break
                time.sleep(10)
            else:raise RuntimeError('RAM did not recover to 3 GiB within one hour; saved cases preserved')
        else:raise RuntimeError('Bounded resource retries exhausted')
        state.update(stage='audit_and_report');atomic_json(state_path,state)
        with (root/'report.log').open('a',encoding='utf-8') as log:
            process=subprocess.Popen([sys.executable,'-u','-m','analysis.residual_map_report',str(root)],
                cwd=CODE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            state.update(worker_pid=process.pid);atomic_json(state_path,state)
            if process.wait()!=0:raise RuntimeError('Final audit/report failed; inspect report.log')
        state.update(status='complete',stage='finished',cases=manifest['completed_cases'])
    except BaseException as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    finally:
        state['elapsed_seconds']=time.time()-start;atomic_json(state_path,state)
        if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args();root=args.out.resolve();root.mkdir(parents=True,exist_ok=True)
    if args.worker:supervise(root);return
    # Never create a second experiment while any recognized campaign is live.
    for process in psutil.process_iter(['pid','name','cmdline']):
        if process.pid==os.getpid():continue
        try:
            command=process.info['cmdline'] or []
            if 'python' in (process.info['name'] or '').lower() and any(name in command for name in
                ['experiments.residual_map_comparison','campaigns.comprehensive_durable_worker','campaigns.run_residual_maps']):
                raise RuntimeError(f'Campaign process {process.pid} is already active; it was not interrupted')
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    job=dict(threads=4,arguments=['--m','64','128','256','--k','5','10','--seeds','3','--starts','3',
        '--seed-offset','70','--start-offset','70','--fit-repetitions','3','--timing-repetitions','3',
        '--queries','32','--dbscan-points','256','--threads','4','--max-rss-gib','2.5','--max-seconds','43200'])
    if (root/'job.json').exists() and json.loads((root/'job.json').read_text())!=job:
        raise ValueError('Existing job differs; choose a new output directory')
    atomic_json(root/'job.json',job)
    atomic_json(root/'supervisor_source.json',dict(sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ram_retry_limit=3,ram_recovery_gib=3,ram_wait_seconds=3600))
    flags=(subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_NO_WINDOW) if os.name=='nt' else 0
    with (root/'supervisor.log').open('a',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,'-u','-m','campaigns.run_residual_maps','--worker','--out',str(root)],
            cwd=CODE,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            creationflags=flags,start_new_session=os.name!='nt',close_fds=True)
    record=dict(pid=process.pid,create_time=psutil.Process(process.pid).create_time(),out=str(root))
    atomic_json(root/'launcher.json',record);print(json.dumps(dict(started=True,**record)))


if __name__=='__main__':main()
