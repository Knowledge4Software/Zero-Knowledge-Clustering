"""Start the complete current-method campaign as an independent background process."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import hashlib
import shutil

import psutil

from commitjl.durable_io import atomic_json

CODE=Path(__file__).resolve().parents[1]


def worker(root):
    state_path=root/'background_status.json'
    awake=False
    if os.name=='nt':
        awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    env=dict(os.environ)
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[name]='4'
    started=time.time()
    state=dict(status='running',stage='experiment',supervisor_pid=os.getpid(),started_unix=started,
        idle_sleep_inhibited=awake,output=str(root))
    atomic_json(state_path,state)
    try:
        command=[sys.executable,'-u','-m','campaigns.comprehensive_durable_worker','--out',str(root),
            '--max-seconds','28800','--max-rss-gib','2.5']
        with (root/'run.log').open('a',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=CODE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            state['worker_pid']=process.pid;atomic_json(state_path,state)
            returncode=process.wait()
        state.update(experiment_returncode=returncode)
        if returncode!=0:raise RuntimeError(f'Experiment exited with code {returncode}; see run.log')
        manifest=json.loads((root/'manifest.json').read_text())
        if not manifest['complete']:raise RuntimeError('Experiment ended with incomplete checkpoints')
        state.update(stage='audit_and_report');atomic_json(state_path,state)
        with (root/'report.log').open('a',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-u','-m','campaigns.comprehensive_durable_worker','--report',str(root)],
                cwd=CODE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Report/audit failed; see report.log')
        state.update(status='complete',stage='finished',cases=manifest['completed_cases'])
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}')
    finally:
        state['elapsed_seconds']=time.time()-started;atomic_json(state_path,state)
        if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    root=args.out.resolve();root.mkdir(parents=True,exist_ok=True)
    if args.worker:worker(root);return
    previous=root/'launcher.json'
    if previous.exists():
        data=json.loads(previous.read_text())
        try:
            process=psutil.Process(data['pid'])
            if abs(process.create_time()-data['create_time'])<.1:
                print(json.dumps(dict(already_running=True,**data)));return
        except psutil.NoSuchProcess:pass
    flags=(subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_NO_WINDOW) if os.name=='nt' else 0
    io_sources={}
    for relative in ['campaigns/run_comprehensive_native.py','campaigns/comprehensive_durable_worker.py','commitjl/durable_io.py']:
        source=CODE/relative;target=root/'execution_io_sources'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        io_sources[relative]=hashlib.sha256(source.read_bytes()).hexdigest()
    atomic_json(root/'execution_io.json',dict(mode='fsync checkpoint contents before replacement; previous-manifest backup',
        scope='Evidence I/O only, outside numerical timings; frozen experiment source and numerical configuration unchanged',
        sources=io_sources))
    command=[sys.executable,'-u','-m','campaigns.run_comprehensive_native','--worker','--out',str(root)]
    with (root/'supervisor.log').open('a',encoding='utf-8') as log:
        process=subprocess.Popen(command,cwd=CODE,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            creationflags=flags,start_new_session=os.name!='nt',close_fds=True)
    data=dict(pid=process.pid,create_time=psutil.Process(process.pid).create_time(),command=command,output=str(root))
    atomic_json(previous,data)
    print(json.dumps(dict(started=True,**data)))


if __name__=='__main__':main()
