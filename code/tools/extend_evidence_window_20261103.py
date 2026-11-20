"""Extend the authorized local downloader; never persist SSH credentials."""
from pathlib import Path
import argparse
import getpass
import json
import os
import subprocess
import sys
import time
import psutil

REPO=Path(__file__).resolve().parents[2]
DEST=REPO/'output/server_run_20261101/final_delivery'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--deadline',type=float,required=True)
    args=parser.parse_args()
    assert time.time()+3600<args.deadline<time.time()+36*3600
    password=getpass.getpass('SSH password: ')
    retired=[]
    for process in psutil.process_iter(['pid','cmdline']):
        if process.pid==os.getpid():continue
        command=process.info['cmdline'] or []
        if any('download_final_evidence_v3_20261103.py' in item for item in command):
            process.terminate();process.wait(timeout=15);retired.append(process.pid)
    env=dict(os.environ,PYTHONPATH=str(REPO/'code'))
    with (DEST/'download_extended_20261103.log').open('ab') as log:
        process=subprocess.Popen([sys.executable,'-u',str(REPO/'code/tools/download_final_evidence_v3_20261103.py'),
             '--deadline',str(args.deadline)],cwd=REPO,env=env,stdin=subprocess.PIPE,stdout=log,
             stderr=subprocess.STDOUT,text=True,
             creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
        process.stdin.write(password+'\n');process.stdin.close()
    password=None
    receipt=dict(pid=process.pid,launch_unix=time.time(),download_deadline_unix=args.deadline,
                 retired_downloader_pids=retired,partial_files_preserved=True,credentials_saved=False)
    (DEST/'downloader_extended_launch.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
