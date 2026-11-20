"""Replace the identified stalled downloader, retaining every partial file."""
from pathlib import Path
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
    password=getpass.getpass('SSH password: ')
    retired=None
    try:
        previous=psutil.Process(22240)
        assert any('download_final_evidence_v2_20261102' in arg for arg in previous.cmdline()),'PID belongs to another process'
        previous.terminate();previous.wait(timeout=10);retired=22240
    except psutil.NoSuchProcess:pass
    deadline=json.loads((DEST/'verified_downloads.json').read_text())['download_deadline_unix']
    env=dict(os.environ,PYTHONPATH=str(REPO/'code'))
    with (DEST/'download_v3_20261103.log').open('ab') as log:
        p=subprocess.Popen([sys.executable,'-u',str(REPO/'code/tools/download_final_evidence_v3_20261103.py'),
              '--deadline',str(deadline)],cwd=REPO,env=env,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,
              creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP,text=True)
        p.stdin.write(password+'\n');p.stdin.close()
    password=None
    receipt=dict(pid=p.pid,retired_stalled_pid=retired,launch_unix=time.time(),
         download_deadline_unix=deadline,partial_transfers_preserved=True,channel_open_timeout_seconds=45,
         channel_read_timeout_seconds=60,compressed_metadata=True)
    (DEST/'downloader_v3_launch.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
