"""Detached local supervision of the authorized reviewer campaign.

Run new seed confirmation first, then the serial proof campaign. Unexpected
process loss can resume checked caches. Scientific/programming failures stop
visibly rather than silently changing experiments or filtering outcomes.
"""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
import psutil
from campaigns.local_fast_tradeoff import save,sha

REPO=Path(__file__).resolve().parents[2]


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();session=root/'autonomous_session';session.mkdir(exist_ok=True)
    status=session/'status.json'
    if status.exists():
        old=read(status)
        for pid in [old.get('pid'),old.get('child_pid')]:
            if pid and psutil.pid_exists(pid):
                args=psutil.Process(pid).cmdline()
                if any('autonomous_referee' in x or str(root) in x for x in args):raise RuntimeError('Supervisor or child still live; no duplicate launch')
    env=dict(os.environ,PYTHONPATH=str(REPO/'code'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',COMMITJL_MEMORY_PRESSURE_POLICY='windows_os')
    previous=read(status) if status.exists() else {}
    state=dict(pid=os.getpid(),child_pid=None,complete=False,status='starting',started_unix=previous.get('started_unix',time.time()),resumed_unix=time.time(),requested_work_window_hours=10.,
        memory_policy='Read-only Windows low-memory notification,1GiB minimum commit headroom,3.5GiB RSS cap; Go1200MiB soft limit and8workers unchanged. Earlier650MiB physical-reserve interruptions retained. No persistent OS setting changed.',private=True)
    save(status,state)
    def run(command,label,recover=False):
        retries=0
        while True:
            with (session/(label+f'_{retries}.stdout.log')).open('a',encoding='utf-8') as stdout,(session/(label+f'_{retries}.stderr.log')).open('a',encoding='utf-8') as stderr:
                child=subprocess.Popen(command,cwd=REPO,env=env,stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                state.update(status='running',active_task=label,child_pid=child.pid);save(status,state)
                while child.poll() is None:
                    state.update(heartbeat_unix=time.time(),available_bytes=psutil.virtual_memory().available)
                    queue=root/'queue.json'
                    if queue.exists():state['proof_queue']=read(queue)
                    save(status,state);time.sleep(15)
            state.update(child_pid=None,last_returncode=child.returncode);save(status,state)
            if child.returncode==0:return
            queue=read(root/'queue.json') if (root/'queue.json').exists() else {}
            if not recover or queue.get('status')=='stopped' or retries>=3:
                raise RuntimeError(f'{label} failed; preserved logs require inspection, no scientific fallback or outcome filtering')
            retries+=1;state.update(status='recovering_process_loss',restart_count=retries);save(status,state)
            time.sleep(15)
    try:
        confirmation=REPO/'code/validation/outputs/refinement_confirmation_20261025'
        if not (confirmation/'summary.json').exists() or not read(confirmation/'summary.json').get('complete'):
            command=[sys.executable,'-u','-m','experiments.confirm_bounded_refinement','--out',str(confirmation)]
            if (confirmation/'protocol.json').exists():command.append('--resume')
            run(command,'new_seed_confirmation')
        binary=REPO/'code/zk/bin/commitjl-bound-referee-final-20261025.exe'
        command=[sys.executable,'-u','-m','campaigns.referee_final','--root',str(root),'--source',str(REPO/'paper_results/review_controls/rank79_independent_replication_20261025'),'--binary',str(binary)]
        for name,suffix in [('compact','compact'),('moments','moments'),('affine','affine')]:command+=['--'+name+'-smoke',str(REPO/f'code/validation/outputs/referee_final_20261025_{suffix}_smoke/checks.json')]
        if (root/'queue.json').exists():command.append('--resume')
        run(command,'matched_proof_campaign',recover=True)
        assert read(root/'summary.json')['complete']
        state.update(complete=True,status='complete',finished_unix=time.time(),paper_visual_review_pending=True);save(status,state)
    except BaseException as error:
        state.update(status='stopped',error=str(error),child_pid=None,finished_unix=time.time());save(status,state);raise


if __name__=='__main__':main()
