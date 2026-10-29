"""Keep one live proof through a documented laptop sleep, without restarting it.

The old scheduler's calendar cap counts sleep. Hold only that scheduler while
a resource-guarded monitor lets this already-running proof phase complete.
Resume the scheduler after the child exits. The raw timing remains untouched.
"""
from pathlib import Path
import argparse,json,time
import psutil
from campaigns.local_fast_tradeoff import save
from campaigns.proof_runtime import memory_stop
from campaigns.execution_admission import require_execution


def main(root,scheduler_pid,child_pid):
    state=json.loads((root/'queue.json').read_text());scheduler=psutil.Process(scheduler_pid);child=psutil.Process(child_pid)
    if state['supervisor_pid']!=scheduler_pid or state['child_pid']!=child_pid or state['active_phase']!='backend_r1':raise RuntimeError('Active phase differs')
    if 'campaigns.referee_two_reviews_local' not in scheduler.cmdline() or 'full-plonk-control-referee-20261027.exe' not in Path(child.exe()).name or child.ppid()!=scheduler_pid:
        raise RuntimeError('Scheduler/prover identity differs')
    policy=Path(__file__).resolve().parents[2]/'code/validation/outputs/referee_round2_20261026/execution_policy.json'
    receipt=dict(started_unix=time.time(),scheduler_pid=scheduler_pid,child_pid=child_pid,child_created_unix=child.create_time(),
        command=child.cmdline(),reason='Windows sleep interval recorded independently; retain raw interrupted timing and finish the same proof worker.',
        proof_restarted=False,peak_rss_bytes=0)
    save(root/'sleep_guardian.json',receipt)
    scheduler.suspend()
    try:
        while child.is_running():
            try:
                require_execution(json.loads(policy.read_text()),target='laptop')
                rss=child.memory_info().rss;receipt['peak_rss_bytes']=max(receipt['peak_rss_bytes'],rss);stop=memory_stop(rss)
                if (root/'STOP').exists():stop='user stop file'
                if psutil.disk_usage(str(root)).free<2**30:stop='disk reserve'
                if time.time()-receipt['started_unix']>12*3600:stop='guardian phase wall limit'
                if stop:receipt['stop']=stop;child.kill();break
                try:receipt['returncode']=child.wait(timeout=.5);break
                except psutil.TimeoutExpired:continue
            except psutil.NoSuchProcess:break
    finally:
        try:scheduler.resume();receipt['scheduler_resumed']=True
        except psutil.NoSuchProcess:receipt['scheduler_resumed']=False
        receipt['finished_unix']=time.time();save(root/'sleep_guardian.json',receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--scheduler-pid',type=int,required=True);p.add_argument('--child-pid',type=int,required=True)
    a=p.parse_args();main(a.root.resolve(),a.scheduler_pid,a.child_pid)
