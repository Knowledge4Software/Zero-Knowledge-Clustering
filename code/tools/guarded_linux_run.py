"""Run one serial laptop comparison phase in Linux/WSL with explicit guards.

The same wrapper and CPU affinity apply to both implementations. This is
execution supervision, not an independent proof verifier.
"""
from pathlib import Path
import argparse,json,os,signal,subprocess,time
import psutil


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--log',type=Path,required=True)
    p.add_argument('--timeout',type=float,default=1800)
    p.add_argument('--rss-mib',type=int,default=3584)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--expect-failure',action='store_true')
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();cmd=a.command[1:] if a.command[:1]==['--'] else a.command
    assert cmd and os.name=='posix' and 1<=a.workers<=8
    assert not a.receipt.exists(),'Never overwrite a measured attempt'
    a.receipt.parent.mkdir(parents=True,exist_ok=True);a.log.parent.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,RAYON_NUM_THREADS=str(a.workers),GOMAXPROCS=str(a.workers),
        GOMEMLIMIT='1200MiB',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        COMMITJL_RESUME_PROOFS='1')
    affinity=sorted(os.sched_getaffinity(0))[:a.workers]
    os.sched_setaffinity(0,affinity)
    start=time.monotonic();started_unix=time.time();peak=0;stop=None
    with a.log.open('wb') as handle:
        child=subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        proc=psutil.Process(child.pid)
        while child.poll() is None:
            try:
                processes=[proc]+proc.children(recursive=True)
                rss=sum(q.memory_info().rss for q in processes if q.is_running())
                peak=max(peak,rss)
            except psutil.Error:pass
            if peak>a.rss_mib*2**20:stop='process tree RSS cap'
            elif psutil.virtual_memory().available<650*2**20:stop='RAM reserve'
            elif psutil.disk_usage(a.receipt.parent).free<12*2**30:stop='disk reserve'
            elif time.monotonic()-start>a.timeout:stop='phase wall limit'
            elif any((folder/'STOP').exists() for folder in [a.receipt.parent,a.receipt.parent.parent]):stop='user stop'
            if stop:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                break
            time.sleep(.2)
    code=child.wait();passed=(code!=0 if a.expect_failure else code==0) and stop is None
    value=dict(complete=passed,command=cmd,started_unix=started_unix,
        wall_seconds=time.monotonic()-start,returncode=code,stop=stop,
        peak_process_tree_rss_bytes=peak,workers=a.workers,cpu_affinity=affinity,
        os='Linux/WSL',expected_failure=a.expect_failure)
    a.receipt.write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps(value),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
