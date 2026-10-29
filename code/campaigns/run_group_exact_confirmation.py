"""Recheck every persisted fit/update for the exact-old preprocessing profile."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
from campaigns.local_fast_tradeoff import save


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();root.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[2];sources=[('historical',repo/'paper_results/review_controls/local_group_moments_extended_20261023'),('fresh',repo/'paper_results/review_controls/group_change_confirmation_20261024/source')]
    state=dict(complete=False,status='running',started_unix=time.time(),selection='All80 persisted archives, both methods and all actual updates. No refitting or outcome selection.',sources=[str(p) for _,p in sources]);save(root/'status.json',state)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    try:
        for name,source in sources:
            while __import__('psutil').virtual_memory().available<1100*2**20:
                state.update(status='waiting_for_ram',phase=name);save(root/'status.json',state);time.sleep(15)
            command=[sys.executable,'-m','experiments.pilot_group_change_bounds','--exact-old','--source',str(source),'--out',str(root/name)]
            if name=='fresh':command.append('--fresh-confirmation')
            with (root/(name+'.log')).open('w',encoding='utf-8') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));state.update(status='running',phase=name,child_pid=child.pid);save(root/'status.json',state)
                if child.wait()!=0:raise RuntimeError(f'{name} failed; preserve partial outputs and inspect log')
        state.update(complete=True,status='complete',finished_unix=time.time());save(root/'status.json',state)
    except BaseException as error:
        state.update(status='stopped',error=str(error));save(root/'status.json',state);raise


if __name__=='__main__':main()
