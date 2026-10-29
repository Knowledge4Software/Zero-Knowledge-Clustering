"""Predeclared independent-fit confirmation, retaining every update/outcome."""
from pathlib import Path
import argparse,os,subprocess,sys,time
from campaigns.local_fast_tradeoff import save


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();root.mkdir(parents=True,exist_ok=False)
    state=dict(complete=False,started_unix=time.time(),phase='fit_and_oracle',seeds=[167,193],datasets=['mnist','fashion_mnist'],k=[3,5],ages=[1,8],all_actual_updates_retained=True)
    save(root/'status.json',state)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    stages=[('fit_and_oracle',[sys.executable,'-m','experiments.pilot_group_moments','--new-confirmation','--threads','1','--out',str(root/'source')]),
        ('new_cost_bounds',[sys.executable,'-m','experiments.pilot_group_change_bounds','--fresh-confirmation','--source',str(root/'source'),'--out',str(root/'bounds')])]
    try:
        for phase,command in stages:
            state['phase']=phase;save(root/'status.json',state)
            with (root/(phase+'.log')).open('w',encoding='utf-8') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                state['child_pid']=child.pid;save(root/'status.json',state)
                if child.wait()!=0:raise RuntimeError(f'{phase} failed; inspect its log')
        state.update(complete=True,phase='complete',finished_unix=time.time());save(root/'status.json',state)
    except BaseException as error:
        state.update(phase='failed',error=str(error));save(root/'status.json',state);raise


if __name__=='__main__':main()
