"""Precision screening on every stored fit, preserving all negative outcomes."""
from pathlib import Path
import argparse,json,subprocess,sys,os,time
import pandas as pd
import psutil
from campaigns.local_fast_tradeoff import save


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();root.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[2];sources=[('historical',repo/'paper_results/review_controls/local_group_moments_extended_20261023'),('fresh',repo/'paper_results/review_controls/group_change_confirmation_20261024/source')]
    state=dict(complete=False,status='running',pid=os.getpid(),started_unix=time.time(),candidate_bits=[8,10,14],gate='Head conclusive>=151/160 and regressions>=30/36; no incorrect conclusions. Pick first passing precision in8,10,otherwise retain14. Stored-fit development, not new independent validation.',results=[]);save(root/'status.json',state)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    try:
        for bits in [8,10]:
            parts=[]
            for name,source in sources:
                while psutil.virtual_memory().available<1100*2**20:
                    state.update(status='waiting_for_ram',bits=bits,phase=name);save(root/'status.json',state);time.sleep(15)
                command=[sys.executable,'-m','experiments.pilot_group_change_bounds','--exact-old','--direct-only','--certificate-scale-bits',str(bits),'--source',str(source),'--out',str(root/f'bits{bits}'/name)]
                if name=='fresh':command.append('--fresh-confirmation')
                (root/f'bits{bits}').mkdir(exist_ok=True)
                with (root/f'bits{bits}'/(name+'.log')).open('w') as log:
                    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));state.update(status='running',bits=bits,phase=name,child_pid=child.pid);save(root/'status.json',state)
                    if child.wait()!=0:raise RuntimeError(f'precision{bits} {name} stopped; inspect log and preserve partial results')
                parts.append(pd.read_csv(root/f'bits{bits}'/name/'results.csv'))
            frame=pd.concat(parts,ignore_index=True);head=frame[(frame.method=='learned')&(frame.variant=='group_exact_old')];assert len(head)==160
            assert all((head.decision=='inconclusive')|(head.decision==head.truth))
            coverage=int((head.decision!='inconclusive').sum());sensitivity=int((head.decision=='worsens').sum());result=dict(bits=bits,conclusive=coverage,regressions_detected=sensitivity,passes=coverage>=151 and sensitivity>=30);state['results'].append(result);save(root/'status.json',state)
            if result['passes']:state['selected_bits']=bits;break
        else:state['selected_bits']=14
        state.update(complete=True,status='complete',finished_unix=time.time());save(root/'status.json',state);print(json.dumps(state),flush=True)
    except BaseException as error:state.update(status='stopped',error=str(error));save(root/'status.json',state);raise


if __name__=='__main__':main()
