"""Start the exact-old proof timing only after the complete native safety gate."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
import pandas as pd
from campaigns.local_fast_tradeoff import save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['confirmation','parent','root','binary','smoke-receipt']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    confirmation=a.confirmation.resolve();root=a.root.resolve();queue_file=root.with_name(root.name+'_queue.json')
    state=dict(complete=False,status='waiting_for_native_confirmation',pid=os.getpid(),started_unix=time.time(),confirmation=str(confirmation),proof_root=str(root),binary_sha256=sha(a.binary),gate='Both cohorts complete, all intervals contain Full, no false conclusive decisions, head direct coverage >=151/160 and regression sensitivity >=30/36. This is a development gate on stored fits, not new independent validation.')
    save(queue_file,state)
    try:
        while True:
            native=json.loads((confirmation/'status.json').read_text())
            if native.get('complete'):break
            if native.get('status')=='stopped':raise RuntimeError('Native confirmation stopped; inspect its logs before proving.')
            time.sleep(15)
        frames=[]
        for cohort in ['historical','fresh']:
            summary=json.loads((confirmation/cohort/'summary.json').read_text())
            assert summary['complete'] and summary['all_intervals_contain_truth'] and summary['false_conclusive_decisions']==0
            frames.append(pd.read_csv(confirmation/cohort/'results.csv'))
        frame=pd.concat(frames,ignore_index=True)
        direct=frame[(frame.method=='learned')&(frame.variant=='group_exact_old')&(frame.budget_fraction==0)]
        assert len(direct)==160 and len(direct[direct.truth=='worsens'])==36
        assert all((direct.decision=='inconclusive')|(direct.decision==direct.truth))
        conclusive=int((direct.decision!='inconclusive').sum());regressions=int(((direct.truth=='worsens')&(direct.decision=='worsens')).sum())
        state.update(native_head_conclusive=conclusive,native_head_regressions=regressions)
        if conclusive<151 or regressions<30:raise RuntimeError('Exact-old profile lost sensitivity at the predefined gate. Preserve outputs and inspect before timing.')
        command=[sys.executable,'-m','campaigns.run_group_fast_control','--exact-old']
        for name in ['parent','root','binary','smoke_receipt']:command+=['--'+name.replace('_','-'),str(getattr(a,name).resolve())]
        if a.resume:command.append('--resume')
        state['proof_attempts']=[]
        while True:
            child=subprocess.Popen(command,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            attempt=dict(child_pid=child.pid,started_unix=time.time());state['proof_attempts'].append(attempt)
            state.update(status='proving',child_pid=child.pid,proof_started_unix=attempt['started_unix']);save(queue_file,state)
            code=child.wait();attempt.update(returncode=code,finished_unix=time.time())
            if code==0:
                state.update(complete=True,status='complete',returncode=0,finished_unix=time.time());save(queue_file,state);break
            proof_state=json.loads((root/'status.json').read_text())
            phase=proof_state.get('active_phase');attempts=proof_state.get('resources',{}).get(phase,{}).get('attempts',[])
            if proof_state.get('paused') or not attempts or attempts[-1].get('stop')!='RAM reserve':
                state.update(status='stopped',returncode=code);save(queue_file,state)
                raise RuntimeError('Proof timing stopped; inspect status and retain caches.')
            # Only retry a low-free-RAM stop. Circuit, pin, resource-cap and
            # validation failures remain visible errors. The phase worker waits
            # for its observed peak plus reserve before using saved proof caches.
            state.update(status='waiting_for_ram_retry',child_pid=None,returncode=code);save(queue_file,state)
            if '--resume' not in command:command.append('--resume')
            time.sleep(15)
    except BaseException as error:
        state.update(status='stopped',error=str(error));save(queue_file,state);raise


if __name__=='__main__':main()
