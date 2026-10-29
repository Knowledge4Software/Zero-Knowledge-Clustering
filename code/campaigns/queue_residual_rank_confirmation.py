"""Wait for the measured proof campaign to end before a larger native screen."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
from campaigns.local_fast_tradeoff import save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--proof-queue',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args();queue=a.root.with_name(a.root.name+'_queue.json');state=dict(complete=False,status='waiting_for_proof_campaign',pid=os.getpid(),proof_queue=str(a.proof_queue.resolve()),root=str(a.root.resolve()),started_unix=time.time(),selection='All160 stored head updates, all36 natural regressions; fixed auxiliary ranks79/95. No new prediction-model fits. No proof or speed claims. Wait for proof queue completion to avoid deliberately overlapping this numerical screen with measured proving.');save(queue,state)
    try:
        while True:
            parent=json.loads(a.proof_queue.read_text())
            if parent.get('complete'):break
            if parent.get('status')=='stopped':raise RuntimeError('Proof queue stopped; do not run a competing screen automatically')
            time.sleep(15)
        script=Path(__file__).resolve().parents[1]/'experiments/pilot_residual_certificate_rank.py';state['pilot_source_sha256']=sha(script)
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        command=[sys.executable,'-m','experiments.pilot_residual_certificate_rank','--all-stored-updates','--ranks','79','95','--out',str(a.root.resolve())]
        with a.root.with_name(a.root.name+'.log').open('w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));state.update(status='running',child_pid=child.pid);save(queue,state)
            if child.wait()!=0:raise RuntimeError('Native screen stopped; preserve partial evidence and inspect the adjacent log')
        state.update(complete=True,status='complete',child_pid=None,finished_unix=time.time());save(queue,state)
    except BaseException as error:state.update(status='stopped',error=str(error));save(queue,state);raise


if __name__=='__main__':main()
