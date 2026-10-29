"""Adopt the current matched worker without restarting its scientific work.

The first supervisor's 12-hour phase cap is appropriate for one prover
command, not an entire many-case campaign. Inner commands keep their own
12-hour cap. This supervisor retains RAM/disk/stop guards across the series.
"""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
import psutil
from campaigns.local_fast_tradeoff import save
from campaigns.proof_runtime import memory_stop
from campaigns.execution_admission import require_execution

REPO=Path(__file__).resolve().parents[2]
POLICY=REPO/'code/validation/outputs/referee_round2_20261026/execution_policy.json'


def process_matches(process,root):
    args=process.cmdline()
    return ('campaigns.referee_final_local' in args and '--timing-only' in args
            and '--root' in args and Path(args[args.index('--root')+1]).resolve()==root)


def run(root):
    require_execution(json.loads(POLICY.read_text()),target='laptop')
    state=json.loads((root/'queue.json').read_text());old_pid=state['pid'];child_pid=state['child_pid']
    if psutil.pid_exists(old_pid):raise RuntimeError('Old supervisor still live; never duplicate it')
    child=psutil.Process(child_pid)
    if state['active_phase']!='matched_controls' or not process_matches(child,root):raise RuntimeError('Existing child identity differs; do not adopt')
    state.setdefault('supervisor_changes',[]).append(dict(old_pid=old_pid,new_pid=os.getpid(),recorded_unix=time.time(),
        reason='Continue the same live matched worker. Apply wall guards to each inner command, not to the whole multi-case series. No proof, key, input, attempt or timing is restarted.'))
    state.update(pid=os.getpid(),status='running');save(root/'queue.json',state)
    env=dict(os.environ,PYTHONPATH=str(REPO/'code'),COMMITJL_EXECUTION_TARGET='laptop',COMMITJL_EXECUTION_POLICY=str(POLICY),
        COMMITJL_FINAL_CAMPAIGN_ROOT=str(root),COMMITJL_MEMORY_PRESSURE_POLICY='windows_os',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def monitor(process):
        birth=process.create_time();stop=None;peak=0
        while process.is_running():
            try:
                if process.create_time()!=birth:raise RuntimeError('PID reused')
                tree=[process]+process.children(recursive=True)
                rss=max((p.memory_info().rss for p in tree),default=0);peak=max(peak,rss)
                stop=memory_stop(rss)
                policy=json.loads(POLICY.read_text())
                try:require_execution(policy,target='laptop')
                except RuntimeError:stop='execution authorization held'
                if (root/'STOP').exists():stop='user stop file'
                if psutil.disk_usage(str(root)).free<1024*2**20:stop='disk reserve'
                if stop:
                    for p in reversed(tree):
                        try:p.kill()
                        except psutil.NoSuchProcess:pass
                    raise RuntimeError(stop)
            except psutil.NoSuchProcess:break
            time.sleep(1)
        return peak
    def phase(name,command):
        require_execution(json.loads(POLICY.read_text()),target='laptop')
        started=time.time()
        state.update(active_phase=name,status='running');save(root/'queue.json',state)
        with (root/(name+'.supervised.log')).open('xb') as log:
            process=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            state['child_pid']=process.pid;save(root/'queue.json',state)
            try:peak=monitor(psutil.Process(process.pid))
            finally:
                if process.poll() is None:
                    try:
                        owner=psutil.Process(process.pid)
                        for p in reversed([owner]+owner.children(recursive=True)):
                            try:p.kill()
                            except psutil.NoSuchProcess:pass
                    except psutil.NoSuchProcess:pass
            rc=process.wait()
        state['attempts'].append(dict(phase=name,wall_seconds=time.time()-started,peak_process_rss_bytes=peak,returncode=rc))
        state['child_pid']=None;save(root/'queue.json',state)
        if rc:raise RuntimeError(name+' failed; preserve all attempts')
        state['completed'].append(name);save(root/'queue.json',state)
    try:
        started=child.create_time();peak=monitor(child)
        matched=json.loads((root/'matched_controls/queue.json').read_text())
        state['attempts'].append(dict(phase='matched_controls',wall_seconds=time.time()-started,peak_process_rss_bytes=peak,
            complete=matched.get('complete',False),supervisor_adopted=True))
        if not matched.get('complete'):raise RuntimeError('Adopted matched worker stopped; inspect its preserved ledger')
        state['completed'].append('matched_controls');state['child_pid']=None;save(root/'queue.json',state)
        py=[sys.executable,'-u','-m']
        phase('extended_proofs',py+['campaigns.referee_final_local','--root',str(root),'--extended-only'])
        phase('final_evidence_export',py+['analysis.update_optimized_evidence','--root',str(root)])
        phase('final_paper_candidate',py+['analysis.release_referee_final_paper','--reuse-checked-tables','--automatic','--tag','prototype_audit_optimized_20261026'])
        extended=json.loads((root/'extended_status.json').read_text())
        state.update(complete=True,status='complete' if extended['complete'] else 'complete_with_resource_exclusions',all_requested_proof_cases_complete=extended['complete'],finished_unix=time.time())
        save(root/'queue.json',state)
    except BaseException as error:
        state.update(status='stopped',error=str(error),child_pid=None);save(root/'queue.json',state);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root.resolve())
