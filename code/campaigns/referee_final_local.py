"""Serial laptop campaign after explicit 26 October authorization.

Frozen scientific profile, matched runtime, real disjoint rows, retained
failures. No network server is used. Progress survives the interactive turn.
"""
from pathlib import Path
import argparse,json,os,shutil,subprocess,sys,time
import psutil
from campaigns.local_fast_tradeoff import save,sha
from campaigns.execution_admission import require_execution
from campaigns.proof_runtime import memory_stop

REPO=Path(__file__).resolve().parents[2]
POLICY=REPO/'code/validation/outputs/referee_round2_20261026/execution_policy.json'


def backend_binary(root,plan):
    amendment=root/'backend_runtime_amendment.json'
    value=json.loads(amendment.read_text()) if amendment.exists() else plan
    path=REPO/value['plonk_binary']
    if 'plonk_binary_sha256' in value:assert sha(path)==value['plonk_binary_sha256'],'backend binary changed'
    return path


def execute(root):
    require_execution(json.loads(POLICY.read_text()),target='laptop')
    plan=json.loads((root/'plan.json').read_text())
    state=dict(complete=False,status='running',pid=os.getpid(),started_unix=time.time(),attempts=[],completed=[],plan_sha256=sha(root/'plan.json'))
    if (root/'queue.json').exists():raise RuntimeError('Existing queue: preserve its attempts and inspect before resuming')
    save(root/'queue.json',state)
    env=dict(os.environ,PYTHONPATH=str(REPO/'code'),COMMITJL_EXECUTION_TARGET='laptop',COMMITJL_EXECUTION_POLICY=str(POLICY),
        COMMITJL_FINAL_CAMPAIGN_ROOT=str(root),COMMITJL_MEMORY_PRESSURE_POLICY='windows_os',GOMAXPROCS=str(plan['workers']),GOMEMLIMIT='1200MiB',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def invoke(name,args):
        require_execution(json.loads(POLICY.read_text()),target='laptop')
        if (root/'STOP').exists():raise RuntimeError('User stop file present')
        if shutil.disk_usage(root).free<2*2**30:raise RuntimeError('Disk reserve: less than 2 GiB free; no new phase started')
        state.update(active_phase=name,status='running',child_pid=None);save(root/'queue.json',state)
        started=time.monotonic();peak=0;stop=None
        with (root/(name+'.log')).open('xb') as log:
            child=subprocess.Popen(args,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            state['child_pid']=child.pid;save(root/'queue.json',state)
            while child.poll() is None:
                try:
                    proc=psutil.Process(child.pid);tree=[proc]+proc.children(recursive=True)
                    largest=max((p.memory_info().rss for p in tree),default=0);peak=max(peak,largest)
                    stop=memory_stop(largest)
                except psutil.NoSuchProcess:continue
                if (root/'STOP').exists():stop='user stop file'
                if shutil.disk_usage(root).free<1024*2**20:stop='disk reserve'
                if time.monotonic()-started>plan['phase_wall_limit_seconds']:stop='phase wall limit'
                if stop:
                    for process in reversed(tree):
                        try:process.kill()
                        except psutil.NoSuchProcess:pass
                    break
                time.sleep(1)
            rc=child.wait()
        state['attempts'].append(dict(phase=name,wall_seconds=time.monotonic()-started,peak_process_rss_bytes=peak,returncode=rc,stop=stop))
        state['child_pid']=None;save(root/'queue.json',state)
        if rc or stop:raise RuntimeError(name+' stopped: '+str(stop)+'; preserve all logs and costs')
        state['completed'].append(name);save(root/'queue.json',state)
    py=[sys.executable,'-u','-m'];binary=REPO/plan['binary'];plonk=REPO/plan['plonk_binary']
    try:
        invoke('compact_smoke',py+['validation.check_group_fast_proofs','--compact','--root',str(root/'compact_smoke'),'--binary',str(binary)])
        invoke('refinement_smoke',py+['validation.check_group_refinement_proofs','--root',str(root/'refinement_smoke'),'--binary',str(binary)])
        # Native study first produces all independent fits before choosing
        # proof cases. Its outcome cannot alter ranks, caps or seed membership.
        invoke('native_confirmation',py+['experiments.referee_round3_native','--out',str(root/'native'),'--design',str(root/'study_plan.json')])
        from campaigns.referee_round3_remote import prepare,execute as timing
        controls=root/'matched_controls';prepare(controls,repetitions=3,workers=plan['workers'],execution_target='laptop')
        invoke('matched_controls',py+['campaigns.referee_final_local','--root',str(root),'--timing-only'])
        invoke('extended_proofs',py+['campaigns.referee_final_local','--root',str(root),'--extended-only'])
        state.update(complete=True,status='complete',finished_unix=time.time());save(root/'queue.json',state)
    except BaseException as error:
        state.update(status='stopped',error=str(error),child_pid=None);save(root/'queue.json',state);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--timing-only',action='store_true');p.add_argument('--extended-only',action='store_true');a=p.parse_args();root=a.root.resolve()
    if a.timing_only:
        from campaigns.referee_round3_remote import execute
        os.environ['COMMITJL_FINAL_CAMPAIGN_ROOT']=str(root)
        plan=json.loads((root/'plan.json').read_text());execute(root/'matched_controls',REPO/plan['binary'],backend_binary(root,plan),execution_target='laptop')
    elif a.extended_only:
        from campaigns.referee_extended_local import execute_extended
        execute_extended(root)
    else:execute(root)
