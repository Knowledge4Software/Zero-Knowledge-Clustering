"""Continue the current cohort after the user's smaller laptop scope change.

Only the two Python schedulers are replaced. The active baseline worker and
its prover keep running; completed scientific phases are never repeated.
"""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
import psutil
from campaigns.local_fast_tradeoff import save,sha
from campaigns.proof_runtime import memory_stop
from campaigns.execution_admission import require_execution

REPO=Path(__file__).resolve().parents[2]
POLICY=REPO/'code/validation/outputs/referee_round2_20261026/execution_policy.json'


def read(path):return json.loads(Path(path).read_text())


def within_scope(case,scope):
    maximum=scope['maximum_proof_catalog_records']
    if type(maximum) is not int or not 1<=maximum<=1024:
        raise ValueError('This user-authorized laptop scope permits at most1024 proof records')
    return case['n']<=maximum


def validate_adopted_process(process,args,handover):
    if args!=handover['child_command'] or process.cmdline()!=args:
        raise RuntimeError('Active phase command differs; do not adopt another process')
    if process.pid!=handover['child_pid'] or abs(process.create_time()-handover['child_created_unix'])>.01:
        raise RuntimeError('Active phase PID or creation time differs')


def adopt_phase(root,label,args,handover,plan):
    from campaigns.referee_round3_remote import adopted_phase_receipt
    started=handover['child_created_unix'];peak=0;rc=None
    try:process=psutil.Process(handover['child_pid'])
    except psutil.NoSuchProcess:process=None
    if process is not None:
        validate_adopted_process(process,args,handover)
        while True:
            try:
                tree=[process]+process.children(recursive=True)
                peak=max(peak,max((p.memory_info().rss for p in tree),default=0))
                stop=memory_stop(max((p.memory_info().rss for p in tree),default=0))
                try:require_execution(read(POLICY),target='laptop')
                except RuntimeError:stop='execution authorization held'
                campaign=Path(handover['campaign_root'])
                if (campaign/'STOP').exists():stop='user stop file'
                if psutil.disk_usage(str(root)).free<1024*2**20:stop='disk reserve'
                if time.time()-started>plan['phase_wall_limit_seconds']:stop='phase wall limit'
                if stop:
                    for p in reversed(tree):
                        try:p.kill()
                        except psutil.NoSuchProcess:pass
                    raise RuntimeError('Adopted phase stopped: '+stop)
                try:rc=process.wait(timeout=.5);break
                except psutil.TimeoutExpired:continue
            except psutil.NoSuchProcess:break
    if rc not in (None,0):raise RuntimeError('Adopted phase exited with code '+str(rc))
    receipt=adopted_phase_receipt(root,label)
    return dict(phase=label,wall_seconds=time.time()-started,returncode=rc,stop=None,
        peak_process_rss_bytes=peak,peak_scope='Observed after coordinator handover',
        coordinator_adopted=True,proof_worker_restarted=False,
        completion_checked_receipt=receipt.relative_to(REPO).as_posix(),completion_receipt_sha256=sha(receipt))


def run(root,handover_path):
    from campaigns.referee_round3_remote import execute
    from campaigns.referee_final_local import backend_binary
    from campaigns.referee_extended_local import execute_extended
    require_execution(read(POLICY),target='laptop')
    scope=read(root/'laptop_scope_amendment_20261027.json');handover=read(handover_path)
    state=read(root/'queue.json');plan=read(root/'plan.json')
    if not within_scope(dict(n=read(Path(handover['cohort_root'])/'plan.json')['catalog_records']),scope):
        raise RuntimeError('Active cohort exceeds the new laptop scope')
    if state['pid']!=handover['old_supervisor_pid'] or psutil.pid_exists(state['pid']):
        raise RuntimeError('Previous supervisor still exists or root identity changed')
    if sha(root/'plan.json')!=handover['campaign_plan_sha256'] or sha(root/'laptop_scope_amendment_20261027.json')!=handover['scope_sha256']:
        raise RuntimeError('Handover campaign/scope pin differs')
    os.environ.update(PYTHONPATH=str(REPO/'code'),COMMITJL_EXECUTION_TARGET='laptop',COMMITJL_EXECUTION_POLICY=str(POLICY),
        COMMITJL_FINAL_CAMPAIGN_ROOT=str(root),COMMITJL_MEMORY_PRESSURE_POLICY='windows_os',
        GOMAXPROCS=str(plan['workers']),GOMEMLIMIT='1200MiB',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    state.setdefault('supervisor_changes',[]).append(dict(old_pid=state['pid'],new_pid=os.getpid(),recorded_unix=time.time(),
        reason=scope['reason'],handover_receipt_sha256=sha(handover_path),active_proof_worker_restarted=False))
    state.update(pid=os.getpid(),status='running',active_phase='extended_proofs',active_cohort=handover['cohort_root'],
        child_pid=handover['child_pid'],maximum_proof_catalog_records=scope['maximum_proof_catalog_records'])
    save(root/'queue.json',state)
    try:
        cohort=Path(handover['cohort_root'])
        execute(cohort,REPO/plan['binary'],backend_binary(root,plan),execution_target='laptop',handover=handover)
        # Record the cohort that the old extended scheduler was waiting for.
        prior=read(root/'extended_status.json');name=cohort.name
        if any(r['case']==name for r in prior['cases']):raise RuntimeError('Current cohort already recorded')
        prior['cases'].append(dict(case=name,status='complete',plan_sha256=sha(cohort/'plan.json')))
        save(root/'extended_status.json',prior)
        state.update(child_pid=None,active_cohort=None);save(root/'queue.json',state)
        execute_extended(root,scope=scope)
        state['attempts'].append(dict(phase='extended_proofs',wall_seconds=time.time()-handover['old_extended_created_unix'],
            complete=True,scope_amended=True,supervisor_adopted=True))
        state['completed'].append('extended_proofs')
        for phase,module in [('final_evidence_export','analysis.update_optimized_evidence'),('final_paper_candidate','analysis.release_referee_final_paper')]:
            args=[sys.executable,'-u','-m',module]
            args+=['--root',str(root)] if phase=='final_evidence_export' else ['--reuse-checked-tables','--automatic','--tag','prototype_audit_optimized_20261026']
            state.update(active_phase=phase,child_pid=None);save(root/'queue.json',state)
            with (root/(phase+'.bounded.log')).open('xb') as log:
                child=subprocess.Popen(args,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                state['child_pid']=child.pid;save(root/'queue.json',state)
                if child.wait():raise RuntimeError(phase+' failed; inspect preserved log')
            state['completed'].append(phase)
        extended=read(root/'extended_status.json')
        state.update(complete=True,status='complete' if extended['complete'] else 'complete_with_scope_or_resource_exclusions',
            all_original_proof_cases_complete=extended['complete'],authorized_scheduling_complete=True,
            child_pid=None,finished_unix=time.time())
        save(root/'queue.json',state)
    except BaseException as error:
        state.update(status='stopped',error=str(error),child_pid=None);save(root/'queue.json',state);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--handover',type=Path,required=True)
    a=p.parse_args();run(a.root.resolve(),a.handover.resolve())
