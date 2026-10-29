"""Read live local process identities; never reconnect to the retired server."""
from pathlib import Path
import json,time
import psutil
from campaigns.local_fast_tradeoff import save,sha

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'code/validation/outputs/referee_optimized_20261026'


def main():
    state=json.loads((ROOT/'queue.json').read_text());live=[]
    try:
        parent=psutil.Process(state['pid']);args=parent.cmdline()
        if not any(v in args for v in ['campaigns.referee_final_local','campaigns.final_local_supervisor','campaigns.bounded_local_campaign']):raise RuntimeError('Supervisor PID differs')
        for p in [parent]+parent.children(recursive=True):
            try:live.append(dict(pid=p.pid,name=p.name(),created_unix=p.create_time(),rss_bytes=p.memory_info().rss,command=p.cmdline()))
            except psutil.NoSuchProcess:pass
        # A phase kept alive through the user's scheduler handover retains its
        # original Windows parent PID. Validate its recorded identity instead
        # of assuming it appears among the new scheduler's descendants.
        cohort=state.get('active_cohort')
        if cohort:
            case=json.loads((Path(cohort)/'queue.json').read_text())
            handovers=case.get('coordinator_handovers',[])
            if handovers and case.get('child_pid')==handovers[-1]['child_pid']:
                from campaigns.bounded_local_campaign import validate_adopted_process
                handover=handovers[-1]
                try:
                    worker=psutil.Process(handover['child_pid']);validate_adopted_process(worker,handover['child_command'],handover)
                    seen={v['pid'] for v in live}
                    for p in [worker]+worker.children(recursive=True):
                        if p.pid not in seen:
                            try:live.append(dict(pid=p.pid,name=p.name(),created_unix=p.create_time(),rss_bytes=p.memory_info().rss,command=p.cmdline()));seen.add(p.pid)
                            except psutil.NoSuchProcess:pass
                except psutil.NoSuchProcess:pass
    except psutil.NoSuchProcess:pass
    status=dict(updated_unix=time.time(),running=bool(live),phase=state.get('active_phase'),queue_status=state['status'],active_processes=live,
        native_study_complete=(ROOT/'native/summary.json').exists(),mobin_retired=True,all_scientific_issues_fixed=False,
        remaining='Repeated matched proofs, online-affine timing, exact backend timing, independent-fit/large-catalog proofs and paid fallback controls.')
    if (ROOT/'matched_controls/queue.json').exists():status['matched']=json.loads((ROOT/'matched_controls/queue.json').read_text())
    scope_path=ROOT/'laptop_scope_amendment_20261027.json'
    if scope_path.exists():
        scope=json.loads(scope_path.read_text());status.update(maximum_proof_catalog_records=scope['maximum_proof_catalog_records'],
            deferred_proof_cases=scope['deferred_proof_cases'],
            remaining='Finish the active1024-record matched cohort and the declared <=1024 center-count/fallback cases that pass resource preflight. The4096/16384 proof cases are deferred by user request; their native checks remain available.')
    save(REPO/'output/optimized_campaign_status_20261026.json',status)
    registry=REPO/'code/validation/outputs/research_jobs_20261024.json';r=json.loads(registry.read_text())
    policy=json.loads((REPO/'code/validation/outputs/referee_round2_20261026/execution_policy.json').read_text())
    r.update(updated_unix=time.time(),active_processes=live,current_status_file=ROOT.relative_to(REPO).as_posix()+'/queue.json',
        proof_queue_status_file=ROOT.relative_to(REPO).as_posix()+'/matched_controls/queue.json',
        current_status=('Local optimized campaign running: '+str(state.get('active_phase')) if live else 'Local optimized campaign '+state['status'])+'. Native study56fits/280records complete. Consult the queue for remaining proof controls.',
        heavy_runs_allowed=policy['heavy_runs_allowed'],replacement_execution_target='laptop',mobin_server_retired=True,mobin_connection_attempts_allowed=False,remote_heavy_runs_allowed=False,
        instructions='Keep research private, preserve all evidence and honor the current local campaign policy. Mobin is permanently retired. Do not resume historical queues. Read actual process identities before reporting a job running.')
    validation=REPO/'output/prototype_audit_optimized_20261026_validation.json'
    newer=REPO/'output/prototype_audit_optimized_20261027_validation.json'
    if newer.exists() and json.loads(newer.read_text()).get('visual_review')=='PASS':validation=newer
    if validation.exists():
        v=json.loads(validation.read_text())
        if v['visual_review']=='PASS':r.update(latest_paper=v['pdf'],latest_paper_validation=validation.relative_to(REPO).as_posix(),latest_paper_canonical_sha256=v['pdf_sha256'],paper_ready_for_private_review=True,submission_ready=False)
    save(registry,r);print(json.dumps(dict(running=status['running'],phase=status['phase'],processes=len(live),native_complete=status['native_study_complete'])))


if __name__=='__main__':main()
