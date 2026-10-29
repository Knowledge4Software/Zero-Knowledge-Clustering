"""Outcome-independent extended proof cases, after matched small controls."""
import json,os,shutil,subprocess,sys,time
from pathlib import Path
import psutil
from campaigns.local_fast_tradeoff import save,sha
from campaigns.execution_admission import require_execution
from campaigns.proof_runtime import memory_stop

REPO=Path(__file__).resolve().parents[2]
POLICY=REPO/'code/validation/outputs/referee_round2_20261026/execution_policy.json'


def bounded_preflight(command,root,log_path):
    """Compile only, with the same laptop memory reserves and a five-minute cap."""
    started=time.monotonic();peak=0;stop=None
    with log_path.open('xb') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        while child.poll() is None:
            try:
                process=psutil.Process(child.pid);rss=process.memory_info().rss;peak=max(peak,rss)
                stop=memory_stop(rss)
            except psutil.NoSuchProcess:continue
            if (root/'STOP').exists():stop='user stop file'
            if shutil.disk_usage(root).free<2**30:stop='disk reserve'
            if time.monotonic()-started>300:stop='compile preflight wall limit'
            if stop:child.kill();break
            time.sleep(.25)
        rc=child.wait()
    return dict(returncode=rc,stop=stop,peak_rss_bytes=peak,wall_seconds=time.monotonic()-started,proofs_generated=0)


def execute_extended(root,scope=None):
    require_execution(json.loads(POLICY.read_text()),target='laptop')
    os.environ['COMMITJL_FINAL_CAMPAIGN_ROOT']=str(root)
    from campaigns.main_rank79 import prepare_case
    from campaigns.referee_round3_remote import prepare,execute
    from campaigns.referee_final_local import backend_binary
    plan=json.loads((root/'plan.json').read_text());results=[]
    if scope is not None and (root/'extended_status.json').exists():
        results=json.loads((root/'extended_status.json').read_text())['cases']
    amendment=root/'extended_stress_amendment.json'
    extra=json.loads(amendment.read_text())['additional_cases'] if amendment.exists() else []
    for case in plan['extended_proof_cases']+extra:
        # The outer coordinator monitors and kills only this process tree.
        # Resource failures are retained as exclusions, not favorable outcomes.
        require_execution(json.loads(POLICY.read_text()),target='laptop')
        name=case['id'];target=root/'extended'/name
        previous=next((r for r in results if r['case']==name),None)
        if previous is not None:
            if previous['status']=='complete':
                if sha(target/'plan.json')!=previous['plan_sha256'] or not json.loads((target/'queue.json').read_text()).get('complete'):
                    raise RuntimeError('Earlier completed cohort does not match its receipt: '+name)
            continue
        if scope is not None:
            from campaigns.bounded_local_campaign import within_scope
            if not within_scope(case,scope):
                results.append(dict(case=name,status='deferred_by_user_scope',reason=scope['reason'],
                    native_results_retained=True,proofs_generated=0))
                save(root/'extended_status.json',dict(complete=False,cases=results));continue
        if shutil.disk_usage(root).free<3*2**30:
            results.append(dict(case=name,status='not_started_disk_reserve',free_bytes=shutil.disk_usage(root).free))
            save(root/'extended_status.json',dict(complete=False,cases=results));continue
        parents=prepare_case(root/'extended_inputs',root/'native',case,workers=plan['workers'])
        parent=parents['ordered_head'];parent.parent.mkdir(parents=True,exist_ok=True)
        receipt=parent.parent/'constraint_preflight.json'
        command=[str(REPO/'code/zk/bin/audit-shape-inspect-20261026.exe'),'-input',str(parent/'head_r63/input.json'),
                '-auxiliary',str(parent/'auxiliary_basis.json'),'-out',str(receipt)]
        if scope is not None:
            check=bounded_preflight(command,root,parent.parent/'constraint_preflight.log')
            save(parent.parent/'constraint_preflight.resource.json',check)
            if check['stop']:
                if check['stop']=='user stop file':raise RuntimeError('User stop file present')
                results.append(dict(case=name,status='not_started_preflight_resource_guard',resource=check))
                save(root/'extended_status.json',dict(complete=False,cases=results));continue
            if check['returncode']:raise RuntimeError('Constraint preflight failed: '+name)
        else:
            with (parent.parent/'constraint_preflight.log').open('xb') as log:
                subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
        inspected=json.loads(receipt.read_text())
        if not all(v['within_guard'] for v in inspected['shapes']):
            results.append(dict(case=name,status='not_started_constraint_guard',preflight_sha256=sha(receipt)))
            save(root/'extended_status.json',dict(complete=False,cases=results));continue
        controls=prepare(target,source=parents['ordered_head'].parent,repetitions=3,workers=plan['workers'],execution_target='laptop',
            phase_wall_limit_seconds=(72 if case['n']>=4096 else 12)*3600,long_catalog_phase=case['n']>=4096)
        # Independent fresh keys and Full controls at each N and k. Never
        # relabel an old timing or reuse cached baseline proofs as generation.
        execute(target,REPO/plan['binary'],backend_binary(root,plan),execution_target='laptop')
        results.append(dict(case=name,status='complete',plan_sha256=sha(target/'plan.json')))
        save(root/'extended_status.json',dict(complete=False,cases=results))
    save(root/'extended_status.json',dict(complete=all(r['status']=='complete' for r in results),
        authorized_scheduling_complete=True,cases=results))
