# Versioned all-source coordinator: compiler resource ceiling only; proof relations unchanged.
"""Matched group-certificate timing, with prospective Full fallback generation.

Preserve every distinct proposed center set from the frozen source workload.
An identical deployed set is a public equality, not a proposed update. Full's
matched20-set baseline includes the old cost plus all19 new costs. No selection
uses held-out outcomes. All real proof work and unsuccessful attempts are charged.
"""
from pathlib import Path
import argparse,copy,json,os,subprocess,time
import numpy as np
import psutil
from campaigns.server_all_sources_full_20261101 import save,sha,resource_record,adopt_running_child,public_bundle_complete,run as run_full
from validation.check_group_fast_proofs import issue_challenge,export_public
from analysis.audit_local_fast_tradeoff import audit_case
from commitjl.group_moment_reference import prepare_groups
from commitjl.group_change_bounds import correction_bounds,exact_old_correction_bounds
from commitjl.reuse_certificates import certificate
from campaigns.proof_runtime import proof_environment,proof_workers,check_reference_runtime,launch_headroom,memory_stop,phase_limit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['parent','root','binary','smoke-receipt']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--proof-backbone-only',action='store_true',help='Verify compact evidence for a subsequent bounded refinement stage; do not present unresolved base evidence as a complete service comparison')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--exact-old',action='store_true');parser.add_argument('--compact',action='store_true');parser.add_argument('--certificate-scale-bits',type=int,choices=[8,10,14],default=14);parser.add_argument('--auxiliary-basis',type=Path);a=parser.parse_args()
    if a.compact:a.exact_old=True
    if a.auxiliary_basis and not a.compact:parser.error('auxiliary basis requires the compact profile')
    parent=a.parent.resolve();root=a.root.resolve();binary=a.binary.resolve();root.mkdir(parents=True,exist_ok=a.resume)
    parent_protocol=json.loads((parent/'protocol.json').read_text())
    workers=proof_workers(parent_protocol)
    if (parent/'status.json').exists():
        check_reference_runtime(json.loads((parent/'status.json').read_text()),parent_protocol)
    phase_wall_limit=phase_limit(parent_protocol)
    source=parent/'head_r63/input.json';inp=json.loads(source.read_text());model=inp['model'];old=model['centers'];q=16
    frozen_prediction_model=copy.deepcopy(model)
    auxiliary=None
    if a.auxiliary_basis:
        auxiliary=json.loads(a.auxiliary_basis.read_text());basis=auxiliary['certificate_basis']
        assert auxiliary['source_input_sha256']==sha(source) and auxiliary['basis_scale']==model['basis_scale']
        assert len(model['basis'])<len(basis)<=128 and basis[:len(model['basis'])]==model['basis']
        assert all(len(row)==model['d'] and all(type(v) is int and abs(v)<=model['basis_scale'] for v in row) for row in basis)
        assert auxiliary['training_sha256']==json.loads((parent/'protocol.json').read_text())['training_sha256']
        model['basis']=basis
    if a.certificate_scale_bits!=14:
        scale=2**a.certificate_scale_bits;model['basis']=np.rint(np.asarray(model['basis'],float)*scale/model['basis_scale']).astype(np.int64).tolist();model['basis_scale']=scale
    proposal_ids=[i for i,c in enumerate(inp['reuse_centers']) if c!=old]
    task=dict(reuse_centers=[inp['reuse_centers'][i] for i in proposal_ids],queries=inp['queries'],comparison_base=old)
    assert len(task['reuse_centers'])==len({json.dumps(c) for c in task['reuse_centers']})
    smoke=json.loads(a.smoke_receipt.read_text());assert smoke['complete'] and smoke['binary_sha256']==sha(binary)
    if a.exact_old:assert smoke.get('exact_old_profile') or (a.compact and smoke.get('compact_profile'))
    if a.compact:assert smoke.get('compact_profile')
    metadata=dict(binary_sha256=sha(binary),source_sha256=sha(source),q=q,proposal_source_indices=proposal_ids,
        scope='Full fixed committed catalog. All19 distinct nonidentical source updates; old public identity omitted. No selection by audit outcomes.',
        comparison='Cost-change audit against reusable public-center Full. The group relation certifies an interval for the exact cost change, not individual approximate assignments.',
        fitting_seconds=parent_protocol['fitting_seconds'],smoke_sha256=sha(a.smoke_receipt))
    if 'phase_wall_limit_seconds' in parent_protocol:metadata['phase_wall_limit_seconds']=phase_wall_limit
    if workers!=2:metadata['proof_runtime_workers']=workers
    if a.exact_old:metadata.update(profile='group-exact-old-v2',preprocessing='Every row proves its exact old nearest group. A<0 certifies improvement using aggregate proof only; all other updates use new-switch bounds. Four cold key setups are charged even when switch proofs are not needed.',interval_scope='Assigned-upper fast path uses conservative global lower bound; it certifies direction, not a narrow cost interval.')
    if a.compact:metadata.update(profile='group-compact-v3',preprocessing='Exact old membership is authenticated inside each head/energy leaf. Each update proof combines moment-link authentication with the assigned change. Three cold setups. All approved updates use full switching bounds; no broad direction-only shortcut.',interval_scope='Same exact-old switching algebra and original certificate precision; independent numerical reference checks all endpoints.')
    if a.compact or a.certificate_scale_bits!=14:metadata.update(certificate_scale_bits=a.certificate_scale_bits,frozen_prediction_basis_scale=frozen_prediction_model['basis_scale'],prediction_model_unchanged=True)
    if auxiliary:
        metadata.update(auxiliary_basis_sha256=sha(a.auxiliary_basis),certificate_rank=len(model['basis']),frozen_prediction_rank=len(frozen_prediction_model['basis']),auxiliary_fitting_seconds=auxiliary['fitting_seconds'],auxiliary_scope='Standalone higher-rank cost-change certificate. Prediction basis stays frozen. Separate fresh setup/proofs; no adaptive-rank sharing or joint63/79 proof-cost claim.')
        metadata['fitting_seconds']+=auxiliary['fitting_seconds']
    if a.certificate_scale_bits!=14:metadata['interval_scope']='Auxiliary certificate basis rounded from the frozen prediction basis; original prediction outputs unchanged. Rounded-basis correction retained. Independent original-space Full truth checks every endpoint.'
    if len(proposal_ids)!=19:metadata['scope']=f'Full fixed committed catalog. All{len(proposal_ids)} distinct nonidentical source updates; old public identity omitted. No update selection by outcomes.'
    if a.resume:
        assert json.loads((root/'protocol.json').read_text())==metadata
        assert json.loads((root/'input.json').read_text())==inp and json.loads((root/'approved.json').read_text())==task
    else:
        save(root/'protocol.json',metadata);save(root/'input.json',inp);save(root/'approved.json',task)
        if a.compact or a.certificate_scale_bits!=14:save(root/'frozen_prediction_model.json',frozen_prediction_model)
        if auxiliary:save(root/'auxiliary_basis_source.json',auxiliary)
    status=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else dict(started_unix=time.time(),resources={})
    status.pop('error',None)
    status.update(complete=False,status='running',paused=False);save(root/'status.json',status)
    env=proof_environment(parent_protocol)
    bundle=root/'bundle'
    def invoke(phase,target):
        record=root/(phase+'.resource.json');prior=json.loads(record.read_text()) if record.exists() else {}
        result=target/(phase+'.json')
        if prior.get('complete') and result.exists() and json.loads(result.read_text()).get('complete'):
            status['resources'][phase]=prior;return
        command=[str(binary),'-phase',phase,'-instance',str(root/'input.json'),'-out',str(target),'-q',str(q),'-reps','1','-warmup','0','-max-constraints','24000000','-trust',str(root/'owner_pin.json'),'-public-task',str(root/'approved.json')]
        attempts=prior.get('attempts',[]);adopted=adopt_running_child(prior,command)
        if adopted is None:
            # Wait before allocating a new proving/setup process. Do not close
            # the user's applications or lower the machine's RAM reserve.
            # On a retry, budget the observed child peak plus the unchanged
            # reserve and headroom; a flat 1500 MiB was too low for this phase.
            required=launch_headroom(prior)
            while psutil.virtual_memory().available<required or memory_stop(0):
                status.update(status='waiting_for_ram',active_phase=phase,child_pid=None,required_available_bytes=required,available_bytes=psutil.virtual_memory().available)
                save(root/'status.json',status)
                time.sleep(15)
            status.update(status='running',required_available_bytes=required,available_bytes=psutil.virtual_memory().available)
        with (root/(phase+'.log')).open('a',encoding='utf-8') as handle:
            if adopted is None:
                child=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                attempt=dict(complete=False,running=True,pid=child.pid,started_unix=time.time(),wall_seconds=0.,peak_rss_bytes=0);attempts.append(attempt)
                start=time.monotonic()
            else:
                child=adopted;attempt=attempts[-1];start=time.monotonic()-(time.time()-attempt['started_unix'])
            status.update(active_phase=phase,child_pid=child.pid);save(root/'status.json',status);last=0.;stop=None
            def poll():
                if adopted is None:return child.poll()
                try:
                    value=child.wait(timeout=0)
                    if value is None and not child.is_running():return 0 if result.exists() and json.loads(result.read_text()).get('complete') else 1
                    return value
                except psutil.TimeoutExpired:return None
                except psutil.NoSuchProcess:return 0 if result.exists() and json.loads(result.read_text()).get('complete') else 1
            while poll() is None:
                try:attempt['peak_rss_bytes']=max(attempt['peak_rss_bytes'],psutil.Process(child.pid).memory_info().rss)
                except psutil.NoSuchProcess:pass
                stop=memory_stop(attempt['peak_rss_bytes'])
                if time.monotonic()-start>phase_wall_limit:stop='configured phase wall guard' if 'phase_wall_limit_seconds' in parent_protocol else 'four-hour phase guard'
                if stop:
                    child.terminate()
                    try:child.wait(timeout=15)
                    except (psutil.TimeoutExpired,subprocess.TimeoutExpired):child.kill();child.wait()
                    break
                if time.monotonic()-last>5:
                    attempt['wall_seconds']=time.monotonic()-start;resource=resource_record(attempts);save(record,resource)
                    status.update(active_elapsed=attempt['wall_seconds']);status['resources'][phase]=resource;save(root/'status.json',status);last=time.monotonic()
                time.sleep(1)
        code=poll();data=json.loads(result.read_text()) if result.exists() else {}
        attempt.update(complete=code==0 and data.get('complete',False),running=False,wall_seconds=time.monotonic()-start,returncode=code,stop=stop)
        resource=resource_record(attempts);save(record,resource);status['resources'][phase]=resource;save(root/'status.json',status)
        if not resource['complete']:raise RuntimeError(f'{phase} stopped; preserve cache and inspect {root/(phase+".log")}')
        print(json.dumps(dict(phase=phase,wall_seconds=resource['wall_seconds'],complete=True)),flush=True)
    try:
        if a.compact:invoke('group_compact_prepare',bundle)
        else:invoke('prepare',bundle);invoke('group_commit_exact' if a.exact_old else 'group_commit',bundle)
        if not (root/'owner_pin.json').exists():issue_challenge(bundle,root/'owner_pin.json',q)
        invoke('group_keys',bundle)
        # The setup authority pins its four generated keys before proof workers.
        pin=json.loads((root/'owner_pin.json').read_text());trust=json.loads((bundle/'trusted.json').read_text())
        if pin['owner']['verifying_key_sha256']:assert pin['owner']==trust,'trusted setup changed after owner pin'
        else:pin['owner']=trust;save(root/'owner_pin.json',pin)
        for phase in (['group_rows','group_query'] if a.compact else ['group_rows','group_link','group_query']):invoke(phase,bundle)
        public=root/'public'
        if not public_bundle_complete(public):
            # An interrupted export can leave its directory without a complete
            # manifest. Retain that evidence, then export into a clean path.
            if public.exists():public.rename(public.with_name(public.name+f'_partial_{time.time_ns()}'))
            export_public(bundle,public)
        invoke('group_verify',public)
        status['active_phase']='independent_native_check';save(root/'status.json',status);native_start=time.monotonic()
        # Independent reference uses the same fixed approximate group labels;
        # Full truth is consulted only after the public decisions were verified.
        rows=np.asarray(inp['x'],dtype=np.int64)
        if a.exact_old:labels=((rows[:,None,:]-np.asarray(old)[None,:,:])**2).sum(2).argmin(1).tolist()
        else:labels=certificate(model,rows,old)['labels']
        groups=prepare_groups(rows,labels,model['k'])
        records=json.loads((bundle/'group_query_records.json').read_text());decisions=json.loads((public/'group_verified_decisions.json').read_text())
        def cost(centers):
            c=np.asarray(centers,dtype=np.int64);return int(((rows[:,None,:]-c[None,:,:])**2).sum(2).min(1).sum())
        old_cost=cost(old);checked=[];fallback_ids=[]
        for i,(record,decision,centers) in enumerate(zip(records,decisions,task['reuse_centers'])):
            ref=(exact_old_correction_bounds if a.exact_old else correction_bounds)(model,rows,groups,old,centers);truth=cost(centers)-old_cost
            if record.get('mode')=='assigned_upper_bound':
                assert a.exact_old and ref['assigned_scaled']<0
                expected_hi=ref['assigned_scaled'];expected_lo=expected_hi-4*model['n']*model['d']*(2**model['input_bits'])**2*ref['cost_denominator']
            else:expected_lo,expected_hi=ref['change_lower_scaled'],ref['change_upper_scaled']
            assert int(record['Lower'])==expected_lo and int(record['Upper'])==expected_hi
            assert int(decision['lower'])<=truth*int(decision['denominator'])<=int(decision['upper'])
            if decision['decision']=='inconclusive':fallback_ids.append(i)
            else:assert decision['decision']==('worsening' if truth>0 else 'improvement' if truth<0 else 'equal')
            checked.append(dict(set=i,source_set=proposal_ids[i],truth=truth,**{k:v for k,v in decision.items() if k!='set'}))
        save(root/'native_checks.json',dict(complete=True,elapsed_seconds=time.monotonic()-native_start,decisions=checked,fallback_ids=fallback_ids))
        if a.proof_backbone_only:
            assert a.compact,'Backbone-only execution requires compact refinement profile'
            save(root/'backbone_complete.json',dict(complete=True,scope='Verified compact backbone only. No claim of a completed fallback service.',pending_refinement_sets=fallback_ids,public_proofs=json.loads((public/'group_verify.json').read_text())['verified'],public_manifest_sha256=sha(public/'public_manifest.json')))
            status.update(backbone_complete=True,status='awaiting_refinement',child_pid=None)
            save(root/'status.json',status)
            return
        # Generate fresh exact proofs only for unresolved updates, never reuse
        # already generated baseline proof files as if they were generation.
        fallback_wall=0.;fallback_proofs=0
        if fallback_ids:
            status['active_phase']='prospective_full_fallback';save(root/'status.json',status)
            fb=root/'fallback';fb.mkdir(exist_ok=True);fi=copy.deepcopy(inp)
            fi['model'].update(method='full',basis=[],seed='');fi['queries']=[list(range(i,i+8)) for i in range(0,model['n'],8)]
            fi['reuse_centers']=[old]+[task['reuse_centers'][i] for i in fallback_ids]
            if (fb/'full/input.json').exists():assert json.loads((fb/'full/input.json').read_text())==fi
            else:save(fb/'full/input.json',fi)
            protocol=json.loads((parent/'protocol.json').read_text());protocol.update(fitting_seconds=0.,fallback_generated_after_verified_inconclusive=True,selection_rule='All public inconclusive outcomes; no Full-truth selection',source_fallback_sets=fallback_ids)
            if not (fb/'protocol.json').exists():save(fb/'protocol.json',protocol)
            run_full(fb,binary,['full']);fallback,_,_=audit_case(fb,'full');fallback_wall=fallback['active_phase_wall_seconds'];fallback_proofs=fallback['public_proofs']
        resources=list(status['resources'].values());active=sum(v['wall_seconds'] for v in resources)+metadata['fitting_seconds']+fallback_wall
        full,_,_=audit_case(parent,'full')
        summary=dict(complete=True,proposal_updates=len(task['reuse_centers']),conclusive_without_fallback=len(task['reuse_centers'])-len(fallback_ids),
            fallback_updates=len(fallback_ids),fresh_fallback_proofs=fallback_proofs,fallback_active_seconds=fallback_wall,
            group_public_proofs=json.loads((public/'group_verify.json').read_text())['verified'],
            active_phase_seconds_including_fitting_and_fallback=active,full_reference_active_seconds=full['active_phase_wall_seconds'],
            full_over_group_active_phase=full['active_phase_wall_seconds']/active,all_intervals_contain_full=True,no_wrong_decisions=True,
            interrupted=any(v.get('resumed') or v.get('interrupted_checkpoint') for v in resources),
            timing_caveat='Serial laptop measurements across changing clock/load conditions, one fit and one repetition. Active phase total includes all attempts, setup, native preparation and fresh public replay. Export/file-monitor gaps are outside this phase sum. Native independent validation is outside deployment time.',
            statement=metadata['comparison'],source_contract=metadata['scope'])
        if a.exact_old:
            summary.update(profile='group-exact-old-v2',assigned_upper_fast_updates=sum(v.get('mode')=='assigned_upper_bound' for v in records),switch_updates=sum(v.get('mode')=='switch' for v in records),interval_scope=metadata['interval_scope'])
        if a.compact:summary.update(profile='group-compact-v3',certificate_precision_unchanged=a.certificate_scale_bits==14,certificate_scale_bits=a.certificate_scale_bits,prediction_model_unchanged=True,three_cold_key_setups=True)
        if auxiliary:summary.update(certificate_rank=len(model['basis']),frozen_prediction_rank=len(frozen_prediction_model['basis']),auxiliary_scope=metadata['auxiliary_scope'])
        save(root/'checked_comparison.json',summary);status.update(complete=True,status='complete',finished_unix=time.time());save(root/'status.json',status)
        print(json.dumps(summary),flush=True)
    except BaseException as error:
        status.update(status='stopped',error=str(error));save(root/'status.json',status);raise


if __name__=='__main__':main()
