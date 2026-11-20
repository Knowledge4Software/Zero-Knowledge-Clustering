# Versioned all-source coordinator: compiler resource ceiling only; proof relations unchanged.
"""Costed bounded refinement attached to a complete compact proof backbone.

The base proofs are reusable evidence, not regenerated for this extension.
Charge their measured preparation/query/replay cost, new refinement work and
any newly generated fallback. Never charge or silently reuse an older fallback.
"""
from pathlib import Path
import argparse, copy, json, os, shutil, subprocess, time
import numpy as np
import psutil
from campaigns.server_all_sources_full_20261101 import save, sha, resource_record, public_bundle_complete, run as run_full
from campaigns.proof_runtime import proof_environment, launch_headroom, check_reference_runtime, memory_stop
from analysis.audit_local_fast_tradeoff import audit_case
from commitjl.group_change_bounds import exact_old_correction_bounds
from commitjl.group_moment_reference import prepare_groups
from commitjl.jl_certificate_pilots import hint_order


def read(path): return json.loads(Path(path).read_text())


def run(base, parent, root, binary, smoke, resume=False, compare_full=True, reuse_key_bundle=None):
    root.mkdir(parents=True, exist_ok=resume)
    check=read(smoke);assert check['complete'] and check['binary_sha256']==sha(binary) and check['resume_keeps_proofs']
    protocol=read(parent/'protocol.json');base_protocol=read(base/'protocol.json')
    check_reference_runtime(base_protocol,protocol)
    base_resources=read(base/'status.json')['resources']
    assert all(base_resources[name]['complete'] for name in ['group_compact_prepare','group_keys','group_rows','group_query','group_verify'])
    assert public_bundle_complete(base/'public')
    metadata=dict(binary_sha256=sha(binary),base_binary_sha256=base_protocol['binary_sha256'],
        parent=str(parent),base=str(base),base_protocol_sha256=sha(base/'protocol.json'),
        base_owner_pin_sha256=sha(base/'owner_pin.json'),base_task_sha256=sha(base/'approved.json'),
        base_public_manifest_sha256=sha(base/'public/public_manifest.json'),smoke_sha256=sha(smoke),
        proof_runtime_workers=protocol.get('proof_runtime_workers',2),budget=16,
        ordering='Descending canonical row interval width, index tie breaks; stop on first conclusive interval.',
        timing='Measured reusable base component sum plus measured selection, lazy refinement setup, proof phase, full public replay and fresh residual Full fallback. Base work is shared, not a newly repeated end-to-end trial. Historical base fallback is excluded because this algorithm refines before fallback.',
        privacy='Public refinement indices and summed endpoint corrections add explicit output leakage. No aggregate-hiding or individual-assignment privacy claim is made.')
    if not compare_full:metadata['full_reference']='No matched Full timing at this request count; no Full speed ratio reported.'
    if reuse_key_bundle is not None:
        reuse_key_bundle=Path(reuse_key_bundle).resolve()
        assert not list(reuse_key_bundle.rglob('*.proof')),'Key-only reuse must not contain correction proofs'
        metadata['reused_key_bundle']=str(reuse_key_bundle)
        metadata['reused_key_trust_sha256']=sha(reuse_key_bundle/'trusted.json')
    if resume: assert read(root/'protocol.json')==metadata
    else: save(root/'protocol.json',metadata)
    state=read(root/'status.json') if (root/'status.json').exists() else dict(resources={},started_unix=time.time())
    state.update(status='running',complete=False,pid=os.getpid(),paused=False);state.pop('error',None);save(root/'status.json',state)
    env=proof_environment(protocol);bundle=root/'bundle'
    if reuse_key_bundle is not None and not bundle.exists():shutil.copytree(reuse_key_bundle,bundle)
    bundle.mkdir(exist_ok=True)
    def invoke(phase,target,source):
        path=root/(phase+'.resource.json');old=read(path) if path.exists() else {}
        if old.get('complete') and (target/(phase+'.json')).exists() and read(target/(phase+'.json'))['complete']:
            state['resources'][phase]=old;return
        required=launch_headroom(old)
        while psutil.virtual_memory().available<required or memory_stop(0):
            state.update(status='waiting_for_ram',active_phase=phase,child_pid=None,available_bytes=psutil.virtual_memory().available,required_available_bytes=required);save(root/'status.json',state);time.sleep(5)
        args=[str(binary),'-phase',phase,'-out',str(target),'-trust',str(base/'owner_pin.json'),
            '-public-task',str(base/'approved.json'),'-refinement-base',str(source),
            '-refinement-trust',str(root/'owner_refinement_pin.json'),'-refinement-request',str(root/'requests.json'),'-max-constraints','24000000']
        attempts=old.get('attempts',[])
        if attempts and attempts[-1].get('running') and psutil.pid_exists(attempts[-1]['pid']):
            raise RuntimeError('Prior refinement child is live; inspect instead of duplicating it')
        with (root/(phase+'.log')).open('a',encoding='utf-8') as log:
            child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            started=time.monotonic();attempt=dict(pid=child.pid,started_unix=time.time(),running=True,complete=False,wall_seconds=0.,peak_rss_bytes=0);attempts.append(attempt)
            state.update(active_phase=phase,child_pid=child.pid,status='running');save(root/'status.json',state)
            stop=None;last=0.
            while child.poll() is None:
                try: attempt['peak_rss_bytes']=max(attempt['peak_rss_bytes'],psutil.Process(child.pid).memory_info().rss)
                except psutil.NoSuchProcess: pass
                stop=memory_stop(attempt['peak_rss_bytes'])
                if time.monotonic()-started>43200:stop='phase wall guard'
                attempt['wall_seconds']=time.monotonic()-started
                if stop:
                    child.terminate();child.wait(timeout=15);break
                if time.monotonic()-last>5:
                    save(path,resource_record(attempts));state['resources'][phase]=resource_record(attempts);save(root/'status.json',state);last=time.monotonic()
                time.sleep(1)
        result=read(target/(phase+'.json')) if (target/(phase+'.json')).exists() else {}
        attempt.update(running=False,complete=child.returncode==0 and result.get('complete',False),wall_seconds=time.monotonic()-started,returncode=child.returncode,stop=stop)
        value=resource_record(attempts);save(path,value);state['resources'][phase]=value;save(root/'status.json',state)
        if not value['complete']:raise RuntimeError(f'{phase} failed: {stop}; preserve logs and cache')
    try:
        if not (root/'requests.json').exists():
            start=time.monotonic();task=read(base/'approved.json');initial=read(base/'public/group_verified_decisions.json');orders=[[] for _ in initial]
            if any(decision['decision']=='inconclusive' for decision in initial):
                inp=read(base/'input.json');model=inp['model'];x=np.asarray(inp['x'],dtype=np.int64);old=np.asarray(task['comparison_base'],dtype=np.int64)
                dist=((x[:,None,:]-old[None,:,:])**2).sum(2);groups=dist.argmin(1);prepared=prepare_groups(x,groups.tolist(),len(old))
                for index,(centers,decision) in enumerate(zip(task['reuse_centers'],initial)):
                    if decision['decision']!='inconclusive':continue
                    b=exact_old_correction_bounds(model,x,prepared,old,centers)
                    assert int(decision['lower'])==b['change_lower_scaled'] and int(decision['upper'])==b['change_upper_scaled']
                    orders[index]=hint_order(b,None,'width')[:16]
            save(root/'requests.json',dict(sets=orders));save(root/'selection.json',dict(complete=True,wall_seconds=time.monotonic()-start,native_only_selection=True,oracle_values_not_used_for_ordering=True))
        task=read(base/'approved.json');initial=read(base/'public/group_verified_decisions.json')
        needed=any(r['decision']=='inconclusive' for r in initial)
        if needed:
            invoke('refinement_setup',bundle,base/'public')
            setup=read(bundle/'trusted.json')
            if (root/'owner_refinement_pin.json').exists():assert read(root/'owner_refinement_pin.json')==setup
            else:save(root/'owner_refinement_pin.json',setup)
            invoke('refinement_prove',bundle,base/'bundle')
            public=root/'public'
            if not public_bundle_complete(public):
                if public.exists():public.rename(public.with_name('public_partial_'+str(time.time_ns())))
                public.mkdir();files=['refinement_records.json','keys/group_refinement_1.vk']+[p.relative_to(bundle).as_posix() for p in (bundle/'refinement').glob('*.proof')]
                for name in files:
                    target=public/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(bundle/name,target)
                save(public/'public_manifest.json',dict(sha256={name:sha(public/name) for name in files}))
            invoke('refinement_verify',public,base/'public')
            decisions=read(public/'refinement_verified_decisions.json');new_proofs=sum(r['refined_rows'] for r in decisions)
        else:
            decisions=[dict(**r,refined_rows=0) for r in initial];new_proofs=0
        native=read(base/'native_checks.json')['decisions'];fallback_ids=[]
        for r,truth in zip(decisions,native):
            value=truth['truth']*int(r['denominator']);assert int(r['lower'])<=value<=int(r['upper'])
            expected='worsening' if value>0 else ('improvement' if value<0 else 'equal')
            assert r['decision'] in ['inconclusive',expected]
            if r['decision']=='inconclusive':fallback_ids.append(r['set'])
        fallback_seconds=0.;fallback_proofs=0
        if fallback_ids:
            state['active_phase']='prospective_full_fallback';save(root/'status.json',state)
            fb=root/'fallback';fb.mkdir(exist_ok=True);full_input=read(parent/'full/input.json');full_input['reuse_centers']=[task['comparison_base']]+[task['reuse_centers'][i] for i in fallback_ids]
            save(fb/'full/input.json',full_input);fp=copy.deepcopy(protocol);fp.update(fitting_seconds=0.,fallback_generated_after_verified_inconclusive=True,source_fallback_sets=fallback_ids);save(fb/'protocol.json',fp)
            run_full(fb,binary,['full']);f,_,_=audit_case(fb,'full');fallback_seconds=f['active_phase_wall_seconds'];fallback_proofs=f['public_proofs']
        full=None
        if compare_full:full,_,_=audit_case(parent,'full')
        base_seconds=sum(v['wall_seconds'] for v in base_resources.values())+base_protocol['fitting_seconds']
        extra_seconds=sum(v['wall_seconds'] for v in state['resources'].values())+read(root/'selection.json')['wall_seconds']
        total=base_seconds+extra_seconds+fallback_seconds
        result=dict(complete=True,all_intervals_contain_full=True,no_wrong_decisions=True,decisions=decisions,
            proposal_updates=len(decisions),conclusive_without_fallback=len(decisions)-len(fallback_ids),fallback_updates=len(fallback_ids),
            new_refinement_proofs=new_proofs,base_public_proofs=read(base/'public/group_verify.json')['verified'],fresh_fallback_proofs=fallback_proofs,
            base_active_seconds=base_seconds,refinement_active_seconds=extra_seconds,fallback_active_seconds=fallback_seconds,
            active_phase_seconds_including_fitting_and_fallback=total,full_reference_active_seconds=(full['active_phase_wall_seconds'] if full else None),full_over_group_active_phase=(full['active_phase_wall_seconds']/total if full else None),
            timing_scope=metadata['timing'])
        save(root/'checked_comparison.json',result);state.update(complete=True,status='complete',child_pid=None,finished_unix=time.time());save(root/'status.json',state);print(json.dumps(result),flush=True)
    except BaseException as e:
        state.update(status='stopped',error=str(e),child_pid=None);save(root/'status.json',state);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['base','parent','root','binary','smoke-receipt']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--no-full-reference',action='store_true');p.add_argument('--reuse-key-bundle',type=Path);a=p.parse_args();run(a.base.resolve(),a.parent.resolve(),a.root.resolve(),a.binary.resolve(),a.smoke_receipt.resolve(),a.resume,not a.no_full_reference,a.reuse_key_bundle)
