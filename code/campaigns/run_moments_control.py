"""Projection-free authenticated moments baseline plus paid exact fallback."""
from pathlib import Path
import argparse,copy,json,os,subprocess,time
import numpy as np
import psutil
from campaigns.local_fast_tradeoff import save,sha,resource_record,public_bundle_complete,run as run_full
from campaigns.proof_runtime import proof_environment,proof_workers,check_reference_runtime,launch_headroom,memory_stop
from validation.check_group_fast_proofs import issue_challenge
from validation.check_moments_only_proofs import export_public
from analysis.audit_local_fast_tradeoff import audit_case


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','parent','binary','smoke-receipt']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args();root=a.root.resolve();parent=a.parent.resolve();binary=a.binary.resolve()
    root.mkdir(parents=True,exist_ok=a.resume);protocol=read(parent/'protocol.json');check_reference_runtime(read(parent/'status.json'),protocol)
    smoke=read(a.smoke_receipt);assert smoke['complete'] and smoke['projection_free'] and smoke['binary_sha256']==sha(binary)
    inp=read(parent/'head_r63/input.json');old=inp['model']['centers'];inp['model'].update(method='affine',basis=[],seed='projection-free moments baseline')
    task=dict(queries=inp['queries'],comparison_base=old,reuse_centers=[c for c in inp['reuse_centers'] if c!=old])
    metadata=dict(profile='group-moments-only-v1',source_sha256=sha(parent/'head_r63/input.json'),binary_sha256=sha(binary),smoke_sha256=sha(a.smoke_receipt),proof_runtime_workers=proof_workers(protocol),selection='Every distinct nonidentity source update; no outcome filtering.',service='Authenticated best-target group upper bound; negative certifies improvement. Every inconclusive triggers newly generated Full old+candidate proofs. No compact statistics or individual predictions.')
    if a.resume:assert read(root/'protocol.json')==metadata and read(root/'input.json')==inp and read(root/'approved.json')==task
    else:save(root/'protocol.json',metadata);save(root/'input.json',inp);save(root/'approved.json',task)
    state=read(root/'status.json') if (root/'status.json').exists() else dict(resources={},started_unix=time.time())
    state.pop('error',None)
    state.update(complete=False,status='running',paused=False);save(root/'status.json',state);bundle=root/'bundle';env=proof_environment(protocol)
    def invoke(phase,target):
        record=root/(phase+'.resource.json');prior=read(record) if record.exists() else {};result=target/(phase+'.json')
        if prior.get('complete') and result.exists() and read(result).get('complete'):state['resources'][phase]=prior;return
        attempts=prior.get('attempts',[])
        if attempts and attempts[-1].get('running'):
            previous=attempts[-1]
            try:
                process=psutil.Process(previous['pid'])
                if str(binary) in process.cmdline():raise RuntimeError('existing moments worker is still running; do not duplicate it')
            except psutil.NoSuchProcess:pass
            previous['running']=False;previous['stop']='interrupted before resume'
        required=launch_headroom(prior)
        while psutil.virtual_memory().available<required or memory_stop(0):
            state.update(status='waiting_for_ram',active_phase=phase,child_pid=None);save(root/'status.json',state);time.sleep(15)
        command=[str(binary),'-phase',phase,'-instance',str(root/'input.json'),'-out',str(target),'-q','16','-max-constraints','1600000','-trust',str(root/'owner_pin.json'),'-public-task',str(root/'approved.json')]
        with (root/(phase+'.log')).open('a',encoding='utf-8') as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));started=time.monotonic();attempt=dict(pid=child.pid,started_unix=time.time(),running=True,complete=False,wall_seconds=0.,peak_rss_bytes=0);attempts.append(attempt)
            state.update(status='running',active_phase=phase,child_pid=child.pid);save(root/'status.json',state);last=0.;stop=None
            while child.poll() is None:
                try:attempt['peak_rss_bytes']=max(attempt['peak_rss_bytes'],psutil.Process(child.pid).memory_info().rss)
                except psutil.NoSuchProcess:pass
                stop=memory_stop(attempt['peak_rss_bytes'])
                if time.monotonic()-started>protocol.get('phase_wall_limit_seconds',12*3600):stop='phase wall guard'
                if stop:child.terminate();child.wait(timeout=20);break
                if time.monotonic()-last>5:
                    attempt['wall_seconds']=time.monotonic()-started;save(record,resource_record(attempts));last=time.monotonic()
                time.sleep(.5)
        attempt.update(complete=child.returncode==0 and result.exists() and read(result).get('complete',False),running=False,wall_seconds=time.monotonic()-started,returncode=child.returncode,stop=stop);resource=resource_record(attempts);save(record,resource);state['resources'][phase]=resource;save(root/'status.json',state)
        if not resource['complete']:raise RuntimeError(f'{phase} stopped; keep caches and costs')
    try:
        invoke('moments_prepare',bundle)
        if not (root/'owner_pin.json').exists():issue_challenge(bundle,root/'owner_pin.json',16)
        invoke('moments_keys',bundle);pin=read(root/'owner_pin.json');trust=read(bundle/'trusted.json')
        if pin['owner']['verifying_key_sha256']:assert pin['owner']==trust
        else:pin['owner']=trust;save(root/'owner_pin.json',pin)
        for phase in ['moments_rows','moments_query']:invoke(phase,bundle)
        public=root/'public'
        if not public_bundle_complete(public):
            if public.exists():public.rename(public.with_name(public.name+f'_partial_{time.time_ns()}'))
            export_public(bundle,public)
        invoke('moments_verify',public)
        x=np.asarray(inp['x'],dtype=np.int64);old=np.asarray(old,dtype=np.int64);old_dist=((x[:,None,:]-old[None,:,:])**2).sum(2);labels=old_dist.argmin(1);old_cost=int(old_dist.min(1).sum())
        records=read(bundle/'moments_query_records.json');decisions=read(public/'moments_verified_decisions.json');checked=[];fallback_ids=[]
        for index,centers in enumerate(task['reuse_centers']):
            centers=np.asarray(centers,dtype=np.int64);new_dist=((x[:,None,:]-centers[None,:,:])**2).sum(2);upper=0
            for g in range(len(old)):
                mask=labels==g;upper+=int((new_dist[mask].sum(0)-old_dist[mask,g].sum()).min())
            truth=int(new_dist.min(1).sum())-old_cost;record=records[index];decision=decisions[index]['decision'];assert int(record['assigned_change'])==upper and int(record['Lower'])<=truth<=upper
            assert decision==('improvement' if upper<0 else 'inconclusive')
            if decision=='inconclusive':fallback_ids.append(index)
            checked.append(dict(set=index,truth=truth,upper=upper,decision=decision))
        save(root/'native_checks.json',dict(complete=True,decisions=checked,fallback_ids=fallback_ids))
        fallback_seconds=0.;fallback_proofs=0
        if fallback_ids:
            state.update(active_phase='prospective_full_fallback');save(root/'status.json',state);fb=root/'fallback';fb.mkdir(exist_ok=True);fi=read(parent/'full/input.json');fi['reuse_centers']=[old.tolist()]+[task['reuse_centers'][i] for i in fallback_ids]
            if (fb/'full/input.json').exists():assert read(fb/'full/input.json')==fi
            else:save(fb/'full/input.json',fi)
            fp=copy.deepcopy(protocol);fp.update(fitting_seconds=0.,fallback_generated_after_verified_inconclusive=True,source_fallback_sets=fallback_ids)
            if not (fb/'protocol.json').exists():save(fb/'protocol.json',fp)
            run_full(fb,binary,['full']);ft,_,_=audit_case(fb,'full');fallback_seconds=ft['active_phase_wall_seconds'];fallback_proofs=ft['public_proofs']
        total=sum(r['wall_seconds'] for r in state['resources'].values())+fallback_seconds;full,_,_=audit_case(parent,'full')
        summary=dict(complete=True,profile=metadata['profile'],proposal_updates=len(records),conclusive_without_fallback=len(records)-len(fallback_ids),fallback_updates=len(fallback_ids),fresh_fallback_proofs=fallback_proofs,fallback_active_seconds=fallback_seconds,group_public_proofs=read(public/'moments_verify.json')['verified'],active_phase_seconds_including_fitting_and_fallback=total,full_reference_active_seconds=full['active_phase_wall_seconds'],full_over_group_active_phase=full['active_phase_wall_seconds']/total,all_intervals_contain_full=True,no_wrong_decisions=True,prediction_agreement=None,assignment_certified_fraction=None,proof_runtime_workers=proof_workers(protocol),scope=metadata['service'])
        save(root/'checked_comparison.json',summary);state.update(complete=True,status='complete',child_pid=None,finished_unix=time.time());save(root/'status.json',state)
    except BaseException as error:
        state.update(status='stopped',error=str(error));save(root/'status.json',state);raise


if __name__=='__main__':main()
