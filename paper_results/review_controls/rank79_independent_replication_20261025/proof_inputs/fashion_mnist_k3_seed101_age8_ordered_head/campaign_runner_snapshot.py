"""Resumable Windows laptop trade-off campaign, matched original-space Full.

All distinct training-only Lloyd iterates1..24 are retained, never selected by
audit outcomes. Head ranks39/63 are compared with equally reusable Full.
"""
from pathlib import Path
import argparse,copy,hashlib,json,os,subprocess,sys,time
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits
from data.loaders import load_dataset
from experiments.review_update_decisions import lloyd_snapshots,array_sha
from commitjl.enriched_jl import enrich_centroid_basis
from commitjl.protected_jl import centroid_basis
from commitjl.reuse_certificates import certificate,integer_stats
from commitjl.relative_certificates import relative_certificate
from experiments.review_controls import public_export

ROOT=Path(__file__).resolve().parents[2]


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
    tmp.write_text(json.dumps(data,indent=2),encoding='utf-8')
    started=time.monotonic()
    try:
        while True:
            try:
                os.replace(tmp,path);return
            except PermissionError:
                # Windows readers/antivirus can briefly deny FILE_SHARE_DELETE.
                if time.monotonic()-started>10:raise
                time.sleep(.05)
    finally:
        if tmp.exists():tmp.unlink()


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def resource_record(attempts):
    """Keep work from every invocation; resuming must not reset its cost."""
    return dict(complete=bool(attempts and attempts[-1].get('complete')),
        returncode=attempts[-1].get('returncode') if attempts else None,
        wall_seconds=sum(x['wall_seconds'] for x in attempts),
        peak_rss_bytes=max((x.get('peak_rss_bytes',0) for x in attempts),default=0),
        attempts=attempts,resumed=len(attempts)>1,
        timing_basis='Sum of active invocation wall times; interrupted attempts retain their last checkpoint. Offline gaps excluded.',
        interrupted_checkpoint=any(x.get('running') for x in attempts[:-1]))


def public_bundle_complete(path):
    manifest=path/'public_manifest.json'
    if not manifest.exists():return False
    data=json.loads(manifest.read_text())
    return all((path/name).is_file() and sha(path/name)==digest for name,digest in data['sha256'].items())


def adopt_running_child(previous,command):
    attempts=previous.get('attempts',[])
    if not attempts or not attempts[-1].get('running'):return None
    attempt=attempts[-1]
    try:child=psutil.Process(attempt['pid'])
    except psutil.NoSuchProcess:return None
    if abs(child.create_time()-attempt['started_unix'])>10 or child.cmdline()!=command:
        raise RuntimeError('Saved live child identity differs; refusing to launch a duplicate phase.')
    return child


def prepare(root,n):
    root.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();dataset='mnist';k=5;seed=11
    with threadpool_limits(limits=2):
        source=load_dataset(dataset,n_max=5000,seed=0);x=np.asarray(source['X'],float);x=x[np.linalg.norm(x,axis=1)>0]
        idx=np.random.default_rng(741+seed).permutation(len(x));cut=int(.6*len(x));train=x[idx[:cut]]
        scale=4095/float(np.abs(train).max());training=np.rint(train*scale).astype(np.int64)
        audit=np.clip(np.rint(x[idx[cut:cut+1024]]*scale),-4096,4096).astype(np.int64)
        trace=lloyd_snapshots(training,k,seed);old=trace[1]
        start=time.monotonic();basis=enrich_centroid_basis(training.astype(float),old,extra=63-centroid_basis(old).shape[1],random_state=seed,preserve_pca_order=True)[0].T
        fit_seconds=time.monotonic()-start;basis=np.rint(basis*16384).astype(np.int64)
        sets=[];steps=[];seen=set();duplicates=[]
        for step,centers in trace.items():
            key=array_sha(centers)
            if key in seen:duplicates.append(step);continue
            seen.add(key);sets.append(centers.tolist());steps.append(step)
        quality=[]
        for rank in [39,63]:
            model=dict(version='bound-audit-v1',method='learned_energy',n=n,d=audit.shape[1],k=k,input_bits=12,basis_scale=16384,input_scale=str(scale),centers=old.tolist(),basis=basis[:rank].tolist(),tail_width=0,tail_rows=[],tail_signs=[],seed='')
            for size,rows in [('native1024',audit),('proof_catalog',audit[:n])]:
                for s,centers in enumerate(sets):
                    ref=certificate(model,rows,centers);rel=relative_certificate(model,rows,centers)
                    distances=((rows[:,None,:]-np.asarray(centers)[None,:,:])**2).sum(2);labels=distances.argmin(1);true=int(distances.min(1).sum())
                    assert all(not flag or lab==t for flag,lab,t in zip(rel['certified_flags'],rel['labels'],labels))
                    quality.append(dict(rank=rank,scope=size,set=s,lloyd_step=steps[s],n=len(rows),agreement_percent=100*np.mean(labels==rel['labels']),
                        relative_coverage_percent=100*rel['certified_count']/len(rows),absolute_coverage_percent=100*ref['certified_count']/len(rows),
                        absolute_cost_error_percent=100*abs(ref['psi']/16384**4/true-1),
                        cost_interval_width_percent=100*(ref['cost_upper_scaled']-ref['cost_lower_scaled'])/ref['cost_denominator']/true))
            inp=dict(model=model,x=audit[:n].tolist(),salt_x='1701',salt_z='1703',queries=[list(range(i,i+16)) for i in range(0,n,16)],reuse_centers=sets,comparison_base=old.tolist())
            save(root/f'head_r{rank}'/'input.json',inp)
        inp=copy.deepcopy(inp);inp['model'].update(method='full',basis=[],seed='');inp['queries']=[list(range(i,i+8)) for i in range(0,n,8)]
        save(root/'full'/'input.json',inp)
    pd.DataFrame(quality).to_csv(root/'quality.csv',index=False)
    protocol=dict(dataset=dataset,k=k,fit_seed=seed,n=n,native_n=1024,basis_scale=16384,ranks=[39,63],
        center_sets=len(sets),lloyd_steps=steps,duplicate_steps_cached=duplicates,
        family='All distinct training-only Lloyd snapshots1..24; deduplicated before auditing, with no accuracy-based update selection.',
        batches=dict(head=16,full=8),constraint_guard=1600000,reps=1,warmup=0,
        accuracy_requirement='User-selected99% agreement with exact quantized Full. Report every set and failures.',
        certificate='Tighter relative assignment count plus conservative absolute original-cost interval; no exact cost-change-decision guarantee.',
        timing='Actual phase workflow including compilation/setup, native answer, witness/proof, worker checks and fresh public-only replay. Projection fitting recorded separately and charged to head.',
        fitting_seconds=fit_seconds,input_preparation_seconds=time.monotonic()-started,
        training_sha256=array_sha(training),audit_sha256=array_sha(audit),proof_audit_sha256=array_sha(audit[:n]),
        scope='Exploratory native/actual-proof workload, not a universal speed or accuracy guarantee.',all_distinct_sets_retained=True)
    save(root/'protocol.json',protocol)
    print(json.dumps(dict(prepared=True,center_sets=len(sets),proof_catalog_n=n,quality=quality)),flush=True)


def run(root,binary,cases=None):
    cases=cases or ['head_r39','head_r63','full']
    protocol=json.loads((root/'protocol.json').read_text())
    pins={case:sha(root/case/'input.json') for case in cases}
    state=dict(complete=False,started=time.time(),binary_sha256=sha(binary),input_sha256=pins,
        protocol_sha256=sha(root/'protocol.json'),runner_sha256=sha(Path(__file__)),requested_cases=cases,cases={})
    if (root/'status.json').exists():
        old=json.loads((root/'status.json').read_text());assert old['binary_sha256']==state['binary_sha256'],'binary changed; use a new campaign directory'
        assert old.get('input_sha256')==pins and old.get('protocol_sha256')==state['protocol_sha256'],'campaign input/protocol changed'
        state=old
    state.update(paused=False,status='running')
    snapshot=root/'campaign_runner_snapshot.py'
    if not snapshot.exists():snapshot.write_bytes(Path(__file__).read_bytes())
    env=dict(os.environ,COMMITJL_RESUME_PROOFS='1',GOMAXPROCS='2',OMP_NUM_THREADS='2',GOMEMLIMIT='1200MiB')
    def execute(case,phase,batch,target,pin=None):
        source=root/case/'input.json';record=root/case/(phase+'.resource.json')
        name=f'{phase}_{batch}.json' if phase in ['relative_reuse','reuse','full_reuse','family_affine'] else phase+'.json'
        previous=json.loads(record.read_text()) if record.exists() else {}
        if record.exists() and (target/name).exists():
            result=json.loads((target/name).read_text())
            if previous.get('complete') and result.get('complete'):return previous
        attempts=previous.get('attempts',[])
        if previous and not attempts:attempts=[previous]
        command=[str(binary),'-phase',phase,'-instance',str(source),'-out',str(target),'-q',str(batch),'-reps','1','-warmup','0','-max-constraints','1600000']
        if phase=='verify':command+=['-trust',str(pin),'-public-task',str(root/case/'approved_public_task.json')]
        log=root/case/(phase+'.log');started=time.monotonic();peak=0;stop=None
        adopted=adopt_running_child(previous,command)
        state.update(active_case=case,active_phase=phase,active_started=time.time());save(root/'status.json',state)
        if adopted is None:
            required=max(1500*2**20,previous.get('peak_rss_bytes',0)+(650+256)*2**20)
            while psutil.virtual_memory().available<required:
                state.update(status='waiting_for_ram',child_pid=None,required_available_bytes=required,available_bytes=psutil.virtual_memory().available);save(root/'status.json',state);time.sleep(15)
            state.update(status='running');started=time.monotonic()
        with log.open('a',encoding='utf-8') as handle:
            if adopted is not None:
                child=adopted;attempt=attempts[-1];peak=attempt.get('peak_rss_bytes',0)
                started=time.monotonic()-(time.time()-attempt['started_unix'])
                attempt.setdefault('monitor_recoveries',[]).append(dict(recovered_unix=time.time(),runner_sha256=sha(Path(__file__))))
                state['active_started']=attempt['started_unix']
            else:
                flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
                child=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,env=env,creationflags=flags)
                attempt=dict(complete=False,running=True,pid=child.pid,started_unix=time.time(),wall_seconds=0.,peak_rss_bytes=0)
                attempts.append(attempt)
            save(record,resource_record(attempts))
            def poll():
                if adopted is None:return child.poll()
                try:
                    code=child.wait(timeout=0)
                    if code is None and not child.is_running():
                        return 0 if (target/name).exists() and json.loads((target/name).read_text()).get('complete') else 1
                    return code
                except psutil.TimeoutExpired:return None
                except psutil.NoSuchProcess:return 0 if (target/name).exists() and json.loads((target/name).read_text()).get('complete') else 1
            last=0
            while poll() is None:
                try:
                    proc=psutil.Process(child.pid);rss=proc.memory_info().rss;peak=max(peak,rss)
                    if rss>3.5*2**30 or psutil.virtual_memory().available<650*2**20:stop='RAM reserve'
                except psutil.NoSuchProcess:pass
                if time.monotonic()-started>4*3600:stop='phase wall guard'
                if stop:
                    child.terminate()
                    try:child.wait(timeout=15)
                    except (subprocess.TimeoutExpired,psutil.TimeoutExpired):child.kill();child.wait()
                    break
                if time.monotonic()-last>5:
                    attempt.update(wall_seconds=time.monotonic()-started,peak_rss_bytes=peak);save(record,resource_record(attempts))
                    state.update(child_pid=child.pid,active_elapsed=time.monotonic()-started,active_peak_rss=peak);save(root/'status.json',state);last=time.monotonic()
                time.sleep(1)
        result=json.loads((target/name).read_text()) if (target/name).exists() else {}
        returncode=poll()
        attempt.update(complete=bool(result.get('complete')) and returncode==0,running=False,returncode=returncode,wall_seconds=time.monotonic()-started,peak_rss_bytes=peak,stop=stop)
        resource=resource_record(attempts)
        save(record,resource)
        if not resource['complete']:raise RuntimeError(f'{case}/{phase} failed: {resource}; inspect {log}')
        print(json.dumps(dict(case=case,phase=phase,**resource)),flush=True);return resource
    for case in cases:
        case_started=time.monotonic()
        inp=json.loads((root/case/'input.json').read_text());batch=len(inp['queries'][0]) if case=='full' else 16;kind='full_reuse' if case=='full' else ('family_affine' if case=='family_affine' else 'relative_reuse')
        save(root/case/'approved_public_task.json',{k:inp[k] for k in ['queries','reuse_centers','comparison_base']})
        bundle=root/case/'bundle'
        resources=[]
        for phase in ['prepare']+([] if case=='full' else ['stage1'])+[kind]:resources.append(execute(case,phase,batch,bundle))
        pub=root/case/'public_bundle'
        if not public_bundle_complete(pub):
            if pub.exists():
                pub.rename(pub.with_name(pub.name+f'_partial_{time.time_ns()}'))
            public_export(bundle,pub)
        save(root/case/'owner_pin.json',json.loads((bundle/'trusted.json').read_text()))
        resources.append(execute(case,'verify',batch,pub,root/case/'owner_pin.json'))
        total=sum(x['wall_seconds'] for x in resources)+(protocol['fitting_seconds'] if case!='full' else 0)
        state['cases'][case]=dict(complete=True,wall_seconds=total,phase_resources=resources,
            enclosing_case_wall_seconds=time.monotonic()-case_started+(protocol['fitting_seconds'] if case!='full' else 0),
            resumed=any(x.get('resumed') for x in resources),
            interrupted_checkpoint=any(x.get('interrupted_checkpoint') for x in resources))
        save(root/'status.json',state)
    if 'full' in state['cases']:
        full=state['cases']['full']['wall_seconds']
        state['speed_ratios']={case:full/state['cases'][case]['wall_seconds'] for case in cases if case!='full'}
    state.update(complete=True,status='complete',finished=time.time());save(root/'status.json',state);print(json.dumps(state),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--n',type=int,default=32);p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--binary',type=Path,default=ROOT/'code/zk/bin/commitjl-bound.exe');p.add_argument('--cases',nargs='+',choices=['head_r39','head_r63','full','family_affine']);a=p.parse_args()
    if a.n<16 or a.n%16:raise ValueError('complete divisible-by16 proof catalog required')
    root=a.root.resolve()
    if a.prepare:prepare(root,a.n)
    if a.run:run(root,a.binary.resolve(),a.cases)


if __name__=='__main__':main()
