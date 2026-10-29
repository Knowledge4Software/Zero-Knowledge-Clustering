"""Serial, resumable main rank79 scale campaign after fresh-fit controls.

The plan is pinned before the new N256 audit is evaluated. Cases share existing
fits deliberately: scale is not claimed as independent model replication.
"""
from pathlib import Path
import argparse,copy,json,os,shutil,subprocess,sys,time
import numpy as np
from threadpoolctl import threadpool_limits
from commitjl.pilot_io import save,sha,array_sha,sign

REPO=Path(__file__).resolve().parents[2]
CASES=[dict(id='fashion_mature_n256',fit='fashion_mnist_k3_seed101_age8',n=256,profiles=['head79','head63','pca79']),dict(id='mnist_early_n1024',fit='mnist_k5_seed113_age1',n=1024,profiles=['head79','pca79'])]


def read(path):return json.loads(Path(path).read_text())


def pin_plan(root,source,binary,prerequisite):
    fits=read(source/'fit_records.json');pins={}
    for case in CASES:
        for method in ['ordered_head','pca']:
            fit=next(r for r in fits if r['fit']==case['fit'] and r['representation']==method);path=source/'fits'/f"{case['fit']}_{method}.npz"
            assert sha(path)==fit['archive_sha256'];pins[path.relative_to(REPO).as_posix()]=fit['archive_sha256']
    plan=dict(cases=CASES,source=str(source),source_summary_sha256=sha(source/'summary.json'),source_fit_hashes=pins,binary_sha256=sha(binary),prerequisite=str(prerequisite),precision=14,head_prediction_rank=63,full_batch=8,compact_batch=16,repetitions=1,phase_wall_limit_seconds=12*3600,selection='N256 Fashion mature and N1024 MNIST early, firstN pinned rows and every distinct original natural update. Same fits as the predeclaredN128 replication; no outcome-dependent filtering. Exact duplicate centers cached and disclosed. No held-out accuracy gate.',service='Exact quantized cost-change sign, with public certified intervals where conclusive and fresh Full old+candidate generation/replay for every inconclusive request. Frozen approximate predictions and their assignment certificate coverage reported separately.',accounting='Cold setup, preprocessing, all actual proof attempts, fitting, public-only replay and prospective fallback are included. Baseline Full is shared as a reference across paired methods, never counted as new fallback generation. Active invocation time excludes queue/RAM/offline waits and inter-phase file export; laptop load varies.',scope='Two scale configurations, one repetition, existing fresh-fit sources. Not new independent fits. No claimed speedup until complete matched measurements and verification.')
    if (root/'plan.json').exists():assert read(root/'plan.json')==plan,'pinned main plan changed'
    else:save(root/'plan.json',plan)
    return plan


def prepare_case(root,source,case,workers=2):
    fits=read(source/'fit_records.json');parents={};checks=[]
    from commitjl.reuse_certificates import integer_stats
    from commitjl.relative_certificates import relative_certificate
    from commitjl.group_change_bounds import exact_old_correction_bounds
    from commitjl.group_moment_reference import prepare_groups
    from experiments.pilot_jl_certificate_ideas import distances,prediction_labels
    with threadpool_limits(limits=1):
        for method in ['ordered_head','pca']:
            meta=next(r for r in fits if r['fit']==case['fit'] and r['representation']==method);archive=source/'fits'/f"{case['fit']}_{method}.npz";assert sha(archive)==meta['archive_sha256']
            with np.load(archive) as frozen:
                all_rows=frozen['audit'].copy();rows=all_rows[:case['n']];basis=frozen['basis'].copy();aux=frozen['auxiliary_basis'].copy();all_sets=frozen['centers'].copy()
            assert array_sha(all_rows)==meta['audit_sha256'] and basis.shape==(63,784) and np.array_equal(aux[:63],basis)
            old=all_sets[0];sets=[];source_indices=[];duplicates=[];seen=set()
            for i,c in enumerate(all_sets):
                digest=array_sha(c)
                if digest in seen:duplicates.append(i);continue
                seen.add(digest);sets.append(c.tolist());source_indices.append(i)
            assert len(sets)>1 and sets[0]==old.tolist()
            n=case['n'];model=dict(version='bound-audit-v1',method='learned_energy',n=n,d=784,k=len(old),input_bits=12,basis_scale=16384,input_scale=meta['input_scale'],centers=old.tolist(),basis=basis.tolist(),tail_width=0,tail_rows=[],tail_signs=[],seed='frozen fresh fit; main scale campaign')
            inp=dict(model=model,x=rows.tolist(),salt_x='20261101',salt_z='20261103',queries=[list(range(i,i+16)) for i in range(0,n,16)],reuse_centers=sets,comparison_base=old.tolist());parent=root/case['id']/method;parents[method]=parent
            full=copy.deepcopy(inp);full['model'].update(method='full',basis=[],seed='');full['queries']=[list(range(i,i+8)) for i in range(0,n,8)]
            if len(rows)!=n:raise ValueError('Catalog exceeds real frozen held-out rows; padding is forbidden')
            metadata=dict(dataset=meta['dataset'],k=meta['k'],fit_seed=meta['fit_seed'],age=meta['age'],n=n,native_n=len(all_rows),representation=method,fitting_seconds=meta['prediction_fitting_seconds'],training_sha256=meta['training_sha256'],audit_sha256=meta['audit_sha256'],proof_audit_sha256=array_sha(rows),source_archive_sha256=meta['archive_sha256'],source_fit=case['fit'],selected_source_sets=source_indices,duplicate_source_sets=duplicates,phase_wall_limit_seconds=12*3600,selection='Every distinct supplied natural update; public identity and duplicates cached before audit; fixed firstN audit prefix, no outcome selection.',fit_scope='Frozen fresh-seed fit; recorded original fitting is charged to each standalone deployment. Scale comparison is not new independent fitting.')
            if workers!=2:metadata['proof_runtime_workers']=workers
            auxiliary=dict(source_input_sha256=None,training_sha256=meta['training_sha256'],audit_sha256=meta['audit_sha256'],basis_scale=16384,certificate_basis=aux.tolist(),rank=79,fitting_seconds=meta['auxiliary_fitting_seconds'],recipe='Frozen training-only residual SVD64,4 iterations,seed20261025, first16 directions appended to byte-identical original63 prefix.')
            for path,value in [(parent/'head_r63/input.json',inp),(parent/'full/input.json',full),(parent/'protocol.json',metadata)]:
                if path.exists():assert read(path)==value,'source input changed'
                else:save(path,value)
            auxiliary['source_input_sha256']=sha(parent/'head_r63/input.json')
            if (parent/'auxiliary_basis.json').exists():assert read(parent/'auxiliary_basis.json')==auxiliary
            else:save(parent/'auxiliary_basis.json',auxiliary)
            # Native quality checks are after the fixed plan is pinned; failures
            # remain in the evidence and do not remove a proof workload.
            od=distances(rows,old);labels=od.argmin(1);prepared=prepare_groups(rows,labels.tolist(),len(old));h,_=integer_stats(model,rows);old_cost=int(od.min(1).sum())
            for s,new in enumerate(all_sets[1:],1):
                nd=distances(rows,new);truth=int(nd.min(1).sum()-old_cost);pred=prediction_labels(model,h,new);assignment=relative_certificate(model,rows,new)
                assert assignment['labels']==pred.tolist() and not any(flag and p!=t for flag,p,t in zip(assignment['certified_flags'],pred,nd.argmin(1)))
                for rank in [63,79]:
                    cm=model if rank==63 else dict(model,basis=aux.tolist());b=exact_old_correction_bounds(cm,rows,prepared,old,new);lo,hi=b['change_lower_scaled'],b['change_upper_scaled'];assert lo<=truth*b['cost_denominator']<=hi and sign(lo,hi) in ['inconclusive',sign(truth,truth)]
                    checks.append(dict(case=case['id'],representation=method,n=n,source_set=s,rank=rank,truth=truth,decision=sign(lo,hi),lower=str(lo),upper=str(hi),denominator=str(b['cost_denominator']),unchanged=bool(np.array_equal(old,new)),prediction_agreement_percent=100*float(np.mean(pred==nd.argmin(1))),assignment_certified_percent=100*assignment['certified_count']/n,width_percent_old_cost=100*(hi-lo)/b['cost_denominator']/old_cost))
    assert read(parents['ordered_head']/'full/input.json')==read(parents['pca']/'full/input.json')
    save(root/case['id']/'native_quality.json',dict(complete=True,checks=checks,new_proofs=0));return parents


def share_full_reference(source,target):
    from campaigns.shared_full_reference import share_full_reference as share
    return share(source,target)


def verify_group(root):
    result=read(root/'checked_comparison.json');assert result['complete'] and result['all_intervals_contain_full'] and result['no_wrong_decisions']
    manifest=read(root/'public/public_manifest.json')['sha256'];assert all(sha(root/'public'/name)==digest for name,digest in manifest.items())
    replay=read(root/'public/group_verify.json');assert replay['complete'] and replay['verified']==result['group_public_proofs']
    native=read(root/'native_checks.json');assert native['complete']
    for r in native['decisions']:assert int(r['lower'])<=int(r['truth'])*int(r['denominator'])<=int(r['upper'])
    if result['fallback_updates']:
        from analysis.audit_local_fast_tradeoff import audit_case
        fallback,_,_=audit_case(root/'fallback','full');assert fallback['public_proofs']==result['fresh_fallback_proofs'] and result['fallback_active_seconds']==fallback['active_phase_wall_seconds']
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['root','source','binary','smoke-receipt','prerequisite']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args();root=a.root.resolve();root.mkdir(parents=True,exist_ok=a.resume);source=a.source.resolve();binary=a.binary.resolve();plan=pin_plan(root,source,binary,a.prerequisite.resolve())
    state=read(root/'queue.json') if a.resume else dict(complete=False,started_unix=time.time(),completed_phases=[],plan_sha256=sha(root/'plan.json'))
    assert state['plan_sha256']==sha(root/'plan.json');state.update(pid=os.getpid(),status='waiting_for_prerequisite',paused=False,child_pid=None);save(root/'queue.json',state)
    snapshot=root/'main_runner_snapshot.py'
    if not snapshot.exists():shutil.copy2(Path(__file__),snapshot)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def invoke(command,phase,status_path,full_parent=None):
        while True:
            with (root/(phase+'.stdout.log')).open('a',encoding='utf-8') as stdout,(root/(phase+'.stderr.log')).open('a',encoding='utf-8') as stderr:
                child=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));state.update(status='running',phase=phase,child_pid=child.pid);save(root/'queue.json',state);code=child.wait()
            if code==0:state['completed_phases'].append(phase);save(root/'queue.json',state);return
            status=read(status_path);active=status.get('active_phase');attempts=status.get('resources',{}).get(active,{}).get('attempts',[])
            if full_parent:
                record=full_parent/'full'/(str(active)+'.resource.json');attempts=read(record).get('attempts',[]) if record.exists() else []
            elif active=='prospective_full_fallback':
                fb=status_path.parent/'fallback';fs=read(fb/'status.json');record=fb/'full'/(str(fs.get('active_phase'))+'.resource.json');attempts=read(record).get('attempts',[]) if record.exists() else []
                if fs.get('paused'):status['paused']=True
            if status.get('paused') or not attempts or attempts[-1].get('stop')!='RAM reserve':raise RuntimeError(f'{phase} stopped; inspect preserved attempts/logs')
            state.update(status='waiting_for_ram_retry',child_pid=None);save(root/'queue.json',state)
            if full_parent is None and '--resume' not in command:command.append('--resume')
            time.sleep(15)
    try:
        while True:
            prior=read(a.prerequisite/'queue.json')
            if prior.get('complete'):break
            if prior.get('paused') or prior.get('status')=='stopped':raise RuntimeError('Prerequisite paused/stopped; main campaign waits for explicit repair/resume')
            time.sleep(15)
        prior=read(a.prerequisite/'summary.json');assert prior['complete']
        for case in CASES:
            for path,digest in plan['source_fit_hashes'].items():assert sha(REPO/path)==digest
            state.update(phase=case['id']+'_preparation',status='running');save(root/'queue.json',state);parents=prepare_case(root,source,case);parent=parents['ordered_head'];full_ref=root/case['id']/'full_reference.json'
            if not full_ref.exists():
                invoke([sys.executable,'-u','-m','campaigns.local_fast_tradeoff','--run','--root',str(parent),'--cases','full','--binary',str(binary)],case['id']+'_full',parent/'status.json',parent)
                from analysis.audit_local_fast_tradeoff import audit_case
                full,_,_=audit_case(parent,'full');save(full_ref,full)
            for profile in case['profiles']:
                method='pca' if profile=='pca79' else 'ordered_head';rank=63 if profile=='head63' else 79;target=root/case['id']/profile
                if (target/'checked_comparison.json').exists() and read(target/'checked_comparison.json').get('complete'):verify_group(target);continue
                if method=='pca':share_full_reference(parent,parents[method])
                command=[sys.executable,'-u','-m','campaigns.run_group_fast_control','--compact','--parent',str(parents[method]),'--root',str(target),'--binary',str(binary),'--smoke-receipt',str(a.smoke_receipt.resolve())]
                if rank==79:command+=['--auxiliary-basis',str(parents[method]/'auxiliary_basis.json')]
                if target.exists():command.append('--resume')
                invoke(command,case['id']+'_'+profile,target/'status.json');verify_group(target)
        results=[]
        for case in CASES:
            for profile in case['profiles']:results.append(dict(case=case['id'],source_fit=case['fit'],n=case['n'],variant=profile,**verify_group(root/case['id']/profile)))
        save(root/'summary.json',dict(complete=True,results=results,plan_sha256=sha(root/'plan.json'),source_summary_sha256=plan['source_summary_sha256'],prerequisite_summary_sha256=sha(a.prerequisite/'summary.json')));state.update(complete=True,status='complete',child_pid=None,finished_unix=time.time());save(root/'queue.json',state)
    except BaseException as error:state.update(status='stopped',error=str(error),child_pid=None);save(root/'queue.json',state);raise


if __name__=='__main__':main()
