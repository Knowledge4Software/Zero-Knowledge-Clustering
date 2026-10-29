"""Serial matched controls on an explicitly authorized execution target.

Every invocation, including failures, is charged in a separate ledger. A
coordinator handover can retain one live phase without restarting its work.
"""
from pathlib import Path
import argparse,copy,json,os,shutil,subprocess,sys,time
import psutil
from campaigns.local_fast_tradeoff import save,sha
from campaigns.remote_execution import require_mobin
from campaigns.execution_admission import require_execution
from campaigns.proof_runtime import memory_stop

REPO=Path(__file__).resolve().parents[2]
POLICY=REPO/'code/validation/outputs/referee_round2_20261026/execution_policy.json'
SOURCE=REPO/'code/validation/outputs/referee_final_20261025/inputs/mnist_switching_mature_n128'


def read(path):return json.loads(Path(path).read_text())


def prepare(root,source=None,repetitions=3,workers=4,execution_target='mobin',phase_wall_limit_seconds=12*3600,long_catalog_phase=False):
    source=SOURCE if source is None else source
    if type(repetitions) is not int or repetitions<3:raise ValueError('predeclared timing controls require at least three repetitions')
    root.mkdir(parents=True,exist_ok=True)
    pins={}
    for method in ['ordered_head','pca']:
        origin=source/method;target=root/'staged'/method
        for name in ['head_r63/input.json','full/input.json','auxiliary_basis.json']:
            original=origin/name;value=read(original);pins[str(original)]=sha(original)
            if name=='auxiliary_basis.json':
                value['excluded_source_fitting_seconds']=value.get('fitting_seconds',0.)
                value['fitting_seconds']=0.
            path=target/name
            if path.exists():assert read(path)==value
            elif name.endswith('input.json'):
                # Preserve owner-pinned bytes across Windows CRLF / Linux LF.
                path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,path)
            else:save(path,value)
        protocol=read(origin/'protocol.json');protocol.update(proof_runtime_workers=workers,fitting_seconds=0.,
            phase_wall_limit_seconds=phase_wall_limit_seconds,
            fit_scope='Existing frozen fitted representations are service inputs. Fitting is excluded for all controls, not replaced by a laptop fitting time.',
            timing=f'New serial {workers}-worker {execution_target} controls. Compare this campaign only; all preparation, keys, attempts, query and public replay are charged. No cross-host ratio.')
        if long_catalog_phase:protocol['long_catalog_phase']='local_20261026'
        path=target/'protocol.json'
        if path.exists():assert read(path)==protocol
        else:save(path,protocol)
    staged={p.relative_to(root).as_posix():sha(p) for p in (root/'staged').rglob('*.json')}
    n=read(root/'staged/ordered_head/full/input.json')['model']['n']
    plan=dict(version=1,source_sha256=pins,staged_sha256=staged,workers=workers,repetitions=repetitions,catalog_records=n,
        execution_target=execution_target,
        phase_wall_limit_seconds=phase_wall_limit_seconds,
        methods=['reusable_exact_full','moment_switch_centroid','moment_switch_pca','moments_only_paid_fallback','online_affine_rebuild','exact_plonk_backend_control'],
        ranks=dict(prediction=63,certificate=79),precision_bits=14,refinement_cap=16,
        before_timing=['compact_real_proof_tamper_smoke','refinement_real_proof_tamper_smoke','plonk_small_complete_catalog_smoke'],
        ordering='Each repetition generates a fresh Full control required by subsequent checks. Compact basis order alternates. Online affine and PLONK run after the first matched repetition. Failures stop the queue and remain charged.',
        accounting='One host only. Fresh keys/proofs for each timing repetition; fresh fallback generation if needed. Frozen fitted models are inputs to this control, and training/fitting is excluded consistently. This differs from the historical laptop cold-fitting totals.',
        inference='Three timing repetitions of one diagnostic fit do not establish fit independence, production scale or superiority to optimized matrix-product protocols.',
        remaining_scientific_work=['Frozen-profile independent seeds/splits','N>=10000 (16384 real rows for the power-of-two implementation)','k=10,20 sweep','A specialized optimized exact matrix-product protocol','Stronger output privacy if aggregate hiding is required'])
    path=root/'plan.json'
    if path.exists():assert read(path)==plan
    else:save(path,plan)
    return plan


def copy_staged_parent(source,target):
    for name in ['head_r63/input.json','full/input.json','auxiliary_basis.json','protocol.json']:
        path=target/name;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            if sha(path)!=sha(source/name):raise RuntimeError('Existing repetition input differs: '+str(path))
        else:shutil.copyfile(source/name,path)
    assert read(target/'auxiliary_basis.json')['source_input_sha256']==sha(target/'head_r63/input.json')


def resume_state(root,binary,plonk_binary,handover):
    """Explicit scope-change handover only; never resume an arbitrary old queue."""
    state=read(root/'queue.json')
    if state.get('complete'):raise RuntimeError('Completed queue must not be resumed')
    expected=dict(plan_sha256=sha(root/'plan.json'),binary_sha256=sha(binary),plonk_binary_sha256=sha(plonk_binary))
    if any(state.get(k)!=v for k,v in expected.items()):raise RuntimeError('Handover plan or binary pin differs')
    if Path(handover['cohort_root']).resolve()!=root.resolve() or handover['old_coordinator_pid']!=state['pid']:
        raise RuntimeError('Handover does not identify this cohort coordinator')
    if psutil.pid_exists(state['pid']):raise RuntimeError('Previous coordinator still exists; do not duplicate it')
    if state['active_phase']!=handover['active_phase'] or state['child_pid']!=handover['child_pid']:
        raise RuntimeError('Handover phase changed')
    state.setdefault('coordinator_handovers',[]).append(handover)
    state.update(pid=os.getpid(),status='running')
    save(root/'queue.json',state)
    return state


def adopted_phase_receipt(root,label):
    if label=='online_affine':path=root/'repetition_01/online_affine/checked_comparison.json'
    elif label.startswith('r') and label.endswith('_moments'):
        repetition=int(label[1:label.index('_')]);path=root/f'repetition_{repetition:02d}/moments_only/checked_comparison.json'
    else:raise RuntimeError('Unsupported phase for scope-change adoption: '+label)
    if not path.exists() or not read(path).get('complete'):raise RuntimeError('Adopted phase has no completed checked receipt')
    value=read(path)
    if label!='online_affine' and not (value.get('all_intervals_contain_full') and value.get('no_wrong_decisions')):
        raise RuntimeError('Adopted phase certificate checks failed')
    return path


def execute(root,binary,plonk_binary,execution_target='mobin',handover=None):
    require_execution(read(POLICY),target=execution_target);plan=read(root/'plan.json')
    assert plan.get('execution_target','mobin')==execution_target,'execution target differs from pinned plan'
    assert all(sha(root/name)==digest for name,digest in plan['staged_sha256'].items()),'staged inputs differ from the pinned plan'
    # Validate portable staged files; do not require the original Windows path.
    for method in ['ordered_head','pca']:
        parent=root/'staged'/method
        assert read(parent/'protocol.json')['proof_runtime_workers']==plan['workers']
        assert read(parent/'auxiliary_basis.json')['source_input_sha256']==sha(parent/'head_r63/input.json')
    if handover is not None:state=resume_state(root,binary,plonk_binary,handover)
    else:
        if (root/'queue.json').exists():raise RuntimeError('Preserve earlier attempt; inspect its ledger before a new execution')
        state=dict(complete=False,status='running',pid=os.getpid(),completed=[],attempts=[],plan_sha256=sha(root/'plan.json'),
            binary_sha256=sha(binary),plonk_binary_sha256=sha(plonk_binary),started_unix=time.time())
    env=dict(os.environ,PYTHONPATH=str(REPO/'code'),COMMITJL_EXECUTION_TARGET=execution_target,COMMITJL_EXECUTION_POLICY=str(POLICY),GOMAXPROCS=str(plan['workers']),GOMEMLIMIT='1200MiB',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def invoke(label,args):
        if label in state['completed']:return
        require_execution(read(POLICY),target=execution_target)
        if shutil.disk_usage(root).free<2*2**30:raise RuntimeError('Disk reserve; no new proof phase started')
        if handover is not None and label==handover['active_phase']:
            from campaigns.bounded_local_campaign import adopt_phase
            attempt=adopt_phase(root,label,args,handover,plan)
            state['attempts'].append(attempt);state['child_pid']=None
            state['completed'].append(label);save(root/'queue.json',state);return
        state.update(active_phase=label,status='running');save(root/'queue.json',state);started=time.monotonic()
        peak_rss=0;stop=None;child=None
        with (root/(label+'.log')).open('xb') as log:
            try:
                child=subprocess.Popen(args,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
                state['child_pid']=child.pid;save(root/'queue.json',state)
                while child.poll() is None:
                    try:
                        process=psutil.Process(child.pid);tree=[process]+process.children(recursive=True)
                        rss=max((p.memory_info().rss for p in tree),default=0);peak_rss=max(peak_rss,rss)
                    except psutil.NoSuchProcess:continue
                    stop=memory_stop(rss)
                    campaign_root=os.environ.get('COMMITJL_FINAL_CAMPAIGN_ROOT')
                    if campaign_root and (Path(campaign_root)/'STOP').exists():stop='user stop file'
                    if shutil.disk_usage(root).free<1024*2**20:stop='disk reserve'
                    if time.monotonic()-started>plan.get('phase_wall_limit_seconds',12*3600):stop='phase wall limit'
                    if stop:
                        for p in reversed(tree):
                            try:p.kill()
                            except psutil.NoSuchProcess:pass
                        break
                    time.sleep(.25)
                rc=child.wait()
            except BaseException:
                if child and child.poll() is None:
                    try:
                        process=psutil.Process(child.pid)
                        for p in reversed([process]+process.children(recursive=True)):
                            try:p.kill()
                            except psutil.NoSuchProcess:pass
                    except psutil.NoSuchProcess:pass
                    child.wait()
                state['attempts'].append(dict(phase=label,wall_seconds=time.monotonic()-started,complete=False,stop='coordinator interrupted',peak_process_rss_bytes=peak_rss));save(root/'queue.json',state)
                raise
        state['attempts'].append(dict(phase=label,wall_seconds=time.monotonic()-started,returncode=rc,peak_process_rss_bytes=peak_rss,stop=stop));state['child_pid']=None
        save(root/'queue.json',state)
        if rc or stop:raise RuntimeError(label+' failed; preserve logs and charge this attempt: '+str(stop))
        state['completed'].append(label);save(root/'queue.json',state)
    py=[sys.executable,'-u','-m']
    try:
        compact=root/'compact_smoke';refinement=root/'refinement_smoke'
        invoke('compact_smoke',py+['validation.check_group_fast_proofs','--compact','--root',str(compact),'--binary',str(binary)])
        invoke('refinement_smoke',py+['validation.check_group_refinement_proofs','--root',str(refinement),'--binary',str(binary)])
        moments=root/'moments_smoke'
        invoke('moments_smoke',py+['validation.check_moments_only_proofs','--root',str(moments),'--binary',str(binary)])
        invoke('online_affine_smoke',py+['validation.check_online_affine_proofs','--root',str(root/'online_affine_smoke'),'--binary',str(binary)])
        tiny=root/'plonk_smoke_input.json'
        model=dict(version='bound-audit-v1',method='full',n=4,d=3,k=2,input_bits=4,basis_scale=8,input_scale='1',centers=[[0,0,0],[2,0,0]],basis=[],tail_width=0,tail_rows=[],tail_signs=[],seed='round3 backend smoke')
        save(tiny,dict(model=model,x=[[0,1,0],[3,0,0],[-2,2,1],[4,2,3]],salt_x='11',salt_z='22',reuse_centers=[model['centers'],[[-1,2,0],[4,1,3]]]))
        smoke=root/'plonk_smoke'
        invoke('plonk_smoke',[str(plonk_binary),'-phase','run','-input',str(tiny),'-out',str(smoke),'-q','2','-allow-benchmark-srs'])
        invoke('plonk_smoke_verify',[str(plonk_binary),'-phase','verify','-out',str(smoke),'-trust',str(smoke/'owner_pin.json'),'-public-task',str(smoke/'approved_task.json')])
        from campaigns.shared_full_reference import share_full_reference
        for repetition in range(1,plan['repetitions']+1):
            case=root/f'repetition_{repetition:02d}';case.mkdir(exist_ok=handover is not None);parents={}
            for method in ['ordered_head','pca']:
                parent=case/method;parents[method]=parent
                copy_staged_parent(root/'staged'/method,parent)
            full=parents['ordered_head']
            invoke(f'r{repetition}_full',py+['campaigns.local_fast_tradeoff','--root',str(full),'--binary',str(binary),'--run','--cases','full'])
            if not any(label in state['completed'] for label in [f'r{repetition}_pca_base',f'r{repetition}_ordered_head_base']):
                share_full_reference(full,parents['pca'])
            order=['ordered_head','pca'] if repetition%2 else ['pca','ordered_head']
            for method in order:
                parent=parents[method];base=case/(method+'_base');extra=case/(method+'_refined')
                invoke(f'r{repetition}_{method}_base',py+['campaigns.run_group_fast_control','--compact','--proof-backbone-only','--parent',str(parent),'--root',str(base),'--binary',str(binary),'--smoke-receipt',str(compact/'checks.json'),'--auxiliary-basis',str(parent/'auxiliary_basis.json')])
                invoke(f'r{repetition}_{method}_refined',py+['campaigns.run_group_refinement_control','--base',str(base),'--parent',str(parent),'--root',str(extra),'--binary',str(binary),'--smoke-receipt',str(refinement/'checks.json')])
                assert read(extra/'checked_comparison.json')['complete']
            invoke(f'r{repetition}_moments',py+['campaigns.run_moments_control','--parent',str(full),'--root',str(case/'moments_only'),'--binary',str(binary),'--smoke-receipt',str(moments/'checks.json')])
            if repetition==1:
                invoke('online_affine',py+['campaigns.run_online_affine_control','--parent',str(full),'--root',str(case/'online_affine'),'--binary',str(binary),'--run-proofs'])
                out=case/'plonk_full'
                inspect=case/'plonk_preflight'
                invoke('plonk_preflight',[str(plonk_binary),'-phase','inspect','-input',str(full/'full/input.json'),'-out',str(inspect),'-q','8'])
                if read(inspect/'compile_only.json')['within_guard']:
                    invoke('plonk_full',[str(plonk_binary),'-phase','run','-input',str(full/'full/input.json'),'-out',str(out),'-q','8','-allow-benchmark-srs'])
                    invoke('plonk_full_verify',[str(plonk_binary),'-phase','verify','-out',str(out),'-trust',str(out/'owner_pin.json'),'-public-task',str(out/'approved_task.json')])
                else:save(case/'plonk_exclusion.json',dict(status='constraint_guard',preflight_sha256=sha(inspect/'compile_only.json'),no_proofs_generated=True))
            # Export only fully completed matched repetitions. PDF candidates
            # are explicitly unreviewed; the previous reviewed PDF is preserved.
            final_root=os.environ.get('COMMITJL_FINAL_CAMPAIGN_ROOT')
            if final_root:
                invoke(f'r{repetition}_export',py+['analysis.update_optimized_evidence','--root',final_root])
                invoke(f'r{repetition}_paper_candidate',py+['analysis.release_referee_final_paper','--reuse-checked-tables','--automatic','--tag','prototype_audit_optimized_20261026'])
        state.update(complete=True,status='complete',finished_unix=time.time());save(root/'queue.json',state)
    except BaseException as error:
        state.update(status='stopped',error=str(error),child_pid=None);save(root/'queue.json',state);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--source-parent',type=Path);p.add_argument('--repetitions',type=int,default=3);p.add_argument('--execute',action='store_true');p.add_argument('--binary',type=Path);p.add_argument('--plonk-binary',type=Path);a=p.parse_args();root=a.root.resolve()
    if a.execute:
        if not a.binary or not a.plonk_binary:p.error('execution requires both frozen binaries')
        execute(root,a.binary.resolve(),a.plonk_binary.resolve())
    else:prepare(root,a.source_parent.resolve() if a.source_parent else None,a.repetitions);print(json.dumps(dict(prepared=True,proofs_generated=False,remote_queue_started=False)))
