"""One fresh Head proof-pipeline repetition between timed workloads.

The second identical-task Full execution is the paid moments fallback.
No proof is borrowed from either first execution. Representation fitting
remains the same recorded component, not a new independent model fit.
"""
from pathlib import Path
import json,os,subprocess,sys,time
from campaigns.local_fast_tradeoff import save,sha

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'code/validation/outputs/referee_final_20261025'
OWNER='timing_repeat_20261025'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    root=ROOT/'timing_repeat';root.mkdir(exist_ok=True);hold=ROOT/'hold_for_validation.json'
    assert read(hold)['owner']==OWNER,'Do not take another validation hold'
    state=dict(complete=False,status='waiting_for_phase_boundary',pid=os.getpid(),started_unix=time.time());save(root/'status.json',state)
    env=dict(os.environ,PYTHONPATH=str(REPO/'code'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',COMMITJL_MEMORY_PRESSURE_POLICY='windows_os')
    def invoke(command,label,target):
        if (target/'protocol.json').exists():command.append('--resume')
        while True:
            with (root/(label+'.log')).open('ab') as log:
                child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                state.update(status='running',phase=label,child_pid=child.pid);save(root/'status.json',state);code=child.wait()
            if code==0:return
            memory_stop=False
            for p in target.rglob('*.resource.json'):
                r=read(p);attempts=r.get('attempts',[])
                if not r.get('complete') and attempts and attempts[-1].get('stop')=='RAM reserve':memory_stop=True
            if not memory_stop:raise RuntimeError(label+' failed; retained logs require inspection')
            if '--resume' not in command:command.append('--resume')
            state.update(status='waiting_for_ram_retry',child_pid=None);save(root/'status.json',state);time.sleep(15)
    try:
        while read(ROOT/'queue.json').get('status')!='waiting_for_validation_window':
            q=read(ROOT/'queue.json')
            if q.get('status')=='stopped' or q.get('paused'):raise RuntimeError('Main campaign stopped before validation boundary')
            if time.time()-state['started_unix']>1800:raise RuntimeError('Validation boundary timeout')
            time.sleep(2)
        case='mnist_switching_mature_n128';parent=Path(read(ROOT/'parents.json')[case]['ordered_head']);binary=REPO/'code/zk/bin/commitjl-bound-referee-final-20261025.exe';extension=read(ROOT/'refinement_extension.json')
        base=root/'head79';refined=root/'refined';smoke=REPO/'code/validation/outputs/referee_final_20261025_compact_smoke/checks.json'
        save(root/'protocol.json',dict(selection='One predeclared timing repetition of the first diagnostic workload after its first baseline cycle. Same fit, rows, four distinct natural updates, ranks63/79,precision14,8workers,cap16. All fresh cold proof stages. Not another independent fitted workload.',first_base=str(ROOT/case/'head79'),first_refined=str(Path(extension['root'])/case/'head79'),source_input_sha256=sha(parent/'head_r63/input.json'),base_binary_sha256=sha(binary),refinement_binary_sha256=extension['binary_sha256'],fitting_scope='Same frozen representation and its recorded fitting component charged again; fitting itself is not independently repeated.'))
        invoke([sys.executable,'-u','-m','campaigns.run_group_fast_control','--compact','--proof-backbone-only','--parent',str(parent),'--root',str(base),'--binary',str(binary),'--smoke-receipt',str(smoke),'--auxiliary-basis',str(parent/'auxiliary_basis.json')],'base',base)
        invoke([sys.executable,'-u','-m','campaigns.run_group_refinement_control','--base',str(base),'--parent',str(parent),'--root',str(refined),'--binary',extension['binary'],'--smoke-receipt',extension['smoke']],'refinement',refined)
        from analysis.refinement_paper_evidence import checked_refinement
        from analysis.audit_local_fast_tradeoff import audit_case
        first=checked_refinement(Path(extension['root'])/case/'head79');second=checked_refinement(refined)
        fb=ROOT/case/'moments_only/fallback';assert sha(parent/'full/input.json')==sha(fb/'full/input.json')
        full,_,_=audit_case(fb,'full')
        def proof_hashes(base,extra):return {sha(p) for folder in [base/'public',extra/'public'] for p in folder.rglob('*.proof')}
        original_hashes=proof_hashes(ROOT/case/'head79',Path(extension['root'])/case/'head79');new_hashes=proof_hashes(base,refined)
        assert len(original_hashes)==len(new_hashes)==166 and not original_hashes.intersection(new_hashes),'Fresh pipeline must not borrow old proof bytes'
        original_full={sha(p) for p in (parent/'full/public_bundle').rglob('*.proof')};new_full={sha(p) for p in (fb/'full/public_bundle').rglob('*.proof')}
        assert len(original_full)==len(new_full)==80 and not original_full.intersection(new_full)
        result=dict(complete=True,case=case,first_method_seconds=first['active_phase_seconds_including_fitting_and_fallback'],second_method_seconds=second['active_phase_seconds_including_fitting_and_fallback'],first_full_seconds=first['full_reference_active_seconds'],second_full_seconds=full['active_phase_wall_seconds'],second_pair_ratio=full['active_phase_wall_seconds']/second['active_phase_seconds_including_fitting_and_fallback'],first_pair_ratio=first['full_over_group_active_phase'],fresh_method_proofs=166,fresh_full_proofs=80,proof_hash_sets_disjoint=True,full_input_sha256=sha(fb/'full/input.json'),scope=__doc__)
        save(root/'checked_timing_repeat.json',result);state.update(complete=True,status='building_paper',child_pid=None);save(root/'status.json',state)
        with (root/'paper_refresh.log').open('wb') as log:r=subprocess.run([sys.executable,'-m','analysis.release_referee_final_paper','--campaign',str(ROOT)],env=env,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
        assert r.returncode==0,'Paper build failed; proof results retained'
        state.update(status='awaiting_visual_review',paper_visual_review_pending=True);save(root/'status.json',state)
        started=time.time()
        while not (root/'visual_review_release.json').exists() and time.time()-started<600:time.sleep(2)
        state.update(status='complete',finished_unix=time.time(),paper_visual_review_pending=not (root/'visual_review_release.json').exists());save(root/'status.json',state)
    except BaseException as error:
        state.update(status='stopped',error=str(error),child_pid=None);save(root/'status.json',state);raise
    finally:
        if hold.exists() and read(hold).get('owner')==OWNER:hold.unlink()


if __name__=='__main__':main()
