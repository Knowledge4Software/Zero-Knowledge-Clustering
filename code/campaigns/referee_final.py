"""Targeted matched reviewer campaign; serial eight-worker laptop execution.

These challenge workloads are selected from an already observed native grid.
They demonstrate difficult cases, not an unbiased estimate of deployment rates.
Every distinct source update, all failures and paid fallbacks are retained.
"""
from pathlib import Path
import argparse,copy,json,os,subprocess,sys,time
import psutil
from campaigns.local_fast_tradeoff import save,sha
from campaigns.main_rank79 import prepare_case,verify_group
from campaigns.shared_full_reference import share_full_reference


REPO=Path(__file__).resolve().parents[2]
CASES=[dict(id='mnist_switching_mature_n128',fit='mnist_k3_seed101_age8',n=128),
       dict(id='fashion_early_n128',fit='fashion_mnist_k3_seed101_age1',n=128),
       dict(id='mnist_early_n1024',fit='mnist_k5_seed113_age1',n=1024)]


def read(path):return json.loads(Path(path).read_text())


def prepare(root,source,binary):
    root.mkdir(parents=True,exist_ok=True);fits=read(source/'fit_records.json');paths=[]
    for case in CASES:
        for representation in ['ordered_head','pca']:
            meta=next(r for r in fits if r['fit']==case['fit'] and r['representation']==representation)
            archive=source/'fits'/f"{case['fit']}_{representation}.npz";assert sha(archive)==meta['archive_sha256'];paths.append(archive)
    plan=dict(cases=CASES,binary_sha256=sha(binary),source=str(source),source_sha256={p.relative_to(REPO).as_posix():sha(p) for p in paths},workers=8,profiles=['full','head79','pca79','moments_only','known_family_affine'],selection='Diagnostic workloads selected from the existing, already observed native grid: mature MNIST with switching/fallback, early Fashion, and a predeclared1024-row image scale. Every distinct supplied update is retained. Not independent new fitting or unbiased deployment-rate sampling.',timing='One serial repetition; every method and its fresh Full fallback uses8Go workers,1200MiB soft memory,3.5GiB RSS guard,650MiB free RAM reserve. Cold setup, native preparation, proof generation, worker checking and fresh public replay charged. Component sums exclude queue/RAM waits and between-phase export gaps.',baseline_replacement='Groups/basis-derived commitments are tied to the deployed baseline. Report measured cold preparation separately and model each baseline replacement as a new preparation; Full public-center circuit setup remains reusable. Modeled reset costs are not presented as a measured rebase experiment.')
    if (root/'plan.json').exists():assert read(root/'plan.json')==plan
    else:save(root/'plan.json',plan)
    parents={}
    for case in CASES:
        cases=prepare_case(root/'inputs',source,case,workers=8)
        parents[case['id']]={key:str(path) for key,path in cases.items()}
    save(root/'parents.json',parents)
    return plan


def verified_moments(root):
    result=read(root/'checked_comparison.json');assert result['complete'] and result['all_intervals_contain_full'] and result['no_wrong_decisions']
    manifest=read(root/'public/public_manifest.json')['sha256'];assert all(sha(root/'public'/name)==h for name,h in manifest.items());replay=read(root/'public/moments_verify.json');assert replay['complete'] and replay['verified']==result['group_public_proofs'];return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','source','binary','compact-smoke','moments-smoke','affine-smoke']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args();root=a.root.resolve();binary=a.binary.resolve()
    if (root/'queue.json').exists():
        previous=read(root/'queue.json')
        for pid in [previous.get('pid'),previous.get('child_pid')]:
            if not pid:continue
            try:
                args=psutil.Process(pid).cmdline()
                if str(root) in args:raise RuntimeError('Existing reviewer coordinator/worker is still live; inspect it instead of launching a duplicate')
            except psutil.NoSuchProcess:pass
    plan=prepare(root,a.source.resolve(),binary)
    for path in [a.compact_smoke,a.moments_smoke,a.affine_smoke]:assert read(path)['complete'] and read(path)['binary_sha256']==sha(binary)
    if a.prepare_only:print(json.dumps(dict(prepared=True,root=str(root))));return
    state=read(root/'queue.json') if a.resume else dict(complete=False,completed_phases=[],started_unix=time.time(),plan_sha256=sha(root/'plan.json'))
    if 'error' in state:state['previous_error']=state.pop('error')
    assert state['plan_sha256']==sha(root/'plan.json');state.update(pid=os.getpid(),status='running',paused=False,child_pid=None);save(root/'queue.json',state)
    env=dict(os.environ,PYTHONPATH=str(REPO/'code'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def invoke(command,phase,target_status):
        if phase in state['completed_phases']:assert read(target_status).get('complete');return
        # Brief validation/pilot windows occur only between timed phases.
        # No active prover is suspended and no artificial wait enters timings.
        while (root/'hold_for_validation.json').exists():
            state.update(status='waiting_for_validation_window',child_pid=None,next_phase=phase)
            save(root/'queue.json',state);time.sleep(2)
        resumable='campaigns.local_fast_tradeoff' not in command
        if resumable and '--root' in command and command[command.index('--root')+1]!=str(root):
            child_root=Path(command[command.index('--root')+1])
            if child_root.exists():
                if (child_root/'protocol.json').exists():command.append('--resume')
                else:child_root.rename(child_root.with_name(child_root.name+f'_preinit_{time.time_ns()}'))
        while True:
            with (root/(phase+'.stdout.log')).open('a',encoding='utf-8') as stdout,(root/(phase+'.stderr.log')).open('a',encoding='utf-8') as stderr:
                child=subprocess.Popen(command,env=env,stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));state.update(status='running',active_phase=phase,child_pid=child.pid,child_started_unix=time.time());save(root/'queue.json',state);code=child.wait()
            result=read(target_status) if target_status.exists() else {}
            if code==0 and result.get('complete'):state['completed_phases'].append(phase);state['child_pid']=None;save(root/'queue.json',state);return
            if result.get('paused'):raise RuntimeError('User paused child; no automatic resume')
            # Only recover a RAM-reserve stop; all other failures stay visible.
            stopped=False
            for resource in target_status.parent.rglob('*.resource.json'):
                attempts=read(resource).get('attempts',[])
                if attempts and attempts[-1].get('stop')=='RAM reserve' and not read(resource).get('complete'):stopped=True
            if not stopped:raise RuntimeError(f'{phase} failed; see preserved logs')
            state.update(status='waiting_for_ram_retry',child_pid=None);save(root/'queue.json',state)
            if resumable and '--root' in command and '--resume' not in command:command.append('--resume')
            time.sleep(15)
    try:
        parents=read(root/'parents.json');results=[]
        for case in CASES:
            source=Path(parents[case['id']]['ordered_head']);pca=Path(parents[case['id']]['pca'])
            for path,h in plan['source_sha256'].items():assert sha(REPO/path)==h
            invoke([sys.executable,'-u','-m','campaigns.local_fast_tradeoff','--root',str(source),'--binary',str(binary),'--run','--cases','full'],case['id']+'_full',source/'status.json')
            share_full_reference(source,pca)
            for variant,parent in [('head79',source),('pca79',pca)]:
                target=root/case['id']/variant
                extension=root/'refinement_extension.json'
                command=[sys.executable,'-u','-m','campaigns.run_group_fast_control','--compact','--parent',str(parent),'--root',str(target),'--binary',str(binary),'--smoke-receipt',str(a.compact_smoke.resolve()),'--auxiliary-basis',str(parent/'auxiliary_basis.json')]
                if extension.exists():command.append('--proof-backbone-only')
                invoke(command,case['id']+'_'+variant,target/('backbone_complete.json' if extension.exists() else 'status.json'))
                if not extension.exists():results.append(dict(case=case['id'],variant=variant,**verify_group(target)))
                # Optional versioned extension, pinned separately from the
                # original matched plan. Never rewrite the base measurements.
                if extension.exists():
                    config=read(extension);extra=Path(config['root'])/case['id']/variant
                    exe=Path(config['binary']);smoke=Path(config['smoke'])
                    assert sha(exe)==config['binary_sha256'] and sha(smoke)==config['smoke_sha256']
                    invoke([sys.executable,'-u','-m','campaigns.run_group_refinement_control','--base',str(target),'--parent',str(parent),'--root',str(extra),'--binary',str(exe),'--smoke-receipt',str(smoke)],case['id']+'_'+variant+'_refined',extra/'status.json')
                    checked=read(extra/'checked_comparison.json');assert checked['complete'] and checked['all_intervals_contain_full'] and checked['no_wrong_decisions']
                    results.append(dict(case=case['id'],variant=variant+'_bounded_refinement',**checked))
            target=root/case['id']/'moments_only'
            invoke([sys.executable,'-u','-m','campaigns.run_moments_control','--parent',str(source),'--root',str(target),'--binary',str(binary),'--smoke-receipt',str(a.moments_smoke.resolve())],case['id']+'_moments',target/'status.json')
            results.append(dict(case=case['id'],variant='moments_only',**verified_moments(target)))
            target=root/case['id']/'known_family_affine'
            invoke([sys.executable,'-u','-m','campaigns.run_family_affine_control','--parent',str(source),'--root',str(target),'--binary',str(binary),'--smoke-receipt',str(a.affine_smoke.resolve())],case['id']+'_affine',target/'control_status.json')
            result=read(target/'checked_comparison.json');assert result['complete'] and result['all_outputs_match_full'];results.append(dict(case=case['id'],variant='known_family_affine',**result))
            save(root/'completed_cases.json',dict(complete=False,results=results))
        save(root/'summary.json',dict(complete=True,results=results,plan_sha256=sha(root/'plan.json'),selection=plan['selection'],new_independent_fits=0))
        state.update(complete=True,status='complete',child_pid=None,finished_unix=time.time());save(root/'queue.json',state)
        # Text/tables may be refreshed from complete checked evidence. A newly
        # rendered PDF must still receive a visual review before final delivery.
        release=REPO/'code/analysis/release_referee_final_paper.py'
        if release.exists():
            with (root/'paper_refresh.log').open('a',encoding='utf-8') as log:
                r=subprocess.run([sys.executable,str(release),'--campaign',str(root),'--automatic'],env=env,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
            state['paper_refresh_returncode']=r.returncode;state['paper_visual_review_pending']=r.returncode==0;save(root/'queue.json',state)
    except BaseException as error:
        state.update(status='stopped',error=str(error),child_pid=None);save(root/'queue.json',state);raise


if __name__=='__main__':main()
