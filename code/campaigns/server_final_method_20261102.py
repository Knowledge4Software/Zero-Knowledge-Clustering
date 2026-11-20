"""Final-day frozen method, matched baselines, automatic reports and archives."""
from pathlib import Path
import argparse
from dataclasses import asdict
import fcntl
import json
import os
import signal
import shutil
import subprocess
import sys
import tarfile
import time

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from analysis.final_server_profiles_20261102 import collect_profiles
from campaigns.cost_aware_pilot import prove_case
from campaigns.proof_runtime import server_profile
from campaigns.published_baselines_laptop import GO, RUST
from campaigns.server_campaign_20261101 import ServerCampaign
from campaigns.server_pca_advantage_20261102 import Campaign
from commitjl.calibrated_routing_20261102 import BatchCosts, CalibratedPlanner
from commitjl.final_method_20261102 import fit_bases, validation_choice, RANKS
from commitjl.pilot_io import array_sha
from experiments.pca_advantage_20261102 import (read, save, sha, source, trace,
                                              evaluate, model_for, proposal_banks)
from tools.server_pca_advantage_lease_20261102 import matches, stop_study
from validation.check_group_fast_proofs import issue_challenge

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'code/validation/outputs/server_final_method_20261102'
MANIFEST=REPO/'final_method_deployment_manifest_20261102.json'
DATASETS=['fashion','har','mnist','text']
SEEDS=[907,911,919]
ORDER=[(d,s) for s in SEEDS for d in DATASETS]
UPDATES=8


def go(campaign,label,phase,out,inp=None,extra=()):
    command=[GO,'-phase',phase,'-out',out,'-max-constraints','24000000','-reps','1','-warmup','0']
    if inp is not None:command+=['-instance',inp]
    return campaign.invoke(label,command+list(extra),timeout=5400)


def fit_case(root,dataset,seed,profiles):
    case=root/'cases'/f'{dataset}_seed{seed}_n128_q8'
    if case.exists():raise RuntimeError('Existing fit retained; no unrecorded retry')
    case.mkdir(parents=True)
    training,audit,k,scale,pin=source(dataset,seed)
    past=training[:512];old=trace(past,k,seed)[8]
    bases,fit_seconds=fit_bases(past,old,seed)
    validation_centers,validation_recipe=proposal_banks(training[512:768],old,8)
    validation=[];preparation={}
    for family,basis in bases.items():
        tick=time.perf_counter()
        for rank in RANKS:
            validation+=evaluate(basis,old,audit[:128],validation_centers,scale,family,rank)
        preparation[family]=fit_seconds[family]+time.perf_counter()-tick
    choices={m:validation_choice(validation,f,profiles[dataset])
             for m,f in [('ordered_head','centroid'),('pca','pca')]}
    save(case/'validation.json',validation)
    save(case/'selection.json',dict(choices=choices,preparation_seconds=preparation,
          frozen_unix=time.time(),profiles_sha256=sha(root/'cost_profiles.json'),
          decision_made_before_test_candidates_and_evaluation=True))
    selected_pin=sha(case/'selection.json')
    # No test candidate or test-bound computation occurs before the pin above.
    test_centers,test_recipe=proposal_banks(training[768:1024],old,UPDATES)
    test=[]
    for family,basis in bases.items():
        for rank in RANKS:test+=evaluate(basis,old,audit[128:256],test_centers,scale,family,rank)
    assert sha(case/'selection.json')==selected_pin
    save(case/'test_numerical.json',test)
    np.savez_compressed(case/'arrays.npz',past=past,rows=audit[128:256],old=old,centers=test_centers,
         validation_rows=audit[:128],validation_centers=validation_centers,
         validation_incoming=training[512:768],test_incoming=training[768:1024],**bases)
    save(case/'provenance.json',dict(dataset=dataset,seed=seed,scale=scale,k=k,source=pin,
         training_sha256=array_sha(past),validation_rows_sha256=array_sha(audit[:128]),
         test_rows_sha256=array_sha(audit[128:256]),validation_recipe=validation_recipe,
         test_recipe=test_recipe,selection_sha256=selected_pin,
         scope='New fits on inspected public corpora; source records can overlap older cohorts. '
               'Prediction accuracy is outside the cost-change proof contract.'))
    return case,choices,preparation,scale


def write_inputs(case,choices,scale):
    with np.load(case/'arrays.npz') as a:
        rows,old,centers=[a[k].copy() for k in ['rows','old','centers']]
        for method,c in choices.items():
            model=model_for(a[c['family']][:c['rank']],old,len(rows),scale)
            inp=dict(model=model,x=rows.tolist(),salt_x='202611029071',salt_z='202611029073',
                     comparison_base=old.tolist(),queries=[list(range(i,min(i+16,len(rows)))) for i in range(0,len(rows),16)],
                     reuse_centers=np.concatenate((old[None],centers)).tolist())
            save(case/method/'input.json',inp)
        full=dict(inp,model=dict(inp['model'],method='full',basis=[],seed=''),
                  queries=[list(range(i,min(i+8,len(rows)))) for i in range(0,len(rows),8)],
                  reuse_centers=[old.tolist()])
        save(case/'shared_full/full/input.json',full)
        save(case/'shared_full/protocol.json',dict(fitting_seconds=0.,proof_runtime_workers=12,phase_wall_limit_seconds=5400))
        adapter=case/(case.name+'_adapter')
        full_new=dict(full,reuse_centers=np.concatenate((old[None],centers)).tolist())
        save(adapter/'ordered_head/full/input.json',full_new)
        save(adapter/'approved_task.json',dict(reuse_centers=full_new['reuse_centers']))
    return adapter


def setup_method(campaign,case,method,profile):
    """Authenticate once, and prepare only the branches selected by preflight."""
    name=case.name+'_'+method
    parent=case/method;base=case/(method+'_base');ref=case/(method+'_refined')
    inp=read(parent/'input.json');old=inp['model']['centers']
    m=inp['model']
    if 'shape' in profile:
        assert profile['shape']==[m['n'],m['d'],m['k'],len(m['basis']),m['input_bits'],m['basis_scale']]
    task=dict(comparison_base=old,reuse_centers=inp['reuse_centers'][1:],queries=inp['queries'])
    tick=time.perf_counter()
    plan=CalibratedPlanner(inp['model'],inp['x'],BatchCosts(**profile['costs'])).batch(task['reuse_centers'])
    save(parent/'preflight.json',plan)
    save(base/'input.json',inp);save(base/'approved.json',task)
    resources={}
    resources['prepare']=go(campaign,name+'_prepare','group_compact_prepare',base/'bundle',base/'input.json',
                            ['-q','16','-public-task',base/'approved.json'])
    issue_challenge(base/'bundle',base/'owner_pin.json',16)
    if plan['certificate']:
        common=['-q','16','-trust',base/'owner_pin.json','-public-task',base/'approved.json']
        resources['keys']=go(campaign,name+'_keys','group_keys',base/'bundle',base/'input.json',common)
        pin=read(base/'owner_pin.json');pin['owner']=read(base/'bundle/trusted.json');save(base/'owner_pin.json',pin)
        resources['rows']=go(campaign,name+'_rows','group_rows',base/'bundle',base/'input.json',common)
    if any(plan['records'][i]['refinement_indices'] for i in plan['certificate']):
        resources['refinement_setup']=go(campaign,name+'_refinement_setup','refinement_setup',ref/'bundle',
            extra=['-trust',base/'owner_pin.json','-public-task',base/'approved.json','-refinement-base',base/'bundle'])
        save(ref/'owner_refinement_pin.json',read(ref/'bundle/trusted.json'))
    alias=case/'sources'/method
    alias.mkdir(parents=True)
    (parent/'full').symlink_to((case/'shared_full/full').resolve(),target_is_directory=True)
    for role,target in [('ordered_head',parent),('ordered_head_base',base),('ordered_head_refined',ref)]:
        (alias/role).symlink_to(target.resolve(),target_is_directory=True)
    out=case/'warm'/(case.name+'_'+method+'_final')
    owner=dict(binary_sha256=sha(GO),certificate=read(base/'owner_pin.json'),
               full=read(case/'shared_full/full/owner_pin.json'))
    if (ref/'owner_refinement_pin.json').exists():owner['refinement']=read(ref/'owner_refinement_pin.json')
    save(out/'owner_pin.json',owner);save(out/'approved.json',task);save(out/'planner.json',plan)
    save(parent/'setup_checked.json',dict(complete=True,resources=resources,
          active_seconds=sum(r['wall_seconds'] for r in resources.values()),
          plan_sha256=sha(parent/'preflight.json'),scope='Fresh prepare, conditional key/row authentication, lazy refinement key setup.'))
    return alias,out


def adapter_warm(campaign,case,adapter):
    private,public=adapter/'warm_private',adapter/'warm_public'
    private.mkdir();public.mkdir()
    for n in ['params','vk','pk','owner_pin.json']:os.link(adapter/'zkml_private'/n,private/n)
    for n in ['params','vk']:os.link(private/n,public/n)
    old_public=public/'bank_000';old_public.mkdir()
    for n in ['proof','public.json']:os.link(adapter/'zkml_public/bank_000'/n,old_public/n)
    source_path=adapter/'ordered_head/full/input.json';resources=[]
    resources.append(campaign.invoke(case.name+'_adapter_warm_verify000',
        [RUST,'verify',public,0,adapter/'zkml_owner_pin.json',adapter/'approved_task.json']))
    for i in range(1,UPDATES+1):
        resources.append(campaign.invoke(case.name+f'_adapter_warm_query{i:03}',[RUST,'query',source_path,private,i]))
        dest=public/f'bank_{i:03}';dest.mkdir()
        for n in ['proof','public.json']:shutil.copy2(private/f'bank_{i:03}'/n,dest/n)
        assert sha(dest/'proof')!=sha(adapter/'zkml_public'/f'bank_{i:03}'/'proof')
        resources.append(campaign.invoke(case.name+f'_adapter_warm_verify{i:03}',
            [RUST,'verify',public,i,adapter/'zkml_owner_pin.json',adapter/'approved_task.json']))
        assert read(private/f'bank_{i:03}'/'result.json')['cost']==read(adapter/'zkml_checked.json')['costs'][i]
    save(adapter/'warm_checked.json',dict(complete=True,requests=UPDATES,
         active_seconds=sum(r['wall_seconds'] for r in resources),all_costs_match=True,
         fresh_candidate_proofs=True,cached_reference_proof_freshly_replayed=True,
         contract='Published ZKML primitive adapter, stronger exact-cost output and different commitment; not unmodified compiler.'))


def summarize(root):
    records=[]
    for d,s in ORDER:
        case=root/'cases'/f'{d}_seed{s}_n128_q8'
        if (case/'result.json').exists():records.append(read(case/'result.json'))
    save(root/'summary.json',dict(updated_unix=time.time(),results=records,
        planned_fits=12,complete_fits=len(records),unique_updates_per_fit=UPDATES,
        interpretation='New matched configuration replications on inspected corpora. '
          'Warm phase invocation sums include native preflight and public replay. '
          'Cold results are sums of separately measured fresh setup and request components, not standalone RPC latency.'))
    lines=['Final frozen server method: completed results','',
           'All timings are seconds for eight updates. Ratios compare matched invocation sums. '
           'Cold totals reconstruct service activation from measured setup plus fresh request work. '
           'Each fit has one warm repetition; range across fits measures configuration variability.','',
           '| Dataset | Seed | Our rank | PCA rank | Ours warm | PCA warm | Full warm | Adapter warm |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['dataset']} | {r['seed']} | {r['ranks']['ordered_head']} | {r['ranks']['pca']} | {r['ordered_head']['warm_seconds']:.1f} | {r['pca']['warm_seconds']:.1f} | {r['full']['warm_seconds']:.1f} | {r['adapter']['warm_seconds']:.1f} |")
    lines+=['','Neither superiority over every baseline nor classification accuracy is inferred. '
            'All incomplete attempts and time/disk exclusions are retained separately. '
            'The adapter certifies a different exact-cost output contract.']
    (root/'results.md').write_text('\n'.join(lines)+'\n')


def retire_and_archive(root,case):
    assert case.resolve().parent==(root/'cases').resolve()
    assert read(case/'result.json')['complete']
    retired=[]
    for p in case.rglob('*'):
        if p.is_file() and not p.is_symlink() and (p.name=='pk' or p.suffix in ['.pk','.r1cs']):
            assert p.resolve().is_relative_to(root.resolve())
            retired.append(dict(path=str(p.relative_to(case)),bytes=p.stat().st_size,sha256=sha(p)))
            p.unlink()
    save(case/'retired_rebuildable_keys.json',dict(files=retired,public_verification_evidence_preserved=True))
    folder=root/'archives';folder.mkdir(exist_ok=True)
    path=folder/(case.name+'.tar.gz')
    members=[p for p in case.rglob('*') if p.is_file() and not p.is_symlink()
             and p.suffix not in ['.pk','.r1cs','.gob'] and p.name!='pk']
    save(case/'portable_manifest.json',{str(p.relative_to(case)):sha(p) for p in members})
    with tarfile.open(path,'w:gz',compresslevel=1,dereference=False) as archive:
        for p in members+[case/'portable_manifest.json']:archive.add(p,arcname=str(Path(case.name)/p.relative_to(case)),recursive=False)
    save(folder/(case.name+'.sha256.json'),dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path)))


def run_case(campaign,root,dataset,seed,profiles):
    case,choices,preparation,scale=fit_case(root,dataset,seed,profiles)
    adapter=write_inputs(case,choices,scale)
    campaign.invoke(case.name+'_full_initialize',[sys.executable,'-m','campaigns.server_all_sources_full_20261101',
        '--root',case/'shared_full','--run','--binary',GO,'--cases','full'])
    full_init=read(case/'shared_full/status.json')['cases']['full']['wall_seconds']
    order=['pca','ordered_head'] if seed!=911 else ['ordered_head','pca']
    save(case/'execution_order.json',dict(order=order,full_reference='Shared fresh old-only initialization; new baseline request proofs below.'))
    for method in order:
        rank=choices[method]['rank']
        alias,out=setup_method(campaign,case,method,profiles[dataset][str(rank)])
        prove_case(campaign,case/'warm',dict(name=out.name,requests=UPDATES),source=alias,run_baseline=method=='ordered_head')
    results={}
    for method,family in [('ordered_head','centroid'),('pca','pca')]:
        out=case/'warm'/(case.name+'_'+method+'_final')
        checked=read(out/'checked.json');plan=read(out/'planner.json')
        assert checked['all_requests_verified'] and checked['all_decisions_correct']
        setup=read(case/method/'setup_checked.json')['active_seconds']
        results[method]=dict(rank=choices[method]['rank'],warm_seconds=checked['measured_warm_routed_seconds'],
             cold_component_seconds=preparation[family]+setup+checked['measured_warm_routed_seconds']+(full_init if plan['full'] else 0.),
             certificate_requests=checked['certificate_requests'],full_requests=checked['direct_full_requests'],
             correction_proofs=checked['selected_correction_proofs'],verified=True)
    full_warm=read(case/'warm'/(case.name+'_ordered_head_final/checked.json'))['measured_warm_full_seconds']
    if not ServerCampaign.rust_case(campaign,adapter):raise RuntimeError('Pinned adapter excluded by shape')
    adapter_cold=sum(read(p)['wall_seconds'] for p in (root/'resources').glob(adapter.name+'_zkml_*.attempt_01.json')
                     if read(p).get('complete') and '_inspect_' not in p.name)
    adapter_warm(campaign,case,adapter)
    result=dict(complete=True,dataset=dataset,seed=seed,ranks={m:c['rank'] for m,c in choices.items()},
        **results,full=dict(warm_seconds=full_warm,cold_component_seconds=full_init+full_warm),
        adapter=dict(warm_seconds=read(adapter/'warm_checked.json')['active_seconds'],cold_component_seconds=adapter_cold),
        finished_unix=time.time(),all_decisions_correct=True)
    save(case/'result.json',result)
    summarize(root)
    retire_and_archive(root,case)


def smoke(campaign,root):
    # Exercise the new setup-only path, then actual fresh proofs and public replay.
    case=root/'smoke'
    rng=np.random.default_rng(11);rows=rng.integers(-3,4,size=(16,3),dtype=np.int64)
    old=np.array([[-2,0,0],[2,0,0]],dtype=np.int64)
    centers=np.array([[[-2,-2,1],[2,2,1]]],dtype=np.int64)
    npth=case/'arrays.npz';case.mkdir();np.savez(npth,rows=rows,old=old,centers=centers,
          centroid=np.array([[16384,0,0]],dtype=np.int64),pca=np.array([[0,16384,0]],dtype=np.int64))
    choices={m:dict(family=f,rank=1) for m,f in [('ordered_head','centroid'),('pca','pca')]}
    write_inputs(case,choices,1.)
    campaign.invoke('smoke_full_initialize',[sys.executable,'-m','campaigns.server_all_sources_full_20261101',
        '--root',case/'shared_full','--run','--binary',GO,'--cases','full'])
    for method in ['ordered_head','pca']:
        alias,out=setup_method(campaign,case,method,dict(costs=asdict(BatchCosts(1e6,1.,1.,margin=0.))))
        assert read(out/'planner.json')['certificate']==[0]
        prove_case(campaign,case/'warm',dict(name=out.name,requests=1),source=alias,run_baseline=method=='ordered_head')
    save(root/'smoke_complete.json',dict(complete=True,excluded_from_scientific_results=True))


def watchdog(root):
    while True:
        state=read(root/'watchdog.json')
        if state.get('complete'):return
        try:live=matches(psutil.Process(state['owner']['pid']),state['owner'])
        except psutil.NoSuchProcess:live=False
        if not live or time.time()>=state['compute_deadline_unix']:
            (root/'STOP').write_text('Final-day computation deadline or owner death.\n')
            stop_study(root,[os.getpid()])
            save(root/'watchdog_recovery.json',dict(unix=time.time(),owner_live=live))
            return
        time.sleep(3)


def run(root,compute_deadline,expiry):
    assert root.resolve()==ROOT.resolve() and server_profile()
    root.mkdir(parents=True,exist_ok=True)
    assert not(root/'plan.json').exists(),'Existing frozen cohort retained'
    assert time.time()<compute_deadline<expiry and expiry-compute_deadline>=6*3600-1
    for n,h in read(MANIFEST).items():assert sha(REPO/n)==h,n
    save(root/'plan.json',dict(created_unix=time.time(),compute_deadline_unix=compute_deadline,
        rental_expiry_estimate_unix=expiry,export_reserve_hours=6,cases=ORDER,updates=UPDATES,
        ranks=RANKS,default_rank=79,rank_gate='All8validation conclusive; compatible measured shape profile;10% service saving.',
        sources='Four previously inspected corpora, three new fitted configurations each.',
        method='Centroid-protected residual PCA, integer-safe bounds,16 endpoint-gain corrections maximum, measured-cost batch routing.',
        baseline_rule='PCA gets the identical rank gate and router. Fresh Full and ZKML primitive adapter controls.',
        timing='Cold component sums and one eight-update warm repetition per fit. Setup/calibration provenance retained.',
        native_checks='All ranks and outcomes retained. No classification accuracy or universal baseline superiority claim.',
        source_manifest_sha256=sha(MANIFEST),resource_profile=server_profile()))
    save(root/'plan_pin.json',dict(plan_sha256=sha(root/'plan.json'),manifest_sha256=sha(MANIFEST)))
    owner=psutil.Process()
    save(root/'watchdog.json',dict(complete=False,compute_deadline_unix=compute_deadline,
         owner=dict(pid=owner.pid,created=owner.create_time(),module='campaigns.server_final_method_20261102')))
    watch=subprocess.Popen([sys.executable,'-u','-m','campaigns.server_final_method_20261102','--root',root,'--watchdog'],
                           stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    lock=(root.parent/'server_campaign_20261101/coordinator.lock').open('a');campaign=None
    def interrupted(signum,frame):raise RuntimeError('Final study interrupted by signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        while True:
            if time.time()+4320>compute_deadline or(root/'STOP').exists():raise RuntimeError('No safe admission time remains')
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                save(root/'waiting.json',dict(unix=time.time(),status='waiting_for_prior_phase_drain'));time.sleep(5)
        narrow=root.parent/'server_calibrated_routing_20261102'
        assert (narrow/'finished.json').exists(),'Prior coordinator must release its owned work first'
        save(root/'cost_profiles.json',collect_profiles(REPO))
        campaign=Campaign(root,compute_deadline-20)
        smoke(campaign,root)
        forecast=3600.
        with threadpool_limits(limits=1):
            for dataset,seed in ORDER:
                name=f'{dataset}_seed{seed}_n128_q8'
                if time.time()+1.2*forecast>compute_deadline-20 or psutil.disk_usage(root).free<24*2**30:
                    save(root/'exclusions'/(name+'.json'),dict(reason='Predeclared time/disk admission',complete=False));continue
                started=time.monotonic()
                try:run_case(campaign,root,dataset,seed,read(root/'cost_profiles.json')['profiles'])
                except Exception as error:
                    save(root/'failures'/(name+'.json'),dict(complete=False,error=str(error),unix=time.time()))
                    if (root/'STOP').exists():raise
                forecast=max(forecast,time.monotonic()-started)
        campaign.state.update(complete=True,status='queue_finished',active_phase=None,finished_unix=time.time());campaign.flush()
    except BaseException as error:
        save(root/'failure.json',dict(error=str(error),unix=time.time()))
        if campaign:
            campaign.state.update(complete=False,status='stopped_with_evidence',error=str(error));campaign.flush()
        raise
    finally:
        stop_study(root,[os.getpid(),watch.pid])
        state=read(root/'watchdog.json');state.update(complete=True,unix=time.time());save(root/'watchdog.json',state)
        watch.wait(timeout=10);summarize(root);lock.close()
        save(root/'finished.json',dict(unix=time.time(),queue_complete=bool(campaign and campaign.state.get('complete'))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--compute-deadline',type=float)
    p.add_argument('--expiry',type=float);p.add_argument('--watchdog',action='store_true');a=p.parse_args()
    assert a.root.resolve()==ROOT.resolve()
    watchdog(a.root) if a.watchdog else run(a.root,a.compute_deadline,a.expiry)
