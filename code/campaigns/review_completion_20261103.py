"""Measured moments-first fallback and native release-workflow checks.

Borrow the stopped gap coordinator's exclusive reservation only after its
N512 result, draining its current guarded child normally. An independent
watchdog always stops this new cohort before resuming that coordinator.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import fcntl
import numpy as np
import psutil
from campaigns import paper_completion_20261102 as completion
from campaigns import server_final_method_20261102 as final
from campaigns.cost_aware_pilot import copy_setup,full_export
from campaigns.proof_runtime import server_profile
from campaigns.published_baselines_laptop import GO,RUST
from campaigns.server_pca_advantage_20261102 import Campaign
from experiments.pca_advantage_20261102 import read,save,sha
from experiments.release_workflow_20261103 import moments_upper
from validation.check_group_fast_proofs import issue_challenge
from validation.check_moments_only_proofs import export_public
from tools.server_pca_advantage_lease_20261102 import matches,stop_study

REPO=Path(__file__).resolve().parents[2]
BASE=REPO/'code/validation/outputs'
MAIN=BASE/'server_final_method_20261102'
GAP=BASE/'paper_gap_checks_20261103'
ROOT=BASE/'review_completion_20261103'
MODULE='campaigns.review_completion_20261103'


def restore_gap(state):
    pin=state.get('gap_owner')
    if not state.get('gap_paused') or not pin:return False
    try:
        process=psutil.Process(pin['pid'])
        if matches(process,pin):process.send_signal(signal.SIGCONT);return True
    except psutil.NoSuchProcess:pass
    return False


def watchdog():
    while True:
        s=read(ROOT/'lease.json')
        if s.get('complete'):return
        try:alive=matches(psutil.Process(s['owner']['pid']),s['owner'])
        except psutil.NoSuchProcess:alive=False
        limit=min(s['deadline_unix'],s['priority_deadline_unix']) if s.get('gap_paused') else s['deadline_unix']
        if not alive or time.time()>=limit:
            (ROOT/'STOP').write_text('Review queue deadline or owner death.\n')
            stop_study(ROOT,[os.getpid()])
            s.update(complete=True,recovered_unix=time.time(),gap_resumed=restore_gap(s))
            save(ROOT/'lease.json',s);return
        time.sleep(2)


def freeze():
    assert server_profile() and not ROOT.exists()
    assert GO.is_file() and RUST.is_file()
    gap=read(GAP/'plan.json');expiry=gap['rental_expiry_estimate_unix']
    assert time.time()<expiry-3600
    sources=dict(gap['source_sha256'])
    for name in ['code/campaigns/review_completion_20261103.py',
                 'code/experiments/release_workflow_20261103.py',
                 'code/tests/test_review_completion_20261103.py',
                 'code/docs/review_completion_20261103.md',
                 'code/validation/check_moments_only_proofs.py']:
        sources[name]=sha(REPO/name)
    for name,h in gap['source_sha256'].items():assert sha(REPO/name)==h,name
    ROOT.mkdir()
    save(ROOT/'plan.json',dict(created_unix=time.time(),source_sha256=sources,
        compute_deadline_unix=expiry-45*60,rental_expiry_estimate_unix=expiry,
        priority_lease_deadline_unix=min(expiry-145*60,time.time()+90*60),
        first_baseline=dict(dataset='fashion',seed=919,n=128,requests=8,forecast_seconds=1500),
        second_baseline=dict(dataset='har',seed=919,n=128,requests=8,forecast_seconds=1200),
        workflow_cases=[f'{d}_seed{s}_n128_q8' for s in [907,911,919] for d in ['fashion','har','mnist','text']],
        gap_pid=read(GAP/'launch.json')['pid'],gap_plan_sha256=sha(GAP/'plan.json'),
        method_unchanged=True,go_sha256=sha(GO),
        priority='Wait for completedN512 result; pause only gap coordinator, drain current child normally; '
                 'one matched moments-first/Full baseline, then resume it. Remaining checks wait for its lock.',
        timing='Projection-free authenticated old groups and best-target moment upper bound. '
               'Fresh query proof and complete public replay; all ambiguous updates get freshly generated '
               'Full candidate proofs with an equally reusable exact old reference. '
               'Setup-inclusive totals sum measured stages, matching main accounting.',
        scope='Two fixed inspected final fits chosen after review; no outcome filtering within a fit. '
              'An additional comparator, not a method retuning or a full12fit timing replication. '
              'Native simulated releases report improvements/worsenings/margins separately; '
              'no cryptographic reset latency or production field trial is inferred.',
        admission='20% forecast headroom. Existing12worker/RAM/disk guards;45minute export reserve. '
                  'No interruption of an active proof child, no rental renewal, no silent retries.'))
    save(ROOT/'plan_pin.json',dict(sha256=sha(ROOT/'plan.json')))
    for name in sources:
        dest=ROOT/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/name,dest)
    completion.publish(ROOT,'review_completion_sources_and_plan',[ROOT/'plan.json',ROOT/'plan_pin.json',ROOT/'sources'])


def moments_case(campaign,dataset,seed):
    if not (ROOT/'smoke/checks.json').exists():
        campaign.invoke('moments_public_smoke',[sys.executable,'-m','validation.check_moments_only_proofs',
                        '--root',ROOT/'smoke','--binary',GO],timeout=180)
    original=MAIN/'cases'/f'{dataset}_seed{seed}_n128_q8'
    case=ROOT/'cases'/original.name;case.mkdir(parents=True,exist_ok=False)
    inp=read(original/'ordered_head/input.json')
    inp['model'].update(method='affine',basis=[],seed='projection-free moments baseline')
    task=dict(queries=inp['queries'],comparison_base=inp['model']['centers'],reuse_centers=inp['reuse_centers'][1:])
    save(case/'input.json',inp);save(case/'approved.json',task)
    save(case/'source_pin.json',dict(arrays_sha256=sha(original/'arrays.npz'),selection_sha256=sha(original/'selection.json')))
    bundle=case/'bundle';resources={}
    def go(label,phase,target,input=None,extra=()):
        r=final.go(campaign,case.name+'_'+label,phase,target,input,extra)
        resources[label]=r;return r
    common=['-q','16','-trust',case/'owner_pin.json','-public-task',case/'approved.json']
    go('prepare','moments_prepare',bundle,case/'input.json',common)
    issue_challenge(bundle,case/'owner_pin.json',16)
    go('keys','moments_keys',bundle,case/'input.json',common)
    pin=read(case/'owner_pin.json');pin['owner']=read(bundle/'trusted.json');save(case/'owner_pin.json',pin)
    go('rows','moments_rows',bundle,case/'input.json',common)
    go('query','moments_query',bundle,case/'input.json',common)
    export_public(bundle,case/'public')
    go('replay','moments_verify',case/'public',extra=common)
    decisions=read(case/'public/moments_verified_decisions.json')
    with np.load(original/'arrays.npz') as a:rows,old,centers=[a[k].copy() for k in ['rows','old','centers']]
    records=[];fallback=[]
    for i,(new,decision) in enumerate(zip(centers,decisions)):
        truth,upper,oldcost=moments_upper(rows,old,new)
        assert int(decision['upper'])==upper and int(decision['lower'])<=truth<=upper
        assert decision['decision']==('improvement' if upper<0 else 'inconclusive')
        records.append(dict(request=i,truth=truth,upper=upper,relative_change=truth/max(oldcost,1),
                            moments_decision=decision['decision']))
        if decision['decision']=='inconclusive':fallback.append(i)
    assert len(decisions)==len(centers)==8
    init=0.;fallback_seconds=0.
    if fallback:
        source=case/'shared_full';full=read(original/'shared_full/full/input.json')
        save(source/'full/input.json',full);save(source/'protocol.json',dict(fitting_seconds=0.,proof_runtime_workers=12,phase_wall_limit_seconds=5400))
        r=campaign.invoke(case.name+'_full_initialize',[sys.executable,'-m','campaigns.server_all_sources_full_20261101',
                          '--root',source,'--run','--binary',GO,'--cases','full'])
        init=read(source/'status.json')['cases']['full']['wall_seconds']
        assert read(source/'full/owner_pin.json')['original_root']==pin['owner']['original_root'], 'Moments/Full catalog commitment differs'
        dest=case/'fallback/private';copy_setup(source/'full/bundle',dest,'full')
        fi=dict(full,reuse_centers=[old.tolist()]+[centers[i].tolist() for i in fallback])
        save(case/'fallback/input.json',fi)
        go('fallback_prove','full_reuse',dest,case/'fallback/input.json',['-q','8'])
        public=case/'fallback/public';full_export(dest,public)
        save(case/'fallback/task.json',dict(reuse_centers=fi['reuse_centers'],queries=fi['queries']))
        go('fallback_replay','verify',public,extra=['-q','8','-trust',source/'full/owner_pin.json','-public-task',case/'fallback/task.json'])
        fallback_seconds=resources['fallback_prove']['wall_seconds']+resources['fallback_replay']['wall_seconds']
        samples=read(dest/'full_reuse_8.json')['samples']
        costs={}
        for sample in samples:
            index=sample.get('centre_set',0);costs[index]=costs.get(index,0)+int(sample['psi'])
        for j,i in enumerate(fallback,1):assert costs[j]-costs[0]==records[i]['truth']
    setup=sum(resources[k]['wall_seconds'] for k in ['prepare','keys','rows'])
    warm=resources['query']['wall_seconds']+resources['replay']['wall_seconds']+fallback_seconds
    result=dict(complete=True,dataset=dataset,seed=seed,requests=8,records=records,
         moments_certificates=8-len(fallback),full_fallback_requests=len(fallback),
         warm_seconds=warm,cold_component_seconds=setup+init+warm,setup_seconds=setup,full_initialization_seconds=init,
         all_decisions_verified=True,all_decisions_correct=True,
         main_comparators=read(original/'result.json'),
         scope='Fresh measured moments-first/Full service on one fixed final fit. '
               'Setup-inclusive sum and recurring work share one execution. No new independent fit is claimed.')
    save(case/'result.json',result)
    # No more queries need these private caches; retain every public proof/key/pin.
    for p in case.rglob('*'):
        if p.is_file() and not p.is_symlink() and (p.name=='pk' or p.suffix in ['.pk','.r1cs']):
            assert p.resolve().is_relative_to(case.resolve());p.unlink()
    completion.publish(ROOT,'review_completion_'+case.name,[case,ROOT/'plan.json',ROOT/'plan_pin.json'])


def run():
    plan=read(ROOT/'plan.json');assert sha(ROOT/'plan.json')==read(ROOT/'plan_pin.json')['sha256']
    for name,h in plan['source_sha256'].items():assert sha(REPO/name)==h,name
    owner=psutil.Process();gap=psutil.Process(plan['gap_pid'])
    gap_pin=dict(pid=gap.pid,created=gap.create_time(),module='campaigns.paper_gap_checks_20261103')
    assert matches(gap,gap_pin)
    state=dict(complete=False,owner=dict(pid=owner.pid,created=owner.create_time(),module=MODULE),
               gap_owner=gap_pin,gap_paused=False,deadline_unix=plan['compute_deadline_unix'],
               priority_deadline_unix=plan['priority_lease_deadline_unix'])
    save(ROOT/'lease.json',state)
    watch=subprocess.Popen([sys.executable,'-u','-m',MODULE,'--watchdog'],stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    campaign=Campaign(ROOT,plan['compute_deadline_unix']-30);lock=None
    def stop(signum,frame):raise RuntimeError('Review queue interrupted by signal '+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def task(label,forecast,callback,deadline=None):
        limit=min(deadline or plan['compute_deadline_unix'],plan['compute_deadline_unix'])
        if time.time()+1.2*forecast>=limit-60 or psutil.disk_usage(ROOT).free<24*2**30:
            save(ROOT/'exclusions'/(label+'.json'),dict(reason='Whole-task time/disk admission',unix=time.time()));return
        previous=campaign.deadline;campaign.deadline=limit-30
        try:callback()
        except Exception as error:
            save(ROOT/'failures'/(label+'.json'),dict(error=str(error),unix=time.time()))
            stop_study(ROOT,[os.getpid(),watch.pid])
        finally:campaign.deadline=previous
    try:
        # Borrow only after N512 proofs finish; do not suspend an active child.
        while time.time()+1800<plan['priority_lease_deadline_unix']:
            if (GAP/'finished.json').exists():break
            if (GAP/'cases/fashion_seed907_n512_q1/result.json').exists():
                assert matches(gap,gap_pin)
                # Persist the recovery obligation before sending SIGSTOP.
                state.update(gap_paused=True,pause_unix=time.time());save(ROOT/'lease.json',state)
                gap.send_signal(signal.SIGSTOP)
                while [p for p in gap.children(recursive=True) if p.status()!=psutil.STATUS_ZOMBIE]:
                    if time.time()+1800>=plan['priority_lease_deadline_unix']:break
                    time.sleep(1)
                if not [p for p in gap.children(recursive=True) if p.status()!=psutil.STATUS_ZOMBIE]:
                    def primary():
                        moments_case(campaign,'fashion',919)
                    task('moments_fashion919_priority',1500,primary,plan['priority_lease_deadline_unix'])
                state.update(gap_resumed=restore_gap(state),gap_paused=False,resumed_unix=time.time());save(ROOT/'lease.json',state)
                break
            save(ROOT/'waiting.json',dict(unix=time.time(),reason='N512 proof cell remains active'));time.sleep(3)
        restore_gap(state);state['gap_paused']=False;save(ROOT/'lease.json',state)
        if (ROOT/'STOP').exists():raise RuntimeError('Stopped at priority/deadline recovery; retain evidence')
        # All subsequent work takes the ordinary lock; never overlaps timings.
        lock=(BASE/'server_campaign_20261101/coordinator.lock').open('a')
        while True:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.time()>plan['compute_deadline_unix']-90:raise RuntimeError('No remaining post-gap compute window')
                time.sleep(3)
        if not (ROOT/'cases/fashion_seed919_n128_q8').exists():
            task('moments_fashion919',1500,lambda:moments_case(campaign,'fashion',919))
        task('moments_har919',1200,lambda:moments_case(campaign,'har',919))
        for name in plan['workflow_cases']:
            task('workflow_'+name,45,lambda name=name:campaign.invoke('workflow_'+name,[sys.executable,'-m',
                 'experiments.release_workflow_20261103','--source',MAIN/'cases'/name,'--out',ROOT/'workflow'/name,
                 '--profiles',MAIN/'cost_profiles.json'],timeout=300))
        campaign.state.update(complete=True,status='queue_finished',active_phase=None);campaign.flush()
    except BaseException as error:save(ROOT/'failure.json',dict(error=str(error),unix=time.time()))
    finally:
        stop_study(ROOT,[os.getpid(),watch.pid])
        resumed=restore_gap(state);state.update(complete=True,gap_resumed=state.get('gap_resumed',False) or resumed,gap_paused=False)
        save(ROOT/'lease.json',state);watch.wait(timeout=15)
        if lock:lock.close()
        save(ROOT/'finished.json',dict(unix=time.time(),queue_complete=campaign.state.get('complete',False)))
        completion.publish(ROOT,'review_completion_complete_evidence',[p for p in ROOT.iterdir() if p.name!='archives'])
        save(ROOT/'sealed.json',dict(complete=True,unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--watchdog',action='store_true')
    a=p.parse_args()
    if a.freeze:freeze()
    elif a.watchdog:watchdog()
    else:run()
