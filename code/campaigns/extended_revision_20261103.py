"""Renewal-bounded revision evidence; reuse pinned methods without retuning."""
from pathlib import Path
import argparse
import datetime
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import numpy as np
import psutil
from threadpoolctl import threadpool_limits
from campaigns import paper_completion_20261102 as completion
from campaigns import paper_gap_checks_20261103 as timing
from campaigns import review_completion_20261103 as moments
from campaigns import server_final_method_20261102 as final
from campaigns.cost_aware_pilot import copy_setup,full_export
from campaigns.proof_runtime import server_profile
from campaigns.published_baselines_laptop import GO,RUST
from campaigns.server_pca_advantage_20261102 import Campaign
from commitjl.final_method_20261102 import fit_bases,validation_choice,RANKS
from commitjl.pilot_io import array_sha
from experiments.pca_advantage_20261102 import read,save,sha,source,trace,evaluate,proposal_banks
from experiments.release_workflow_20261103 import candidate,moments_upper
from tools.server_pca_advantage_lease_20261102 import stop_study

REPO=Path(__file__).resolve().parents[2]
BASE=REPO/'code/validation/outputs'
MAIN=BASE/'server_final_method_20261102'
ROOT=BASE/'extended_revision_20261103'
MODULE='campaigns.extended_revision_20261103'
PREVIOUS=['paper_gap_checks_20261103','review_completion_20261103','review_fit_replications_20261103']
FITS=[(d,s) for s in [907,911,919] for d in ['fashion','har','mnist','text']]
PUBLISH=completion.publish


def publish(root,name,paths):
    return PUBLISH(root,'extended_revision_'+name,paths)


def freeze(expiry):
    assert server_profile() and not ROOT.exists() and GO.is_file() and RUST.is_file()
    assert time.time()+12*3600<expiry<time.time()+36*3600
    prior=read(BASE/'review_fit_replications_20261103/plan.json')
    sources=dict(prior['source_sha256'])
    for name,h in sources.items():assert sha(REPO/name)==h,name
    sources[str(Path(__file__).relative_to(REPO))]=sha(__file__)
    sources['code/tests/test_extended_revision_20261103.py']=sha(REPO/'code/tests/test_extended_revision_20261103.py')
    inputs={str((MAIN/'cost_profiles.json').relative_to(REPO)):sha(MAIN/'cost_profiles.json')}
    for dataset,seed in FITS:
        original=MAIN/'cases'/f'{dataset}_seed{seed}_n128_q8'
        for name in ['arrays.npz','selection.json','provenance.json','result.json']:
            inputs[str((original/name).relative_to(REPO))]=sha(original/name)
    ROOT.mkdir()
    save(ROOT/'plan.json',dict(created_unix=time.time(),source_sha256=sources,
         sealed_input_sha256=inputs,
         compute_deadline_unix=expiry-3*3600,rental_expiry_estimate_unix=expiry,
         expiry_source='User says renewed expiry is8pm tomorrow,4November2026,Tehran.',export_reserve_hours=3,
         go_sha256=sha(GO),rust_sha256=sha(RUST),main_plan_sha256=sha(MAIN/'plan.json'),
         fits=[dict(dataset=d,seed=s) for d,s in FITS],method_unchanged=True,
         tasks=['Measured moments-first/Full baseline on every one of12main fits; reuse only completed identical earlier cells.',
                'Three fresh paired recurring executions of ours/PCA on all12fits; alternate arm order and retain ranges. '
                'Completed identical3-repeat blocks from earlier gap queue may be referenced without rerunning.',
                'Two proof-verified release checkpoints on HAR929, each with four fixed-rate minibatch proposals and all four arms.',
                'Fixed-configuration four-arm scale: Fashion256/Q1,HAR256/Q1,HAR512/Q1; Fashion512 is the preceding queue.',
                'If admitted, fresh four-arm control execution plus two additional Full/adapter warm executions on seed907 of each source.',
                'If admitted, two proof-verified release checkpoints on Fashion929.',
                'Seal portable per-cell evidence and a final metadata/source/index archive; keep failures/exclusions.'],
         workflow='At each checkpoint, fit/authenticate for the deployed bank and audit four0.25-rate '
                  'training-minibatch candidates on the held-out catalog. Install the first publicly verified '
                  'strict improvement in the fixed candidate order, otherwise retain the bank. '
                  'Each checkpoint starts a fresh audit worker; activation cost is charged for all arms, '
                  'even if the previous reference is retained. No production chronology or encoder correctness claim.',
         scale_scope='Fixed N128 validation-selected ranks; routing profiles at other catalog sizes explicitly extrapolate. '
                     'New proving/replay times are measured; shape extrapolation is not new measured-profile rank validation.',
         admission='Serial12worker guarded phases.20% whole-block forecast headroom,24GiB disk (32GiB forN512). '
                   'RSS108GiB cap,RAM12GiB/disk16GiB reserves. Independent deadline/owner watchdog. '
                   'No same-cohort retries or outcome filtering. Incomplete earlier attempts remain '
                   'separately referenced when a renewed-budget replication starts.',
         interpretation='Post-review diagnostics on inspected corpora. Technical repeats are not new fits. '
                        'The separately queued five new native fit seeds per source measure coverage variation only. '
                        'Original study plans, deadlines and exclusions remain unchanged.'))
    save(ROOT/'plan_pin.json',dict(sha256=sha(ROOT/'plan.json')))
    for name in sources:
        dest=ROOT/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/name,dest)
    publish(ROOT,'sources_and_plan',[ROOT/'plan.json',ROOT/'plan_pin.json',ROOT/'sources'])


def copied_case(case,dataset,seed):
    original=MAIN/'cases'/f'{dataset}_seed{seed}_n128_q8'
    case.mkdir(parents=True,exist_ok=False)
    for name in ['arrays.npz','selection.json','provenance.json']:shutil.copy2(original/name,case/name)
    selected=read(case/'selection.json')
    return case,selected['choices'],selected['preparation_seconds'],read(case/'provenance.json')['scale']


def exact_repeat(campaign,case,rep):
    source=case/'shared_full/full';out=case/'control_repeats'/f'full_rep{rep:02}'
    inp=read(source/'input.json')
    with np.load(case/'arrays.npz') as a:inp['reuse_centers']=[a['old'].tolist()]+a['centers'].tolist()
    copy_setup(source/'bundle',out/'private','full');save(out/'input.json',inp)
    one=final.go(campaign,case.name+f'_full_control_rep{rep:02}','full_reuse',out/'private',out/'input.json',['-q','8'])
    full_export(out/'private',out/'public')
    save(out/'task.json',dict(reuse_centers=inp['reuse_centers'],queries=inp['queries']))
    two=final.go(campaign,case.name+f'_full_control_verify{rep:02}','verify',out/'public',extra=[
        '-q','8','-trust',source/'owner_pin.json','-public-task',out/'task.json'])
    costs={}
    for sample in read(out/'private/full_reuse_8.json')['samples']:
        index=sample.get('centre_set',0);costs[index]=costs.get(index,0)+int(sample['psi'])
    with np.load(case/'arrays.npz') as a:
        for index,new in enumerate(a['centers'],1):
            assert costs[index]-costs[0]==moments_upper(a['rows'],a['old'],new)[0]
    return one['wall_seconds']+two['wall_seconds']


def controls(campaign,dataset):
    case=ROOT/'cases'/f'{dataset}_seed907_n128_q8_controls'
    oldfit,oldsummarize,oldretire=final.fit_case,final.summarize,final.retire_and_archive
    original_archive=oldretire
    def retire(root,folder):
        initial=read(folder/'result.json');adapter=folder/(folder.name+'_adapter');records=[]
        for rep in [1,2]:
            order=['full','adapter'] if rep==1 else ['adapter','full']
            for method in order:
                if method=='full':seconds=exact_repeat(campaign,folder,rep)
                else:
                    target=folder/'control_repeats'/f'adapter_rep{rep:02}';target.mkdir(parents=True)
                    for name in ['zkml_private','zkml_public','ordered_head']:
                        (target/name).symlink_to((adapter/name).resolve(),target_is_directory=True)
                    for name in ['zkml_owner_pin.json','approved_task.json','zkml_checked.json']:
                        shutil.copy2(adapter/name,target/name)
                    class Prefix:
                        def invoke(self,label,command,*args,**kwargs):
                            return campaign.invoke(f'control_rep{rep:02}_'+label,command,*args,**kwargs)
                    final.adapter_warm(Prefix(),folder,target)
                    seconds=read(target/'warm_checked.json')['active_seconds']
                records.append(dict(repetition=rep,method=method,seconds=seconds,order=order))
                save(folder/'control_repeats.json',dict(complete=False,records=records))
                timing.retire_owned_keys(folder/'control_repeats')
        save(folder/'control_repeats.json',dict(complete=True,initial=initial,records=records,
             scope='Two additional warm Full/adapter executions after the four-arm execution on one frozen fit. '
                   'Original adapter cold/warm process-policy difference is retained.'))
        original_archive(root,folder)
        publish(root,'controls_'+dataset,[folder,root/'plan.json',root/'plan_pin.json'])
    try:
        final.fit_case=lambda *_:copied_case(case,dataset,907)
        final.summarize=lambda *_:None;final.retire_and_archive=retire
        final.run_case(campaign,ROOT,dataset,907,read(MAIN/'cost_profiles.json')['profiles'])
    finally:final.fit_case,final.summarize,final.retire_and_archive=oldfit,oldsummarize,oldretire


def workflow(campaign,dataset):
    seed=929;training,audit,k,scale,pin=source(dataset,seed)
    reference=trace(training[:512],k,seed)[8];events=[]
    profiles=read(MAIN/'cost_profiles.json')['profiles']
    for checkpoint in [0,1]:
        case=ROOT/'cases'/f'{dataset}_seed929_n128_q4_checkpoint{checkpoint}'
        case.mkdir(parents=True,exist_ok=False)
        with threadpool_limits(limits=1):
            bases,fitting=fit_bases(training[:512],reference,seed)
            validation,_=proposal_banks(training[512:768],reference,8);records=[];preparation={}
            for family,basis in bases.items():
                tick=time.perf_counter()
                for rank in RANKS:records+=evaluate(basis,reference,audit[:128],validation,scale,family,rank)
                preparation[family]=fitting[family]+time.perf_counter()-tick
            choices={m:validation_choice(records,f,profiles[dataset])
                     for m,f in [('ordered_head','centroid'),('pca','pca')]}
        save(case/'selection.json',dict(choices=choices,preparation_seconds=preparation,
             frozen_unix=time.time(),before_test_candidate_construction=True))
        incoming=training[768+checkpoint*128:896+checkpoint*128]
        centers=np.array([candidate(batch,reference,.25) for batch in np.array_split(incoming,4)])
        np.savez_compressed(case/'arrays.npz',past=training[:512],rows=audit[128:256],old=reference,
             centers=centers,validation_rows=audit[:128],validation_centers=validation,**bases)
        save(case/'provenance.json',dict(dataset=dataset,seed=seed,scale=scale,source=pin,
             checkpoint=checkpoint,selection_sha256=sha(case/'selection.json'),reference_sha256=array_sha(reference),
             scope='Four fixed0.25rate minibatch candidates; fresh audit worker per checkpoint; all four arms charged setup.'))
        oldfit,oldsum,oldq=final.fit_case,final.summarize,final.UPDATES
        try:
            final.fit_case=lambda *_:(case,choices,preparation,scale);final.summarize=lambda *_:None;final.UPDATES=4
            final.run_case(campaign,ROOT,dataset,seed,profiles)
        finally:final.fit_case,final.summarize,final.UPDATES=oldfit,oldsum,oldq
        checked=read(case/'warm'/(case.name+'_ordered_head_final')/'checked.json')
        assert checked['all_requests_verified'] and checked['all_decisions_correct']
        # Use the verifier-produced receipt already sealed with the proof;
        # never mutate a case after its portable manifest has been generated.
        public=case/'warm'/(case.name+'_ordered_head_final')/'routed/public'
        fresh=read(public/'verified.json')
        assert fresh['complete'] and fresh['all_requests_verified']
        verified=fresh['decisions'];selected=None;strata=[]
        for i,(new,decision) in enumerate(zip(centers,verified)):
            truth,upper,oldcost=moments_upper(audit[128:256],reference,new)
            expected='improvement' if truth<0 else 'worsening' if truth>0 else 'equal'
            assert decision['decision']==expected
            if selected is None and decision['decision']=='improvement':selected=i
            strata.append(dict(request=i,truth=truth,relative_change=truth/max(oldcost,1),verified_decision=decision['decision']))
        events.append(dict(checkpoint=checkpoint,selected_improvement=selected,records=strata,
                           proof_checked_sha256=sha(case/'warm'/(case.name+'_ordered_head_final')/'checked.json'),
                           changed_reference=selected is not None,result=read(case/'result.json')))
        if selected is not None:reference=centers[selected].copy()
        save(ROOT/'workflow'/dataset/'events.json',dict(complete=checkpoint==1,events=events))
        publish(ROOT,f'workflow_{dataset}_checkpoint{checkpoint}',[case,ROOT/'workflow'/dataset,ROOT/'plan.json'])


def report():
    save(ROOT/'coverage.json',dict(unix=time.time(),completed=[str(p.relative_to(ROOT)) for p in (ROOT/'cases').glob('*/result.json')],
         imported=read(ROOT/'imported.json') if (ROOT/'imported.json').exists() else [],
         failures=[dict(name=p.name,**read(p)) for p in (ROOT/'failures').glob('*.json')],
         exclusions=[dict(name=p.name,**read(p)) for p in (ROOT/'exclusions').glob('*.json')]))


def run():
    plan=read(ROOT/'plan.json');deadline=plan['compute_deadline_unix']
    assert sha(ROOT/'plan.json')==read(ROOT/'plan_pin.json')['sha256']
    for name,h in plan['source_sha256'].items():assert sha(REPO/name)==h,name
    for name,h in plan['sealed_input_sha256'].items():assert sha(REPO/name)==h,name
    assert sha(MAIN/'plan.json')==plan['main_plan_sha256']
    assert sha(GO)==plan['go_sha256'] and sha(RUST)==plan['rust_sha256']
    # Namespace only this process's reused helpers; on-disk earlier code/plans stay pinned.
    timing.ROOT=ROOT;moments.ROOT=ROOT;completion.publish=publish
    owner=psutil.Process()
    save(ROOT/'watchdog.json',dict(complete=False,compute_deadline_unix=deadline,
         owner=dict(pid=owner.pid,created=owner.create_time(),module=MODULE)))
    watch=subprocess.Popen([sys.executable,'-u','-m',MODULE,'--watchdog'],stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    lock=(BASE/'server_campaign_20261101/coordinator.lock').open('a');campaign=Campaign(ROOT,deadline-30)
    imported=[]
    def stop(signum,frame):raise RuntimeError('Extended revision interrupted by signal '+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def task(label,forecast,callback,n=128):
        if time.time()+1.2*forecast>=deadline-60 or psutil.disk_usage(ROOT).free<(32 if n>=512 else 24)*2**30:
            save(ROOT/'exclusions'/(label+'.json'),dict(reason='Whole-block time/disk admission',unix=time.time()));return
        try:callback()
        except Exception as error:
            save(ROOT/'failures'/(label+'.json'),dict(error=str(error),unix=time.time()))
            stop_study(ROOT,[os.getpid(),watch.pid])
            for p in (ROOT/'cases').glob('*'):
                if p.is_dir():timing.retire_owned_keys(p)
        report()
    try:
        while not all((BASE/name/'finished.json').exists() for name in PREVIOUS):
            if time.time()>deadline-90:raise RuntimeError('Prior queues did not finish in time')
            save(ROOT/'waiting.json',dict(unix=time.time(),reason='Prior pinned queues remain active'));time.sleep(10)
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for dataset,seed in FITS:
            previous=BASE/'review_completion_20261103/cases'/f'{dataset}_seed{seed}_n128_q8/result.json'
            if previous.exists() and read(previous).get('complete'):
                imported.append(dict(kind='moments',source=str(previous.relative_to(REPO)),sha256=sha(previous)))
            else:task(f'moments_{dataset}_{seed}',1800,lambda d=dataset,s=seed:moments.moments_case(campaign,d,s))
            previous=BASE/'paper_gap_checks_20261103/cases'/f'{dataset}_seed{seed}_n128_q8_timing/result.json'
            if previous.exists() and read(previous).get('complete') and read(previous).get('repetitions')==3:
                imported.append(dict(kind='paired_timing',source=str(previous.relative_to(REPO)),sha256=sha(previous)))
            else:
                if previous.parent.exists():
                    imported.append(dict(kind='earlier_incomplete_timing',source=str(previous.parent.relative_to(REPO)),
                                         policy='Preserved separately; new bounded three-repetition cohort.'))
                task(f'timing_{dataset}_{seed}',4200,lambda d=dataset,s=seed:timing.paired_repeats(campaign,ROOT,d,s))
            save(ROOT/'imported.json',imported)
        task('proof_release_workflow_har929',4200,lambda:workflow(campaign,'har'))
        for dataset,n in [('fashion',256),('har',256),('har',512)]:
            task(f'scale_{dataset}_{n}',4500,lambda d=dataset,n=n:completion.proof_cell(campaign,ROOT,d,n,1),n)
        for dataset in ['fashion','har','mnist','text']:
            task('controls_'+dataset,7200,lambda d=dataset:controls(campaign,d))
        task('proof_release_workflow_fashion929',4800,lambda:workflow(campaign,'fashion'))
        campaign.state.update(complete=True,status='queue_finished',active_phase=None);campaign.flush()
    except BaseException as error:save(ROOT/'failure.json',dict(error=str(error),unix=time.time()))
    finally:
        stop_study(ROOT,[os.getpid(),watch.pid])
        state=read(ROOT/'watchdog.json');state['complete']=True;save(ROOT/'watchdog.json',state)
        watch.wait(timeout=15);lock.close();report()
        save(ROOT/'finished.json',dict(unix=time.time(),queue_complete=campaign.state.get('complete',False)))
        # Per-cell archives already contain proofs. The closing archive avoids
        # downloading a duplicate copy of every large parameter/proof bundle.
        paths=[p for p in ROOT.iterdir() if p.name not in ['cases','archives']]
        publish(ROOT,'closing_metadata_and_sources',paths)
        save(ROOT/'sealed.json',dict(complete=True,unix=time.time(),proofs_in_per_cell_archives=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--expiry',type=float)
    p.add_argument('--watchdog',action='store_true');a=p.parse_args()
    if a.freeze:freeze(a.expiry)
    elif a.watchdog:final.watchdog(ROOT)
    else:run()
