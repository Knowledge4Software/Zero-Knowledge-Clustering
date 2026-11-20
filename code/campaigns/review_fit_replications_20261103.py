"""Five additional pinned fit seeds per source; native coverage, not proof time."""
from pathlib import Path
import argparse
import fcntl
import os
import shutil
import signal
import subprocess
import sys
import time
import numpy as np
import psutil
from threadpoolctl import threadpool_limits
from campaigns import server_final_method_20261102 as final
from campaigns.paper_completion_20261102 import publish
from campaigns.published_baselines_laptop import GO
from campaigns.server_pca_advantage_20261102 import Campaign
from commitjl.calibrated_routing_20261102 import BatchCosts,CalibratedPlanner
from experiments.pca_advantage_20261102 import read,save,sha
from experiments.release_workflow_20261103 import moments_upper
from tools.server_pca_advantage_lease_20261102 import stop_study

REPO=Path(__file__).resolve().parents[2]
BASE=REPO/'code/validation/outputs'
MAIN=BASE/'server_final_method_20261102'
REVIEW=BASE/'review_completion_20261103'
ROOT=BASE/'review_fit_replications_20261103'
MODULE='campaigns.review_fit_replications_20261103'
SEEDS=[929,937,947,953,967]


def freeze():
    assert not ROOT.exists() and set(SEEDS).isdisjoint([907,911,919])
    previous=read(REVIEW/'plan.json')
    assert time.time()<previous['compute_deadline_unix']-120
    ROOT.mkdir()
    sources=dict(previous['source_sha256'])
    sources[str(Path(__file__).relative_to(REPO))]=sha(__file__)
    for name,h in previous['source_sha256'].items():assert sha(REPO/name)==h,name
    save(ROOT/'plan.json',dict(created_unix=time.time(),source_sha256=sources,
         compute_deadline_unix=previous['compute_deadline_unix'],go_sha256=sha(GO),
         seeds=SEEDS,cases=[dict(dataset=d,seed=s) for s in SEEDS for d in ['fashion','har','mnist','text']],
         method_unchanged=True,cost_profiles_sha256=sha(MAIN/'cost_profiles.json'),
         scope='Five additional seed-selected fits per source, frozen before new outcomes. '
               'Native validation/test intervals and routing coverage only; no new proof time, '
               'independent-corpus or disjoint-row population claim. Eight updates per fit. '
               'Source rows may overlap earlier fits; same frozen training/validation/test role recipe.',
         order='Wait for review-completion queue and acquire exclusive benchmark lock.',
         admission='24seconds remaining per new fit plus30second reserve,24GiB disk; '
                   'native BLAS1 and guarded120second maximum per fit. Retain every failure/exclusion.'))
    save(ROOT/'plan_pin.json',dict(sha256=sha(ROOT/'plan.json')))
    shutil.copy2(MAIN/'cost_profiles.json',ROOT/'cost_profiles.json')
    for name in sources:
        dest=ROOT/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/name,dest)
    publish(ROOT,'review_fit_replications_sources_and_plan',[ROOT/'plan.json',ROOT/'plan_pin.json',ROOT/'sources'])


def fit(dataset,seed):
    profiles=read(ROOT/'cost_profiles.json')['profiles']
    with threadpool_limits(limits=1):
        case,choices,preparation,scale=final.fit_case(ROOT,dataset,seed,profiles)
        with np.load(case/'arrays.npz') as z:a={k:z[k].copy() for k in z.files}
        routes={}
        for method,choice in choices.items():
            model=final.model_for(a[choice['family']][:choice['rank']],a['old'],128,scale)
            routes[method]=CalibratedPlanner(model,a['rows'],BatchCosts(**profiles[dataset][str(choice['rank'])]['costs'])).batch(a['centers'])
        records=[]
        for i,new in enumerate(a['centers']):
            truth,upper,oldcost=moments_upper(a['rows'],a['old'],new)
            records.append(dict(request=i,truth=truth,relative_change=truth/max(oldcost,1),
                                moments_conclusive=upper<0,
                                our_certificate=i in routes['ordered_head']['certificate'],
                                pca_certificate=i in routes['pca']['certificate']))
        save(case/'result.json',dict(complete=True,dataset=dataset,seed=seed,requests=8,records=records,
             ranks={m:c['rank'] for m,c in choices.items()},native_only=True,
             all_numerical_intervals_checked=True,selection_sha256=sha(case/'selection.json')))


def run():
    plan=read(ROOT/'plan.json');deadline=plan['compute_deadline_unix']
    assert sha(ROOT/'plan.json')==read(ROOT/'plan_pin.json')['sha256']
    for name,h in plan['source_sha256'].items():assert sha(REPO/name)==h,name
    assert sha(ROOT/'cost_profiles.json')==plan['cost_profiles_sha256']
    owner=psutil.Process()
    save(ROOT/'watchdog.json',dict(complete=False,compute_deadline_unix=deadline,
         owner=dict(pid=owner.pid,created=owner.create_time(),module=MODULE)))
    watch=subprocess.Popen([sys.executable,'-u','-m',MODULE,'--watchdog'],stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    lock=(BASE/'server_campaign_20261101/coordinator.lock').open('a');campaign=Campaign(ROOT,deadline-30)
    def stop(signum,frame):raise RuntimeError('Fit queue interrupted by signal '+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        while not (REVIEW/'finished.json').exists():
            if time.time()>deadline-60:raise RuntimeError('No remaining native-fit window')
            save(ROOT/'waiting.json',dict(unix=time.time(),reason='Review completion is still active'));time.sleep(5)
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for cell in plan['cases']:
            dataset,seed=cell['dataset'],cell['seed'];label=f'{dataset}_seed{seed}'
            if time.time()+24>=deadline-30 or psutil.disk_usage(ROOT).free<24*2**30:
                save(ROOT/'exclusions'/(label+'.json'),dict(reason='Time/disk admission'));continue
            try:
                campaign.invoke(label,[sys.executable,'-m',MODULE,'--dataset',dataset,'--seed',str(seed)],timeout=120)
            except Exception as error:
                save(ROOT/'failures'/(label+'.json'),dict(error=str(error),unix=time.time()))
        campaign.state.update(complete=True,status='queue_finished',active_phase=None);campaign.flush()
    except BaseException as error:save(ROOT/'failure.json',dict(error=str(error),unix=time.time()))
    finally:
        stop_study(ROOT,[os.getpid(),watch.pid])
        state=read(ROOT/'watchdog.json');state['complete']=True;save(ROOT/'watchdog.json',state)
        watch.wait(timeout=15);lock.close()
        save(ROOT/'finished.json',dict(unix=time.time(),queue_complete=campaign.state.get('complete',False)))
        publish(ROOT,'review_fit_replications_complete_evidence',[p for p in ROOT.iterdir() if p.name!='archives'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--watchdog',action='store_true')
    p.add_argument('--dataset');p.add_argument('--seed',type=int);a=p.parse_args()
    if a.freeze:freeze()
    elif a.watchdog:final.watchdog(ROOT)
    elif a.dataset:fit(a.dataset,a.seed)
    else:run()
