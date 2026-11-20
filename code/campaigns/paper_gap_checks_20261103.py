"""Separate, bounded post-study scale and paired timing checks.

Never modify the sealed November 2 cohorts or tune the frozen method.
"""
from pathlib import Path
import argparse
import fcntl
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time

import numpy as np
import psutil

from campaigns import paper_completion_20261102 as completion
from campaigns import server_final_method_20261102 as final
from campaigns.cost_aware_pilot import prove_case
from campaigns.proof_runtime import server_profile
from campaigns.published_baselines_laptop import GO, RUST
from campaigns.server_pca_advantage_20261102 import Campaign
from commitjl.calibrated_routing_20261102 import BatchCosts, CalibratedPlanner
from experiments.pca_advantage_20261102 import read, save, sha
from tools.server_pca_advantage_lease_20261102 import stop_study

REPO = Path(__file__).resolve().parents[2]
BASE = REPO/'code/validation/outputs'
MAIN = BASE/'server_final_method_20261102'
ROOT = BASE/'paper_gap_checks_20261103'
MODULE = 'campaigns.paper_gap_checks_20261103'
REPETITIONS = 3
CELLS = [dict(kind='scale', dataset='fashion', seed=907, n=512, q=1,
              forecast_seconds=5400, disk_gib=32),
         dict(kind='timing', dataset='har', seed=919, n=128, q=8,
              forecast_seconds=4200, disk_gib=24),
         dict(kind='timing', dataset='fashion', seed=907, n=128, q=8,
              forecast_seconds=2100, disk_gib=24)]


def admitted(cell, now, deadline, free_bytes):
    return (now+1.2*cell['forecast_seconds'] < deadline-60
            and free_bytes >= cell['disk_gib']*2**30)


def order_for(repetition, initial):
    return list(initial if repetition % 2 else reversed(initial))


def freeze(root, deadline):
    assert root.resolve() == ROOT.resolve() and server_profile()
    assert not root.exists(), 'Keep all earlier attempts and exclusions'
    expiry = read(MAIN/'plan.json')['rental_expiry_estimate_unix']
    assert time.time()+600 < deadline <= expiry-3600
    assert read(MAIN/'finished.json')['queue_complete']
    assert read(MAIN/'seal_status.json')['complete']
    assert read(completion.ROOT/'finished.json')['queue_complete']
    assert GO.is_file() and RUST.is_file(), 'Supply the pinned server binary paths before initialization'
    root.mkdir(parents=True)
    dependencies = set(read(REPO/'final_method_deployment_manifest_20261102.json'))
    dependencies.update(read(REPO/'paper_completion_deployment_manifest_20261102.json'))
    dependencies.update([str(Path(__file__).relative_to(REPO)),
                         'code/campaigns/cost_aware_pilot.py',
                         'code/campaigns/server_pca_advantage_20261102.py',
                         'code/tools/server_guard_20261101.py'])
    sources = {name:sha(REPO/name) for name in sorted(dependencies)}
    for manifest in ['final_method_deployment_manifest_20261102.json',
                     'paper_completion_deployment_manifest_20261102.json']:
        for name, digest in read(REPO/manifest).items():
            assert sources[name] == digest, name
    pins = {str(p.relative_to(REPO)):sha(p) for p in
            [MAIN/'plan.json', MAIN/'plan_pin.json', MAIN/'cost_profiles.json',
             MAIN/'summary.json', completion.ROOT/'plan.json',
             completion.ROOT/'finished.json']}
    for dataset, seed in [('fashion',907),('har',919)]:
        original = MAIN/'cases'/f'{dataset}_seed{seed}_n128_q8'
        assert read(original/'result.json')['complete']
        for name in ['arrays.npz','selection.json','provenance.json','result.json']:
            pins[str((original/name).relative_to(REPO))] = sha(original/name)
    save(root/'plan.json', dict(created_unix=time.time(),compute_deadline_unix=deadline,
         rental_expiry_estimate_unix=expiry,export_reserve_hours=1,
         authorization='User requested any still-needed paper experiments on November3. '
                       'Use a new bounded window; preserve the original expired compute cutoff.',
         cells=CELLS,paired_warm_repetitions=REPETITIONS,source_sha256=sources,
         sealed_input_sha256=pins,go_sha256=sha(GO),rust_sha256=sha(RUST),
         method_unchanged=True,original_deadlines_unchanged=True,
         timing='Fresh proof generation and fresh public replay; setup reused within a block. '
                'Native preflight charged anew. Copies/export and independent exact checks excluded equally.',
         repetition_order='PCA/ours,ours/PCA,PCA/ours, fixed before measurements.',
         scope='Post-study fixed-fit diagnostics; timing repetitions are not independent fits. '
               'N512 uses frozen N128-selected ranks and explicitly extrapolated routing costs. '
               'Fresh proof times are measured. No unseen-data, universal superiority or significance claim.',
         admission='20% forecast headroom and32GiB disk for scale,24GiB otherwise. '
                   'Serial12worker phases,108GiB RSS cap,12GiB RAM and16GiB disk reserves. '
                   'Skip a block if the whole forecast does not fit; retain every partial attempt.',
         no_original_result_edits=True))
    save(root/'plan_pin.json',dict(plan_sha256=sha(root/'plan.json')))
    for name in sources:
        dest=root/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(REPO/name,dest)
    completion.publish(root,'paper_gap_checks_sources_and_plan',
                       [root/'sources',root/'plan.json',root/'plan_pin.json'])


def retire_owned_keys(folder):
    folder=folder.resolve()
    assert folder.is_relative_to(ROOT.resolve())
    records=[]
    for p in sorted(folder.rglob('*')):
        if p.is_file() and not p.is_symlink() and (p.name=='pk' or p.suffix in ['.pk','.r1cs']):
            assert p.resolve().is_relative_to(folder)
            records.append(dict(path=str(p.relative_to(folder)),bytes=p.stat().st_size,sha256=sha(p)))
            p.unlink()
    save(folder/'retired_timing_copy_keys.json',dict(files=records,public_evidence_preserved=True))


def paired_repeats(campaign,root,dataset,seed):
    original=MAIN/'cases'/f'{dataset}_seed{seed}_n128_q8'
    case=root/'cases'/f'{dataset}_seed{seed}_n128_q8_timing'
    case.mkdir(parents=True,exist_ok=False)
    for name in ['arrays.npz','selection.json','provenance.json']:
        shutil.copy2(original/name,case/name)
    selection=read(case/'selection.json');choices=selection['choices']
    scale=read(case/'provenance.json')['scale']
    final.write_inputs(case,choices,scale)
    campaign.invoke(case.name+'_full_initialize',
        [sys.executable,'-m','campaigns.server_all_sources_full_20261101',
         '--root',case/'shared_full','--run','--binary',GO,'--cases','full'])
    profiles=read(MAIN/'cost_profiles.json')['profiles'][dataset]
    sources={};templates={};plans={}
    for method in ['pca','ordered_head']:
        alias,out=final.setup_method(campaign,case,method,profiles[str(choices[method]['rank'])])
        sources[method]=alias;templates[method]=out
        plans[method]=read(out/'planner.json')
    records=[]
    with np.load(case/'arrays.npz') as a:rows=a['rows'].copy();centers=a['centers'].copy()
    for rep in range(1,REPETITIONS+1):
        for method in order_for(rep,['pca','ordered_head']):
            out=case/'warm'/f'{case.name}_{method}_rep{rep:02}'
            out.mkdir(parents=True)
            for name in ['owner_pin.json','approved.json']:
                shutil.copy2(templates[method]/name,out/name)
            model=read(case/method/'input.json')['model']
            plan=CalibratedPlanner(model,rows,BatchCosts(**profiles[str(choices[method]['rank'])]['costs'])).batch(centers)
            for key in ['certificate','full']:
                assert plan[key]==plans[method][key], 'Frozen policy changed across repeats'
            assert [r['refinement_indices'] for r in plan['records']]==[
                r['refinement_indices'] for r in plans[method]['records']]
            save(out/'planner.json',plan)
            prove_case(campaign,case/'warm',dict(name=out.name,requests=8),
                       source=sources[method],run_baseline=False)
            checked=read(out/'checked.json')
            assert checked['all_requests_verified'] and checked['all_decisions_correct']
            records.append(dict(repetition=rep,method=method,execution_order=order_for(rep,['pca','ordered_head']),
                                seconds=checked['measured_warm_routed_seconds'],
                                certificate_requests=checked['certificate_requests'],
                                correction_proofs=checked['selected_correction_proofs'],
                                checked_sha256=sha(out/'checked.json')))
            save(case/'timing_repetitions.json',dict(complete=False,records=records))
            # Keep source proving keys until every fresh repetition completes.
            retire_owned_keys(out)
    result=dict(complete=True,dataset=dataset,seed=seed,repetitions=REPETITIONS,
                all_decisions_correct=True,records=records,original_fit=str(original.relative_to(REPO)),
                original_arrays_sha256=sha(original/'arrays.npz'),
                original_selection_sha256=sha(original/'selection.json'),
                ranks={m:choices[m]['rank'] for m in ['ordered_head','pca']},
                summary={m:dict(mean=statistics.mean(r['seconds'] for r in records if r['method']==m),
                               minimum=min(r['seconds'] for r in records if r['method']==m),
                               maximum=max(r['seconds'] for r in records if r['method']==m))
                         for m in ['ordered_head','pca']},
                scope='Three technical timing repetitions of one already inspected fixed fit, '
                      'not new fit replications. No new Full/adapter timing is inferred.',finished_unix=time.time())
    save(case/'result.json',result);save(case/'timing_repetitions.json',result)
    retire_owned_keys(case)
    completion.publish(root,'paper_gap_checks_'+case.name,[case,root/'plan.json',root/'plan_pin.json'])


def report(root):
    results=[dict(case=p.parent.name,**read(p)) for p in sorted((root/'cases').glob('*/result.json'))]
    save(root/'coverage.json',dict(updated_unix=time.time(),results=results,
         failures=[dict(case=p.stem,**read(p)) for p in sorted((root/'failures').glob('*.json'))],
         exclusions=[dict(case=p.stem,**read(p)) for p in sorted((root/'exclusions').glob('*.json'))]))


def run(root):
    plan=read(root/'plan.json');deadline=plan['compute_deadline_unix']
    assert sha(root/'plan.json')==read(root/'plan_pin.json')['plan_sha256']
    for name,digest in {**plan['source_sha256'],**plan['sealed_input_sha256']}.items():
        assert sha(REPO/name)==digest,name
    assert sha(GO)==plan['go_sha256'] and sha(RUST)==plan['rust_sha256']
    owner=psutil.Process()
    save(root/'watchdog.json',dict(complete=False,compute_deadline_unix=deadline,
         owner=dict(pid=owner.pid,created=owner.create_time(),module=MODULE)))
    watch=subprocess.Popen([sys.executable,'-u','-m',MODULE,'--watchdog'],
                           stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,start_new_session=True)
    lock=(BASE/'server_campaign_20261101/coordinator.lock').open('a');campaign=None
    def interrupted(signum,frame):raise RuntimeError('Gap checks interrupted by signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        campaign=Campaign(root,deadline-30)
        for cell in plan['cells']:
            label=f"{cell['dataset']}_seed{cell['seed']}_n{cell['n']}_q{cell['q']}_{cell['kind']}"
            if not admitted(cell,time.time(),deadline,psutil.disk_usage(root).free):
                save(root/'exclusions'/(label+'.json'),dict(reason='Whole-block time/disk admission',cell=cell,unix=time.time()))
                report(root);continue
            try:
                if cell['kind']=='scale':
                    completion.proof_cell(campaign,root,cell['dataset'],cell['n'],cell['q'])
                else:paired_repeats(campaign,root,cell['dataset'],cell['seed'])
            except Exception as error:
                save(root/'failures'/(label+'.json'),dict(error=str(error),unix=time.time()))
                # Do not overlap any orphaned phase with the next timing block.
                stop_study(root,[os.getpid(),watch.pid])
                # Retire only this new campaign's rebuildable private key caches.
                for case in (root/'cases').glob('*'):
                    if case.is_dir():retire_owned_keys(case)
            report(root)
        campaign.state.update(complete=True,status='queue_finished',active_phase=None);campaign.flush()
    except BaseException as error:
        save(root/'failure.json',dict(error=str(error),unix=time.time()))
    finally:
        stop_study(root,[os.getpid(),watch.pid])
        state=read(root/'watchdog.json');state.update(complete=True);save(root/'watchdog.json',state)
        watch.wait(timeout=15);lock.close();report(root)
        save(root/'finished.json',dict(unix=time.time(),queue_complete=bool(campaign and campaign.state.get('complete'))))
        completion.publish(root,'paper_gap_checks_complete_evidence',
                           [p for p in root.iterdir() if p.name!='archives'])
        save(root/'sealed.json',dict(complete=True,unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true')
    p.add_argument('--deadline',type=float);p.add_argument('--watchdog',action='store_true')
    a=p.parse_args()
    if a.freeze:freeze(ROOT,a.deadline)
    elif a.watchdog:final.watchdog(ROOT)
    else:run(ROOT)
