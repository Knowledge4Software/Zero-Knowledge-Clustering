"""Non-preempting, deadline-bounded completion of the paper's four RQs."""
from pathlib import Path
import argparse
import copy
import csv
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time

import numpy as np
import psutil

from campaigns import server_final_method_20261102 as final
from campaigns.server_pca_advantage_20261102 import Campaign
from campaigns.published_baselines_laptop import GO, RUST
from experiments.pca_advantage_20261102 import read, save, sha, source
from commitjl.pilot_io import array_sha
from experiments.paper_completion_20261102 import diagnostic_arrays, scaled_profiles
from tools.server_pca_advantage_lease_20261102 import matches, stop_study

REPO = Path(__file__).resolve().parents[2]
BASE = REPO/'code/validation/outputs'
MAIN = BASE/'server_final_method_20261102'
ROOT = BASE/'paper_completion_20261102'
MANIFEST = REPO/'paper_completion_deployment_manifest_20261102.json'
SOURCES = ['fashion', 'har', 'mnist', 'text', 'gisette', 'arcene', 'text_tfidf',
           'har_neural256', 'gaussian_easy', 'gaussian_hard', 'dbscan_margin']
# These are diagnostic fixed-configuration cells. Original choices are reused,
# never reselected on a prefix, stress case or larger test catalog.
PROOFS = [('fashion', 128, 1, 1800), ('har', 128, 1, 1500),
          ('fashion', 128, 4, 2700), ('har', 128, 4, 2400),
          ('fashion', 32, 1, 1500), ('fashion', 512, 1, 5400)]


def inventory(root):
    broad = BASE/'server_all_sources_20261101'
    native = []
    for p in sorted((broad/'native_cases').glob('*/native_results.json')):
        v = read(p)
        native.append(dict(case=p.parent.name, complete=v.get('complete'), sha256=sha(p)))
    cold = [dict(case=p.parent.name, sha256=sha(p))
            for p in sorted((broad/'matched').glob('*/cold_complete.json'))]
    exclusions = [dict(case=p.stem, reason=read(p).get('reason'))
                  for p in sorted((broad/'exclusions').glob('*.json'))]
    v = dict(unix=time.time(), main_completed=[p.parent.name for p in (MAIN/'cases').glob('*/result.json')],
             historical_native=native, historical_cold=cold, historical_exclusions=exclusions,
             corpus_count=6, workload_count=11,
             final_method_scope='Four core datasets, three fits; historical controls on all cached corpora.',
             missing_optional_source='CIFAR10 cached embeddings are unavailable.',
             arcene_limit='Only64 historical past rows; frozen rank79 basis unsupported. Retain existing Arcene controls.')
    save(root/'coverage_inventory.json', v)
    return v


def freeze(root):
    assert root.resolve() == ROOT.resolve()
    root.mkdir(parents=True, exist_ok=True)
    assert not (root/'plan.json').exists()
    for name, digest in read(MANIFEST).items(): assert sha(REPO/name) == digest, name
    main = read(MAIN/'plan.json')
    save(root/'plan.json', dict(created_unix=time.time(), compute_deadline_unix=main['compute_deadline_unix'],
         rental_expiry_estimate_unix=main['rental_expiry_estimate_unix'], export_reserve_hours=6,
         source_manifest_sha256=sha(MANIFEST), main_plan_sha256=sha(MAIN/'plan.json'),
         method_unchanged=True, rq1='Rank16/79, cap0/4/8/16, endpoint/width, assigned-only, matched native controls.',
         rq2='Main cold/warm comparison plus fresh matched request-prefix1/4 proofs;8 from main. Larger request counts remain models.',
         rq3='Native catalog32/128/512/896; resource receipts and existing verified scale controls; fresh fixed-configuration32/512 proof cells if admitted.',
         rq4='Equal/near-reference/large updates and rebuilt-reference episodes, all outcomes retained; final-bundle tamper replay.',
         final_native_cases=[f'{d}_seed{s}_n128_q8' for s in [907,911,919] for d in ['fashion','har','mnist','text']],
         historical_fixed_rank_cases=[f'{d}_seed{s}' for s in [701,709,719] for d in SOURCES],
         proof_cells=[dict(dataset=d,n=n,requests=q,forecast_seconds=t) for d,n,q,t in PROOFS],
         scale_interpretation='Frozen N128 validation-selected basis/rank and candidate1. '
             'N32/N512 route costs extrapolate the N128 profile; not a fresh measured-profile rank selection. '
             'Measured proof times, public checks and extrapolated routing estimates stay separate.',
         order='Wait for main study and its evidence seal; exclusive lock; native/stress/tamper, then request-size and scale proof cells.',
         admission='20% forecast headroom,32GiB disk forN512,24GiB otherwise; guarded phases and independent deadline watchdog.',
         queue_policy='Never interrupt main proofs, extend rental, change frozen algorithm or silently retry/discard outcomes.',
         reporting='Exploratory diagnostics on inspected sources; shared rows and nested request prefixes are not independent replicates. '
                   'Raw projections/coresets are numerical controls and have no cryptographic sign guarantee. '
                   'ZKML primitive adapter has a different backend and stronger exact-cost output contract.',
         no_manuscript_edits=True, go_sha256=sha(GO), rust_sha256=sha(RUST)))
    save(root/'plan_pin.json', dict(plan_sha256=sha(root/'plan.json'), manifest_sha256=sha(MANIFEST)))
    inventory(root)
    for name in read(MANIFEST):
        target=root/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(REPO/name,target)
    shutil.copy2(MANIFEST,root/'sources'/MANIFEST.name)
    publish(root,'paper_completion_sources_and_plan',
            [root/'sources',root/'plan.json',root/'plan_pin.json',root/'coverage_inventory.json'])


def publish(root, name, paths):
    folder = MAIN/'archives'; folder.mkdir(exist_ok=True)
    target = folder/(name+'.tar.gz')
    if target.exists():
        assert (folder/(name+'.sha256.json')).exists()
        return
    temporary = folder/(name+'.tmp')
    with tarfile.open(temporary, 'w:gz', compresslevel=1, dereference=False) as a:
        for parent in paths:
            selected = [parent] if parent.is_file() else sorted(parent.rglob('*'))
            for p in selected:
                if not p.is_file() or p.is_symlink() or p.name=='pk' or p.suffix in ['.pk','.r1cs','.gob']:
                    continue
                assert p.resolve().is_relative_to(root.resolve())
                a.add(p, arcname=str(p.relative_to(root)), recursive=False)
    temporary.replace(target)
    save(folder/(name+'.sha256.json'), dict(file=target.name,bytes=target.stat().st_size,sha256=sha(target)))


def prepare_diagnostic(root, dataset, n, q, profiles):
    original = MAIN/'cases'/f'{dataset}_seed907_n128_q8'
    assert read(original/'result.json')['complete']
    case = root/'cases'/f'{dataset}_seed907_n{n}_q{q}'
    case.mkdir(parents=True, exist_ok=False)
    selected = read(original/'selection.json')
    provenance = read(original/'provenance.json')
    a = diagnostic_arrays(original,dataset,n,q)
    np.savez_compressed(case/'arrays.npz', **a)
    save(case/'selection.json', dict(selected, reused_without_reselection=True,
         original_selection_sha256=sha(original/'selection.json')))
    save(case/'provenance.json', dict(provenance, diagnostic_n=n, diagnostic_requests=q,
         original_arrays_sha256=sha(original/'arrays.npz'), rows_sha256=array_sha(a['rows']),
         scope='Nested request prefixes or fixed-basis catalog scaling, not independent new fits.'))
    return case, selected['choices'], selected['preparation_seconds'], provenance['scale']


def profile_for_diagnostic(profiles, dataset, n):
    return scaled_profiles(profiles,dataset,n)


def proof_cell(campaign, root, dataset, n, q):
    profiles = profile_for_diagnostic(read(MAIN/'cost_profiles.json')['profiles'],dataset,n)
    save(root/f'profiles_{dataset}_n{n}_q{q}.json', profiles)
    case, choices, prep, scale = prepare_diagnostic(root,dataset,n,q,profiles)
    old_fit, old_summary, old_updates = final.fit_case, final.summarize, final.UPDATES
    try:
        final.fit_case = lambda *_: (case,choices,prep,scale)
        final.summarize = lambda *_: None
        final.UPDATES = q
        final.run_case(campaign, root, dataset, 907, profiles)
    finally:
        final.fit_case, final.summarize, final.UPDATES = old_fit, old_summary, old_updates
    # The final helper already retained public evidence and retired owned keys.
    publish(root, 'paper_completion_'+case.name, [case, root/'plan.json',root/'plan_pin.json'])


def export_tables(root):
    reports = root/'reports'; reports.mkdir(exist_ok=True)
    rows=[]
    for p in sorted((root/'native').glob('*/result.json')):
        v=read(p)
        for key in ['ablations','operating_limits','scale','numerical_baselines']:
            rows += [dict(case=p.parent.name, kind=key, **r) for r in v.get(key,[])]
    if rows:
        keys=sorted({k for r in rows for k in r})
        with (reports/'native_controls.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    history=[]
    for p in sorted((root/'historical').glob('*/result.json')):
        history += [dict(case=p.parent.name,**r) for r in read(p).get('records',[])]
    if history:
        keys=sorted({k for r in history for k in r})
        with (reports/'historical_basis_diagnostics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(history)
    proofs=[]
    for p in sorted((root/'cases').glob('*/result.json')):
        v=read(p)
        for method in ['ordered_head','pca','full','adapter']:
            proofs.append(dict(case=p.parent.name,method=method,**v[method]))
    if proofs:
        keys=sorted({k for r in proofs for k in r})
        with (reports/'request_and_scale_proofs.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(proofs)
    save(reports/'coverage.json',dict(updated_unix=time.time(), main_complete_fits=len(list((MAIN/'cases').glob('*/result.json'))),
         final_native_cases=len(list((root/'native').glob('*/result.json'))),
         historical_diagnostics=[dict(case=p.parent.name,complete=read(p).get('complete'),excluded=read(p).get('excluded',False))
             for p in sorted((root/'historical').glob('*/result.json'))],
         fresh_diagnostic_proof_cases=len(list((root/'cases').glob('*/result.json'))),
         failures=[p.name for p in (root/'failures').glob('*.json')],
         exclusions=[p.name for p in (root/'exclusions').glob('*.json')]))
    summary=read(reports/'coverage.json')
    lines=['Paper experiment coverage','',
        f"Main final-method comparisons complete: {summary['main_complete_fits']}/12 fitted configurations.",
        f"Final-method native ablations/operating-limit cases: {summary['final_native_cases']}/12.",
        f"Fresh request-size/scale diagnostic proof cells: {summary['fresh_diagnostic_proof_cases']}/6.",
        '', 'RQ1: native rank/refinement/order/assigned-only ablations and numerical baseline signs.',
        'RQ2: main measured cold/warm8-update costs, fresh1/4-update prefixes when complete; large request counts remain explicit models.',
        'RQ3: native catalog scaling and measured proof/resource receipts; distinguish original verified scale controls from new fixed-configuration diagnostics.',
        'RQ4: equal/near-reference/large updates, rebuilt-reference episodes, and public tamper replay.',
        '', 'Use completed cells only. Read failures/exclusions before making claims. '
        'Nested prefixes and exploratory fixed-rank diagnostics are not independent fits. '
        'Arcene rank79 is unsupported by its64 cached past rows. '
        'Experiment completion does not establish a formal security proof, literature novelty or journal readiness.']
    (reports/'coverage.md').write_text('\n'.join(lines)+'\n')


def free_completed_key_caches(root, target_gib=42):
    """Retire rebuildable caches only in finished, already archived older cohorts."""
    index=read(MAIN/'backups/index.json')
    assert index.get('complete')
    retired=[]
    for name in ['server_campaign_20261101','server_diversity_20261101']:
        old=(BASE/name).resolve()
        assert old.parent==BASE.resolve() and read(old/'matched_queue.json').get('complete')
        archive=next(v for v in index['files'] if v['file']==name+'.tar.gz')
        assert (MAIN/'backups'/archive['file']).stat().st_size==archive['bytes']
        assert sha(MAIN/'backups'/archive['file'])==archive['sha256']
        for p in sorted(old.rglob('*')):
            if psutil.disk_usage(root).free>=target_gib*2**30:break
            if p.is_file() and not p.is_symlink() and (p.name=='pk' or p.suffix in ['.pk','.r1cs','.gob']):
                assert p.resolve().is_relative_to(old)
                retired.append(dict(path=str(p.relative_to(BASE)),bytes=p.stat().st_size,sha256=sha(p)))
                p.unlink()
        if psutil.disk_usage(root).free>=target_gib*2**30:break
    save(root/'old_rebuildable_cache_retirement.json',dict(files=retired,
         public_evidence_preserved=True,completed_cohort_archives_preserved=True,free_gib=psutil.disk_usage(root).free/2**30))


def run(root):
    plan=read(root/'plan.json'); deadline=plan['compute_deadline_unix']
    assert sha(root/'plan.json')==read(root/'plan_pin.json')['plan_sha256']
    for name,digest in read(MANIFEST).items(): assert sha(REPO/name)==digest,name
    owner=psutil.Process()
    save(root/'watchdog.json',dict(complete=False,compute_deadline_unix=deadline,
         owner=dict(pid=owner.pid,created=owner.create_time(),module='campaigns.paper_completion_20261102')))
    watch=subprocess.Popen([sys.executable,'-u','-m','campaigns.paper_completion_20261102','--root',root,'--watchdog'],
          stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    lock=(BASE/'server_campaign_20261101/coordinator.lock').open('a'); campaign=None
    def interrupted(signum,frame): raise RuntimeError('Completion queue interrupted by signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted)
    try:
        while True:
            if time.time()>deadline-90: raise RuntimeError('No supplemental compute window remains')
            if (MAIN/'finished.json').exists() and (MAIN/'seal_status.json').exists() and read(MAIN/'seal_status.json').get('complete'):
                try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); break
                except BlockingIOError: pass
            save(root/'waiting.json',dict(unix=time.time(),status='waiting_for_main_and_verified_evidence_seal'))
            time.sleep(15)
        campaign=Campaign(root,deadline-30)
        # This occurs after the main proof benchmark and its evidence seal.
        free_completed_key_caches(root)
        def task(label,kind,src,out,forecast=90):
            if time.time()+1.2*forecast>deadline-60:
                save(root/'exclusions'/(label+'.json'),dict(reason='Time admission before fixed compute deadline')); return
            try:
                campaign.invoke(label,[sys.executable,'-m','experiments.paper_completion_20261102',
                    '--task',kind,'--source',src,'--out',out,'--binary',GO],timeout=max(600,forecast*2))
            except Exception as e:
                save(root/'failures'/(label+'.json'),dict(error=str(e),unix=time.time()))
        first=MAIN/'cases/fashion_seed907_n128_q8'
        task('final_public_tamper','tamper',first,root/'tamper',180)
        for name in plan['final_native_cases']:
            case=MAIN/'cases'/name
            if (case/'arrays.npz').exists(): task('native_'+name,'native',case,root/'native'/name,90)
            else: save(root/'exclusions'/('native_'+name+'.json'),dict(reason='Main fit arrays unavailable'))
        export_tables(root)
        publish(root,'paper_completion_final_method_native',[root/'native',root/'tamper',root/'reports',root/'plan.json'])
        for name in plan['historical_fixed_rank_cases']:
            dataset=name.split('_seed')[0]
            cohort='server_campaign_20261101' if dataset in ['fashion','text'] else 'server_diversity_20261101' if dataset in ['har','mnist'] else 'server_all_sources_20261101'
            fitted=BASE/cohort/'fitted'/name
            if fitted.exists(): task('historical_'+name,'historical',fitted,root/'historical'/name,90)
            else: save(root/'exclusions'/('historical_'+name+'.json'),dict(reason='Historical fit unavailable'))
        export_tables(root)
        publish(root,'paper_completion_all_source_diagnostics',[root/'historical',root/'reports',root/'coverage_inventory.json',root/'plan.json'])
        for dataset,n,q,forecast in PROOFS:
            label=f'{dataset}_seed907_n{n}_q{q}'
            required=32 if n==512 else 24
            if time.time()+1.2*forecast>deadline-60 or psutil.disk_usage(root).free<required*2**30:
                save(root/'exclusions'/(label+'.json'),dict(reason='Time/disk admission before fixed compute deadline',forecast_seconds=forecast));continue
            try: proof_cell(campaign,root,dataset,n,q)
            except Exception as e: save(root/'failures'/(label+'.json'),dict(error=str(e),unix=time.time()))
            export_tables(root)
        campaign.state.update(complete=True,status='queue_finished',active_phase=None);campaign.flush()
    except BaseException as e:
        save(root/'failure.json',dict(error=str(e),unix=time.time()))
        if campaign:campaign.state.update(complete=False,status='stopped_with_evidence');campaign.flush()
    finally:
        stop_study(root,[os.getpid(),watch.pid])
        state=read(root/'watchdog.json');state.update(complete=True);save(root/'watchdog.json',state)
        watch.wait(timeout=10);lock.close();inventory(root);export_tables(root)
        save(root/'finished.json',dict(unix=time.time(),queue_complete=bool(campaign and campaign.state.get('complete'))))


def seal(root):
    deadline=read(root/'plan.json')['compute_deadline_unix']
    while not(root/'finished.json').exists() and time.time()<deadline+60:time.sleep(10)
    time.sleep(15)
    # Only metadata/export work after the compute deadline; no new proofs.
    export_tables(root)
    paths=[p for p in root.iterdir() if p.name!='archives']
    publish(root,'paper_completion_complete_evidence',paths)
    save(root/'sealed.json',dict(complete=True,unix=time.time()))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--freeze',action='store_true');p.add_argument('--watchdog',action='store_true');p.add_argument('--seal',action='store_true')
    a=p.parse_args();assert a.root.resolve()==ROOT.resolve()
    if a.freeze:freeze(a.root)
    elif a.watchdog:final.watchdog(a.root)
    elif a.seal:seal(a.root)
    else:run(a.root)
