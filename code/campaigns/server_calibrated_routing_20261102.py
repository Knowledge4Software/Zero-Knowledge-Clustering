"""Queued prospective routing repair; old cohorts and proof relations frozen."""
from pathlib import Path
import argparse
from dataclasses import asdict
import fcntl
import os
import signal
import subprocess
import sys
import time

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from analysis.server_routing_calibration_20261102 import calibrate_case
from campaigns.cost_aware_pilot import prove_case
from campaigns.proof_runtime import server_profile
from campaigns.server_pca_advantage_20261102 import Campaign
from commitjl.calibrated_routing_20261102 import BatchCosts, CalibratedPlanner
from commitjl.pilot_io import array_sha, frozen_image_sample
from experiments.pca_advantage_20261102 import proposal_banks, read, save, sha
from tools.server_pca_advantage_lease_20261102 import matches, stop_study

REPO = Path(__file__).resolve().parents[2]
OLD = REPO/'code/validation/outputs/server_pca_advantage_20261102'
ROOT = REPO/'code/validation/outputs/server_calibrated_routing_20261102'
MANIFEST = REPO/'calibrated_routing_deployment_manifest_20261102_v2.json'
ABSOLUTE_END = 1793524934.1857634+36*3600
CASES = ['fashion_seed811_n128_q4', 'fashion_seed821_n128_q4',
         'fashion_seed829_n128_q4', 'har_seed811_n128_q4']
UPDATES = 8


def future_rows(dataset, seed, pin):
    """Previously unused source rows, later in the same frozen permutation."""
    if dataset == 'fashion':
        raw = np.asarray(frozen_image_sample(REPO, 'fashion_mnist'))
        assert array_sha(raw) == pin['source']['source_sha256']
        ids = np.flatnonzero(np.linalg.norm(raw, axis=1) > 0)
        ids = ids[np.random.default_rng(741+seed).permutation(len(ids))]
        new_ids = ids[2048:2304]
        assert not set(new_ids) & set(pin['source']['training_ids']+pin['source']['audit_ids'])
        data = raw[new_ids]
        provenance = dict(row_ids=new_ids.tolist(), permutation_segment=[2048, 2304],
                          excluded_both_prior_training_and_audit=True)
    elif dataset == 'har':
        p = REPO/'code/validation/outputs/referee_two_reviews_20261027/data/har561.npz'
        assert sha(p) == pin['source']['source_sha256']
        with np.load(p) as z:
            ids = np.random.default_rng(seed).permutation(len(z['training']))[1024:1280]
            assert not set(ids) & set(pin['source']['training_ids'])
            assert not np.intersect1d(np.unique(z['training_subject'][ids]),
                                     pin['source']['audit_subjects']).size
            data = z['training'][ids]
        provenance = dict(row_ids=ids.tolist(), permutation_segment=[1024, 1280],
                          official_training_partition=True, disjoint_audit_subjects=True)
    else:
        raise ValueError('Unscheduled source')
    assert len(data) == 256
    return np.clip(np.rint(data*pin['scale']), -4095, 4095).astype(np.int64), provenance


def prepare(root, name):
    case = root/'cases'/name
    if case.exists():
        raise RuntimeError('Existing evidence retained; recovery requires a new pin')
    dataset, seed_text = name.split('_')[:2]
    seed = int(seed_text.removeprefix('seed'))
    native = OLD/'native'/f'{dataset}_seed{seed}'
    pin = read(native/'inputs_pin.json')
    # Freeze both calibrations before constructing or evaluating new candidates.
    calibrated = {m: calibrate_case(OLD, name, m) for m in ['ordered_head', 'pca']}
    save(case/'calibration.json', {m: dict(costs=asdict(c), **p) for m, (c, p) in calibrated.items()})
    with np.load(native/'arrays.npz') as a:
        old = a['old'].copy()
        historical = np.concatenate((a['validation_centers'], a['test_centers']))
    rows, provenance = future_rows(dataset, seed, pin)
    proposals, recipe = proposal_banks(rows, old, count=16)
    kept = [i for i, c in enumerate(proposals)
            if not any(np.array_equal(c, h) for h in historical)][:UPDATES]
    assert len(kept) == UPDATES
    proposals = proposals[kept]
    save(case/'new_updates.json', dict(centers=proposals.tolist(), recipe=[recipe[i] for i in kept],
         provenance=provenance, previous_selection_sha256=sha(native/'selection.json'),
         fixed_catalog=True, no_new_basis_selection=True,
         interpretation='New candidate updates on previously inspected authenticated catalogs; '
                        'not independent dataset or catalog confirmation.'))
    for method, (costs, _) in calibrated.items():
        src = OLD/'matched'/name
        alias = case/'sources'/method
        parent = alias/'ordered_head'
        full = parent/'full'
        full.mkdir(parents=True)
        inp = read(src/method/'full/input.json')
        inp['reuse_centers'] = np.concatenate((old[None], proposals)).tolist()
        save(full/'input.json', inp)
        (full/'bundle').symlink_to((src/method/'full/bundle').resolve(), target_is_directory=True)
        for role in ['base', 'refined']:
            (alias/('ordered_head_'+role)).symlink_to((src/(method+'_'+role)).resolve(), target_is_directory=True)
        out = case/'warm'/(name+'_'+method+'_calibrated')
        base = read(src/(method+'_base/input.json'))
        original_task = read(src/(method+'_base/approved.json'))
        task = dict(original_task, reuse_centers=proposals.tolist())
        owner = dict(binary_sha256=sha(REPO/'tmp/external_baselines_20261027/commitjl-bound-linux'),
                     certificate=read(src/(method+'_base/owner_pin.json')),
                     full=read(src/method/'full/owner_pin.json'),
                     refinement=read(src/(method+'_refined/owner_refinement_pin.json')))
        plan = CalibratedPlanner(base['model'], base['x'], costs).batch(proposals.tolist())
        save(out/'approved.json', task)
        save(out/'owner_pin.json', owner)
        save(out/'planner.json', plan)
    save(case/'plan_pin.json', {str(p.relative_to(case)): sha(p) for p in case.rglob('*.json')})
    return case, seed


def summarize(root):
    results = []
    for name in CASES:
        case = root/'cases'/name
        arms = {}
        for method in ['ordered_head', 'pca']:
            p = case/'warm'/(name+'_'+method+'_calibrated/checked.json')
            if p.exists():
                c = read(p)
                arms[method] = {k: c.get(k) for k in ['complete', 'requests', 'certificate_requests',
                    'direct_full_requests', 'selected_correction_proofs', 'measured_warm_routed_seconds',
                    'measured_warm_full_seconds', 'all_requests_verified', 'all_decisions_correct']}
        if arms:
            results.append(dict(case=name, arms=arms, pair_complete=len(arms) == 2))
    save(root/'summary.json', dict(updated_unix=time.time(), cases=results,
         interpretation='Prospective new-update warm routing only. Cached setup, fixed earlier '
                        'basis/rank selections. Both arms use measured calibration. No guarantee '
                        'of superiority over PCA or Full; no new ZKML timings in this cohort.'))


def smoke(campaign, root):
    """Exercise actual fresh certificate/refinement and Full public verification."""
    source = OLD/'smoke'
    for method, route in [('ordered_head', 'certificate'), ('pca', 'certificate'), ('ordered_head', 'full')]:
        target = root/'smoke'/(method+'_'+route)
        alias = target/'source'
        for role, old_role in [('ordered_head', method), ('ordered_head_base', method+'_base'),
                              ('ordered_head_refined', method+'_refined')]:
            (alias/role).parent.mkdir(parents=True, exist_ok=True)
            (alias/role).symlink_to((source/old_role).resolve(), target_is_directory=True)
        inp = read(source/(method+'_base/input.json'))
        task = read(source/(method+'_base/approved.json'))
        costs = BatchCosts(1e6, 1., 1., margin=0.) if route == 'certificate' else BatchCosts(1., 1e6, 1., margin=0.)
        plan = CalibratedPlanner(inp['model'], inp['x'], costs).batch(task['reuse_centers'])
        assert len(plan[route]) == len(task['reuse_centers']), 'Smoke did not activate intended route'
        name = method+'_'+route+'_smoke'
        out = target/name
        save(out/'planner.json', plan)
        save(out/'approved.json', task)
        save(out/'owner_pin.json', dict(binary_sha256=sha(REPO/'tmp/external_baselines_20261027/commitjl-bound-linux'),
             certificate=read(source/(method+'_base/owner_pin.json')),
             full=read(source/method/'full/owner_pin.json'),
             refinement=read(source/(method+'_refined/owner_refinement_pin.json'))))
        prove_case(campaign, target, dict(name=name, requests=len(task['reuse_centers'])),
                   source=alias, run_baseline=False)
    save(root/'smoke_complete.json', dict(complete=True, excluded_from_scientific_timings=True,
                                        routes=['centroid_certificate', 'pca_certificate', 'full']))


def watch(root):
    while True:
        state = read(root/'watchdog.json')
        if state.get('complete'):
            return
        try:
            live = matches(psutil.Process(state['owner']['pid']), state['owner'])
        except psutil.NoSuchProcess:
            live = False
        if not live or time.time() >= state['deadline_unix']:
            (root/'STOP').write_text('Prospective study owner-death/deadline stop.\n')
            stop_study(root, [os.getpid()])
            save(root/'watchdog_recovery.json', dict(finished_unix=time.time(), owner_live=live))
            return
        time.sleep(3)


def run(root):
    assert root.resolve() == ROOT.resolve(), 'Only the isolated, pinned output root is authorized'
    root.mkdir(parents=True, exist_ok=True)
    if (root/'plan.json').exists():
        raise RuntimeError('Existing cohort; no unrecorded resume')
    assert server_profile() and time.time() < ABSOLUTE_END
    manifest = read(MANIFEST)
    for name, digest in manifest.items():
        assert sha(REPO/name) == digest, name
    save(root/'plan.json', dict(created_unix=time.time(), deadline_unix=ABSOLUTE_END,
         cases=CASES, updates=UPDATES, exclusive_hours=2.5,
         queue_policy='Wait for older all-source primary lock; never pause earlier work.',
         resources=server_profile(), manifest_sha256=sha(MANIFEST),
         fixed_before_new_updates=True, maximum_case_forecast_seconds=2400,
         cost_scope='New warm proofs plus private planning and public verification; '
                    'cached earlier setup and calibration excluded equally, disclosed separately.',
         basis_scope='Prior validation-selected basis/rank choices retained; isolate routing repair.',
         comparison='Same measured calibration and global batch optimizer for both bases. '
                    'Fresh common Full baseline. ZKML is not rerun here.',
         claims='All results and exclusions retained; timing improvements are hypotheses.'))
    save(root/'plan_pin.json', dict(plan_sha256=sha(root/'plan.json'), manifest_sha256=sha(MANIFEST)))
    owner = psutil.Process()
    save(root/'watchdog.json', dict(complete=False, deadline_unix=ABSOLUTE_END,
         owner=dict(pid=owner.pid, created=owner.create_time(), module='campaigns.server_calibrated_routing_20261102')))
    watcher = subprocess.Popen([sys.executable, '-u', '-m', 'campaigns.server_calibrated_routing_20261102',
                                '--watchdog', '--root', str(root)], start_new_session=True,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    lock = (root.parent/'server_campaign_20261101/coordinator.lock').open('a')
    campaign = None
    try:
        def interrupted(signum, frame):
            raise RuntimeError('Study received termination signal '+str(signum))
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        while True:
            if (root/'STOP').exists() or time.time()+2880 >= ABSOLUTE_END:
                save(root/'not_admitted.json', dict(reason='Insufficient time after older queue', complete=False))
                return
            older = read(root.parent/'server_all_sources_20261101/matched_queue.json')
            if not older.get('complete'):
                if older.get('status') == 'stopped_with_preserved_evidence' and older.get('error') != "name 'shutil' is not defined":
                    raise RuntimeError('Older queue stopped unexpectedly; retain new queued evidence')
                save(root/'waiting.json', dict(updated_unix=time.time(), status='waiting_for_original_queue',
                                               older_status=older.get('status'), older_error=older.get('error')))
                time.sleep(10)
                continue
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                save(root/'waiting.json', dict(updated_unix=time.time(), status='waiting_for_original_queue'))
                time.sleep(10)
        # Inspect the old queue's terminal receipt, not only a potentially idle lock.
        older = read(root.parent/'server_all_sources_20261101/matched_queue.json')
        assert older.get('complete'), 'Earlier queue needs recovery; retain new queued evidence'
        deadline = min(ABSOLUTE_END-20, time.time()+2.5*3600)
        state = read(root/'watchdog.json'); state['deadline_unix'] = deadline+10
        save(root/'watchdog.json', state)
        campaign = Campaign(root, deadline)
        smoke(campaign, root)
        observed = 2400.
        with threadpool_limits(limits=1):
            for name in CASES:
                if time.time()+1.2*observed > deadline or psutil.disk_usage(root).free < 24*2**30:
                    save(root/'exclusions'/(name+'.json'), dict(reason='Predeclared time/disk admission', complete=False))
                    continue
                started = time.monotonic()
                case, seed = prepare(root, name)
                order = ['ordered_head', 'pca'] if seed == 821 else ['pca', 'ordered_head']
                save(case/'execution_order.json', dict(order=order, common_full='Fresh eight-update Full reference'))
                for method in order:
                    prove_case(campaign, case/'warm',
                        dict(name=name+'_'+method+'_calibrated', requests=UPDATES),
                        source=case/'sources'/method, run_baseline=method == 'ordered_head')
                save(case/'complete.json', dict(complete=True, finished_unix=time.time()))
                observed = max(observed, time.monotonic()-started)
                summarize(root)
        campaign.state.update(complete=True, status='queue_finished', active_phase=None, finished_unix=time.time())
        campaign.flush()
    except BaseException as error:
        save(root/'failure.json', dict(error=str(error), finished_unix=time.time()))
        if campaign:
            campaign.state.update(complete=False, status='stopped_with_evidence', error=str(error))
            campaign.flush()
        raise
    finally:
        stop_study(root, [os.getpid(), watcher.pid])
        state = read(root/'watchdog.json'); state.update(complete=True, finished_unix=time.time())
        save(root/'watchdog.json', state)
        watcher.wait(timeout=10)
        summarize(root)
        lock.close()
        save(root/'finished.json', dict(finished_unix=time.time(), queue_complete=bool(campaign and campaign.state.get('complete'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--watchdog', action='store_true')
    args = parser.parse_args()
    assert args.root.resolve() == ROOT.resolve()
    watch(args.root) if args.watchdog else run(args.root)
