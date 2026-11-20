"""Bounded new PCA comparison; frozen validation choices, fresh real proofs.

The existing broad server campaign yields between proof phases and resumes
automatically. Every comparison receives identical resources and corrections.
"""
from pathlib import Path
import argparse
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

from campaigns.proof_runtime import proof_environment, server_profile
from campaigns.published_baselines_laptop import GO
from experiments.pca_advantage_20261102 import (
    DATASETS, SEEDS, EndpointPlanner, model_for, read, save, sha)
from tools.server_pca_advantage_lease_20261102 import LegacyLease

REPO = Path(__file__).resolve().parents[2]
BASE = REPO/'code/validation/outputs'
PRIMARY = BASE/'server_campaign_20261101'
MANIFEST = REPO/'pca_advantage_deployment_manifest_20261102.json'
PROOF_ORDER = [('fashion', 811), ('fashion', 821), ('fashion', 829),
               ('har', 811), ('mnist', 811), ('text', 811)]


def freeze(root):
    if (root/'plan.json').exists():
        raise RuntimeError('A new study must use a fresh output directory')
    for name, digest in read(MANIFEST).items():
        if sha(REPO/name) != digest:
            raise RuntimeError('Deployment pin mismatch: '+name)
    plan = dict(created_unix=time.time(), datasets=DATASETS, fit_seeds=SEEDS,
         basis_families=['pca', 'centroid', 'history25', 'history75'],
         ranks='k-1,8,16,32,79; equal-rank numerical controls retained',
         past_training_rows=512, validation_incoming_rows=256, test_incoming_rows=256,
         validation_catalog_rows=128, test_catalog_rows=128,
         validation_updates=8, test_updates=8, proof_updates=4,
         proof_n=128, warm_repetitions=1, correction_cap=16,
         input_bits=12, basis_scale=16384,
         selection='Select each method on disjoint validation rows/updates only, using a declared arithmetic proxy. Freeze all choices before evaluating final test rows. Test every numerical candidate; retain negatives.',
         cost_scope='Cold proof totals charge the complete measured fitting/validation search for the selected method group, native preparation, key setup, authentication, correction work, public replay and newly generated Full fallback. Warm new proofs reuse setup; native planner and public replay charged. Numerical work proxy is not proof timing.',
         representation_scope='Standalone cost-change certificate ranks; no frozen rank63 prediction-accuracy claim or free sharing of authenticated statistics across ranks.',
         proof_order=[dict(dataset=d, seed=s) for d, s in PROOF_ORDER],
         proof_order_selection='Fixed before new validation/test results. First three Fashion fits test the previous low-rank hypothesis; other datasets follow if budget permits. Numerical tests cover all four datasets and three fits.',
         paired_order='PCA first for seed811/829, ours first for seed821; same rule for warm proofs.',
         screening_processes=12, blas_threads_per_screen_process=1,
         screen_wall_limit_seconds=2100, exclusive_study_hours=4,
         case_admission_forecast_seconds=3300, resource_profile=server_profile(),
         lease='Yield the older coordinator only, drain its active guarded phase, run exclusively, then resume it. A separately running watchdog stops new workers before resuming the older coordinator on owner death or deadline.',
         old_time_scope='A yielded old cell may include the study as coordinator queue wait; retain its original per-phase/component times for scientific comparisons.',
         no_manuscript_edits=True, same_binary_sha256=sha(GO),
         deployment_manifest_sha256=sha(MANIFEST),
         success_rule='A PCA advantage is a measured result, not a requirement to discard unsuccessful candidates. Report equal-rank numerical results and validation-selected real-proof comparisons separately.')
    save(root/'plan.json', plan)
    save(root/'plan_pin.json', dict(plan_sha256=sha(root/'plan.json'),
                                  manifest_sha256=sha(MANIFEST), go_sha256=sha(GO)))


class Campaign:
    def __init__(self, root, deadline):
        self.root, self.deadline = Path(root), deadline
        self.env = proof_environment(dict(proof_runtime_workers=12))
        self.state = dict(pid=os.getpid(), complete=False, status='running', completed=[],
                          started_unix=time.time(), deadline_unix=deadline)
        self.flush()

    def flush(self):
        self.state['updated_unix'] = time.time()
        save(self.root/'matched_queue.json', self.state)

    def invoke(self, label, command, timeout=5400, expect_failure=False):
        if (self.root/'STOP').exists() or time.time() > self.deadline-10:
            raise RuntimeError('New study stopped or deadline reached')
        remaining = self.deadline-time.time()-10
        command = list(map(str, command))
        replacements = {
            'campaigns.local_fast_tradeoff': 'campaigns.server_all_sources_full_20261101',
            'campaigns.run_group_fast_control': 'campaigns.server_all_sources_compact_20261101',
            'campaigns.run_group_refinement_control': 'campaigns.server_pca_advantage_refinement_20261102'}
        command = [replacements.get(part, part) for part in command]
        receipt = self.root/'resources'/(label+'.attempt_01.json')
        if receipt.exists():
            raise RuntimeError('Existing attempt retained; do not silently retry: '+label)
        self.state.update(active_phase=label, status='running')
        self.flush()
        args = [sys.executable, '-m', 'tools.server_guard_20261101',
                '--receipt', str(receipt), '--log', str(receipt.with_suffix('.log')),
                '--root', str(self.root), '--timeout', str(min(timeout, remaining))]
        if expect_failure:
            args.append('--expect-failure')
        result = subprocess.run(args+['--']+command, cwd=REPO, env=self.env)
        record = read(receipt) if receipt.exists() else {}
        if result.returncode or not record.get('complete'):
            raise RuntimeError('Guarded study phase failed: '+label)
        self.state['completed'].append(label)
        self.flush()
        return record


def linked_tree(source, target):
    """Link immutable evidence only after Full finishes; never hide a new query."""
    for path in Path(source).rglob('*'):
        dest = Path(target)/path.relative_to(source)
        if path.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
        elif dest.exists():
            if sha(dest) != sha(path):
                raise RuntimeError('Existing Full input differs across representations')
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, dest)


def write_case(case, rows, old, proposals, models, scale, fitting, training_hash, selection):
    for method, basis in models.items():
        parent = case/method
        model = model_for(basis, old, len(rows), scale)
        inp = dict(model=model, x=rows.tolist(), salt_x='20261102811', salt_z='20261102813',
                   queries=[list(range(i, min(i+16, len(rows)))) for i in range(0, len(rows), 16)],
                   reuse_centers=np.concatenate((old[None, :, :], proposals)).tolist(),
                   comparison_base=old.tolist())
        save(parent/'head_r63/input.json', inp)
        full = dict(inp, model=dict(model, method='full', basis=[], seed=''),
                    queries=[list(range(i, min(i+8, len(rows)))) for i in range(0, len(rows), 8)])
        save(parent/'full/input.json', full)
        save(parent/'protocol.json', dict(fitting_seconds=fitting[method],
             proof_runtime_workers=12, phase_wall_limit_seconds=5400,
             training_sha256=training_hash, selection_sha256=selection,
             standalone_certificate_rank=len(basis), prediction_accuracy_claim=False,
             fitting_scope='All fitting and validation candidate-search work in this method group is charged once.'))
    assert read(case/'ordered_head/full/input.json') == read(case/'pca/full/input.json')
    save(case/'approved_task.json', dict(reuse_centers=np.concatenate((old[None, :, :], proposals)).tolist()))
    save(case/'inputs_pin.json', {str(p.relative_to(case)): sha(p) for p in case.rglob('*.json')})


def materialize(root, dataset, seed):
    source = root/'native'/f'{dataset}_seed{seed}'
    complete, pin, selected = [read(source/p) for p in ['complete.json', 'inputs_pin.json', 'selection.json']]
    assert complete['complete'] and sha(source/'selection.json') == complete['selection_sha256']
    assert sha(source/'arrays.npz') == pin['arrays_sha256']
    with np.load(source/'arrays.npz') as arrays:
        choices = {m: selected[k] for m, k in [('ordered_head', 'ours'), ('pca', 'pca')]}
        models = {m: arrays[c['family']][:c['rank']] for m, c in choices.items()}
        case = root/'matched'/f'{dataset}_seed{seed}_n128_q4'
        if case.exists():
            raise RuntimeError('Existing matched case retained')
        write_case(case, arrays['test'], arrays['old'], arrays['test_centers'][:4], models,
                   pin['scale'], {'ordered_head': selected['selection_preparation_seconds']['ours'],
                                  'pca': selected['selection_preparation_seconds']['pca']},
                   pin['training_sha256'], sha(source/'selection.json'))
    save(case/'selection_source.json', dict(source=str(source), choices=choices,
         selected_before_test=True, policy='Both methods independently validation-selected; equal-rank controls are retained in numerical results.'))
    return case


def cold(campaign, case, seed):
    name = case.name
    campaign.invoke(name+'_full', [sys.executable, '-m', 'campaigns.local_fast_tradeoff',
                    '--root', case/'ordered_head', '--run', '--binary', GO, '--cases', 'full'])
    linked_tree(case/'ordered_head/full', case/'pca/full')
    save(case/'pca/status.json', read(case/'ordered_head/status.json'))
    order = ['pca', 'ordered_head'] if seed in [811, 829] else ['ordered_head', 'pca']
    save(case/'execution_order.json', dict(cold=order, warm=order,
         full_reference='One matched common Full run; same original integers and candidate banks.'))
    for method in order:
        campaign.invoke(name+'_'+method+'_base', [sys.executable, '-m',
            'campaigns.run_group_fast_control', '--compact', '--proof-backbone-only',
            '--parent', case/method, '--root', case/(method+'_base'), '--binary', GO,
            '--smoke-receipt', PRIMARY/'compact_smoke/checks.json'])
        campaign.invoke(name+'_'+method+'_refined', [sys.executable, '-m',
            'campaigns.run_group_refinement_control', '--base', case/(method+'_base'),
            '--parent', case/method, '--root', case/(method+'_refined'), '--binary', GO,
            '--smoke-receipt', PRIMARY/'refinement_smoke/checks.json'])
    checks = {m: read(case/(m+'_refined/checked_comparison.json')) for m in order}
    if not all(c['complete'] and c['no_wrong_decisions'] and c['all_intervals_contain_full']
               for c in checks.values()):
        raise RuntimeError('Cold comparison did not verify')
    save(case/'cold_complete.json', dict(complete=True, finished_unix=time.time(),
         ours_seconds=checks['ordered_head']['active_phase_seconds_including_fitting_and_fallback'],
         pca_seconds=checks['pca']['active_phase_seconds_including_fitting_and_fallback'],
         pca_over_ours=checks['pca']['active_phase_seconds_including_fitting_and_fallback']/
                       checks['ordered_head']['active_phase_seconds_including_fitting_and_fallback']))


def warm(campaign, case, seed):
    from campaigns.cost_aware_pilot import calibrate, prove_case
    costs, calibration = calibrate(REPO/'server_inputs/calibration')
    out = campaign.root/'warm'/case.name
    order = read(case/'execution_order.json')['warm']
    for method in order:
        alias = out/'sources'/method
        alias.mkdir(parents=True, exist_ok=True)
        for role, target in [('ordered_head', case/method),
                             ('ordered_head_base', case/(method+'_base')),
                             ('ordered_head_refined', case/(method+'_refined'))]:
            (alias/role).symlink_to(target.resolve(), target_is_directory=True)
        target = out/(case.name+'_'+method+'_warm01')
        inp = read(case/(method+'_base/input.json'))
        task = read(case/(method+'_base/approved.json'))
        pin = dict(binary_sha256=sha(GO), certificate=read(case/(method+'_base/owner_pin.json')),
                   full=read(case/method/'full/owner_pin.json'))
        if (case/(method+'_refined/owner_refinement_pin.json')).exists():
            pin['refinement'] = read(case/(method+'_refined/owner_refinement_pin.json'))
        plan = EndpointPlanner(inp['model'], inp['x'], costs).batch(task['reuse_centers'])
        save(target/'owner_pin.json', pin)
        save(target/'approved.json', task)
        save(target/'planner.json', plan)
        save(target/'calibration.json', dict(source='Frozen earlier seed521 Q8; same for both arms',
                                           sources_sha256=calibration))
        prove_case(campaign, out, dict(name=target.name, requests=len(task['reuse_centers'])),
                   source=alias, run_baseline=method == 'ordered_head')
    save(out/'complete.json', dict(complete=True, finished_unix=time.time(),
         interpretation='Fresh proofs with cached setup, same four future updates, one repetition per fit; not new independent candidate updates.'))


def smoke(campaign):
    rng = np.random.default_rng(11)
    rows = rng.integers(-3, 4, size=(16, 3), dtype=np.int64)
    old = np.array([[-2, 0, 0], [2, 0, 0]], dtype=np.int64)
    proposals = np.array([[[-2, -2, 1], [2, 2, 1]]], dtype=np.int64)
    case = campaign.root/'smoke'
    models = {'ordered_head': np.array([[16384, 0, 0]], dtype=np.int64),
              'pca': np.array([[0, 16384, 0]], dtype=np.int64)}
    write_case(case, rows, old, proposals, models, 1.,
               {'ordered_head': 0., 'pca': 0.}, 'synthetic-smoke', 'synthetic-smoke')
    cold(campaign, case, 811)
    warm(campaign, case, 811)
    save(campaign.root/'smoke_complete.json', dict(complete=True,
         note='Synthetic real-proof correctness check; excluded from scientific speed results.'))


def summarize(root):
    numerical = []
    for case in sorted((root/'native').glob('*')):
        if not (case/'complete.json').exists():
            continue
        selected = read(case/'selection.json')
        records = read(case/'test.json')
        rows = {}
        for key in ['ours', 'pca']:
            c = selected[key]
            rr = [r for r in records if (r['family'], r['rank']) == (c['family'], c['rank'])]
            rows[key] = dict(family=c['family'], rank=c['rank'], requests=len(rr),
                conclusive=sum(r['conclusive'] for r in rr),
                corrections=sum(r['corrections'] for r in rr))
        numerical.append(dict(case=case.name, selected=rows))
    proof = []
    for case in sorted((root/'matched').glob('*')):
        row = dict(case=case.name)
        for filename in ['cold_complete.json', 'incomplete.json']:
            if (case/filename).exists():
                row[filename] = read(case/filename)
        for method in ['ordered_head', 'pca']:
            file = root/'warm'/case.name/(case.name+'_'+method+'_warm01/checked.json')
            if file.exists():
                c = read(file)
                row[method+'_warm'] = {k: c.get(k) for k in ['complete', 'requests',
                    'certificate_requests', 'direct_full_requests', 'selected_correction_proofs',
                    'measured_warm_routed_seconds', 'measured_warm_full_seconds',
                    'all_requests_verified', 'all_decisions_correct']}
        proof.append(row)
    save(root/'summary.json', dict(updated_unix=time.time(), numerical=numerical, proofs=proof,
         interpretation='All test results retained. Real-proof runs use validation-frozen independently selected ranks; do not confuse equal-rank numerical controls with proof timing or claim all PCA variants were timed.'))


def run(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not server_profile():
        raise RuntimeError('Authorized Linux server profile required')
    freeze(root)
    save(root/'launch.json', dict(pid=os.getpid(), started_unix=time.time(), status='acquiring_lease'))
    def interrupted(number, frame):
        raise RuntimeError('Study received termination signal '+str(number))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    campaign = None
    try:
        with LegacyLease(root, hours=4) as lease:
            campaign = Campaign(root, lease.state['deadline_unix']-20)
            smoke(campaign)
            campaign.invoke('native_screen', [sys.executable, '-m',
                'experiments.pca_advantage_20261102', '--root', root], timeout=2100)
            summarize(root)
            observed = 3300.
            for dataset, seed in PROOF_ORDER:
                if time.time()+1.2*observed > campaign.deadline:
                    save(root/'exclusions'/f'{dataset}_seed{seed}.json',
                         dict(reason='Prespecified four-hour paired-case admission', complete=False))
                    continue
                if psutil.disk_usage(root).free < 24*2**30:
                    save(root/'exclusions'/f'{dataset}_seed{seed}.json',
                         dict(reason='Disk admission; retain all existing evidence', complete=False))
                    continue
                case = materialize(root, dataset, seed)
                started = time.monotonic()
                try:
                    cold(campaign, case, seed)
                    warm(campaign, case, seed)
                except Exception as error:
                    save(case/'incomplete.json', dict(complete=False, error=str(error)))
                    raise
                finally:
                    observed = max(observed, time.monotonic()-started)
                    summarize(root)
            campaign.state.update(complete=True, status='queue_finished', active_phase=None,
                                  finished_unix=time.time())
            campaign.flush()
    except BaseException as error:
        if campaign is not None:
            campaign.state.update(complete=False, status='stopped_with_evidence',
                                  error=str(error), finished_unix=time.time())
            campaign.flush()
        save(root/'failure.json', dict(error=str(error), finished_unix=time.time()))
        raise
    finally:
        summarize(root)
        save(root/'finished.json', dict(finished_unix=time.time(),
             queue_complete=campaign.state.get('complete', False) if campaign else False,
             lease=read(root/'lease.json') if (root/'lease.json').exists() else None))
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
