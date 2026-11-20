"""Fresh warm routed-vs-Full proofs, with earlier pilot-only cost calibration.

Authenticated rows, keys and old Full-cost proofs are reused equally where
applicable. No new request proof or request cache is copied into either arm.
"""
from pathlib import Path
import argparse
import copy
import datetime
import json
import os
import shutil
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'code'))
from campaigns.published_baselines_laptop import GO, read, save, sha
from campaigns.laptop_advantage_pilot import DEFAULT as BASIC
from campaigns.laptop_four_hour_research import DeadlineCampaign
from commitjl.cost_aware import Planner, ProofCosts, exact_distances
from validation.check_group_fast_proofs import export_public
from validation.verify_cost_aware import verify

ROOT = REPO/'code/validation/outputs/cost_aware_20261029'


def sample_seconds(sample):
    return (sample.get('native_answer_seconds', 0)+sample.get('witness_seconds', 0)
            +sum(sample.get('prove_seconds', []))+sum(sample.get('verify_seconds', [])))


def calibrate(source):
    """Only the completed earlier Q8 run, never this run's response timings."""
    full = read(source/'ordered_head/full/bundle/full_reuse_8.json')
    base = read(source/'ordered_head_base/bundle/group_query.json')
    ref = read(source/'ordered_head_refined/bundle/refinement_prove.json')
    full_work = sum(sample_seconds(s) for s in full['samples'] if s.get('centre_set', 0)>0)/8
    base_replay = read(source/'ordered_head_base/status.json')['resources']['group_verify']['wall_seconds']/8
    cert_work = sum(sample_seconds(s) for s in base['samples'])/8+base_replay
    # Public correction replay cost plus the repeated base replay are included.
    rresources = read(source/'ordered_head_refined/status.json')['resources']
    replay = rresources['refinement_verify']['wall_seconds']
    correction_work = (sum(sample_seconds(s) for s in ref['samples'])+max(0, replay-base_replay*8))/len(ref['samples'])
    base_resources = read(source/'ordered_head_base/status.json')['resources']
    certificate_overhead = max(0., base_resources['group_query']['wall_seconds']-sum(sample_seconds(s) for s in base['samples']))
    refinement_overhead = max(0., rresources['refinement_prove']['wall_seconds']-sum(sample_seconds(s) for s in ref['samples']))
    costs = ProofCosts(full_work, cert_work, correction_work, base_replay,
                       certificate_batch_seconds=certificate_overhead,
                       refinement_batch_seconds=refinement_overhead)
    paths = ['ordered_head/full/bundle/full_reuse_8.json', 'ordered_head_base/bundle/group_query.json',
             'ordered_head_base/status.json', 'ordered_head_refined/bundle/refinement_prove.json',
             'ordered_head_refined/status.json']
    return costs, {name: sha(source/name) for name in paths}


def initialize(root):
    import numpy as np
    root.mkdir(parents=True, exist_ok=False)
    now = datetime.datetime.now(datetime.timezone.utc)
    deadline = now+datetime.timedelta(minutes=75)
    save(root/'plan.json', dict(created_utc=now.isoformat(), deadline_utc=deadline.isoformat(),
         workers=8, rss_cap_mib=3584, serial=True, stop_directories=[str(root.relative_to(REPO))],
         cases=[dict(name='text_n32', requests=16), dict(name='fashion_n32', requests=8)],
         policy='Width-ordered native preflight, at most16 corrections; require conclusive result and predicted proof work10% below Full. All requests retained.',
         scope='Fresh warm request-proof generation and public replay; existing independently pinned setup/row evidence and old Full cost reused. Native planning is charged. No claim of a fresh cold run.',
         privacy='Path choice and refinement indices are observable. Exact native distances are private prover planning, not certification.',
         calibration='Earlier Q8 timings only, frozen before new proof execution; same catalogs are exploratory, not independent held-out data.',
         order='Text routed then Full; Fashion Full then routed.', no_paper_edits=True, no_remote_access=True))
    papers = [REPO/'manuscript.tex', REPO/'references.bib']+sorted((REPO/'paper_results').rglob('*.tex'))
    save(root/'preservation.json', {p.relative_to(REPO).as_posix(): sha(p) for p in papers})
    for case in read(root/'plan.json')['cases']:
        source = BASIC/'matched'/case['name']
        out = root/case['name']
        inp = read(source/'ordered_head_base/input.json')
        centers = np.load(source/'ordered_head_diagnostic_inputs.npz')['centers'][1:case['requests']+1].tolist()
        costs, calibration = calibrate(source)
        planner = Planner(inp['model'], inp['x'], costs)
        plan = planner.batch(centers)
        # A separate cold-batch prediction discloses the activation decision.
        res = read(source/'ordered_head_base/status.json')['resources']
        extra_setup = sum(res[p]['wall_seconds'] for p in ['group_compact_prepare', 'group_keys', 'group_rows'])
        cold = copy.deepcopy(plan)
        if plan['predicted_gross_savings_seconds'] <= extra_setup:
            cold['certificate'] = []; cold['full'] = list(range(case['requests']))
        cold['extra_setup_seconds'] = extra_setup
        save(out/'planner.json', plan)
        save(out/'cold_activation_prediction.json', dict(certificate=cold['certificate'], full=cold['full'],
             extra_setup_seconds=extra_setup, estimated_gross_savings_seconds=plan['predicted_gross_savings_seconds'],
             measured=False, note='Cost-model activation only; fitting and lazy refinement setup would be additional cold costs.'))
        save(out/'calibration.json', dict(source=str(source), sources_sha256=calibration, costs=plan['costs']))
        save(out/'approved.json', dict(comparison_base=inp['model']['centers'], reuse_centers=centers))
        pin = dict(binary_sha256=sha(GO), certificate=read(source/'ordered_head_base/owner_pin.json'),
                   full=read(source/'ordered_head/full/owner_pin.json'),
                   refinement=read(source/'ordered_head_refined/owner_refinement_pin.json'))
        save(out/'owner_pin.json', pin)
    files = [Path(__file__), REPO/'code/commitjl/cost_aware.py', REPO/'code/validation/verify_cost_aware.py']
    files += list(root.rglob('*.json'))
    save(root/'source_pin.json', {str(p.relative_to(REPO)): sha(p) for p in files})
    print(json.dumps(dict(initialized=True, root=str(root), deadline_utc=deadline.isoformat())), flush=True)


def copy_setup(source, target, kind):
    """Exclude all changing-request proofs, then explicitly retain old Full cost."""
    prefix = 'group_query' if kind == 'certificate' else 'full_reuse' if kind == 'full' else 'refinement'
    shutil.copytree(source, target, ignore=lambda directory, names:
                    [n for n in names if n.startswith(prefix)] if Path(directory)==source else [])
    if kind == 'full':
        (target/'full_reuse').mkdir()
        for p in (source/'full_reuse').glob('s000_*'):
            shutil.copy2(p, target/'full_reuse'/p.name)
    else:
        assert not list(target.rglob(prefix+'/*.proof'))


def full_export(bundle, public):
    # Small allowlist avoids importing training/dataframe dependencies to export.
    public.mkdir()
    names = ['trusted.json', 'prepare.json', 'full_reuse_8.json']
    names += [p.relative_to(bundle).as_posix() for folder, suffix in [('keys', '*.vk'), ('full_reuse', '*.proof')]
              for p in (bundle/folder).glob(suffix)]
    for name in names:
        target = public/name; target.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith('.json'):
            value = read(bundle/name)
            for s in value.get('samples') or []:
                s.pop('assign', None)
            save(target, value)
        else:
            shutil.copy2(bundle/name, target)
    save(public/'public_manifest.json', dict(sha256={n: sha(public/n) for n in names}))


def prove_case(campaign, root, case, source=None, run_baseline=True):
    name, count = case['name'], case['requests']
    source = source or BASIC/'matched'/name
    out = root/name
    task, plan = read(out/'approved.json'), read(out/'planner.json')
    resources = {}

    def invoke(label, args):
        value = campaign.invoke(name+'_'+label, args, 1800)
        resources[label] = value
        return value

    def go(label, phase, private, inp=None, extras=()):
        args = [GO, '-phase', phase, '-out', private, '-max-constraints', '1600000', '-reps', '1', '-warmup', '0']
        if inp is not None:
            args += ['-instance', inp]
        return invoke(label, args+list(extras))

    def exact_arm(arm, selected):
        arm.mkdir()
        private = arm/'private'
        copy_setup(source/'ordered_head/full/bundle', private, 'full')
        inp = read(source/'ordered_head/full/input.json')
        inp['reuse_centers'] = [task['comparison_base']]+[task['reuse_centers'][i] for i in selected]
        save(arm/'input.json', inp)
        go(arm.name+'_prove', 'full_reuse', private, arm/'input.json', ['-q', '8'])
        full_export(private, arm/'public')
        return inp['queries']

    def routed_arm():
        routed = out/'routed'; routed.mkdir()
        public = routed/'public'; public.mkdir()
        save(public/'routes.json', {k: plan[k] for k in ['certificate', 'full']})
        cert_ids = plan['certificate']
        if cert_ids:
            base = routed/'base'; base.mkdir()
            copy_setup(source/'ordered_head_base/bundle', base/'bundle', 'certificate')
            inp = read(source/'ordered_head_base/input.json')
            inp['reuse_centers'] = [task['comparison_base']]+[task['reuse_centers'][i] for i in cert_ids]
            save(base/'input.json', inp)
            certtask = dict(comparison_base=task['comparison_base'], reuse_centers=inp['reuse_centers'][1:], queries=inp['queries'])
            save(base/'approved.json', certtask)
            pin = out/'certificate_pin.json'; save(pin, read(out/'owner_pin.json')['certificate'])
            common = ['-trust', pin, '-public-task', base/'approved.json']
            go('routed_certificate_prove', 'group_query', base/'bundle', base/'input.json', ['-q', '16', *common])
            export_public(base/'bundle', public/'certificate')
            orders = [plan['records'][i]['refinement_indices'] for i in cert_ids]
            if any(orders):
                refined = routed/'refined'; refined.mkdir()
                copy_setup(source/'ordered_head_refined/bundle', refined/'bundle', 'refinement')
                save(refined/'requests.json', dict(sets=orders))
                rpin = out/'refinement_pin.json'; save(rpin, read(out/'owner_pin.json')['refinement'])
                go('routed_refinement_prove', 'refinement_prove', refined/'bundle', extras=[*common,
                   '-refinement-base', base/'bundle', '-refinement-trust', rpin, '-refinement-request', refined/'requests.json'])
                dest = public/'refinement'; dest.mkdir()
                paths = ['refinement_records.json', 'keys/group_refinement_1.vk']
                paths += [p.relative_to(refined/'bundle').as_posix() for p in (refined/'bundle/refinement').glob('*.proof')]
                for file in paths:
                    (dest/file).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(refined/'bundle'/file, dest/file)
                save(dest/'public_manifest.json', dict(sha256={p: sha(dest/p) for p in paths}))
        if plan['full']:
            queries = exact_arm(routed/'routed_full', plan['full'])
            (routed/'routed_full/public').rename(public/'full')
            save(public/'full_queries.json', queries)
        # Exactly one fresh replay of the selected branch evidence. The
        # refinement verifier already replays the compact backbone itself.
        checked = verify(public, out/'owner_pin.json', out/'approved.json', GO,
                         invoke=lambda label, args: invoke('routed_'+label, args))
        return checked

    def baseline_arm():
        arm = out/'baseline_full'
        queries = exact_arm(arm, list(range(count)))
        ft = dict(reuse_centers=[task['comparison_base']]+task['reuse_centers'], queries=queries)
        save(arm/'task.json', ft); save(arm/'pin.json', read(out/'owner_pin.json')['full'])
        go('baseline_verify', 'verify', arm/'public', extras=['-trust', arm/'pin.json', '-public-task', arm/'task.json'])

    if not run_baseline:
        checked = routed_arm()
    elif name == 'fashion_n32':
        baseline_arm(); checked = routed_arm()
    else:
        checked = routed_arm(); baseline_arm()
    rows = read(source/'ordered_head/full/input.json')['x']
    old_cost = int(exact_distances(rows, task['comparison_base']).min(1).sum())
    truth = [int(exact_distances(rows, c).min(1).sum())-old_cost for c in task['reuse_centers']]
    for record, delta in zip(checked['decisions'], truth):
        assert int(record['lower']) <= delta*int(record['denominator']) <= int(record['upper'])
        expected = 'worsening' if delta>0 else 'improvement' if delta<0 else 'equal'
        assert record['decision'] == expected
    routed_seconds = sum(v['wall_seconds'] for k,v in resources.items() if k.startswith('routed'))+plan['native_seconds']
    baseline_seconds = sum(v['wall_seconds'] for k,v in resources.items() if k.startswith('baseline')) if run_baseline else None
    # Fresh candidate proof bytes must differ from the earlier Q8 execution.
    new_proofs = []
    for p in (out/'baseline_full/private/full_reuse').glob('*.proof'):
        if not p.name.startswith('s000_'):
            new_proofs.append(p)
            old = source/'ordered_head/full/bundle/full_reuse'/p.name
            if old.exists(): assert sha(p) != sha(old)
    if run_baseline:
        assert len(new_proofs) == count*len(read(source/'ordered_head/full/input.json')['queries'])
    save(out/'checked.json', dict(complete=True, requests=count, certificate_requests=len(plan['certificate']),
         direct_full_requests=len(plan['full']), selected_correction_proofs=sum(len(r['refinement_indices']) for r in plan['records']),
         measured_warm_routed_seconds=routed_seconds, measured_warm_full_seconds=baseline_seconds,
         full_over_routed=baseline_seconds/routed_seconds if run_baseline else None, planner_seconds=plan['native_seconds'],
         all_requests_verified=True, all_decisions_correct=True, all_intervals_contain_truth=True,
         fresh_baseline_request_proofs=len(new_proofs), resources=resources,
         timing='Invocation wall sums for fresh query/refinement generation and public replay; native planner charged. Existing setup/old evidence excluded equally; export/copy and independent oracle validation excluded.',
         repetitions=1, exploratory=True))
    print(json.dumps(read(out/'checked.json')), flush=True)


def run(root):
    for name, digest in read(root/'source_pin.json').items():
        if sha(REPO/name) != digest:
            raise RuntimeError('frozen source/input changed: '+name)
    deadline = datetime.datetime.fromisoformat(read(root/'plan.json')['deadline_utc']).timestamp()
    os.environ['COMMITJL_RESUME_PROOFS'] = '1'
    campaign = DeadlineCampaign(root, deadline)
    # Campaign's explicit environment was captured after enabling setup reuse.
    try:
        for case in read(root/'plan.json')['cases']:
            prove_case(campaign, root, case)
        preservation = read(root/'preservation.json')
        assert all(sha(REPO/p)==digest for p,digest in preservation.items())
        save(root/'checks.json', dict(complete=True, paper_unchanged=True,
             cases=[read(root/c['name']/'checked.json') for c in read(root/'plan.json')['cases']]))
        campaign.state.update(complete=True, status='complete', active_phase=None, finished_unix=time.time())
        save(root/'matched_queue.json', campaign.state)
    finally:
        (root/'FINISHED').write_text('Bounded optimization proof run ended. Preserve all attempts.\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT); p.add_argument('--initialize', action='store_true')
    a = p.parse_args()
    initialize(a.root.resolve()) if a.initialize else run(a.root.resolve())
