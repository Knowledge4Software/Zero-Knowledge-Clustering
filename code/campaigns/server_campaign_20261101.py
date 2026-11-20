"""New, pinned 12-vCPU server campaign; historical laptop evidence is immutable.

Largest supported Halo2 catalog first, then matched independent-fit controls.
All timing phases run serially with the same CPU/RAM allocation. Small genuine
cryptographic correctness checks precede the first scientific benchmark.
"""
from pathlib import Path
import argparse
import copy
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import psutil
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from campaigns.published_baselines_laptop import Campaign, GO, RUST, read, save, sha
from campaigns.proof_runtime import server_profile, proof_environment
from commitjl.pilot_io import array_sha, frozen_image_sample
from commitjl.protected_jl import centroid_basis
from commitjl.enriched_jl import enrich_centroid_basis
from experiments.rank79_independent_replication import trace, extend_basis
from experiments.pilot_jl_certificate_ideas import distances

REPO = Path(__file__).resolve().parents[2]
SEEDS = [701, 709, 719]


def freeze(root):
    root.mkdir(parents=True, exist_ok=True)
    if (root/'plan.json').exists():
        return
    plan = dict(created_unix=time.time(), datasets=['fashion', 'text'], fit_seeds=SEEDS,
                primary_n=128, primary_q=4, warm_repetitions=2,
                methods=['centroid certificate', 'PCA certificate', 'Full Groth16', 'ZKML primitive adapter'],
                large_first=dict(dataset='fashion', n=1024, q=1, fallback_n=512,
                                 adapter_max_degree=20, advice_columns=17,
                                 selection='Largest catalog that fits the existing adapter degree limit; no selection by outcome.'),
                first_resource_canary='code/validation/outputs/server_canary_20261101; frozen historical seed173 fit, excluded from independent-fit inference.',
                resource_profile=server_profile(), initial_hours=32, proof_start_cutoff_hours=26.5,
                proof_drain_hours=28, phase_timeout_seconds=5400,
                selection='First four distinct training-only chronological proposals; same public duplicate rule for every method.',
                training_rows=512, incoming_rows=512, audit_pool_rows=1024,
                fitted_on_server=True, input_bits=12, prediction_rank=63, certificate_rank=79,
                basis_scale=16384, correction_cap=16,
                interpretation='New independent fits on previously inspected public sources. No unseen-dataset or universal-superiority claim. Fits can share source records.',
                timing='Separate cold phase-wall costs, repeated warm request-proof phases and public replay. Native planning charged; copy/export and independent numerical checks separately scoped.',
                calibration='Frozen earlier seed521 laptop Q8 cost ratios, equal for both bases; no policy tuning to these server outcomes.',
                authorization='User requested remote execution, largest RAM case first and all available server CPU workers.',
                no_manuscript_edits=True, source_manifest_sha256=sha(REPO/'deployment_manifest.json'))
    save(root/'plan.json', plan)
    save(root/'plan_pin.json', dict(plan_sha256=sha(root/'plan.json'),
                                  go_sha256=sha(GO), rust_sha256=sha(RUST)))
    save(root/'environment.json', dict(python=sys.version, numpy=np.__version__,
         kernel=list(os.uname()), cpu_affinity=sorted(os.sched_getaffinity(0)),
         total_memory_bytes=psutil.virtual_memory().total,
         dependency_freeze_sha256=sha(REPO/'environment.freeze.txt'),
         source_manifest_sha256=sha(REPO/'deployment_manifest.json')))


def make_inputs(root, dataset, seed):
    target = root/'fitted'/f'{dataset}_seed{seed}'
    if (target/'pin.json').exists():
        assert sha(target/'arrays.npz') == read(target/'pin.json')['arrays_sha256']
        return target
    started = time.monotonic()
    if dataset == 'fashion':
        raw = np.asarray(frozen_image_sample(REPO, 'fashion_mnist'))
        raw = raw[np.linalg.norm(raw, axis=1) > 0]
        ids = np.random.default_rng(741+seed).permutation(len(raw))
        training, audit = raw[ids[:1024]], raw[ids[1024:2048]]
        source = dict(source='canonical Fashion5000', training_ids=ids[:1024].tolist(),
                      audit_ids=ids[1024:2048].tolist(), source_sha256=array_sha(raw))
    else:
        path = REPO/'code/validation/outputs/referee_two_reviews_20261027/data/text_minilm384.npz'
        archive = np.load(path)
        train_ids = np.random.default_rng(seed).permutation(len(archive['training']))[:1024]
        audit_ids = np.random.default_rng(seed+1).permutation(len(archive['audit']))[:1024]
        training, audit = archive['training'][train_ids], archive['audit'][audit_ids]
        source = dict(source=str(path.relative_to(REPO)), source_sha256=sha(path),
                      training_ids=train_ids.tolist(), audit_ids=audit_ids.tolist())
    scale = 4095/float(np.abs(training[:512]).max())
    training = np.clip(np.rint(training*scale), -4095, 4095).astype(np.int64)
    audit = np.clip(np.rint(audit*scale), -4095, 4095).astype(np.int64)
    assert len(audit) == 1024
    past, incoming = training[:512], training[512:]
    old = trace(past, 5, seed)[8]
    banks = [old]
    recipe = []
    duplicates = []
    for batch_id, batch in enumerate(np.array_split(incoming, 16)):
        labels = distances(batch, old).argmin(1)
        proposal = np.array([batch[labels == j].mean(0) if np.any(labels == j) else old[j]
                             for j in range(5)])
        for rate in [.125, .25, .5, 1.]:
            value = np.rint((1-rate)*old+rate*proposal).astype(np.int64)
            if any(np.array_equal(value, previous) for previous in banks):
                duplicates.append(dict(batch=batch_id, rate=rate))
            else:
                banks.append(value)
                recipe.append(dict(batch=batch_id, rate=rate))
            if len(banks) >= 5:
                break
        if len(banks) >= 5:
            break
    if len(banks) != 5:
        raise RuntimeError('Fewer than four genuine distinct proposals; preserve input failure')
    arrays = dict(audit=audit, training=past, centers=np.array(banks))
    fits = {}
    with threadpool_limits(limits=server_profile()['workers']):
        for method in ['ordered_head', 'pca']:
            tick = time.monotonic()
            if method == 'ordered_head':
                basis = enrich_centroid_basis(past.astype(float), old,
                    extra=63-centroid_basis(old).shape[1], random_state=seed,
                    preserve_pca_order=True)[0].T
            else:
                basis = PCA(63, svd_solver='randomized', random_state=seed).fit(past.astype(float)).components_
            basis = np.rint(basis*16384).astype(np.int64)
            aux, _ = extend_basis(basis, past, old)
            arrays[method+'_basis'] = basis
            arrays[method+'_auxiliary'] = aux
            fits[method] = time.monotonic()-tick
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target/'arrays.npz', **arrays)
    save(target/'pin.json', dict(dataset=dataset, seed=seed, scale=scale,
         arrays_sha256=sha(target/'arrays.npz'), training_sha256=array_sha(past),
         audit_sha256=array_sha(audit), basis_fitting_seconds=fits,
         enclosing_input_preparation_seconds=time.monotonic()-started,
         proposal_recipe=recipe, public_duplicates=duplicates, **source))
    return target


def materialize(root, dataset, seed, n, q):
    fitted = make_inputs(root, dataset, seed)
    data = np.load(fitted/'arrays.npz')
    pin = read(fitted/'pin.json')
    case = root/'matched'/f'{dataset}_seed{seed}_n{n}_q{q}'
    if (case/'input_pin.json').exists():
        existing = read(case/'input_pin.json')
        for relative, digest in existing['files'].items():
            assert sha(case/relative) == digest
        return case
    old = data['centers'][0].tolist()
    for method in ['ordered_head', 'pca']:
        parent = case/method
        model = dict(version='bound-audit-v1', method='learned_energy', n=n,
                     d=data['audit'].shape[1], k=5, input_bits=12,
                     basis_scale=16384, input_scale=str(pin['scale']), centers=old,
                     basis=data[method+'_basis'].tolist(), tail_width=0,
                     tail_rows=[], tail_signs=[], seed=str(seed))
        inp = dict(model=model, x=data['audit'][:n].tolist(), salt_x='20261101701',
                   salt_z='20261101703', queries=[list(range(i, i+16)) for i in range(0, n, 16)],
                   reuse_centers=data['centers'][:q+1].tolist(), comparison_base=old)
        save(parent/'head_r63/input.json', inp)
        full = copy.deepcopy(inp)
        full['model'].update(method='full', basis=[], seed='')
        full['queries'] = [list(range(i, i+8)) for i in range(0, n, 8)]
        save(parent/'full/input.json', full)
        save(parent/'protocol.json', dict(dataset=dataset, fit_seed=seed, n=n, k=5,
             proof_runtime_workers=server_profile()['workers'], phase_wall_limit_seconds=5400,
             execution_os='Linux/server', fitting_seconds=pin['basis_fitting_seconds'][method],
             training_sha256=pin['training_sha256'], audit_sha256=array_sha(data['audit'][:n]),
             source_proposals=pin['proposal_recipe'][:q], selected_source_sets=list(range(q)),
             source_selection=read(root/'plan.json')['selection']))
        save(parent/'auxiliary_basis.json', dict(source_input_sha256=sha(parent/'head_r63/input.json'),
             training_sha256=pin['training_sha256'], audit_sha256=array_sha(data['audit'][:n]),
             basis_scale=16384, certificate_basis=data[method+'_auxiliary'].tolist(),
             rank=79, fitting_seconds=0., recipe='Frozen prediction63 plus16 residual directions; fit time charged in parent.'))
    assert read(case/'ordered_head/full/input.json') == read(case/'pca/full/input.json')
    save(case/'approved_task.json', dict(reuse_centers=inp['reuse_centers']))
    save(case/'input_pin.json', dict(n=n, q=q, fitted_pin_sha256=sha(fitted/'pin.json'),
         files={str(p.relative_to(case)):sha(p) for p in case.rglob('*.json')}))
    return case


class ServerCampaign(Campaign):
    def __init__(self, root):
        self.root = root
        canary_state = REPO/'code/validation/outputs/server_canary_20261101/state.json'
        origin = read(canary_state)['started_unix'] if canary_state.exists() else time.time()
        self.state = read(root/'matched_queue.json') if (root/'matched_queue.json').exists() else dict(
            started_unix=origin, complete=False, completed=[])
        self.env = proof_environment(dict(proof_runtime_workers=server_profile()['workers']))

    def invoke(self, label, command, timeout=5400, expect_failure=False):
        command = list(map(str, command))
        if any(module in command for module in ['campaigns.run_group_fast_control',
                                                'campaigns.run_group_refinement_control']):
            child_root = Path(command[command.index('--root')+1])
            if (child_root/'protocol.json').exists() and '--resume' not in command:
                command.append('--resume')
        resources = self.root/'resources'
        resources.mkdir(exist_ok=True)
        previous = sorted(resources.glob(label+'.attempt_*.json'))
        if previous and read(previous[-1]).get('complete'):
            return read(previous[-1])
        elapsed = time.time()-self.state['started_unix']
        if (self.root/'STOP').exists() or elapsed > 26.5*3600:
            raise RuntimeError('Server queue admission deadline or STOP requested')
        if psutil.disk_usage(self.root).free < 20*2**30:
            raise RuntimeError('Insufficient free disk for another phase; retain evidence')
        receipt = resources/f'{label}.attempt_{len(previous)+1:02}.json'
        log = receipt.with_suffix('.log')
        self.state.update(status='running', active_phase=label, updated_unix=time.time())
        save(self.root/'matched_queue.json', self.state)
        args = [sys.executable, '-m', 'tools.server_guard_20261101', '--receipt', str(receipt),
                '--log', str(log), '--root', str(self.root), '--timeout', str(min(timeout, 5400))]
        if expect_failure:
            args.append('--expect-failure')
        result = subprocess.run(args+['--']+list(map(str, command)), cwd=REPO, env=self.env)
        if result.returncode:
            raise RuntimeError(f'{label} failed; inspect {log}')
        if label not in self.state['completed']:
            self.state['completed'].append(label)
        save(self.root/'matched_queue.json', self.state)
        return read(receipt)

    def rust_case(self, case):
        name = case.name
        source = case/'ordered_head/full/input.json'
        inp = read(source)
        n, d = inp['model']['n'], inp['model']['d']
        inspection = case/'zkml_inspection'
        selection = inspection/'selection.json'
        if not selection.exists():
            if (inspection/'excluded.json').exists():
                return False
            trials = []
            # Start immediately below the size-predicted degree; never above20,
            # which is the existing, source-pinned adapter's admitted maximum.
            first = max(12, min(20, math.ceil(math.log2(n*d))))
            for degree in range(first, 21):
                label = name+f'_zkml_inspect_k{degree}'
                try:
                    self.invoke(label, [RUST, 'inspect', source, inspection/f'k{degree}', degree, 17])
                except RuntimeError:
                    logs = sorted((self.root/'resources').glob(label+'.attempt_*.log'))
                    text = logs[-1].read_text(errors='replace') if logs else ''
                    if not any(t in text for t in ['NotEnoughRowsAvailable', 'usable_rows=']):
                        raise
                    trials.append(dict(k=degree, fit=False))
                    continue
                trials.append(dict(k=degree, fit=True))
                save(selection, dict(k=degree, trials=trials,
                     selection='Smallest inspected fitting degree at17 advice columns; declared size-based lower start.'))
                break
            else:
                save(inspection/'excluded.json', dict(complete=False, trials=trials,
                     reason='Catalog exceeds pinned adapter maximum degree20; no implementation or outcome change.'))
                return False
        degree = read(selection)['k']
        bundle, public = case/'zkml_private', case/'zkml_public'
        public.mkdir(exist_ok=True)
        self.invoke(name+'_zkml_prepare', [RUST, 'prepare', source, bundle, degree, 17])
        shutil.copy2(bundle/'owner_pin.json', case/'zkml_owner_pin.json')
        for filename in ['params', 'vk']:
            if not (public/filename).exists():
                os.link(bundle/filename, public/filename)
        self.invoke(name+'_zkml_queries', [RUST, 'queries', source, bundle])
        costs = []
        for index, centers in enumerate(inp['reuse_centers']):
            private_bank = bundle/f'bank_{index:03}'
            public_bank = public/f'bank_{index:03}'
            public_bank.mkdir(exist_ok=True)
            for filename in ['proof', 'public.json']:
                shutil.copy2(private_bank/filename, public_bank/filename)
            self.invoke(name+f'_zkml_verify{index:03}',
                        [RUST, 'verify', public, index, case/'zkml_owner_pin.json', case/'approved_task.json'])
            exact = sum(min(sum((int(x)-int(c))**2 for x, c in zip(row, center))
                            for center in centers) for row in inp['x'])
            assert exact == read(private_bank/'result.json')['cost']
            costs.append(exact)
        save(case/'zkml_checked.json', dict(complete=True, costs=costs, all_costs_match_integer_oracle=True,
             public_only_replay=True, proof_count=len(costs), adaptation='ZKML primitive adapter, not unmodified compiler.'))
        return True


def smoke_input(root):
    model = dict(version='bound-audit-v1', method='full', n=4, d=3, k=2,
                 input_bits=4, basis_scale=8, input_scale='1', centers=[[0,0,0],[8,8,0]],
                 basis=[], tail_width=0, tail_rows=[], tail_signs=[], seed='')
    save(root/'smoke/input.json', dict(model=model, x=[[0,0,1],[4,0,1],[7,7,2],[1,7,2]],
         salt_x='11', salt_z='13', queries=[[0,1],[2,3]],
         reuse_centers=[model['centers'], [[1,0,0],[7,8,0]]], comparison_base=model['centers']))


def warm(campaign, case, repetition):
    from campaigns.cost_aware_pilot import calibrate, prove_case
    from commitjl.cost_aware import Planner
    costs, calibration = calibrate(REPO/'server_inputs/calibration')
    out = campaign.root/'warm'/case.name/f'repetition_{repetition:02}'
    marker = out/'checks.json'
    if marker.exists() and read(marker).get('complete'):
        return
    for method in ['ordered_head', 'pca']:
        alias = out/'sources'/method
        alias.mkdir(parents=True, exist_ok=True)
        for role, target in [('ordered_head', case/method), ('ordered_head_base', case/(method+'_base')),
                             ('ordered_head_refined', case/(method+'_refined'))]:
            if not (alias/role).exists():
                (alias/role).symlink_to(target.resolve(), target_is_directory=True)
        prefix = case.name+'_'+method+f'_r{repetition:02}'
        previous = sorted(out.glob(prefix+'_attempt*'))
        if previous and (previous[-1]/'checked.json').exists():
            continue
        # A failed warm trial is retained; a retry gets fresh phase labels so
        # successful pieces of an interrupted trial cannot masquerade as new.
        name = prefix+f'_attempt{len(previous)+1:02}'
        target = out/name
        inp = read(case/(method+'_base/input.json'))
        task = read(case/(method+'_base/approved.json'))
        pins = dict(binary_sha256=sha(GO), certificate=read(case/(method+'_base/owner_pin.json')),
                    full=read(case/method/'full/owner_pin.json'))
        refinement_pin = case/(method+'_refined/owner_refinement_pin.json')
        if refinement_pin.exists():
            pins['refinement'] = read(refinement_pin)
        plan = Planner(inp['model'], inp['x'], costs).batch(task['reuse_centers'])
        save(target/'owner_pin.json', pins)
        save(target/'approved.json', task)
        save(target/'planner.json', plan)
        save(target/'calibration.json', dict(source='Frozen laptop seed521 Q8', sources_sha256=calibration))
        prove_case(campaign, out, dict(name=name, requests=len(task['reuse_centers'])),
                   source=alias, run_baseline=method == 'ordered_head')
    # Fresh Halo2 proofs in a new bundle; immutable setup files may be linked.
    private, public = out/'zkml_private', out/'zkml_public'
    private.mkdir(parents=True, exist_ok=True)
    public.mkdir(exist_ok=True)
    for filename in ['params', 'vk', 'pk', 'owner_pin.json']:
        if not (private/filename).exists():
            os.link(case/'zkml_private'/filename, private/filename)
    for filename in ['params', 'vk']:
        if not (public/filename).exists():
            os.link(private/filename, public/filename)
    source = case/'ordered_head/full/input.json'
    records = []
    for index in range(1, len(read(source)['reuse_centers'])):
        label = case.name+f'_warm_r{repetition:02}_zkml_bank{index}'
        record = campaign.invoke(label, [RUST, 'query', source, private, index])
        destination = public/f'bank_{index:03}'
        destination.mkdir(exist_ok=True)
        for filename in ['proof', 'public.json']:
            shutil.copy2(private/f'bank_{index:03}'/filename, destination/filename)
        replay = campaign.invoke(label+'_verify', [RUST, 'verify', public, index,
                    case/'zkml_owner_pin.json', case/'approved_task.json'])
        assert read(private/f'bank_{index:03}'/'result.json')['cost'] == read(case/'zkml_checked.json')['costs'][index]
        records.append(dict(request=index, generation=record, replay=replay))
    save(marker, dict(complete=True, warm_repetition=repetition, zkml_requests=records,
         timing='Warm process-per-request Halo2 key loading is charged; original reference evidence and immutable setup reused.'))


def run(root):
    if not server_profile():
        raise RuntimeError('Server profile is mandatory')
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    freeze(root)
    pin = read(root/'plan_pin.json')
    assert sha(root/'plan.json') == pin['plan_sha256']
    assert sha(GO) == pin['go_sha256'] and sha(RUST) == pin['rust_sha256']
    campaign = ServerCampaign(root)
    try:
        if not (root/'smoke_v2/checks.json').exists():
            smoke_input(root)
            campaign.smoke()
        canary_root = REPO/'code/validation/outputs/server_canary_20261101'
        canary = read(canary_root/'state.json')
        if not canary.get('complete'):
            raise RuntimeError('Largest-catalog resource canary has not completed successfully')
        save(root/'largest_case_complete.json', dict(source=str(canary_root),
             selected_n=canary['selected_n'], canary_state_sha256=sha(canary_root/'state.json'),
             scope='Resource calibration on the old frozen fit, not an independent primary comparison.'))
        for seed in SEEDS:
            for dataset in ['fashion', 'text']:
                case = materialize(root, dataset, seed, 128, 4)
                if not campaign.rust_case(case):
                    raise RuntimeError('Primary comparator does not fit')
                campaign.method_case(case)
                for repetition in [1, 2]:
                    warm(campaign, case, repetition)
                save(case/'complete.json', dict(complete=True, finished_unix=time.time()))
        # Complete the remaining methods for the same largest catalog if admitted.
        if time.time()-campaign.state['started_unix'] < 23*3600:
            largest = materialize(root, 'fashion', SEEDS[0], canary['selected_n'], 1)
            if campaign.rust_case(largest):
                campaign.method_case(largest)
        campaign.state.update(complete=True, status='complete', active_phase=None,
                              finished_unix=time.time())
    except BaseException as error:
        campaign.state.update(complete=False, status='stopped_with_preserved_evidence',
                              error=str(error), finished_unix=time.time())
        raise
    finally:
        save(root/'matched_queue.json', campaign.state)
        files = [p for p in root.rglob('*') if p.is_file() and p.suffix in ['.json', '.proof', '.vk']]
        save(root/'evidence_manifest.json', dict(files={str(p.relative_to(root)):sha(p) for p in files
             if p.name != 'evidence_manifest.json'}, complete=campaign.state.get('complete', False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve())
