"""Matched HAR/MNIST extension, ahead of optional Fashion capacity work.

The primary deployment and completed results remain immutable. Every new
configuration uses the same four proof arms, cold setup and two warm passes.
"""
from pathlib import Path
import argparse
import copy
import fcntl
import os
import subprocess
import sys
import time

import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from campaigns.published_baselines_laptop import GO, RUST, read, save, sha
from campaigns.server_campaign_20261101 import ServerCampaign, REPO, warm
from campaigns.proof_runtime import server_profile
from commitjl.pilot_io import array_sha, frozen_image_sample
from commitjl.protected_jl import centroid_basis
from commitjl.enriched_jl import enrich_centroid_basis
from experiments.rank79_independent_replication import trace, extend_basis
from experiments.pilot_jl_certificate_ideas import distances

PRIMARY = REPO/'code/validation/outputs/server_campaign_20261101'
PREVIOUS_CAPACITY = REPO/'code/validation/outputs/server_capacity_20261101'
CAPACITY_AFTER = REPO/'code/validation/outputs/server_capacity_after_diversity_20261101'
SEEDS = [701, 709, 719]
DATASETS = ['har', 'mnist']
MANIFEST = REPO/'diversity_deployment_manifest_20261101.json'


def source_rows(dataset, seed):
    """Select inputs independently of outcomes and preserve original row IDs."""
    if dataset == 'mnist':
        raw = np.asarray(frozen_image_sample(REPO, 'mnist'))
        nonzero = np.flatnonzero(np.linalg.norm(raw, axis=1) > 0)
        ids = nonzero[np.random.default_rng(741+seed).permutation(len(nonzero))]
        train_ids, audit_ids = ids[:1024], ids[1024:2048]
        assert len(audit_ids) == 1024 and not np.intersect1d(train_ids, audit_ids).size
        return raw[train_ids], raw[audit_ids], dict(
            source='canonical MNIST5000', source_sha256=array_sha(raw),
            training_ids=train_ids.tolist(), audit_ids=audit_ids.tolist(),
            partition='Disjoint training/audit rows within the canonical frozen sample.'), 5
    if dataset != 'har':
        raise ValueError(f'Unscheduled dataset: {dataset}')
    path = REPO/'code/validation/outputs/referee_two_reviews_20261027/data/har561.npz'
    with np.load(path) as data:
        train_ids = np.random.default_rng(seed).permutation(len(data['training']))[:1024]
        audit_ids = np.random.default_rng(seed+1).permutation(len(data['audit']))[:1024]
        train_subjects = np.unique(data['training_subject'][train_ids])
        audit_subjects = np.unique(data['audit_subject'][audit_ids])
        assert not np.intersect1d(train_subjects, audit_subjects).size
        return data['training'][train_ids], data['audit'][audit_ids], dict(
            source=str(path.relative_to(REPO)), source_sha256=sha(path),
            training_ids=train_ids.tolist(), audit_ids=audit_ids.tolist(),
            training_subjects=train_subjects.tolist(), audit_subjects=audit_subjects.tolist(),
            partition='Frozen official training/test sources with disjoint subjects.'), 6


def quantize(training, audit):
    """Past training rows alone choose the scale; audit rows never fit it."""
    maximum = float(np.abs(training[:512]).max())
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError('Invalid past-training quantization scale')
    scale = 4095/maximum
    return (np.clip(np.rint(training*scale), -4095, 4095).astype(np.int64),
            np.clip(np.rint(audit*scale), -4095, 4095).astype(np.int64), scale)


def make_inputs(root, dataset, seed):
    target = root/'fitted'/f'{dataset}_seed{seed}'
    if (target/'pin.json').exists():
        pin = read(target/'pin.json')
        assert pin['dataset'] == dataset and pin['seed'] == seed
        assert sha(target/'arrays.npz') == pin['arrays_sha256']
        return target
    started = time.monotonic()
    training, audit, source, k = source_rows(dataset, seed)
    assert training.shape[0] == audit.shape[0] == 1024
    training, audit, scale = quantize(training, audit)
    past, incoming = training[:512], training[512:]
    old = trace(past, k, seed)[8]
    banks, recipe, duplicates = [old], [], []
    for batch_id, batch in enumerate(np.array_split(incoming, 16)):
        labels = distances(batch, old).argmin(1)
        proposal = np.array([batch[labels == j].mean(0) if np.any(labels == j) else old[j]
                             for j in range(k)])
        for rate in [.125, .25, .5, 1.]:
            value = np.rint((1-rate)*old+rate*proposal).astype(np.int64)
            if any(np.array_equal(value, previous) for previous in banks):
                duplicates.append(dict(batch=batch_id, rate=rate))
            else:
                banks.append(value)
                recipe.append(dict(batch=batch_id, rate=rate))
            if len(banks) == 5:
                break
        if len(banks) == 5:
            break
    if len(banks) != 5:
        raise RuntimeError('Fewer than four distinct training-only proposals')
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
            arrays[method+'_basis'], arrays[method+'_auxiliary'] = basis, aux
            fits[method] = time.monotonic()-tick
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target/'arrays.npz', **arrays)
    save(target/'pin.json', dict(dataset=dataset, seed=seed, k=k, scale=scale,
         arrays_sha256=sha(target/'arrays.npz'), training_sha256=array_sha(past),
         audit_sha256=array_sha(audit), basis_fitting_seconds=fits,
         enclosing_input_preparation_seconds=time.monotonic()-started,
         proposal_recipe=recipe, public_duplicates=duplicates, **source))
    return target


def materialize(root, dataset, seed, n=128, q=4):
    if n != 128 or q != 4:
        raise ValueError('This extension freezes N128/Q4')
    fitted = make_inputs(root, dataset, seed)
    pin = read(fitted/'pin.json')
    case = root/'matched'/f'{dataset}_seed{seed}_n{n}_q{q}'
    if (case/'input_pin.json').exists():
        existing = read(case/'input_pin.json')
        assert existing['fitted_pin_sha256'] == sha(fitted/'pin.json')
        for relative, digest in existing['files'].items():
            assert sha(case/relative) == digest
        return case
    with np.load(fitted/'arrays.npz') as data:
        old = data['centers'][0].tolist()
        for method in ['ordered_head', 'pca']:
            parent = case/method
            model = dict(version='bound-audit-v1', method='learned_energy', n=n,
                         d=data['audit'].shape[1], k=pin['k'], input_bits=12,
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
            save(parent/'protocol.json', dict(dataset=dataset, fit_seed=seed, n=n, k=pin['k'],
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
    save(case/'input_pin.json', dict(n=n, q=q, k=pin['k'], fitted_pin_sha256=sha(fitted/'pin.json'),
         files={str(p.relative_to(case)):sha(p) for p in case.rglob('*.json')}))
    return case


def check_deployment():
    for relative, digest in read(MANIFEST).items():
        assert sha(REPO/relative) == digest, f'Deployment pin mismatch: {relative}'


def freeze(root):
    if not (root/'plan.json').exists():
        save(root/'plan.json', dict(created_unix=time.time(), datasets=DATASETS, fit_seeds=SEEDS,
             primary_n=128, primary_q=4, warm_repetitions=2, clusters={'har':6, 'mnist':5},
             methods=['centroid certificate','PCA certificate','Full Groth16','ZKML primitive adapter'],
             selection='First four distinct training-only chronological proposals; same public duplicate rule for every method.',
             training_rows=512, incoming_rows=512, audit_pool_rows=1024, prediction_rank=63,
             certificate_rank=79, correction_cap=16, input_bits=12, basis_scale=16384,
             resource_profile=server_profile(), proof_start_cutoff_hours=26.5,
             initial_hours=32, phase_timeout_seconds=5400,
             priority='HAR/MNIST matched controls before any additional optional Fashion capacity work.',
             order=[dict(dataset=dataset, seed=seed) for seed in SEEDS for dataset in DATASETS],
             timing='Same cold and two warm protocols as primary; preserve every failed attempt; separate fitting and diagnostic costs.',
             interpretation='New independent fits on previously inspected public sources; six fits are not six datasets.',
             authorization='User requested running our method and baselines on the other datasets too.',
             no_manuscript_edits=True, primary_plan_sha256=sha(PRIMARY/'plan.json'),
             deployment_manifest_sha256=sha(MANIFEST)))
        save(root/'plan_pin.json', dict(plan_sha256=sha(root/'plan.json'),
             deployment_manifest_sha256=sha(MANIFEST), go_sha256=sha(GO), rust_sha256=sha(RUST)))
    pins = read(root/'plan_pin.json')
    assert sha(root/'plan.json') == pins['plan_sha256']
    assert sha(MANIFEST) == pins['deployment_manifest_sha256']
    assert sha(GO) == pins['go_sha256'] and sha(RUST) == pins['rust_sha256']


class DiversityCampaign(ServerCampaign):
    def invoke(self, label, command, timeout=5400, expect_failure=False):
        if (PRIMARY/'STOP').exists():
            raise RuntimeError('Primary STOP requested')
        return super().invoke(label, command, timeout, expect_failure)


def wait_for_primary(campaign):
    while True:
        primary = read(PRIMARY/'matched_queue.json')
        if (campaign.root/'STOP').exists() or (PRIMARY/'STOP').exists():
            raise RuntimeError('STOP requested while waiting')
        if time.time()-campaign.state['started_unix'] >= 26.5*3600:
            raise RuntimeError('Dataset extension admission deadline')
        if primary.get('status') == 'stopped_with_preserved_evidence':
            raise RuntimeError('Primary stopped; retain extension for recovery')
        if primary.get('complete'):
            lock = (PRIMARY/'coordinator.lock').open('a')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
                assert read(PRIMARY/'matched_queue.json')['complete']
                previous = read(PREVIOUS_CAPACITY/'matched_queue.json')
                assert previous['status'] == 'stopped_with_preserved_evidence'
                assert not previous.get('completed'), 'Capacity work unexpectedly started'
                return lock
            except BlockingIOError:
                lock.close()
        campaign.state.update(status='waiting_for_primary', active_phase=None, updated_unix=time.time())
        save(campaign.root/'matched_queue.json', campaign.state)
        time.sleep(15)


def inherit_smoke(root):
    evidence = {}
    for name in ['smoke_v2', 'compact_smoke', 'refinement_smoke']:
        source = PRIMARY/name
        assert read(source/'checks.json')['complete']
        target = root/name
        if not target.exists():
            target.symlink_to(source, target_is_directory=True)
        assert target.resolve() == source.resolve()
        evidence[name] = sha(source/'checks.json')
    save(root/'inherited_correctness_checks.json', dict(
        checks_sha256=evidence, go_sha256=sha(GO), rust_sha256=sha(RUST),
        explanation='Unmodified deployed proof executables; reuse completed primary real-proof negative tests.'))


def launch_capacity_after(root):
    if (root/'capacity_handoff.json').exists():
        return
    if (root/'STOP').exists() or (PRIMARY/'STOP').exists():
        return
    if time.time()-read(root/'matched_queue.json')['started_unix'] >= 26.5*3600:
        save(root/'capacity_handoff.json', dict(launched=False, reason='Original admission deadline reached'))
        return
    env = dict(os.environ, COMMITJL_ZKML_BINARY=str(REPO/'tmp/external_baselines_20261027/prototype-audit-capacity-20261101-linux'))
    with (REPO/'server_capacity_after_diversity_20261101.log').open('ab') as log:
        child = subprocess.Popen([sys.executable, '-u', '-m', 'campaigns.server_capacity_20261101',
             '--root', str(CAPACITY_AFTER)], cwd=REPO, env=env, stdin=subprocess.DEVNULL,
             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(root/'capacity_handoff.json', dict(launched=True, pid=child.pid, launched_unix=time.time(),
         root=str(CAPACITY_AFTER), scope='Optional capacity cells retain original resource/time admission rules.'))


def run(root):
    if not server_profile():
        raise RuntimeError('Authorized server profile required')
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    check_deployment()
    freeze(root)
    campaign = DiversityCampaign(root)
    campaign.state.update(pid=os.getpid(), updated_unix=time.time())
    primary_lock = None
    try:
        primary_lock = wait_for_primary(campaign)
        inherit_smoke(root)
        for seed in SEEDS:
            for dataset in DATASETS:
                campaign.state.update(status='preparing_inputs', dataset=dataset, fit_seed=seed)
                save(root/'matched_queue.json', campaign.state)
                case = materialize(root, dataset, seed)
                if not campaign.rust_case(case):
                    raise RuntimeError('Primary comparator does not fit; preserve exclusion')
                campaign.method_case(case)
                for repetition in [1, 2]:
                    warm(campaign, case, repetition)
                save(case/'complete.json', dict(complete=True, finished_unix=time.time(), methods=4))
        campaign.state.update(complete=True, status='complete', active_phase=None, finished_unix=time.time())
    except BaseException as error:
        campaign.state.update(complete=False, status='stopped_with_preserved_evidence',
                              error=str(error), finished_unix=time.time())
        raise
    finally:
        save(root/'matched_queue.json', campaign.state)
        files = [p for p in root.rglob('*.json') if p.is_file() and not p.is_symlink()]
        save(root/'evidence_manifest.json', dict(files={str(p.relative_to(root)):sha(p) for p in files
             if p.name != 'evidence_manifest.json'}, complete=campaign.state.get('complete', False)))
        if primary_lock is not None:
            primary_lock.close()
    if campaign.state['complete']:
        launch_capacity_after(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve())
