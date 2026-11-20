"""Bounded, all-available-source comparison after the original four sources.

Six real corpora, two additional representations and three synthetic controls
are distinct reporting strata. New sources receive native controls for three
fits and a serial, outcome-independent N32 proof queue. Budget exclusions and
failures remain explicit. Existing deployments and scientific results stay
immutable, and no paper acceptance probability is inferred from dataset count.
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
import psutil
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from campaigns.published_baselines_laptop import GO, RUST, read, save, sha
from campaigns.server_campaign_20261101 import ServerCampaign, REPO
from campaigns.server_all_sources_warm_20261101 import warm
from campaigns.server_diversity_20261101 import inherit_smoke
from campaigns.proof_runtime import server_profile
from commitjl.pilot_io import array_sha
from commitjl.protected_jl import centroid_basis
from commitjl.enriched_jl import enrich_centroid_basis
from experiments.rank79_independent_replication import trace, extend_basis
from experiments.pilot_jl_certificate_ideas import distances

PRIMARY = REPO/'code/validation/outputs/server_campaign_20261101'
DIVERSITY = REPO/'code/validation/outputs/server_diversity_20261101'
PACKAGE = REPO/'server_inputs/all_sources_20261101'
MANIFEST = REPO/'all_sources_deployment_manifest_20261101.json'
SEEDS = [701, 709, 719]
EXTRA = ['gisette', 'arcene', 'text_tfidf', 'har_neural256',
         'gaussian_easy', 'gaussian_hard', 'dbscan_margin']
EXISTING = ['fashion', 'text', 'har', 'mnist']
SPECS = {
    'gisette': dict(past=512, incoming=512, audit=1024, k=5, d=5000, cold_seconds=3600),
    'arcene': dict(past=64, incoming=4, audit=32, k=5, d=10000, cold_seconds=5400),
    'text_tfidf': dict(past=512, incoming=512, audit=1024, k=5, d=2000, cold_seconds=1800),
    'har_neural256': dict(past=512, incoming=512, audit=1024, k=6, d=256, cold_seconds=1200),
    'gaussian_easy': dict(past=512, incoming=512, audit=976, k=5, d=256, cold_seconds=1200),
    'gaussian_hard': dict(past=512, incoming=512, audit=976, k=5, d=512, cold_seconds=1800),
    'dbscan_margin': dict(past=512, incoming=256, audit=232, k=5, d=128, cold_seconds=900),
}


def admissible(elapsed, forecast):
    return elapsed + 1.2*forecast < 32.5*3600


def retire_private_keys(root, case):
    """Retire owned rebuildable proving keys; retain public verification evidence."""
    assert case.resolve().parent == (root/'matched').resolve()
    assert case.resolve().is_relative_to(root.resolve())
    receipt = case/'private_key_retirement.json'
    if receipt.exists():
        return
    records = []
    for directory in [case, root/'warm'/case.name]:
        assert directory.resolve().is_relative_to(root.resolve())
        if not directory.exists():
            continue
        for path in sorted(directory.rglob('*')):
            if path.is_file() and not path.is_symlink() and (path.name == 'pk' or path.suffix == '.pk'):
                assert path.resolve().is_relative_to(root.resolve())
                records.append(dict(path=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=sha(path)))
                path.unlink()
    save(receipt,dict(retired_unix=time.time(),files=records,
         scope='Only rebuildable private proving keys; parameters, verification keys, proofs, inputs, commitments and receipts retained.',
         warm_timing_available=False, recovery='A new key setup needs a separately pinned rerun; never reinterpret old proofs with fresh keys.'))


def ensure_disk_for_case(root, completed, dataset):
    required = {'arcene':60,'gisette':50,'text_tfidf':36}.get(dataset,32)*2**30
    for old in completed:
        if psutil.disk_usage(root).free >= required:
            break
        retire_private_keys(root,old)
    return psutil.disk_usage(root).free >= required


def source_rows(dataset, seed):
    spec = SPECS[dataset]
    package_pin = read(PACKAGE/'pin.json')
    if dataset == 'text_tfidf':
        filename = f'text_tfidf_seed{seed}.npz'
        assert sha(PACKAGE/filename) == package_pin['files'][filename]
        with np.load(PACKAGE/filename) as z:
            training, audit = z['training'], z['audit']
            source = dict(source=filename, source_sha256=sha(PACKAGE/filename),
                 preprocessing_pin_sha256=sha(PACKAGE/f'text_tfidf_seed{seed}_preprocessing.json'),
                 training_ids=z['training_ids'].tolist(), audit_ids=z['audit_ids'].tolist(),
                 preprocessing_fit='Vocabulary and IDF fitted only on first512 past-training documents.',
                 same_corpus_as='text')
        assert not set(source['training_ids']) & set(source['audit_ids'])
        return training[:512], training[512:], audit, source
    if dataset == 'har_neural256':
        path = REPO/'code/validation/outputs/referee_two_reviews_20261027/data/har561.npz'
        encoder = PACKAGE/'har_encoder_seed307.npz'
        assert sha(encoder) == package_pin['files'][encoder.name]
        with np.load(path) as data, np.load(encoder) as weights:
            train_ids = np.random.default_rng(seed).permutation(len(data['training']))[:1024]
            audit_ids = np.random.default_rng(seed+1).permutation(len(data['audit']))[:1024]
            assert not np.intersect1d(data['training_subject'][train_ids], data['audit_subject'][audit_ids]).size
            def encode(x):
                x = np.maximum(x.astype(np.float32) @ weights['encoder_0.weight'].T + weights['encoder_0.bias'], 0)
                return (x/np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float64)
            training, audit = encode(data['training'][train_ids]), encode(data['audit'][audit_ids])
        return training[:512], training[512:], audit, dict(
            source=str(path.relative_to(REPO)), source_sha256=sha(path), encoder_sha256=sha(encoder),
            training_ids=train_ids.tolist(), audit_ids=audit_ids.tolist(), same_corpus_as='har',
            encoder_scope='Historical seed307 encoder frozen before new fits; trained on official training subjects, no test subjects. Encoder inference outside proof.')
    if dataset in ['gisette', 'arcene']:
        filename = dataset+'.npz'
        assert sha(PACKAGE/filename) == package_pin['files'][filename]
        with np.load(PACKAGE/filename) as z:
            raw, original_ids = z['X'].astype(np.float64), z['original_row_ids']
        assert array_sha(raw) == package_pin['provenance'][dataset]['canonical_float64_sha256']
        source = dict(source=filename, source_sha256=sha(PACKAGE/filename),
                      original_float64_sha256=array_sha(raw), original_row_ids=original_ids.tolist())
    else:
        from data.loaders import gaussian_mixture, dbscan_margin_dataset
        if dataset == 'dbscan_margin':
            generated = dbscan_margin_dataset(n=1000, d=128, rho=1., eps=.15, frac_near_band=.5, seed=0)
        else:
            generated = gaussian_mixture(n=2000, d=spec['d'], k=10,
                gamma=1. if dataset == 'gaussian_easy' else .35,
                imbalance=1. if dataset == 'gaussian_easy' else 3., seed=0)
        raw = generated['X']
        source = dict(source='synthetic generator', generator=generated['meta'], synthetic_control=True)
    ids = np.random.default_rng(741+seed).permutation(len(raw))
    a, b, count = spec['past'], spec['incoming'], spec['audit']
    past_ids, incoming_ids, audit_ids = ids[:a], ids[a:a+b], ids[a+b:a+b+count]
    assert len(audit_ids) == count
    assert len(set(past_ids)|set(incoming_ids)|set(audit_ids)) == a+b+count
    source.update(past_ids=past_ids.tolist(), incoming_ids=incoming_ids.tolist(), audit_ids=audit_ids.tolist())
    return raw[past_ids], raw[incoming_ids], raw[audit_ids], source


def quantize(past, incoming, audit):
    maximum = float(np.abs(past).max())
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError('Invalid past-training quantization scale')
    scale = 4095/maximum
    values = [np.clip(np.rint(x*scale), -4095, 4095).astype(np.int64) for x in [past, incoming, audit]]
    return *values, scale


def fitted_path(root, dataset, seed):
    if dataset in ['fashion', 'text']:
        return PRIMARY/'fitted'/f'{dataset}_seed{seed}'
    if dataset in ['har', 'mnist']:
        return DIVERSITY/'fitted'/f'{dataset}_seed{seed}'
    return root/'fitted'/f'{dataset}_seed{seed}'


def make_inputs(root, dataset, seed):
    target = fitted_path(root, dataset, seed)
    if (target/'pin.json').exists():
        assert sha(target/'arrays.npz') == read(target/'pin.json')['arrays_sha256']
        return target
    if dataset not in EXTRA:
        raise RuntimeError(f'Missing prerequisite fit: {dataset}/{seed}')
    started = time.monotonic()
    past, incoming, audit, source = source_rows(dataset, seed)
    past, incoming, audit, scale = quantize(past, incoming, audit)
    k = SPECS[dataset]['k']
    with threadpool_limits(limits=server_profile()['workers']):
        old = trace(past, k, seed)[8]
        banks, recipe, duplicates = [old], [], []
        for batch_id, batch in enumerate(np.array_split(incoming, min(16, len(incoming)))):
            labels = distances(batch, old).argmin(1)
            proposal = np.array([batch[labels == j].mean(0) if np.any(labels == j) else old[j] for j in range(k)])
            for rate in [.125, .25, .5, 1.]:
                value = np.rint((1-rate)*old+rate*proposal).astype(np.int64)
                if any(np.array_equal(value, previous) for previous in banks):
                    duplicates.append(dict(batch=batch_id, rate=rate))
                else:
                    banks.append(value); recipe.append(dict(batch=batch_id, rate=rate))
                if len(banks) == 5:
                    break
            if len(banks) == 5:
                break
        if len(banks) != 5:
            raise RuntimeError('Fewer than four distinct training-only proposals; retain failure')
        arrays = dict(training=past, audit=audit, centers=np.array(banks))
        fits = {}
        for method in ['ordered_head', 'pca']:
            tick = time.monotonic()
            if method == 'ordered_head':
                basis = enrich_centroid_basis(past.astype(float), old,
                    extra=63-centroid_basis(old).shape[1], random_state=seed, preserve_pca_order=True)[0].T
            else:
                basis = PCA(63, svd_solver='randomized', random_state=seed).fit(past.astype(float)).components_
            assert basis.shape == (63, past.shape[1]), 'Report insufficient numerical rank; do not silently change width'
            basis = np.rint(basis*16384).astype(np.int64)
            aux, _ = extend_basis(basis, past, old)
            arrays[method+'_basis'], arrays[method+'_auxiliary'] = basis, aux
            fits[method] = time.monotonic()-tick
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target/'arrays.npz', **arrays)
    save(target/'pin.json', dict(dataset=dataset, seed=seed, k=k, scale=scale,
         arrays_sha256=sha(target/'arrays.npz'), training_sha256=array_sha(past), audit_sha256=array_sha(audit),
         basis_fitting_seconds=fits, enclosing_input_preparation_seconds=time.monotonic()-started,
         past_rows=len(past), incoming_rows=len(incoming), audit_pool_rows=len(audit),
         proposal_recipe=recipe, public_duplicates=duplicates, **source))
    return target


def materialize(root, dataset, seed):
    fitted = make_inputs(root, dataset, seed)
    pin = read(fitted/'pin.json')
    case = root/'matched'/f'{dataset}_seed{seed}_n32_q4'
    if (case/'input_pin.json').exists():
        frozen = read(case/'input_pin.json')
        assert frozen['fitted_pin_sha256'] == sha(fitted/'pin.json')
        for name, digest in frozen['files'].items():
            assert sha(case/name) == digest
        return case
    with np.load(fitted/'arrays.npz') as data:
        old = data['centers'][0].tolist()
        for method in ['ordered_head', 'pca']:
            parent = case/method
            model = dict(version='bound-audit-v1', method='learned_energy', n=32,
                         d=data['audit'].shape[1], k=len(old), input_bits=12, basis_scale=16384,
                         input_scale=str(pin['scale']), centers=old, basis=data[method+'_basis'].tolist(),
                         tail_width=0, tail_rows=[], tail_signs=[], seed=str(seed))
            inp = dict(model=model, x=data['audit'][:32].tolist(), salt_x='20261101701', salt_z='20261101703',
                       queries=[list(range(i,i+16)) for i in range(0,32,16)],
                       reuse_centers=data['centers'].tolist(), comparison_base=old)
            assert len(inp['x']) == 32 and len(inp['reuse_centers']) == 5
            save(parent/'head_r63/input.json', inp)
            full = copy.deepcopy(inp); full['model'].update(method='full', basis=[], seed='')
            full['queries'] = [list(range(i,i+8)) for i in range(0,32,8)]
            save(parent/'full/input.json', full)
            save(parent/'protocol.json', dict(dataset=dataset, fit_seed=seed, n=32, k=len(old),
                 proof_runtime_workers=server_profile()['workers'], phase_wall_limit_seconds=5400,
                 compiler_constraint_cap=24000000,
                 execution_os='Linux/server', fitting_seconds=pin['basis_fitting_seconds'][method],
                 training_sha256=pin['training_sha256'], audit_sha256=array_sha(data['audit'][:32]),
                 source_proposals=pin['proposal_recipe'], selected_source_sets=list(range(4)),
                 source_selection=read(root/'plan.json')['selection']))
            save(parent/'auxiliary_basis.json', dict(source_input_sha256=sha(parent/'head_r63/input.json'),
                 training_sha256=pin['training_sha256'], audit_sha256=array_sha(data['audit'][:32]),
                 basis_scale=16384, certificate_basis=data[method+'_auxiliary'].tolist(), rank=79,
                 fitting_seconds=0., recipe='Same frozen63+16 training-only basis recipe as original server queue.'))
    assert read(case/'ordered_head/full/input.json') == read(case/'pca/full/input.json')
    save(case/'approved_task.json', dict(reuse_centers=inp['reuse_centers']))
    save(case/'input_pin.json', dict(n=32, q=4, fitted_pin_sha256=sha(fitted/'pin.json'),
         files={str(p.relative_to(case)):sha(p) for p in case.rglob('*.json')}))
    return case


def native_case(root, dataset, seed):
    """Reuse the validated published-baseline gate on one immutable fitted task."""
    from experiments.published_baseline_gate import native
    from commitjl.published_audit_baselines import projection_bases
    fitted = make_inputs(root, dataset, seed)
    pin = read(fitted/'pin.json')
    if dataset in EXTRA:
        materialize(root, dataset, seed)  # Freeze proof tasks before viewing native results.
    target = root/'native_cases'/f'{dataset}_seed{seed}'
    if (target/'native_results.json').exists():
        assert read(target/'native_results.json')['complete']
        assert read(target/'source_pin.json')['fitted_pin_sha256'] == sha(fitted/'pin.json')
        return
    case = target/'inputs'/f'{dataset}_seed{seed}'
    case.mkdir(parents=True, exist_ok=True)
    with np.load(fitted/'arrays.npz') as data:
        arrays = dict(audit=data['audit'][:128], centers=data['centers'])
        for method in ['ordered_head', 'pca']:
            arrays[method+'_prediction'] = data[method+'_basis']
            arrays[method+'_certificate'] = data[method+'_auxiliary']
    fits = dict(pin['basis_fitting_seconds'])
    for method in ['sign_rp', 'sparse_rp']:
        tick = time.monotonic()
        arrays[method+'_prediction'], arrays[method+'_certificate'] = projection_bases(
            method, 63, 79, arrays['audit'].shape[1], seed)
        fits[method] = time.monotonic()-tick
    np.savez_compressed(case/'arrays.npz', **arrays)
    save(case/'pin.json', dict(dataset=dataset, fit_seed=seed, scale=pin['scale'],
         basis_fitting_seconds=fits, arrays_sha256=sha(case/'arrays.npz'),
         original_fitted_pin_sha256=sha(fitted/'pin.json')))
    save(target/'source_pin.json', dict(fitted_pin_sha256=sha(fitted/'pin.json')))
    save(target/'plan.json', dict(native_catalog_sizes=[32,128] if len(arrays['audit'])>=128 else [32],
         coreset_draws=[16,32], scope='Native numerical controls, not cryptographic proof timings; identical task for every method.'))
    save(target/'plan_pin.json', dict(plan_sha256=sha(target/'plan.json')))
    native(target)


def freeze(root):
    if not (root/'plan.json').exists():
        save(root/'plan.json', dict(created_unix=time.time(), existing=EXISTING, additional_workloads=EXTRA,
             real_corpora=['Fashion-MNIST','MNIST','20 Newsgroups','UCI HAR','Gisette','Arcene'],
             representations={'text_tfidf':'20 Newsgroups', 'har_neural256':'UCI HAR'},
             synthetic_controls=['gaussian_easy','gaussian_hard','dbscan_margin'], fit_seeds=SEEDS,
             specifications=SPECS, proof_n=32, proof_q=4, native_n=[32,128], arcene_native_n=[32],
             prediction_rank=63, certificate_rank=79, correction_cap=16, warm_repetitions=2,
             proof_methods=['centroid certificate','PCA certificate','Full Groth16','ZKML primitive adapter'],
             native_methods=['exact Full','centroid certificate','PCA certificate','sign RP certificate adapter',
                 'sparse RP certificate adapter','raw sign RP','raw sparse RP','lightweight coreset16/32','uniform coreset16/32'],
             selection='First four distinct training-only chronological proposals; every outcome, failure and exclusion retained.',
             order='Native controls on all11 workloads/three fits; cold proof coverage across all extra sources for seed701, then709,719; two warm passes only after cold coverage.',
             resource_profile=server_profile(), proof_start_cutoff_hours=32.5, proof_drain_hours=34,
             compiler_constraint_cap=24000000,
             compiler_cap_scope='Isolated coordinator copies raise only compiler resource allowance; identical Go/Rust executables and proof relations.',
             disk_policy='Before admitting a cell reserve60GiB forArcene,50GiB forGisette,36GiB forTFIDF,32GiB otherwise. Retire only completed older-case private proving-key caches in chronological order if needed; retain all public evidence and explicitly exclude their warm timings.',
             initial_study_hard_cap_hours=36, rental_hours=72, remaining_recovery_reserve_hours=36,
             phase_timeout_seconds=5400, admission='20% cell forecast margin before32.5h from original canary; use maximum of declared and observed per-dataset cell durations; never extend the paid rental.',
             unavailable_optional_source='CIFAR10 ResNet50 loader has no cached feature archive; not counted as executed or silently downloaded.',
             arcene_limit='Only100 cached rows:64 past,4 incoming,32 audit; three splits share records. Report small-sample limitation and rank79 residual redundancy.',
             timing='Keep N32 extension separate from N128 original server timing cohort; native controls have no proof-time claim.',
             authorisation='User explicitly requested all available dataset benchmarks to strengthen paper evidence.',
             no_manuscript_edits=True, acceptance_probability='Not inferable from dataset count.',
             deployment_manifest_sha256=sha(MANIFEST), package_pin_sha256=sha(PACKAGE/'pin.json')))
        save(root/'plan_pin.json', dict(plan_sha256=sha(root/'plan.json'),
             deployment_manifest_sha256=sha(MANIFEST), go_sha256=sha(GO), rust_sha256=sha(RUST)))
    pins = read(root/'plan_pin.json')
    assert sha(root/'plan.json') == pins['plan_sha256']
    assert sha(MANIFEST) == pins['deployment_manifest_sha256']
    assert sha(GO) == pins['go_sha256'] and sha(RUST) == pins['rust_sha256']


class AllSourcesCampaign(ServerCampaign):
    def invoke(self, label, command, timeout=5400, expect_failure=False):
        """New36-hour study plan; keep the original72-hour rental and guards."""
        command = list(map(str, command))
        if timeout == 1800:
            timeout = 5400
        if any(module in command for module in ['campaigns.run_group_fast_control','campaigns.run_group_refinement_control']):
            child_root = Path(command[command.index('--root')+1])
            if (child_root/'protocol.json').exists() and '--resume' not in command:
                command.append('--resume')
        replacements = {
            'campaigns.local_fast_tradeoff':'campaigns.server_all_sources_full_20261101',
            'campaigns.run_group_fast_control':'campaigns.server_all_sources_compact_20261101',
            'campaigns.run_group_refinement_control':'campaigns.server_all_sources_refinement_20261101',
        }
        command = [replacements.get(part,part) for part in command]
        resources = self.root/'resources'; resources.mkdir(exist_ok=True)
        previous = sorted(resources.glob(label+'.attempt_*.json'))
        if previous and read(previous[-1]).get('complete'):
            return read(previous[-1])
        if any((p/'STOP').exists() for p in [self.root,PRIMARY,DIVERSITY]):
            raise RuntimeError('STOP requested')
        if time.time()-self.state['started_unix'] >= 32.5*3600:
            raise RuntimeError('All-sources admission deadline')
        if psutil.disk_usage(self.root).free < 20*2**30:
            raise RuntimeError('Retain evidence: disk admission limit')
        receipt = resources/f'{label}.attempt_{len(previous)+1:02}.json'
        self.state.update(status='running', active_phase=label, updated_unix=time.time())
        save(self.root/'matched_queue.json', self.state)
        args = [sys.executable,'-m','tools.server_guard_20261101','--receipt',str(receipt),
                '--log',str(receipt.with_suffix('.log')),'--root',str(self.root),'--timeout',str(min(timeout,5400))]
        if expect_failure:
            args.append('--expect-failure')
        result = subprocess.run(args+['--']+command, cwd=REPO, env=self.env)
        if result.returncode:
            raise RuntimeError(f'{label} failed; inspect {receipt.with_suffix(".log")}')
        if label not in self.state['completed']:
            self.state['completed'].append(label)
        save(self.root/'matched_queue.json', self.state)
        return read(receipt)


def wait_for_diversity(campaign):
    while True:
        if any((p/'STOP').exists() for p in [campaign.root,PRIMARY,DIVERSITY]):
            raise RuntimeError('STOP requested while waiting')
        if time.time()-campaign.state['started_unix'] >= 32.5*3600:
            raise RuntimeError('All-sources admission deadline')
        prior = read(DIVERSITY/'matched_queue.json')
        if prior.get('status') == 'stopped_with_preserved_evidence':
            raise RuntimeError('HAR/MNIST queue needs recovery before adding work')
        if prior.get('complete'):
            lock = (PRIMARY/'coordinator.lock').open('a')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
                assert read(PRIMARY/'matched_queue.json')['complete']
                return lock
            except BlockingIOError:
                lock.close()
        campaign.state.update(status='waiting_for_har_mnist', active_phase=None, updated_unix=time.time())
        save(campaign.root/'matched_queue.json', campaign.state)
        time.sleep(20)


def run(root):
    if not server_profile():
        raise RuntimeError('Authorized server profile required')
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'coordinator.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name, digest in read(MANIFEST).items():
        assert sha(REPO/name) == digest, name
    freeze(root)
    campaign = AllSourcesCampaign(root)
    campaign.state.update(pid=os.getpid(), updated_unix=time.time())
    primary_lock = None
    try:
        primary_lock = wait_for_diversity(campaign)
        inherit_smoke(root)
        for dataset in EXTRA+EXISTING:
            for seed in SEEDS:
                campaign.invoke(f'native_{dataset}_seed{seed}', [sys.executable,'-m',
                    'campaigns.server_all_sources_20261101','--root',root,'--native-case',dataset,'--seed',seed])
        save(root/'native_complete.json', dict(complete=True, workloads=11, fits=33,
             scope='Native controls include both favorable and unfavorable results; not33 independent datasets.'))
        cold = []
        forecasts = {dataset:SPECS[dataset]['cold_seconds'] for dataset in EXTRA}
        for seed in SEEDS:
            for dataset in EXTRA:
                name = f'{dataset}_seed{seed}_n32_q4'
                existing = root/'matched'/name/'cold_complete.json'
                if existing.exists() and read(existing).get('complete'):
                    case = materialize(root,dataset,seed)
                    cold.append(case)
                    forecasts[dataset] = max(forecasts[dataset],read(existing).get('wall_seconds',0))
                    continue
                if not admissible(time.time()-campaign.state['started_unix'], forecasts[dataset]):
                    save(root/'exclusions'/(name+'_cold.json'), dict(reason='Prespecified time admission', complete=False))
                    continue
                if not ensure_disk_for_case(root,cold,dataset):
                    save(root/'exclusions'/(name+'_cold.json'),dict(reason='Prespecified disk admission after retiring older private key caches',complete=False))
                    continue
                case = materialize(root, dataset, seed)
                started = time.monotonic()
                try:
                    if not campaign.rust_case(case):
                        raise RuntimeError('Pinned adapter layout does not fit; retain exclusion')
                    campaign.method_case(case)
                    save(case/'cold_complete.json', dict(complete=True, methods=4, finished_unix=time.time(),wall_seconds=time.monotonic()-started))
                    cold.append(case)
                except RuntimeError as error:
                    if any((p/'STOP').exists() for p in [root,PRIMARY,DIVERSITY]):
                        raise
                    save(case/'cold_incomplete.json', dict(complete=False, error=str(error)))
                    retire_private_keys(root,case)
                finally:
                    forecasts[dataset] = max(forecasts[dataset],time.monotonic()-started)
                    save(root/'cell_forecasts.json',forecasts)
        for repetition in [1,2]:
            for case in cold:
                if (case/'private_key_retirement.json').exists():
                    save(root/'exclusions'/(case.name+f'_warm{repetition}.json'),dict(reason='Private proving keys retired for later-source coverage; public proof evidence retained',complete=False))
                    continue
                if not admissible(time.time()-campaign.state['started_unix'],1800):
                    save(root/'exclusions'/(case.name+f'_warm{repetition}.json'), dict(reason='Prespecified time admission', complete=False))
                    continue
                try:
                    warm(campaign,case,repetition)
                except RuntimeError as error:
                    if any((p/'STOP').exists() for p in [root,PRIMARY,DIVERSITY]):
                        raise
                    save(case/f'warm{repetition}_incomplete.json',dict(complete=False,error=str(error)))
        campaign.state.update(complete=True,status='queue_finished',active_phase=None,finished_unix=time.time(),
             completed_cold_cells=len(cold),planned_cold_cells=21,
             interpretation='Queue finished is not a claim that every optional timing cell completed; read cell markers and exclusions.')
    except BaseException as error:
        campaign.state.update(complete=False,status='stopped_with_preserved_evidence',error=str(error),finished_unix=time.time())
        raise
    finally:
        save(root/'matched_queue.json',campaign.state)
        files=[p for p in root.rglob('*.json') if p.is_file() and p.name!='evidence_manifest.json']
        save(root/'evidence_manifest.json',dict(files={str(p.relative_to(root)):sha(p) for p in files},
             queue_complete=campaign.state.get('complete',False)))
        if primary_lock is not None:
            primary_lock.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--native-case',choices=EXTRA+EXISTING)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    args=parser.parse_args()
    if args.native_case:
        if args.seed is None:
            parser.error('--seed is required with --native-case')
        native_case(args.root.resolve(),args.native_case,args.seed)
    else:
        run(args.root.resolve())
