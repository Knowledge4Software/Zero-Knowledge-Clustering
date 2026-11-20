"""Deferred larger-circuit extension, isolated from the frozen primary queue.

The4096-row stress case measures the exact Halo2 adapter only. Complete
four-method comparisons are attempted at1024 and, if admitted,2048 rows.
Larger shared prefixes do not count as additional independent training fits.
"""
from pathlib import Path
import argparse
import fcntl
import os
import shutil
import time

import numpy as np
import psutil

from campaigns.published_baselines_laptop import GO, RUST, read, save, sha
from campaigns.server_campaign_20261101 import ServerCampaign, materialize, smoke_input, REPO
from campaigns.proof_runtime import server_profile
from commitjl.pilot_io import array_sha

PRIMARY = REPO/'code/validation/outputs/server_campaign_20261101'


def admission(elapsed_seconds, forecast_seconds):
    # Includes a20% uncertainty margin and leaves proof drain/replay time.
    return elapsed_seconds + 1.2*forecast_seconds < 26.5*3600


class ScaleCampaign(ServerCampaign):
    def invoke(self, label, command, timeout=5400, expect_failure=False):
        if (PRIMARY/'STOP').exists():
            raise RuntimeError('Primary STOP is also honored by the extension')
        if timeout == 1800:
            timeout = 5400
        return super().invoke(label, command, timeout, expect_failure)

    def rust_case(self, case):
        name = case.name
        source = case/'ordered_head/full/input.json'
        inp = read(source)
        n = inp['model']['n']
        degree = {512:20, 1024:21, 2048:22, 4096:23}[n]
        private, public = case/'zkml_private', case/'zkml_public'
        public.mkdir(exist_ok=True)
        # Size rule fixed from the measured512-row layout; inspection must pass.
        self.invoke(name+'_zkml_inspect', [RUST, 'inspect', source,
                    case/'zkml_inspection', degree, 17])
        save(case/'zkml_inspection/selection.json', dict(k=degree, columns=17,
             selection='Frozen degree20+log2(N/512), validated by MockProver; no outcome-based search.'))
        self.invoke(name+'_zkml_prepare', [RUST, 'prepare', source, private, degree, 17])
        shutil.copy2(private/'owner_pin.json', case/'zkml_owner_pin.json')
        for filename in ['params', 'vk']:
            if not (public/filename).exists():
                os.link(private/filename, public/filename)
        costs = []
        for index, centers in enumerate(inp['reuse_centers']):
            # Separate banks preserve the90-minute limit on each proof phase.
            self.invoke(name+f'_zkml_bank{index:03}', [RUST, 'query', source, private, index])
            destination = public/f'bank_{index:03}'
            destination.mkdir(exist_ok=True)
            for filename in ['proof', 'public.json']:
                shutil.copy2(private/f'bank_{index:03}'/filename, destination/filename)
            self.invoke(name+f'_zkml_verify{index:03}', [RUST, 'verify', public, index,
                        case/'zkml_owner_pin.json', case/'approved_task.json'])
            # Independent Python integer oracle, including the tie rule.
            exact, labels = 0, []
            for row in inp['x']:
                ds = [sum((int(x)-int(c))**2 for x, c in zip(row, center)) for center in centers]
                label = min(range(len(ds)), key=ds.__getitem__)
                labels.append(label)
                exact += ds[label]
            assert exact == read(private/f'bank_{index:03}/result.json')['cost']
            assert labels == read(private/f'bank_{index:03}/private_diagnostics.json')['labels']
            costs.append(exact)
        save(case/'zkml_checked.json', dict(complete=True, costs=costs,
             public_only_replay=True, all_costs_match_integer_oracle=True,
             all_assignments_match_integer_oracle=True, proof_count=len(costs),
             adaptation='Versioned size-bound extension of the ZKML primitive adapter.'))
        return True


def freeze(root):
    if (root/'plan.json').exists():
        return
    inputs = REPO/'server_inputs/scale_20261101'
    save(root/'plan.json', dict(created_unix=time.time(), resource_profile=server_profile(),
         selection='First training-only distinct candidate, same frozen seed701 basis/centers, fixed nested audit prefixes.',
         dataset='fashion', fit_seed=701, q=1, stress_n=4096, stress_degree=23,
         stress_scope='Exact Halo2 adapter resource stress only; no four-method speed comparison at4096.',
         matched_n=[1024,2048], methods=['centroid certificate','PCA certificate','Full Groth16','ZKML primitive adapter'],
         maximum_n=4096, maximum_degree=23, prediction_rank=63, certificate_rank=79,
         primary_completion_required=True, primary_sources_unchanged=True,
         proof_start_cutoff_hours=26.5, phase_timeout_seconds=5400, initial_budget_hours=32,
         planned_stress_seconds=3*3600, initial_matched1024_seconds=3*3600,
         admission_rule='20% forecast margin; no cell starts unless its total forecast finishes before hour26.5. Re-estimate2048 from measured1024 wall time; all per-phase caps remain90min.',
         optional_policy='Record failure or time/resource exclusion; never tune inputs or replace failed results. A failed4096 stress probe does not suppress smaller matched comparisons.',
         no_new_independent_fit=True, no_warm_repetitions=True, no_manuscript_edits=True,
         extra_input_pin_sha256=sha(inputs/'pin.json'), primary_plan_sha256=sha(PRIMARY/'plan.json'),
         deployment_manifest_sha256=sha(REPO/'scale_deployment_manifest_20261101.json')))
    save(root/'plan_pin.json', dict(plan_sha256=sha(root/'plan.json'), rust_sha256=sha(RUST), go_sha256=sha(GO)))


def extend_inputs(root):
    target = root/'fitted/fashion_seed701'
    if (target/'pin.json').exists():
        assert sha(target/'arrays.npz') == read(target/'pin.json')['arrays_sha256']
        return
    source = PRIMARY/'fitted/fashion_seed701'
    pin = read(source/'pin.json')
    assert sha(source/'arrays.npz') == pin['arrays_sha256']
    inputs = REPO/'server_inputs/scale_20261101'
    extra_pin = read(inputs/'pin.json')
    assert sha(inputs/'fashion_extra.npz') == extra_pin['archive_sha256']
    data = {k:v for k,v in np.load(source/'arrays.npz').items()}
    extra = np.load(inputs/'fashion_extra.npz')
    pixels, ids = extra['pixels'], extra['raw_source_ids']
    assert pixels.shape == (3072,784) and array_sha(pixels) == extra_pin['pixels_sha256']
    canonical = np.random.default_rng(0).choice(extra_pin['source_rows'], 5000, replace=False)
    assert len(np.unique(ids)) == 3072 and not np.intersect1d(ids,canonical).size
    expanded = np.clip(np.rint(pixels.astype(float)*pin['scale']),-4095,4095).astype(np.int64)
    data['audit'] = np.concatenate([data['audit'],expanded])
    assert data['audit'].shape == (4096,784)
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target/'arrays.npz', **data)
    pin.update(arrays_sha256=sha(target/'arrays.npz'), audit_sha256=array_sha(data['audit']),
               primary_fitted_pin_sha256=sha(source/'pin.json'),
               appended_input_pin_sha256=sha(inputs/'pin.json'),
               audit_prefix_source='First1024 unchanged from primary; appended3072 exclude every canonical5000 source row.')
    save(target/'pin.json', pin)


def check_deployment():
    for relative, digest in read(REPO/'scale_deployment_manifest_20261101.json').items():
        assert sha(REPO/relative) == digest, relative
    # Ensure this adapter changes only the two resource admission bounds.
    old = (REPO/'code/zk_external/zkml/prototype_audit.rs').read_text()
    new = (REPO/'code/zk_external/zkml/prototype_audit_scale_20261101.rs').read_text()
    before = 'shape.n<=1024 && shape.d<=10000 && shape.centers>=2 && shape.k<=20'
    after = 'shape.n<=4096 && shape.d<=10000 && shape.centers>=2 && shape.k<=23'
    assert old.count(before) == 1 and new == old.replace(before,after)


def run(root):
    if not server_profile():
        raise RuntimeError('Authorized server profile required')
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    check_deployment()
    freeze(root)
    pins = read(root/'plan_pin.json')
    assert sha(root/'plan.json') == pins['plan_sha256']
    assert sha(RUST) == pins['rust_sha256'] and sha(GO) == pins['go_sha256']
    campaign = ScaleCampaign(root)
    origin = campaign.state['started_unix']
    campaign.state['pid'] = os.getpid()
    try:
        # A sleeping coordinator cannot contaminate primary timing measurements.
        while not read(PRIMARY/'matched_queue.json').get('complete'):
            primary = read(PRIMARY/'matched_queue.json')
            if (root/'STOP').exists() or (PRIMARY/'STOP').exists():
                raise RuntimeError('STOP requested while waiting')
            if primary.get('status') == 'stopped_with_preserved_evidence':
                raise RuntimeError('Primary queue stopped; preserve its priority and do not start optional work')
            if time.time()-origin >= 26.5*3600:
                raise RuntimeError('Extension admission deadline reached while waiting')
            campaign.state.update(status='waiting_for_primary', active_phase=None, updated_unix=time.time())
            save(root/'matched_queue.json', campaign.state)
            time.sleep(30)
        # Shared exclusion lock also prevents overlap with a resumed primary.
        primary_lock = (PRIMARY/'coordinator.lock').open('a')
        while True:
            try:
                fcntl.flock(primary_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                # The primary can publish completion before its final manifest
                # is finished. Wait for that export to release the lock.
                if time.time()-origin >= 26.5*3600 or (root/'STOP').exists() or (PRIMARY/'STOP').exists():
                    raise RuntimeError('Deadline or STOP while waiting for primary lock')
                time.sleep(30)
        assert read(PRIMARY/'matched_queue.json')['complete']
        extend_inputs(root)
        smoke_input(root)
        campaign.smoke()
        # New verifier must also accept the old adapter's honest small proof.
        old = PRIMARY/'smoke_v2'
        campaign.invoke('old_binary_proof_new_verifier', [RUST,'verify',old/'public',1,
                        old/'owner_pin.json',old/'approved_task.json'],300)
        # Old verifier must accept the new small proof, validating compatibility.
        legacy = REPO/'tmp/external_baselines_20261027/prototype-audit-linux'
        smoke = root/'smoke_v2'
        campaign.invoke('new_binary_proof_old_verifier', [legacy,'verify',smoke/'public',1,
                        smoke/'owner_pin.json',smoke/'approved_task.json'],300)
        stress_forecast = read(root/'plan.json')['planned_stress_seconds']
        if admission(time.time()-origin, stress_forecast):
            stress = materialize(root,'fashion',701,4096,1)
            try:
                campaign.rust_case(stress)
                save(stress/'stress_complete.json',dict(complete=True,scope='Halo2 resource stress only'))
            except RuntimeError as error:
                save(stress/'stress_incomplete.json',dict(complete=False,error=str(error),
                     scope='Resource exclusion retained; not a successful four-method comparison'))
        else:
            save(root/'stress_excluded.json',dict(reason='Insufficient declared time budget'))
        forecast = read(root/'plan.json')['initial_matched1024_seconds']
        for n in [1024,2048]:
            if not admission(time.time()-origin,forecast):
                save(root/f'n{n}_excluded.json',dict(reason='Insufficient declared time budget',forecast_seconds=forecast))
                continue
            # One large proving key plus existing evidence must fit disk reserves.
            if psutil.disk_usage(root).free < 30*2**30:
                save(root/f'n{n}_excluded.json',dict(reason='Insufficient free disk for another complete cell'))
                continue
            started = time.monotonic()
            case = materialize(root,'fashion',701,n,1)
            try:
                campaign.rust_case(case)
                campaign.method_case(case)
                elapsed = time.monotonic()-started
                save(case/'complete.json',dict(complete=True,wall_seconds=elapsed,methods=4))
                forecast = max(forecast,elapsed)*2.2
            except RuntimeError as error:
                save(case/'incomplete.json',dict(complete=False,error=str(error)))
                # Preserve bounded larger-cell exclusions rather than repeat a failure.
                break
        campaign.state.update(complete=True,status='queue_finished',active_phase=None,
             finished_unix=time.time(),interpretation='Consult each cell marker: queue completion can include time/resource exclusions.')
    except BaseException as error:
        campaign.state.update(complete=False,status='stopped_with_preserved_evidence',error=str(error),finished_unix=time.time())
        raise
    finally:
        save(root/'matched_queue.json',campaign.state)
        files = [p for p in root.rglob('*') if p.is_file() and
                 (p.suffix in ['.json','.proof','.vk'] or p.name == 'proof')]
        save(root/'evidence_manifest.json',dict(files={str(p.relative_to(root)):sha(p) for p in files
             if p.name != 'evidence_manifest.json'},queue_complete=campaign.state.get('complete',False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args = parser.parse_args()
    run(args.root.resolve())
