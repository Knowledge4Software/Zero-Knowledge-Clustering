"""Max-RAM capacity probe followed by bounded, matched scalability controls.

Supersedes only the waiting optional extension. Primary source/timings remain
immutable. A guard-limited8192 probe is a capacity result, not a speedup claim.
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
from campaigns.server_scale_20261101 import ScaleCampaign, admission, PRIMARY
from campaigns.server_campaign_20261101 import materialize, smoke_input, REPO
from campaigns.proof_runtime import server_profile
from commitjl.pilot_io import array_sha


class CapacityCampaign(ScaleCampaign):
    def rust_case(self, case):
        if read(case/'input_pin.json')['n'] != 8192:
            return super().rust_case(case)
        name = case.name
        source = case/'ordered_head/full/input.json'
        inp = read(source)
        private, public = case/'zkml_private', case/'zkml_public'
        public.mkdir(exist_ok=True)
        self.invoke(name+'_zkml_inspect',[RUST,'inspect',source,case/'zkml_inspection',24,17])
        self.invoke(name+'_zkml_prepare',[RUST,'prepare',source,private,24,17])
        shutil.copy2(private/'owner_pin.json',case/'zkml_owner_pin.json')
        for filename in ['params','vk']:
            if not (public/filename).exists():
                os.link(private/filename,public/filename)
        # One fresh update proof suffices for the high-memory capacity probe.
        # It does not supply the reference+update comparison used in other cells.
        self.invoke(name+'_zkml_bank001',[RUST,'query',source,private,1])
        destination = public/'bank_001'
        destination.mkdir(exist_ok=True)
        for filename in ['proof','public.json']:
            shutil.copy2(private/'bank_001'/filename,destination/filename)
        self.invoke(name+'_zkml_verify001',[RUST,'verify',public,1,
                    case/'zkml_owner_pin.json',case/'approved_task.json'])
        exact, labels = 0, []
        for row in inp['x']:
            ds = [sum((int(x)-int(c))**2 for x,c in zip(row,center))
                  for center in inp['reuse_centers'][1]]
            label = min(range(len(ds)),key=ds.__getitem__)
            labels.append(label)
            exact += ds[label]
        assert exact == read(private/'bank_001/result.json')['cost']
        assert labels == read(private/'bank_001/private_diagnostics.json')['labels']
        save(case/'zkml_checked.json',dict(complete=True,proof_count=1,bank=1,cost=exact,
             all_costs_match_integer_oracle=True,all_assignments_match_integer_oracle=True,
             public_only_replay=True,scope='Single-proof capacity probe, not a four-method or reference/update comparison.'))
        return True


def freeze(root):
    if (root/'plan.json').exists():
        return
    inputs = REPO/'server_inputs/capacity_20261101'
    save(root/'plan.json',dict(created_unix=time.time(),resource_profile=server_profile(),
         selection='First training-only distinct candidate, same frozen seed701 basis/centers, fixed nested audit prefixes.',
         dataset='fashion',fit_seed=701,q=1,capacity_n=8192,capacity_degree=24,
         capacity_proofs=1,capacity_bank=1,capacity_forecast_seconds=4.5*3600,
         capacity_scope='Single-proof memory capacity probe; guards may stop before completion. Report the limiting resource and preserve all attempts.',
         stress_n=4096,stress_degree=23,stress_forecast_seconds=3*3600,
         stress_scope='Halo2 reference and one update proof, not a four-method comparison.',
         matched_n=[1024,2048],initial_matched1024_seconds=3*3600,
         methods=['centroid certificate','PCA certificate','Full Groth16','ZKML primitive adapter'],
         maximum_n=8192,maximum_degree=24,advice_columns=17,
         primary_completion_required=True,proof_start_cutoff_hours=26.5,
         phase_timeout_seconds=5400,initial_budget_hours=32,
         admission_rule='20% forecast margin, predicted cell completion before hour26.5 from original canary;2048 forecast is2.2 times measured1024 time.',
         failure_policy='A failed capacity/stress probe is retained and smaller cells continue. No repeated capacity retries or outcome-based input selection.',
         superseded_extension='server_scale_20261101, stopped before any proof phase',
         authorization='User explicitly requested all usable RAM, updated full-queue ETA and current results.',
         private_key_retirement='After each capacity/stress attempt exits, hash and retire only its rebuildable private pk file. Preserve parameters, vk, proof, inputs, public bundles and receipts.',
         no_new_independent_fit=True,no_manuscript_edits=True,
         extra_input_pin_sha256=sha(inputs/'pin.json'),
         primary_plan_sha256=sha(PRIMARY/'plan.json'),
         deployment_manifest_sha256=sha(REPO/'capacity_deployment_manifest_20261101.json')))
    save(root/'plan_pin.json',dict(plan_sha256=sha(root/'plan.json'),rust_sha256=sha(RUST),go_sha256=sha(GO)))


def extend_inputs(root):
    target = root/'fitted/fashion_seed701'
    if (target/'pin.json').exists():
        assert sha(target/'arrays.npz') == read(target/'pin.json')['arrays_sha256']
        return
    source = PRIMARY/'fitted/fashion_seed701'
    pin = read(source/'pin.json')
    assert sha(source/'arrays.npz') == pin['arrays_sha256']
    inputs = REPO/'server_inputs/capacity_20261101'
    extra_pin = read(inputs/'pin.json')
    assert sha(inputs/'fashion_extra.npz') == extra_pin['archive_sha256']
    with np.load(source/'arrays.npz') as archive:
        data = dict(archive.items())
    with np.load(inputs/'fashion_extra.npz') as extra:
        pixels, ids = extra['pixels'], extra['raw_source_ids']
    assert pixels.shape == (7168,784) and array_sha(pixels) == extra_pin['pixels_sha256']
    canonical = np.random.default_rng(0).choice(70000,5000,replace=False)
    assert len(np.unique(ids)) == 7168 and not np.intersect1d(ids,canonical).size
    expanded = np.clip(np.rint(pixels.astype(float)*pin['scale']),-4095,4095).astype(np.int64)
    data['audit'] = np.concatenate([data['audit'],expanded])
    assert data['audit'].shape == (8192,784)
    target.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target/'arrays.npz',**data)
    pin.update(arrays_sha256=sha(target/'arrays.npz'),audit_sha256=array_sha(data['audit']),
         primary_fitted_pin_sha256=sha(source/'pin.json'),appended_input_pin_sha256=sha(inputs/'pin.json'),
         audit_prefix_source='Original1024 + prior3072 extra + new4096, all added IDs outside canonical5000.')
    save(target/'pin.json',pin)


def retire_private_key(root,case):
    key = case/'zkml_private/pk'
    if not key.exists():
        return
    assert key.resolve().is_relative_to(root.resolve()) and key.is_file() and not key.is_symlink()
    receipt = dict(path=str(key.relative_to(root)),bytes=key.stat().st_size,sha256=sha(key),
         retired_unix=time.time(),reason='Rebuildable private key; retain parameters, source, public evidence and all receipts for the bounded rental disk budget.')
    save(case/'private_key_retirement.json',receipt)
    key.unlink()


def check_deployment():
    for relative,digest in read(REPO/'capacity_deployment_manifest_20261101.json').items():
        assert sha(REPO/relative) == digest,relative
    old = (REPO/'code/zk_external/zkml/prototype_audit.rs').read_text()
    new = (REPO/'code/zk_external/zkml/prototype_audit_capacity_20261101.rs').read_text()
    assert new == old.replace('shape.n<=1024 && shape.d<=10000 && shape.centers>=2 && shape.k<=20',
                              'shape.n<=8192 && shape.d<=10000 && shape.centers>=2 && shape.k<=24')


def wait_for_primary(campaign):
    root,origin = campaign.root,campaign.state['started_unix']
    while True:
        primary = read(PRIMARY/'matched_queue.json')
        if (root/'STOP').exists() or (PRIMARY/'STOP').exists():
            raise RuntimeError('STOP requested while waiting')
        if primary.get('status') == 'stopped_with_preserved_evidence':
            raise RuntimeError('Primary stopped; optional work must wait for its recovery')
        if time.time()-origin >= 26.5*3600:
            raise RuntimeError('Extension admission deadline')
        if primary.get('complete'):
            lock = (PRIMARY/'coordinator.lock').open('a')
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                assert read(PRIMARY/'matched_queue.json')['complete']
                return lock
            except BlockingIOError:
                lock.close()
        campaign.state.update(status='waiting_for_primary',active_phase=None,updated_unix=time.time())
        save(root/'matched_queue.json',campaign.state)
        time.sleep(30)


def run(root):
    if not server_profile():
        raise RuntimeError('Authorized server profile required')
    root.mkdir(parents=True,exist_ok=True)
    lock = (root/'coordinator.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    check_deployment()
    freeze(root)
    pins = read(root/'plan_pin.json')
    assert sha(root/'plan.json') == pins['plan_sha256']
    assert sha(RUST) == pins['rust_sha256'] and sha(GO) == pins['go_sha256']
    campaign = CapacityCampaign(root)
    origin = campaign.state['started_unix']
    campaign.state['pid'] = os.getpid()
    try:
        primary_lock = wait_for_primary(campaign)
        extend_inputs(root)
        smoke_input(root)
        campaign.smoke()
        old,new = PRIMARY/'smoke_v2',root/'smoke_v2'
        campaign.invoke('old_proof_new_verifier',[RUST,'verify',old/'public',1,old/'owner_pin.json',old/'approved_task.json'],300)
        legacy = REPO/'tmp/external_baselines_20261027/prototype-audit-linux'
        campaign.invoke('new_proof_old_verifier',[legacy,'verify',new/'public',1,new/'owner_pin.json',new/'approved_task.json'],300)
        plan = read(root/'plan.json')
        for n,forecast,needed_disk in [(8192,plan['capacity_forecast_seconds'],64),(4096,plan['stress_forecast_seconds'],38)]:
            if not admission(time.time()-origin,forecast) or psutil.disk_usage(root).free < needed_disk*2**30:
                save(root/f'n{n}_probe_excluded.json',dict(reason='Declared time or disk admission limit',forecast_seconds=forecast))
                continue
            case = materialize(root,'fashion',701,n,1)
            try:
                campaign.rust_case(case)
                save(case/'probe_complete.json',dict(complete=True,scope=('Single update proof capacity probe' if n==8192 else 'Halo2 reference/update stress probe')))
            except RuntimeError as error:
                if (root/'STOP').exists() or (PRIMARY/'STOP').exists():
                    raise
                save(case/'probe_incomplete.json',dict(complete=False,error=str(error),
                     scope='Resource-limited probe; inspect attempt stop reason, not a successful speed comparison'))
            finally:
                retire_private_key(root,case)
        forecast = plan['initial_matched1024_seconds']
        for n in [1024,2048]:
            if not admission(time.time()-origin,forecast) or psutil.disk_usage(root).free < 30*2**30:
                save(root/f'n{n}_matched_excluded.json',dict(reason='Declared time or disk admission limit',forecast_seconds=forecast))
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
                if (root/'STOP').exists() or (PRIMARY/'STOP').exists():
                    raise
                save(case/'incomplete.json',dict(complete=False,error=str(error)))
                break
        campaign.state.update(complete=True,status='queue_finished',active_phase=None,
             finished_unix=time.time(),interpretation='Read each cell completion/exclusion marker; resource-limited probes remain incomplete.')
    except BaseException as error:
        campaign.state.update(complete=False,status='stopped_with_preserved_evidence',error=str(error),finished_unix=time.time())
        raise
    finally:
        save(root/'matched_queue.json',campaign.state)
        files = [p for p in root.rglob('*') if p.is_file() and
                 (p.suffix in ['.json','.proof','.vk'] or p.name=='proof')]
        save(root/'evidence_manifest.json',dict(files={str(p.relative_to(root)):sha(p) for p in files
             if p.name!='evidence_manifest.json'},queue_complete=campaign.state.get('complete',False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args = parser.parse_args()
    run(args.root.resolve())
