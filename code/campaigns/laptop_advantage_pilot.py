"""Bounded, real-proof laptop gate before renting a server.

Fresh fit seed, existing public benchmark sources; exploratory, not unseen-data
confirmation. Eight chronological requests per dataset are fixed before native
outcomes; a64-request numerical diagnostic never selects the proof requests.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'code'))
from campaigns.published_baselines_laptop import Campaign, RUST, GO, read, save, sha

DEFAULT = REPO/'code/validation/outputs/laptop_advantage_pilot_20261028'
PRIOR = REPO/'code/validation/outputs/published_baselines_20261027'


def prepare(root):
    import numpy as np
    from sklearn.decomposition import PCA
    from threadpoolctl import threadpool_limits
    from commitjl.pilot_io import frozen_image_sample, array_sha
    from commitjl.enriched_jl import enrich_centroid_basis
    from commitjl.protected_jl import centroid_basis
    from commitjl.relative_certificates import relative_certificate
    from commitjl.group_change_bounds import exact_old_correction_bounds
    from commitjl.group_moment_reference import prepare_groups
    from commitjl.jl_certificate_pilots import hint_order, refinement_work
    from experiments.rank79_independent_replication import trace, extend_basis
    from experiments.pilot_jl_certificate_ideas import distances

    root.mkdir(parents=True, exist_ok=False)
    seed = 521
    spec = dict(client_date='2026-10-28', fit_seed=seed,
                datasets=['fashion', 'text'], k=5, old_training_step=8,
                past_training_rows=512, incoming_training_rows=512,
                native_sizes=[32, 128], native_requests_at_n32=64,
                native_requests_at_n128=8, proof_n=32, proof_request_prefix=8,
                proposal_recipe='16 consecutive disjoint incoming batches; rates .125,.25,.5,1.; batch-major order; no audit-based selection.',
                prediction_rank=63, certificate_rank=79, basis_bits=14,
                maximum_exact_rows=16,
                runtime=dict(os='Linux/WSL', serial=True, workers=8,
                             max_process_tree_rss_mib=3584, min_available_ram_mib=650,
                             max_proof_hours=4, phase_seconds=1800),
                timing_repetitions=1,
                inference='Exploratory gate: new fit seed on previously inspected benchmark sources; no unseen-dataset or universal-superiority claim.',
                selection='Every first8 request retained, deduplicated only by public center equality for every method; no native-outcome filtering.',
                fitting='Common old-center/proposal fitting recorded separately; method-specific basis fitting charged once in cold totals.',
                paper_changes=False, remote_access=False)
    save(root/'plan.json', spec)
    papers = [REPO/'manuscript.tex', REPO/'references.bib'] + sorted((REPO/'paper_results').rglob('*.tex'))
    save(root/'authorization_and_scope.json', dict(
        authorization='User explicitly requested basic local experiments before renting a server.',
        paper_sha256={p.relative_to(REPO).as_posix():sha(p) for p in papers},
        maximum_catalog_records=1024, remote_access=False))
    source_text = REPO/'code/validation/outputs/referee_two_reviews_20261027/data/text_minilm384.npz'
    save(root/'plan_pin.json', dict(plan_sha256=sha(root/'plan.json'), script_sha256=sha(Path(__file__)),
                                  text_source_sha256=sha(source_text)))
    numerical = []
    with threadpool_limits(limits=1):
        for name in ['fashion', 'text']:
            started = time.monotonic()
            if name == 'fashion':
                raw = np.asarray(frozen_image_sample(REPO, 'fashion_mnist'))
                raw = raw[np.linalg.norm(raw, axis=1)>0]
                order = np.random.default_rng(741+seed).permutation(len(raw))
                raw_training, raw_audit = raw[order[:1024]], raw[order[1024:2048]]
                source_pin = dict(source='canonical5000 Fashion sample', sample_sha256=array_sha(raw),
                                  training_source_row_positions=order[:1024].tolist(),
                                  audit_source_row_positions=order[1024:2048].tolist())
            else:
                source = np.load(source_text)
                raw_training, raw_audit = source['training'][:1024], source['audit'][:1024]
                source_pin = dict(source=source_text.relative_to(REPO).as_posix(), source_sha256=sha(source_text),
                                  training_source_rows=[0,1024], audit_source_rows=[0,1024])
            scale = 4095/float(np.abs(raw_training[:512]).max())
            train = np.clip(np.rint(raw_training*scale), -4095, 4095).astype(np.int64)
            audit = np.clip(np.rint(raw_audit*scale), -4095, 4095).astype(np.int64)
            past, incoming = train[:512], train[512:]
            old = trace(past, 5, seed)[8]
            banks = [old]
            proposals = []
            for number, batch in enumerate(np.array_split(incoming,16)):
                labels = distances(batch,old).argmin(1)
                target = np.asarray([batch[labels==j].mean(0) if np.any(labels==j) else old[j] for j in range(5)])
                for rate in [.125,.25,.5,1.]:
                    banks.append(np.rint((1-rate)*old+rate*target).astype(np.int64))
                    proposals.append(dict(batch=number, rate=rate))
            common_seconds = time.monotonic()-started
            case = root/'matched'/f'{name}_n32'
            save(case/'source_pin.json', dict(**source_pin, common_center_proposal_fitting_seconds=common_seconds,
                                            training_sha256=array_sha(past), incoming_sha256=array_sha(incoming),
                                            audit_sha256=array_sha(audit), input_scale=scale))
            unique = [old.tolist()]
            selected, duplicates = [], []
            for i, centers in enumerate(banks[1:9]):
                value = centers.tolist()
                if value in unique:
                    duplicates.append(i)
                else:
                    selected.append(i); unique.append(value)
            assert len(unique)>1
            for method in ['ordered_head','pca']:
                tick=time.monotonic()
                if method=='ordered_head':
                    basis=enrich_centroid_basis(past.astype(float),old,
                        extra=63-centroid_basis(old).shape[1],random_state=seed,preserve_pca_order=True)[0].T
                else:
                    basis=PCA(63,svd_solver='randomized',random_state=seed).fit(past.astype(float)).components_
                basis=np.rint(basis*16384).astype(np.int64)
                prediction_fit=time.monotonic()-tick
                aux, auxiliary_fit=extend_basis(basis,past,old)
                model=dict(version='bound-audit-v1',method='learned_energy',n=32,d=audit.shape[1],k=5,
                           input_bits=12,basis_scale=16384,input_scale=str(scale),centers=old.tolist(),
                           basis=basis.tolist(),tail_width=0,tail_rows=[],tail_signs=[],seed=str(seed))
                parent=case/method
                inp=dict(model=model,x=audit[:32].tolist(),salt_x='20261028521',salt_z='20261028523',
                         queries=[list(range(i,i+16)) for i in range(0,32,16)],
                         reuse_centers=unique,comparison_base=old.tolist())
                save(parent/'head_r63/input.json',inp)
                full=copy.deepcopy(inp);full['model'].update(method='full',basis=[],seed='')
                full['queries']=[list(range(i,i+8)) for i in range(0,32,8)]
                save(parent/'full/input.json',full)
                save(parent/'protocol.json',dict(dataset=name,fit_seed=seed,n=32,native_n=128,k=5,
                    representation=method,training_sha256=array_sha(past),audit_sha256=array_sha(audit),
                    fitting_seconds=prediction_fit,prediction_fitting_seconds=prediction_fit,
                    auxiliary_fitting_seconds=auxiliary_fit,proof_runtime_workers=8,execution_os='Linux/WSL',
                    phase_wall_limit_seconds=1800,selected_source_sets=selected,duplicate_source_sets=duplicates,
                    source_selection=spec['selection'],source_proposals=proposals[:8],
                    common_fitting_seconds=common_seconds))
                save(parent/'auxiliary_basis.json',dict(source_input_sha256=sha(parent/'head_r63/input.json'),
                    training_sha256=array_sha(past),audit_sha256=array_sha(audit),basis_scale=16384,
                    certificate_basis=aux.tolist(),rank=79,fitting_seconds=auxiliary_fit,
                    recipe='Frozen prediction63 plus16 training residual directions.'))
                np.savez_compressed(case/f'{method}_diagnostic_inputs.npz',audit=audit[:128],training=past,
                                    basis=basis,auxiliary_basis=aux,centers=np.asarray(banks))
                for n, limit in [(32,64),(128,8)]:
                    x=audit[:n];od=distances(x,old);groups=od.argmin(1)
                    prepared=prepare_groups(x,groups.tolist(),5);old_cost=int(od.min(1).sum())
                    m=dict(model,n=n)
                    for i,new in enumerate(banks[1:limit+1]):
                        nd=distances(x,new);truth=int(nd.min(1).sum())-old_cost
                        assignment=relative_certificate(m,x,new)
                        prediction=np.asarray(assignment['labels']);flags=np.asarray(assignment['certified_flags'],bool)
                        assert not np.any(flags&(prediction!=nd.argmin(1)))
                        b=exact_old_correction_bounds(dict(m,basis=aux.tolist()),x,prepared,old,new)
                        den=b['cost_denominator']
                        corrections=[int(v)*den for v in nd.min(1)-nd[np.arange(n),groups]]
                        refined=refinement_work(b,corrections,hint_order(b,None,'width'),16)
                        assert refined['lower']<=truth*den<=refined['upper']
                        expected='improves' if truth<0 else 'worsens' if truth>0 else 'equal'
                        assert refined['decision'] in ['inconclusive',expected]
                        numerical.append(dict(dataset=name,method=method,n=n,request=i,proposal=proposals[i],
                            truth=truth,decision=refined['decision'],exact_rows=refined['exact_rows'],
                            agreement_percent=100*float(np.mean(prediction==nd.argmin(1))),
                            certified_labels_percent=100*float(flags.mean())))
                    save(root/'native_checkpoint.json',dict(records=numerical,active_dataset=name,active_method=method,n=n))
            left=read(case/'ordered_head/full/input.json');right=read(case/'pca/full/input.json')
            assert left['x']==right['x'] and left['reuse_centers']==right['reuse_centers']
            save(case/'approved_task.json',dict(reuse_centers=unique))
            save(case/'input_pin.json',dict(complete=True,n=32,d=audit.shape[1],k=5,
                full_input_sha256=sha(case/'ordered_head/full/input.json'),
                centroid_input_sha256=sha(case/'ordered_head/head_r63/input.json'),
                pca_input_sha256=sha(case/'pca/head_r63/input.json'),centers=len(unique)))
            print(json.dumps(dict(prepared=name,unique_proposals=len(unique)-1)),flush=True)
    summary=[]
    for name in ['fashion','text']:
        for method in ['ordered_head','pca']:
            for n,request_count in [(32,8),(32,64),(128,8)]:
                rows=[r for r in numerical if r['dataset']==name and r['method']==method and r['n']==n and r['request']<request_count]
                summary.append(dict(dataset=name,method=method,n=n,requests=len(rows),
                    conclusive=sum(r['decision']!='inconclusive' for r in rows),
                    correction_rows=sum(r['exact_rows'] for r in rows),
                    agreement_percent=float(np.mean([r['agreement_percent'] for r in rows]))))
    assert sha(root/'plan.json')==read(root/'plan_pin.json')['plan_sha256']
    save(root/'native_checks.json',dict(complete=True,native_only=True,new_proofs=0,
         all_intervals_contain_truth=True,no_wrong_decisions=True,records=numerical,summary=summary))
    print(json.dumps(dict(native_complete=True,summary=summary)),flush=True)


class Pilot(Campaign):
    def invoke(self,label,command,timeout=1800,expect_failure=False):
        if time.time()-self.state['started_unix']>4*3600:
            raise RuntimeError('Four-hour laptop pilot budget exhausted; preserve all evidence')
        return super().invoke(label,command,timeout,expect_failure)

    def run(self):
        assert os.name=='posix' and RUST.exists() and GO.exists()
        assert sha(self.root/'plan.json')==read(self.root/'plan_pin.json')['plan_sha256']
        # Existing smoke receipts are correctness evidence, never new benchmark work.
        for folder in ['compact_smoke','refinement_smoke']:
            check=read(PRIOR/folder/'checks.json')
            assert check['complete'] and check['binary_sha256']==sha(GO)
            save(self.root/folder/'checks.json',check)
        external=read(PRIOR/'smoke_v2/checks.json')
        assert external['complete'] and external['rust_binary_sha256']==sha(RUST)
        save(self.root/'smoke_reuse.json',dict(prior=str(PRIOR),go_sha256=sha(GO),rust_sha256=sha(RUST),
            meaning='Reuse previously completed real-proof/tampering correctness smoke only; no historical workload timings reused.'))
        try:
            # Fixed order across all four systems; one exploratory timing run.
            for name in ['fashion','text']:
                target=self.root/'matched'/f'{name}_n32'
                self.rust_case(target)
                self.method_case(target)
            self.state.update(complete=True,status='complete',active_phase=None,finished_unix=time.time())
            save(self.root/'matched_queue.json',self.state)
        except BaseException as error:
            self.state.update(complete=False,status='stopped_with_evidence',error=str(error))
            save(self.root/'matched_queue.json',self.state)
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--stage',choices=['prepare','proofs'],required=True)
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.root.resolve())
    else:Pilot(args.root.resolve()).run()
