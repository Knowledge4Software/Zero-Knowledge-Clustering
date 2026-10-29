"""Reproducible local gate for paper-derived prototype-audit baselines.

Freeze inputs before evaluating outcomes. Native results and real-proof adapter
results are separate. No manuscript, methodology or historical evidence edits.
Run with PYTHONPATH=code, preferably in the pinned Linux/WSL environment.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from commitjl.pilot_io import array_sha, frozen_image_sample, save, sha
from commitjl.published_audit_baselines import (
    CoresetAudit, ProjectedAudit, distances, interval_decision,
    point_decision, projection_bases, random_projection,
)
from commitjl.enriched_jl import enrich_centroid_basis
from commitjl.protected_jl import centroid_basis
from commitjl.relative_certificates import relative_certificate
from commitjl.reuse_certificates import integer_stats
from commitjl.group_change_bounds import advantage_bounds, assigned_change
from commitjl.group_moment_reference import prepare_groups
from commitjl.jl_certificate_pilots import hint_order
from experiments.rank79_independent_replication import extend_basis, trace

REPO = Path(__file__).resolve().parents[2]
DEFAULT = REPO / 'code/validation/outputs/published_baseline_gate_20261029'
METHODS = ['ordered_head', 'pca', 'sign_rp', 'sparse_rp']
SEEDS = [557, 563, 569]
SOURCES = {
    'sign_rp': dict(title='Random Projections for k-means Clustering',
                   authors='Boutsidis, Zouzias, Drineas', venue='NeurIPS 2010',
                   url='https://papers.neurips.cc/paper/3901-random-projections-for-k-means-clustering.pdf',
                   implementation='Algorithm 1 random sign embedding; supplied banks, no cluster refit; rank63 is an empirical budget, not a certified theorem parameter.'),
    'sparse_rp': dict(title='Database-friendly random projections: Johnson-Lindenstrauss with binary coins',
                     authors='Achlioptas', venue='JCSS 2003',
                     url='https://doi.org/10.1016/S0022-0000(03)00025-4',
                     implementation='Independent -sqrt(3/m),0,+sqrt(3/m) entries with probabilities1/6,2/3,1/6.'),
    'lightweight_coreset': dict(title='Scalable k-Means Clustering via Lightweight Coresets',
                              authors='Bachem, Lucic, Krause', venue='KDD 2018',
                              url='https://las.inf.ethz.ch/files/bachem18scalable.pdf',
                              implementation='Algorithm1, sampling with replacement, exact importance weights; fixed empirical sample budgets, no claimed epsilon/delta at these sizes.'),
    'ezdps': dict(title='ezDPS: An Efficient and Zero-Knowledge Machine Learning Inference Pipeline',
                 venue='PoPETs 2023', url='https://petsymposium.org/popets/2023/popets-2023-0061.pdf',
                 implementation='Applicability review only: DWT-PCA-SVM model-private inference, not this private-catalog cost-change task. No runtime imported.'),
    'zhang2021': dict(title='Verifiable k-means clustering scheme with privacy-preserving',
                     venue='Journal of Computer Applications 2021',
                     url='https://www.joca.cn/EN/10.11772/j.issn.1001-9081.2020060766',
                     implementation='Applicability review only; outsourced clustering/encryption contract differs. No native or proof runtime claimed.'),
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def model_for(rows, old, basis, scale, seed):
    return dict(version='bound-audit-v1', method='learned_energy', n=len(rows),
                d=rows.shape[1], k=len(old), input_bits=12, basis_scale=16384,
                input_scale=str(scale), centers=old.tolist(), basis=basis.tolist(),
                tail_width=0, tail_rows=[], tail_signs=[], seed=str(seed))


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    paper_paths = [REPO/'manuscript.tex', REPO/'references.bib']
    paper_paths += sorted((REPO/'paper_results').rglob('*.tex'))
    plan = dict(created_client_date='2026-10-29', status='frozen_before_new_outcomes',
                deadline_utc=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=6)).isoformat(),
                stop_directories=[str(root.relative_to(REPO)),str((root/'proofs').relative_to(REPO))],
                datasets=['fashion', 'text', 'har'], fit_seeds=SEEDS, native_catalog_sizes=[32, 128],
                native_requests=8, prediction_rank=63, certificate_rank=79, basis_scale=16384,
                correction_cap=16, native_timing_repetitions=3, coreset_draws=[16,32],
                methods=METHODS, papers=SOURCES, proof_catalog_size=32, proof_requests=4,
                proof_fit_seed=SEEDS[0], proof_datasets=['fashion','text'],
                proof_order='Fashion: centroid,PCA,sign,sparse; text: sparse,sign,PCA,centroid.',
                proof_repetitions=1, proof_budget_hours=6, maximum_phase_seconds=1800,
                resource_policy=dict(serial=True, workers=8, rss_mib=3584, reserve_ram_mib=650),
                inference='Exploratory independent refits on inspected public sources; neither unseen-data confirmation nor a population inference.',
                raw_vs_adapter='Raw published projections and coresets are numerical controls. RP certificate arms reuse our relation and are explicitly adaptations.',
                statement='Exact quantized full-catalog cost-change sign; encoder inference and class accuracy not certified.',
                timing='Native per-bank work with reusable row summaries, plus separately measured fitting/preparation; proof phases measured separately with all fallback paid.',
                no_remote_execution=True, paper_sha256={p.relative_to(REPO).as_posix():sha(p) for p in paper_paths})
    save(root/'plan.json', plan)
    save(root/'plan_pin.json', dict(plan_sha256=sha(root/'plan.json'),
                                  implementation_sha256=sha(Path(__file__)),
                                  baseline_module_sha256=sha(REPO/'code/commitjl/published_audit_baselines.py')))
    data_dir = REPO/'code/validation/outputs/referee_two_reviews_20261027/data'
    text = np.load(data_dir/'text_minilm384.npz')
    har = np.load(data_dir/'har561.npz')
    fashion = np.asarray(frozen_image_sample(REPO, 'fashion_mnist'))
    fashion = fashion[np.linalg.norm(fashion, axis=1)>0]
    with threadpool_limits(limits=1):
        for dataset in plan['datasets']:
            for seed in SEEDS:
                tick=time.perf_counter()
                if dataset=='fashion':
                    order=np.random.default_rng(741+seed).permutation(len(fashion))
                    training, audit=fashion[order[:1024]], fashion[order[1024:1152]]
                    source=dict(source='canonical5000 Fashion sample', source_sha256=array_sha(fashion),
                                training_row_positions=order[:1024].tolist(), audit_row_positions=order[1024:1152].tolist())
                else:
                    archive = text if dataset=='text' else har
                    train_ids=np.random.default_rng(seed).permutation(len(archive['training']))[:1024]
                    audit_ids=np.random.default_rng(seed+1).permutation(len(archive['audit']))[:128]
                    training, audit=archive['training'][train_ids],archive['audit'][audit_ids]
                    source=dict(source=str((data_dir/('text_minilm384.npz' if dataset=='text' else 'har561.npz')).relative_to(REPO)),
                                source_sha256=sha(data_dir/('text_minilm384.npz' if dataset=='text' else 'har561.npz')),
                                training_row_positions=train_ids.tolist(), audit_row_positions=audit_ids.tolist())
                scale=4095/float(np.abs(training[:512]).max())
                train=np.clip(np.rint(training*scale),-4095,4095).astype(np.int64)
                x=np.clip(np.rint(audit*scale),-4095,4095).astype(np.int64)
                past,incoming=train[:512],train[512:]
                old=trace(past,5,seed)[8]
                banks=[old];recipe=[]
                for batch_id,batch in enumerate(np.array_split(incoming,16)[:2]):
                    labels=distances(batch,old).argmin(1)
                    target=np.array([batch[labels==j].mean(0) if np.any(labels==j) else old[j] for j in range(5)])
                    for rate in [.125,.25,.5,1.]:
                        banks.append(np.rint((1-rate)*old+rate*target).astype(np.int64))
                        recipe.append(dict(batch=batch_id,rate=rate))
                common_seconds=time.perf_counter()-tick
                case=root/'inputs'/f'{dataset}_seed{seed}'
                arrays=dict(audit=x,training=past,centers=np.array(banks))
                fits={}
                for method in METHODS:
                    started=time.perf_counter()
                    if method=='ordered_head':
                        basis=enrich_centroid_basis(past.astype(float),old,
                            extra=63-centroid_basis(old).shape[1],random_state=seed,preserve_pca_order=True)[0].T
                        basis=np.rint(basis*16384).astype(np.int64)
                        aux,_=extend_basis(basis,past,old)
                    elif method=='pca':
                        basis=np.rint(PCA(63,svd_solver='randomized',random_state=seed).fit(past.astype(float)).components_*16384).astype(np.int64)
                        aux,_=extend_basis(basis,past,old)
                    else:
                        basis,aux=projection_bases(method,63,79,x.shape[1],seed)
                    fits[method]=time.perf_counter()-started
                    arrays[method+'_prediction']=basis;arrays[method+'_certificate']=aux
                case.mkdir(parents=True)
                np.savez_compressed(case/'arrays.npz',**arrays)
                save(case/'pin.json',dict(dataset=dataset,fit_seed=seed,**source,arrays_sha256=sha(case/'arrays.npz'),
                     training_sha256=array_sha(past),audit_sha256=array_sha(x),scale=scale,
                     proposal_recipe=recipe,common_fitting_seconds=common_seconds,basis_fitting_seconds=fits,
                     duplicates=[i for i,b in enumerate(banks[1:]) if any(np.array_equal(b,p) for p in banks[:i+1])]))
                print(json.dumps(dict(prepared=case.name,seconds=common_seconds)),flush=True)
    save(root/'prepared.json',dict(complete=True,plan_sha256=sha(root/'plan.json'),
         input_pins={p.parent.name:sha(p) for p in sorted((root/'inputs').glob('*/pin.json'))}))


def measured_batch(query,banks,repetitions=3):
    """Return individual complete-pass times; never time the independent oracle."""
    all_times=[];result=None
    for _ in range(repetitions):
        ticks=[];values=[]
        for bank in banks:
            start=time.perf_counter();values.append(query(bank));ticks.append(time.perf_counter()-start)
        all_times.append(ticks)
        if result is None:result=values
    return result,all_times


class CertificateAudit:
    def __init__(self,model,certificate_basis,rows):
        self.model=model;self.rows=rows
        self.cm=dict(model,basis=certificate_basis.tolist())
        self.h,self.e=integer_stats(model,rows)
        self.row_constant=np.array([sum(int(v)**2 for v in row)+int(e) for row,e in zip(self.h,self.e)],dtype=object)
        self.ch,self.ce=integer_stats(self.cm,rows)
        od=distances(rows,model['centers']);self.groups=od.argmin(1)
        self.prepared=prepare_groups(rows,self.groups,len(model['centers']))

    def predict(self,bank):
        hc,ec=integer_stats(self.model,bank)
        scores=self.h.astype(object)@hc.astype(object).T*(-2)
        scores+=np.array([sum(int(v)**2 for v in row)+int(e) for row,e in zip(hc,ec)],dtype=object)[None,:]
        labels=np.asarray(scores.argmin(1),dtype=int)
        cost=sum(int(v)+int(c) for v,c in zip(scores.min(1),self.row_constant))/16384**4
        return cost,labels

    def certify(self,bank):
        lower,upper=advantage_bounds(self.cm,self.ch,self.ce,bank,self.groups)
        den=16384**6
        assigned=assigned_change(self.prepared,self.model['centers'],bank)*den
        bounds=dict(row_lower=lower,row_upper=upper,cost_denominator=den,assigned_scaled=assigned,
                    change_lower_scaled=assigned+sum(map(int,lower)),change_upper_scaled=assigned+sum(map(int,upper)))
        lo,hi=bounds['change_lower_scaled'],bounds['change_upper_scaled']
        base=interval_decision(lo,hi)
        order=hint_order(bounds,None,'width')[:16];used=0
        if base=='inconclusive':
            for i in order:
                d=distances(self.rows[i:i+1],bank)[0]
                correction=(int(d.min())-int(d[self.groups[i]]))*den
                lo+=correction-int(lower[i]);hi+=correction-int(upper[i]);used+=1
                if interval_decision(lo,hi)!='inconclusive':break
        return dict(lower=str(lo),upper=str(hi),denominator=str(den),base_decision=base,
                    decision=interval_decision(lo,hi),corrections=used)


def native(root):
    plan=read(root/'plan.json');pin=read(root/'plan_pin.json')
    assert sha(root/'plan.json')==pin['plan_sha256']
    assert not (root/'native_results.json').exists(),'Use a fresh output directory for new measurements'
    records=[];times=[]
    with threadpool_limits(limits=1):
        for case in sorted((root/'inputs').iterdir()):
            meta=read(case/'pin.json');assert sha(case/'arrays.npz')==meta['arrays_sha256']
            data=np.load(case/'arrays.npz');banks=data['centers'];old=banks[0]
            for n in plan['native_catalog_sizes']:
                x=data['audit'][:n];truth=[distances(x,b) for b in banks]
                costs=[sum(map(int,d.min(1))) for d in truth];changes=[c-costs[0] for c in costs[1:]]
                exact,timing=measured_batch(lambda b:(int(distances(x,b).min(1).sum()),None),banks)
                times.append(dict(case=case.name,n=n,method='exact_full',phase='query',seconds=timing,preparation_seconds=0.))
                for j,delta in enumerate(changes):
                    records.append(dict(case=case.name,dataset=meta['dataset'],fit_seed=meta['fit_seed'],n=n,method='exact_full',request=j,
                        true_change=delta,estimated_change=delta,point_sign_correct=True,assignment_agreement=1.,
                        deterministic_decision=point_decision(delta),base_conclusive=True,conclusive=True,corrections=0,proof_available=False))
                for method in METHODS:
                    model=model_for(x,old,data[method+'_prediction'],meta['scale'],meta['fit_seed'])
                    tick=time.perf_counter();audit=CertificateAudit(model,data[method+'_certificate'],x);prep=time.perf_counter()-tick
                    predictions,ptimes=measured_batch(audit.predict,banks)
                    certificates,ctimes=measured_batch(audit.certify,banks[1:])
                    times.append(dict(case=case.name,n=n,method=method+'_certificate',phase='prediction',seconds=ptimes,preparation_seconds=prep,basis_fitting_seconds=meta['basis_fitting_seconds'][method]))
                    times.append(dict(case=case.name,n=n,method=method+'_certificate',phase='certificate',seconds=ctimes,preparation_seconds=prep,basis_fitting_seconds=meta['basis_fitting_seconds'][method]))
                    for j,(cert,delta) in enumerate(zip(certificates,changes)):
                        assert int(cert['lower'])<=delta*int(cert['denominator'])<=int(cert['upper'])
                        assert cert['decision'] in ['inconclusive',point_decision(delta)]
                        assignments=relative_certificate(model,x,banks[j+1]);flags=np.array(assignments['certified_flags'],bool)
                        labels=predictions[j+1][1]
                        assert np.array_equal(labels,assignments['labels'])
                        assert not np.any(flags&(labels!=truth[j+1].argmin(1)))
                        estimate=predictions[j+1][0]-predictions[0][0]
                        records.append(dict(case=case.name,dataset=meta['dataset'],fit_seed=meta['fit_seed'],n=n,method=method+'_certificate',request=j,
                            true_change=delta,estimated_change=estimate,point_sign_correct=point_decision(estimate)==point_decision(delta),
                            assignment_agreement=float(np.mean(labels==truth[j+1].argmin(1))),label_certificate_coverage=float(flags.mean()),
                            deterministic_decision=cert['decision'],base_conclusive=cert['base_decision']!='inconclusive',conclusive=cert['decision']!='inconclusive',
                            corrections=cert['corrections'],interval=cert,proof_available=False))
                    if method in ['sign_rp','sparse_rp']:
                        tick=time.perf_counter();raw=ProjectedAudit(x,random_projection(method,63,x.shape[1],meta['fit_seed']));prep=time.perf_counter()-tick
                        values,rtimes=measured_batch(raw.query,banks)
                        times.append(dict(case=case.name,n=n,method=method+'_raw',phase='query',seconds=rtimes,preparation_seconds=prep))
                        for j,delta in enumerate(changes):
                            estimate=values[j+1][0]-values[0][0]
                            records.append(dict(case=case.name,dataset=meta['dataset'],fit_seed=meta['fit_seed'],n=n,method=method+'_raw',request=j,
                                true_change=delta,estimated_change=estimate,point_sign_correct=point_decision(estimate)==point_decision(delta),
                                assignment_agreement=float(np.mean(values[j+1][1]==truth[j+1].argmin(1))),
                                deterministic_decision=None,base_conclusive=None,conclusive=None,corrections=None,proof_available=False))
                for lightweight in [True,False]:
                    for size in plan['coreset_draws']:
                        method=('lightweight' if lightweight else 'uniform')+f'_coreset_m{size}'
                        tick=time.perf_counter();audit=CoresetAudit(x,size,meta['fit_seed'],lightweight);prep=time.perf_counter()-tick
                        values,ctimes=measured_batch(audit.query,banks)
                        times.append(dict(case=case.name,n=n,method=method,phase='query',seconds=ctimes,preparation_seconds=prep,unique_rows=len(audit.indices)))
                        for j,delta in enumerate(changes):
                            estimate=values[j+1]-values[0]
                            records.append(dict(case=case.name,dataset=meta['dataset'],fit_seed=meta['fit_seed'],n=n,method=method,request=j,
                                true_change=delta,estimated_change=estimate,point_sign_correct=point_decision(estimate)==point_decision(delta),assignment_agreement=None,
                                deterministic_decision=None,base_conclusive=None,conclusive=None,corrections=None,proof_available=False))
                print(json.dumps(dict(native_complete=case.name,n=n,records=len(records))),flush=True)
    save(root/'native_results.json',dict(complete=True,new_proofs=0,native_only=True,records=records,timings=times,
         all_deterministic_intervals_valid=True,no_wrong_certified_labels=True,
         timing_scope='Three complete serial passes per frozen method/case. Native CPU times, not proof times; request work excludes the separately measured independent oracle.',
         source_pins={p.parent.name:sha(p) for p in sorted((root/'inputs').glob('*/pin.json'))}))


def proof_inputs(root):
    """Create all proof inputs before inspecting the native output file."""
    plan=read(root/'plan.json');proof_root=root/'proofs';proof_root.mkdir(exist_ok=False)
    for dataset in plan['proof_datasets']:
        case=root/'inputs'/f'{dataset}_seed{plan["proof_fit_seed"]}'
        data=np.load(case/'arrays.npz');meta=read(case/'pin.json');n=plan['proof_catalog_size']
        banks=[]
        for b in data['centers'][:plan['proof_requests']+1].tolist():
            if b not in banks:banks.append(b)
        target=proof_root/'matched'/f'{dataset}_n{n}'
        for method in METHODS:
            parent=target/method
            model=model_for(data['audit'][:n],data['centers'][0],data[method+'_prediction'],meta['scale'],meta['fit_seed'])
            inp=dict(model=model,x=data['audit'][:n].tolist(),salt_x='20261029557',salt_z='20261029559',
                     queries=[list(range(i,i+16)) for i in range(0,n,16)],reuse_centers=banks,comparison_base=banks[0])
            save(parent/'head_r63/input.json',inp)
            full=copy.deepcopy(inp);full['model'].update(method='full',basis=[],seed='')
            full['queries']=[list(range(i,i+8)) for i in range(0,n,8)]
            save(parent/'full/input.json',full)
            save(parent/'protocol.json',dict(dataset=dataset,fit_seed=meta['fit_seed'],n=n,native_n=128,k=5,
                representation=method,training_sha256=meta['training_sha256'],audit_sha256=meta['audit_sha256'],
                fitting_seconds=meta['basis_fitting_seconds'][method],proof_runtime_workers=8,execution_os='Linux/WSL',phase_wall_limit_seconds=1800,
                source_selection='Fixed first4 proposals of the first pinned fit, before new native outcomes. Exact public duplicates cached equally.',
                source_pin_sha256=sha(case/'pin.json')))
            save(parent/'auxiliary_basis.json',dict(source_input_sha256=sha(parent/'head_r63/input.json'),
                 training_sha256=meta['training_sha256'],basis_scale=16384,certificate_basis=data[method+'_certificate'].tolist(),
                 fitting_seconds=0.,rank=79,recipe='Frozen63-row prediction prefix,16 additional rows; published random sketches retain their distribution.'))
        save(target/'approved_task.json',dict(reuse_centers=banks))
    # Existing independently checked circuit-relation smokes can be reused by
    # exact binary hash. They are not new timing measurements.
    prior=REPO/'code/validation/outputs/published_baselines_20261027'
    for name in ['compact_smoke','refinement_smoke']:
        save(proof_root/name/'checks.json',read(prior/name/'checks.json'))
    save(proof_root/'input_manifest.json',dict(complete=True,
         inputs_sha256={p.relative_to(proof_root).as_posix():sha(p) for p in sorted((proof_root/'matched').rglob('input.json'))}))


def proofs(root):
    from campaigns.published_baselines_laptop import Campaign, GO, RUST
    from campaigns.shared_full_reference import share_full_reference
    assert os.name=='posix' and GO.exists() and RUST.exists()
    proof_root=root/'proofs';manifest=read(proof_root/'input_manifest.json')
    for file,digest in manifest['inputs_sha256'].items():assert sha(proof_root/file)==digest
    campaign=Campaign(proof_root)
    for smoke in ['compact_smoke','refinement_smoke']:
        assert read(proof_root/smoke/'checks.json')['binary_sha256']==sha(GO)
    try:
        for dataset in ['fashion','text']:
            target=proof_root/'matched'/f'{dataset}_n32'
            parent=target/'ordered_head'
            campaign.invoke(dataset+'_full',[sys.executable,'-m','campaigns.local_fast_tradeoff','--root',parent,'--binary',GO,'--run','--cases','full'])
            order=METHODS if dataset=='fashion' else METHODS[::-1]
            for method in order:
                p=target/method
                if method!='ordered_head':share_full_reference(parent,p)
                campaign.invoke(dataset+'_'+method+'_base',[sys.executable,'-m','campaigns.run_group_fast_control','--compact','--proof-backbone-only',
                    '--parent',p,'--root',target/(method+'_base'),'--binary',GO,'--smoke-receipt',proof_root/'compact_smoke/checks.json','--auxiliary-basis',p/'auxiliary_basis.json'])
                campaign.invoke(dataset+'_'+method+'_refined',[sys.executable,'-m','campaigns.run_group_refinement_control',
                    '--base',target/(method+'_base'),'--parent',p,'--root',target/(method+'_refined'),'--binary',GO,
                    '--smoke-receipt',proof_root/'refinement_smoke/checks.json'])
            # An existing paper-derived external backend remains in this new
            # cohort as a separately measured comparator, not a historical ratio.
            campaign.rust_case(target)
        campaign.state.update(complete=True,status='complete',active_phase=None,finished_unix=time.time())
        save(proof_root/'matched_queue.json',campaign.state)
    except BaseException as error:
        campaign.state.update(complete=False,status='stopped_with_preserved_evidence',error=str(error),finished_unix=time.time())
        save(proof_root/'matched_queue.json',campaign.state)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','native','proofs'])
    parser.add_argument('--root',type=Path,default=DEFAULT)
    args=parser.parse_args();root=args.root.resolve()
    if args.phase=='prepare':prepare(root);proof_inputs(root)
    elif args.phase=='native':native(root)
    else:proofs(root)


if __name__=='__main__':main()
