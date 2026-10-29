"""New seed/split confirmation of the fixed rank79 + at-most16-row policy.

All seeds, ranks, budgets and updates are fixed before outcomes. These are
additional fits on existing public datasets, not independent new datasets.
"""
from pathlib import Path
import argparse, json, time
import numpy as np
import psutil
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from campaigns.local_fast_tradeoff import save,sha
from commitjl.pilot_io import frozen_image_sample,array_sha
from commitjl.enriched_jl import enrich_centroid_basis
from commitjl.protected_jl import centroid_basis
from commitjl.reuse_certificates import integer_stats
from commitjl.relative_certificates import relative_certificate
from commitjl.group_change_bounds import exact_old_correction_bounds
from commitjl.group_moment_reference import prepare_groups
from commitjl.jl_certificate_pilots import hint_order,refinement_work
from experiments.rank79_independent_replication import trace,extend_basis
from experiments.pilot_jl_certificate_ideas import distances,prediction_labels

REPO=Path(__file__).resolve().parents[2]
CONFIGURATIONS=[('mnist',3,137),('fashion_mnist',5,149)]


def run(out,resume):
    out.mkdir(parents=True,exist_ok=resume)
    protocol=dict(configurations=CONFIGURATIONS,ages=[1,8],steps={'1':[2,4,8,16],'8':[9,12,16,24]},
        prediction_rank=63,certificate_rank=79,basis_scale=16384,exact_row_cap=16,
        ordering='Descending canonical certified row width; index breaks ties, stop on a conclusive sign.',
        native_only=True,runner_sha256=sha(Path(__file__)),
        selection='Two new predeclared seed/split configurations after policy selection on the older development grid. Every distinct supplied natural update retained; no accuracy or sign filtering.',
        replication_unit='Two additional dataset/k/seed strata; ages and representations share a split, audit prefixes overlap. Not new datasets or statistically independent update samples.')
    if resume:assert json.loads((out/'protocol.json').read_text())==json.loads(json.dumps(protocol))
    else:save(out/'protocol.json',protocol)
    checkpoint=json.loads((out/'checkpoint.json').read_text()) if (out/'checkpoint.json').exists() else dict(records=[],fits=[],elapsed_seconds=0.)
    records=checkpoint['records'];fits=checkpoint['fits'];done={(f['fit'],f['representation']) for f in fits};started=time.monotonic()
    with threadpool_limits(limits=1):
        for dataset,k,seed in CONFIGURATIONS:
            if all((f'{dataset}_k{k}_seed{seed}_age{age}',method) in done for age in [1,8] for method in ['ordered_head','pca']):continue
            while psutil.virtual_memory().available<1000*2**20:
                save(out/'status.json',dict(complete=False,status='waiting_for_ram',pid=__import__('os').getpid()));time.sleep(10)
            source=frozen_image_sample(REPO,dataset);source_hash=array_sha(source);source=source[np.linalg.norm(source,axis=1)>0]
            index=np.random.default_rng(741+seed).permutation(len(source));cut=int(.6*len(source));tr=source[index[:cut]];scale=4095/float(np.abs(tr).max())
            training=np.rint(tr*scale).astype(np.int64);audit=np.clip(np.rint(source[index[cut:cut+1024]]*scale),-4096,4096).astype(np.int64)
            snapshots=trace(training,k,seed)
            for age in [1,8]:
                fit=f'{dataset}_k{k}_seed{seed}_age{age}';old=snapshots[age];steps=[2,4,8,16] if age==1 else [9,12,16,24]
                unique=[old];source_steps=[age];seen={old.tobytes()}
                for step in steps:
                    candidate=snapshots[step]
                    if candidate.tobytes() not in seen:seen.add(candidate.tobytes());unique.append(candidate);source_steps.append(step)
                for method in ['ordered_head','pca']:
                    if (fit,method) in done:continue
                    tick=time.monotonic()
                    if method=='ordered_head':basis=enrich_centroid_basis(training.astype(float),old,extra=63-centroid_basis(old).shape[1],random_state=seed,preserve_pca_order=True)[0].T
                    else:basis=PCA(63,svd_solver='randomized',random_state=seed).fit(training.astype(float)).components_
                    basis=np.rint(basis*16384).astype(np.int64);fitting=time.monotonic()-tick;aux,extra=extend_basis(basis,training,old)
                    path=out/'fits'/f'{fit}_{method}.npz';path.parent.mkdir(exist_ok=True);np.savez_compressed(path,basis=basis,auxiliary_basis=aux,centers=np.asarray(unique),audit=audit)
                    fits.append(dict(fit=fit,representation=method,dataset=dataset,k=k,fit_seed=seed,age=age,archive_sha256=sha(path),
                        source_sample_sha256=source_hash,training_sha256=array_sha(training),audit_sha256=array_sha(audit),input_scale=str(scale),
                        prediction_fitting_seconds=fitting,auxiliary_fitting_seconds=extra,lloyd_steps=source_steps))
                    for n in [128,1024]:
                        x=audit[:n];od=distances(x,old);groups=od.argmin(1);prepared=prepare_groups(x,groups.tolist(),k);oldcost=int(od.min(1).sum())
                        model=dict(version='bound-audit-v1',method='learned_energy',n=n,d=784,k=k,input_bits=12,basis_scale=16384,input_scale=str(scale),centers=old.tolist(),basis=basis.tolist(),tail_width=0,tail_rows=[],tail_signs=[],seed=str(seed));h,_=integer_stats(model,x)
                        for index,new in enumerate(unique[1:],1):
                            nd=distances(x,new);truth=int(nd.min(1).sum()-oldcost);prediction=prediction_labels(model,h,new);assignment=relative_certificate(model,x,new);flags=np.asarray(assignment['certified_flags'],bool)
                            assert np.array_equal(prediction,np.asarray(assignment['labels'])) and not np.any(flags&(prediction!=nd.argmin(1)))
                            b=exact_old_correction_bounds(dict(model,basis=aux.tolist()),x,prepared,old,new);den=b['cost_denominator'];exact=[int(v)*den for v in nd.min(1)-nd[np.arange(n),groups]]
                            order=hint_order(b,None,'width');initial=refinement_work(b,exact,order,0);refined=refinement_work(b,exact,order,16)
                            expected='worsens' if truth>0 else ('improves' if truth<0 else 'equal')
                            assert refined['lower']<=truth*den<=refined['upper'] and refined['decision'] in ['inconclusive',expected]
                            records.append(dict(fit=fit,representation=method,n=n,source_step=source_steps[index],set=index,true_change=truth,
                                initial_decision=initial['decision'],decision=refined['decision'],exact_rows=refined['exact_rows'],
                                selected=order[:refined['exact_rows']],agreement_percent=100*float(np.mean(prediction==nd.argmin(1))),assignment_certified_percent=100*float(flags.mean()),
                                width_percent_old_cost=100*(refined['upper']-refined['lower'])/den/oldcost,
                                lower=str(refined['lower']),upper=str(refined['upper']),denominator=str(den),strict_switch_rows=int((nd.min(1)<nd[np.arange(n),groups]).sum())))
                    save(out/'checkpoint.json',dict(records=records,fits=fits,elapsed_seconds=checkpoint['elapsed_seconds']+time.monotonic()-started));save(out/'fit_records.json',fits)
                    save(out/'status.json',dict(complete=False,status='running',fits_complete=len(fits),expected_fits=8,active_fit=fit,representation=method));print(json.dumps(dict(fit=fit,representation=method,records=len(records))),flush=True)
    summary=[]
    for n in [128,1024]:
        for method in ['ordered_head','pca']:
            r=[v for v in records if v['n']==n and v['representation']==method];reg=[v for v in r if v['true_change']>0]
            summary.append(dict(n=n,representation=method,updates=len(r),initial_conclusive=sum(v['initial_decision']!='inconclusive' for v in r),
                conclusive=sum(v['decision']!='inconclusive' for v in r),regressions=len(reg),certified_regressions=sum(v['decision']=='worsens' for v in reg),
                mean_agreement_percent=float(np.mean([v['agreement_percent'] for v in r])) if r else None,minimum_agreement_percent=min([v['agreement_percent'] for v in r],default=None),
                mean_assignment_certified_percent=float(np.mean([v['assignment_certified_percent'] for v in r])) if r else None,
                mean_width_percent_old_cost=float(np.mean([v['width_percent_old_cost'] for v in r])) if r else None,
                total_exact_rows=sum(v['exact_rows'] for v in r),maximum_exact_rows=max([v['exact_rows'] for v in r],default=0)))
    result=dict(complete=True,native_only=True,all_intervals_contain_truth=True,false_conclusive=0,new_proofs=0,additional_split_configurations=2,representation_fits=8,records=len(records),results=summary)
    save(out/'records.json',records);save(out/'summary.json',result);save(out/'status.json',dict(complete=True,status='complete'));print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();run(a.out.resolve(),a.resume)
