"""Explicitly authorized frozen-profile confirmation and scaling preparation.

No seed, rank, budget or request is chosen from its audit outcome. Source data
must contain enough real held-out rows; repetition/padding is forbidden.
"""
from pathlib import Path
import argparse,json,time
from campaigns.local_fast_tradeoff import save,sha
from campaigns.execution_admission import require_execution

REPO=Path(__file__).resolve().parents[2]
POLICY=REPO/'code/validation/outputs/referee_round2_20261026/execution_policy.json'
DESIGN=REPO/'code/validation/outputs/referee_round3_20261026/extended_study_plan.json'


def configurations(design):
    c=design['independent_fit_confirmation'];s=design['center_count_sweep']
    return sorted({(d,c['k'],seed) for d in c['datasets'] for seed in c['seeds']} |
                  {(d,k,s['seed']) for d in s['datasets'] for k in s['k']})


def catalog_sizes(design,dataset,k,seed):
    s=design['scaling']
    return s['catalog_sizes'] if (dataset,k,seed)==(s['dataset'],s['k'],s['seed']) else [design['independent_fit_confirmation']['audit_records']]


def heldout_indices(count,training_rows,maximum_audit,seed):
    import numpy as np
    if training_rows<80 or count<training_rows+maximum_audit:raise ValueError('Insufficient real rows; padding/duplication is forbidden')
    indices=np.random.default_rng(741+seed).permutation(count)
    return indices[:training_rows],indices[training_rows:training_rows+maximum_audit]


def run(out,design_path,resume=False):
    require_execution(json.loads(POLICY.read_text()))
    # Delay the fitting stack until remote admission has passed.
    import numpy as np
    from sklearn.decomposition import PCA
    from threadpoolctl import threadpool_limits
    from data.loaders import load_dataset
    from commitjl.pilot_io import array_sha
    from commitjl.enriched_jl import enrich_centroid_basis
    from commitjl.protected_jl import centroid_basis
    from commitjl.reuse_certificates import integer_stats
    from commitjl.relative_certificates import relative_certificate
    from commitjl.group_change_bounds import exact_old_correction_bounds
    from commitjl.group_moment_reference import prepare_groups
    from commitjl.jl_certificate_pilots import hint_order,refinement_work
    from experiments.rank79_independent_replication import trace,extend_basis
    from experiments.pilot_jl_certificate_ideas import distances,prediction_labels
    design=json.loads(design_path.read_text());profile=design['frozen_profile']
    assert profile==dict(prediction_rank=63,certificate_rank=79,basis_precision_bits=14,maximum_exact_rows=16,ordering='Descending canonical width, index ties; stop at first conclusive sign')
    out.mkdir(parents=True,exist_ok=resume)
    protocol=dict(design_sha256=sha(design_path),runner_sha256=sha(Path(__file__)),configurations=configurations(design),source_sample_rows=50000,training_rows=3000,source_sample_seed=0,
        native_only=True,selection='All predeclared fits, both ages/bases and every distinct source request. Retain all outcomes; no accuracy filter.',
        replication_unit='Eight dataset/seed splits; k sweeps, ages, bases, updates and catalog prefixes are paired within these splits.')
    if resume:assert json.loads((out/'protocol.json').read_text())==json.loads(json.dumps(protocol))
    else:save(out/'protocol.json',protocol)
    checkpoint=json.loads((out/'checkpoint.json').read_text()) if (out/'checkpoint.json').exists() else dict(records=[],fits=[])
    records,fits=checkpoint['records'],checkpoint['fits'];done={(f['fit'],f['representation']) for f in fits}
    started=time.monotonic()
    with threadpool_limits(limits=1):
        for dataset,k,seed in configurations(design):
            sizes=catalog_sizes(design,dataset,k,seed);maximum=max(sizes)
            if all((f'{dataset}_k{k}_seed{seed}_age{age}',method) in done for age in [1,8] for method in ['ordered_head','pca']):continue
            loaded=load_dataset(dataset,n_max=50000,seed=0);source=np.asarray(loaded['X'],float);source_hash=array_sha(source)
            source=source[np.linalg.norm(source,axis=1)>0]
            train_index,audit_index=heldout_indices(len(source),3000,maximum,seed)
            scale=4095/float(np.abs(source[train_index]).max())
            training=np.rint(source[train_index]*scale).astype(np.int64)
            audit=np.clip(np.rint(source[audit_index]*scale),-4096,4096).astype(np.int64)
            split_hash=dict(training_sha256=array_sha(training),audit_sha256=array_sha(audit),source_sample_sha256=source_hash,source_meta=loaded['meta'])
            save(out/f'split_{dataset}_{seed}_{maximum}.json',dict(**split_hash,training_source_indices=train_index.tolist(),audit_source_indices=audit_index.tolist()))
            snapshots=trace(training,k,seed)
            for age in [1,8]:
                fit=f'{dataset}_k{k}_seed{seed}_age{age}';old=snapshots[age]
                steps=[2,4,8,16] if age==1 else [9,12,16,24]
                sets=[old];source_steps=[age];seen={old.tobytes()}
                for step in steps:
                    candidate=snapshots[step]
                    if candidate.tobytes() not in seen:seen.add(candidate.tobytes());sets.append(candidate);source_steps.append(step)
                for method in ['ordered_head','pca']:
                    if (fit,method) in done:continue
                    tick=time.monotonic()
                    if method=='ordered_head':basis=enrich_centroid_basis(training.astype(float),old,extra=63-centroid_basis(old).shape[1],random_state=seed,preserve_pca_order=True)[0].T
                    else:basis=PCA(63,svd_solver='randomized',random_state=seed).fit(training.astype(float)).components_
                    basis=np.rint(basis*16384).astype(np.int64);fitting=time.monotonic()-tick;aux,extra=extend_basis(basis,training,old)
                    archive=out/'fits'/f'{fit}_{method}.npz';archive.parent.mkdir(exist_ok=True)
                    np.savez_compressed(archive,basis=basis,auxiliary_basis=aux,centers=np.asarray(sets),audit=audit)
                    pending=[]
                    for n in sizes:
                        rows=audit[:n];assert len(rows)==n
                        od=distances(rows,old);groups=od.argmin(1);prepared=prepare_groups(rows,groups.tolist(),k);oldcost=int(od.min(1).sum())
                        model=dict(version='bound-audit-v1',method='learned_energy',n=n,d=784,k=k,input_bits=12,basis_scale=16384,input_scale=str(scale),centers=old.tolist(),basis=basis.tolist(),tail_width=0,tail_rows=[],tail_signs=[],seed=str(seed))
                        h,_=integer_stats(model,rows)
                        for index,new in enumerate(sets[1:],1):
                            nd=distances(rows,new);truth=int(nd.min(1).sum()-oldcost);prediction=prediction_labels(model,h,new)
                            assignment=relative_certificate(model,rows,new);flags=np.asarray(assignment['certified_flags'],bool)
                            assert np.array_equal(prediction,np.asarray(assignment['labels'])) and not np.any(flags&(prediction!=nd.argmin(1)))
                            b=exact_old_correction_bounds(dict(model,basis=aux.tolist()),rows,prepared,old,new);den=b['cost_denominator']
                            exact=[int(v)*den for v in nd.min(1)-nd[np.arange(n),groups]];order=hint_order(b,None,'width')
                            initial=refinement_work(b,exact,order,0);refined=refinement_work(b,exact,order,16)
                            expected='worsens' if truth>0 else ('improves' if truth<0 else 'equal')
                            assert refined['lower']<=truth*den<=refined['upper'] and refined['decision'] in ['inconclusive',expected]
                            pending.append(dict(fit=fit,representation=method,n=n,source_step=source_steps[index],set=index,true_change=truth,
                                initial_decision=initial['decision'],decision=refined['decision'],exact_rows=refined['exact_rows'],selected=order[:refined['exact_rows']],
                                agreement_percent=100*float(np.mean(prediction==nd.argmin(1))),assignment_certified_percent=100*float(flags.mean()),
                                width_percent_old_cost=100*(refined['upper']-refined['lower'])/den/oldcost,lower=str(refined['lower']),upper=str(refined['upper']),denominator=str(den),
                                strict_switch_rows=int((nd.min(1)<nd[np.arange(n),groups]).sum())))
                    fits.append(dict(fit=fit,representation=method,dataset=dataset,k=k,fit_seed=seed,age=age,archive_sha256=sha(archive),**split_hash,
                        input_scale=str(scale),prediction_fitting_seconds=fitting,auxiliary_fitting_seconds=extra,lloyd_steps=source_steps))
                    records.extend(pending);done.add((fit,method))
                    save(out/'checkpoint.json',dict(records=records,fits=fits));save(out/'fit_records.json',fits)
                    save(out/'status.json',dict(complete=False,status='running',fits_complete=len(fits),fit=fit,representation=method,elapsed_seconds=time.monotonic()-started))
    save(out/'records.json',records)
    save(out/'summary.json',dict(complete=True,native_only=True,new_proofs=0,all_intervals_contain_truth=True,false_conclusive=0,representation_fits=len(fits),split_configurations=8,records=len(records),design_sha256=sha(design_path)))
    save(out/'status.json',dict(complete=True,status='complete'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--design',type=Path,default=DESIGN);p.add_argument('--resume',action='store_true');p.add_argument('--prepare-proof-case',help='After native completion: FIT:N, using the frozen archive');a=p.parse_args()
    if a.prepare_proof_case:
        require_execution(json.loads(POLICY.read_text()));fit,n=a.prepare_proof_case.rsplit(':',1)
        assert json.loads((a.out/'summary.json').read_text())['complete'],'Complete native confirmation before proof input preparation'
        meta=next(v for v in json.loads((a.out/'fit_records.json').read_text()) if v['fit']==fit)
        design=json.loads(a.design.read_text())
        if int(n) not in catalog_sizes(design,meta['dataset'],meta['k'],meta['fit_seed']):raise ValueError('Catalog size outside the predeclared fit profile')
        from campaigns.main_rank79 import prepare_case
        prepare_case(a.out/'proof_inputs',a.out,dict(id=fit+'_n'+n,fit=fit,n=int(n)),workers=4)
    else:run(a.out.resolve(),a.design.resolve(),a.resume)
