"""Fresh training-seed replication, with matched inputs for real proof runs.

All fits/updates are fixed before outcomes. Dataset/seed/k strata are the four
independent split configurations; ages, representations and updates share data.
"""
from pathlib import Path
import argparse,copy,csv,json,os,time
import numpy as np
import psutil
from sklearn.cluster import kmeans_plusplus
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits
from commitjl.pilot_io import save,sha,array_sha,sign,frozen_image_sample
from commitjl.enriched_jl import enrich_centroid_basis
from commitjl.protected_jl import centroid_basis
from commitjl.reuse_certificates import integer_stats
from commitjl.relative_certificates import relative_certificate
from commitjl.group_change_bounds import exact_old_correction_bounds
from commitjl.group_moment_reference import prepare_groups
from experiments.pilot_jl_certificate_ideas import distances,prediction_labels

REPO=Path(__file__).resolve().parents[2]
CONFIGURATIONS=[(d,k,s) for d in ['fashion_mnist','mnist'] for k,s in [(3,101),(5,113)]]
PROOF_FITS=['fashion_mnist_k3_seed101_age8','mnist_k5_seed113_age1']


def trace(training,k,seed):
    centers=kmeans_plusplus(training.astype(float),k,random_state=seed)[0];result={}
    for step in range(1,25):
        # Preserve floating Lloyd distances until assignment (the independent
        # integer Full oracle is used only after quantizing the snapshots).
        labels=np.concatenate([((training[i:i+128,None,:]-centers[None,:,:])**2).sum(2).argmin(1) for i in range(0,len(training),128)])
        centers=np.asarray([training[labels==j].mean(0) if np.any(labels==j) else centers[j] for j in range(k)])
        result[step]=np.rint(centers).astype(np.int64)
    return result


def extend_basis(basis,training,old):
    tick=time.monotonic();q=np.linalg.qr(basis.T.astype(float),mode='reduced')[0];residual=training.astype(float)-old[0];residual-=residual@q@q.T
    _,_,directions=randomized_svd(residual,n_components=64,n_iter=4,random_state=20261025)
    result=np.concatenate((basis,np.rint(directions[:16]*16384).astype(np.int64)))
    assert np.array_equal(result[:63],basis);return result,time.monotonic()-tick


def make_proof_parent(out,fit,method,model,aux,rows,sets,meta):
    parent=out/'proof_inputs'/f'{fit}_{method}'
    unique=[];indices=[];seen=set();duplicates=[]
    for s,c in enumerate(sets):
        digest=array_sha(c)
        if digest in seen:duplicates.append(s);continue
        seen.add(digest);indices.append(s);unique.append(c.tolist())
    assert unique[0]==model['centers']
    m=copy.deepcopy(model);m['n']=128
    inp=dict(model=m,x=rows[:128].tolist(),salt_x='20261101',salt_z='20261103',queries=[list(range(i,i+16)) for i in range(0,128,16)],reuse_centers=unique,comparison_base=m['centers'])
    save(parent/'head_r63/input.json',inp)
    full=copy.deepcopy(inp);full['model'].update(method='full',basis=[],seed='');full['queries']=[list(range(i,i+8)) for i in range(0,128,8)];save(parent/'full/input.json',full)
    save(parent/'protocol.json',dict(**meta,source_fit=fit,representation=method,n=128,native_n=1024,fitting_seconds=meta['prediction_fitting_seconds'],selected_source_sets=indices,duplicate_source_sets=duplicates,selection='First128 pinned audit rows, all4 predeclared training-only updates; exact duplicate center sets cached before evaluating outcomes. No outcome/accuracy selection.',timing='Fresh matched Full and compact63/79, cold setup, generation, public-only replay, and newly generated fallback for every unresolved update. One repetition on a laptop.',batches=dict(full=8,compact=16)))
    save(parent/'auxiliary_basis.json',dict(source_input_sha256=sha(parent/'head_r63/input.json'),training_sha256=meta['training_sha256'],audit_sha256=meta['audit_sha256'],basis_scale=16384,certificate_basis=aux.tolist(),rank=79,fitting_seconds=meta['auxiliary_fitting_seconds'],recipe='Same original63 prefix, training-only residual randomized SVD64,4 iterations,seed20261025; first16 rounded at precision14.',selection='Predeclared fresh fit and every distinct update; no held-out selection. Prediction model frozen at63.'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=a.resume);started=time.monotonic();records=[];fits=[];prior_elapsed=0.
    if a.resume and (out/'checkpoint.json').exists():
        checkpoint=json.loads((out/'checkpoint.json').read_text());records=checkpoint['records'];fits=checkpoint['fits'];prior_elapsed=checkpoint['elapsed_seconds']
        assert json.loads((out/'protocol.json').read_text())['runner_sha256']==sha(Path(__file__)),'runner changed; inspect partial campaign before resuming'
    completed={(r['fit'],r['representation']) for r in fits}
    save(out/'protocol.json',dict(configurations=[dict(dataset=d,k=k,seed=s) for d,k,s in CONFIGURATIONS],ages=[1,8],update_steps={'1':[2,4,8,16],'8':[9,12,16,24]},representations=['ordered_head','pca'],certificate_ranks=[63,79],prediction_rank=63,basis_scale=16384,scopes=[1024,128],proof_fits=PROOF_FITS,proof_profiles=['head79','head63','pca79'],accuracy_target_percent=99.,selection='All configurations/ages/methods/updates fixed before outcomes; retain duplicates, errors, natural regressions and inconclusive outcomes. Proof selection independent of quality.',replication_unit='Four dataset/seed/k split configurations. Eight deployed models and two representations share these splits; updates/ranks/scopes are paired, not independent replications.',scope='Numerical confirmation first. These are new fits on existing public source datasets, not new datasets or production logs. Actual proof timings are a separate campaign.',runner_sha256=sha(Path(__file__))))
    with threadpool_limits(limits=1):
        for dataset,k,seed in CONFIGURATIONS:
            while psutil.virtual_memory().available<1100*2**20:
                save(out/'status.json',dict(complete=False,status='waiting_for_ram',pid=os.getpid(),stage='fitting',dataset=dataset,seed=seed));time.sleep(15)
            source=frozen_image_sample(REPO,dataset);source=source[np.linalg.norm(source,axis=1)>0];idx=np.random.default_rng(741+seed).permutation(len(source));cut=int(.6*len(source));tr=source[idx[:cut]];scale=4095/float(np.abs(tr).max());training=np.rint(tr*scale).astype(np.int64);audit=np.clip(np.rint(source[idx[cut:cut+1024]]*scale),-4096,4096).astype(np.int64)
            snapshots=trace(training,k,seed)
            for age in [1,8]:
                fit=f'{dataset}_k{k}_seed{seed}_age{age}';old=snapshots[age];steps=[2,4,8,16] if age==1 else [9,12,16,24];sets=[old]+[snapshots[s] for s in steps]
                for method in ['ordered_head','pca']:
                    if (fit,method) in completed:
                        previous=next(r for r in fits if r['fit']==fit and r['representation']==method)
                        assert sha(out/'fits'/f'{fit}_{method}.npz')==previous['archive_sha256'];continue
                    tick=time.monotonic()
                    if method=='ordered_head':basis=enrich_centroid_basis(training.astype(float),old,extra=63-centroid_basis(old).shape[1],random_state=seed,preserve_pca_order=True)[0].T
                    else:basis=PCA(63,svd_solver='randomized',random_state=seed).fit(training.astype(float)).components_
                    basis=np.rint(basis*16384).astype(np.int64);fitting=time.monotonic()-tick;assert basis.shape==(63,784);aux,aux_seconds=extend_basis(basis,training,old)
                    meta=dict(dataset=dataset,k=k,fit_seed=seed,age=age,training_sha256=array_sha(training),audit_sha256=array_sha(audit),source_sample_sha256=array_sha(frozen_image_sample(REPO,dataset)),prediction_fitting_seconds=fitting,auxiliary_fitting_seconds=aux_seconds,input_scale=str(scale),lloyd_steps=[age]+steps)
                    m=dict(version='bound-audit-v1',method='learned_energy',n=1024,d=784,k=k,input_bits=12,basis_scale=16384,input_scale=str(scale),centers=old.tolist(),basis=basis.tolist(),tail_width=0,tail_rows=[],tail_signs=[],seed=f'fresh seed{seed}, training-only')
                    archive=out/'fits'/f'{fit}_{method}.npz';archive.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(archive,basis=basis,auxiliary_basis=aux,centers=np.asarray(sets),audit=audit)
                    fits.append(dict(fit=fit,representation=method,archive_sha256=sha(archive),**meta));save(out/'fit_records.json',fits)
                    if fit in PROOF_FITS:make_proof_parent(out,fit,method,m,aux,audit,sets,meta)
                    for n in [1024,128]:
                        rows=audit[:n];dist_old=distances(rows,old);groups=dist_old.argmin(1);prepared=prepare_groups(rows,groups.tolist(),k);h,_=integer_stats(m,rows);old_cost=int(dist_old.min(1).sum())
                        for s,new in enumerate(sets[1:],1):
                            truth_dist=distances(rows,new);truth=int(truth_dist.min(1).sum()-old_cost);pred=prediction_labels(m,h,new);assignment=relative_certificate(m,rows,new)
                            certified=np.asarray(assignment['certified_flags'],dtype=bool);assert np.array_equal(np.asarray(assignment['labels']),pred)
                            assert not np.any(certified&(pred!=truth_dist.argmin(1)))
                            for rank in [63,79]:
                                cm=m if rank==63 else dict(m,basis=aux.tolist());b=exact_old_correction_bounds(cm,rows,prepared,old,new);lo,hi=b['change_lower_scaled'],b['change_upper_scaled'];den=b['cost_denominator'];decision=sign(lo,hi)
                                assert lo<=truth*den<=hi and decision in ['inconclusive',sign(truth,truth)]
                                records.append(dict(fit=fit,representation=method,n=n,set=s,lloyd_step=steps[s-1],rank=rank,true_change=truth,truth=sign(truth,truth),decision=decision,lower=str(lo),upper=str(hi),denominator=str(den),width_percent_old_cost=100*(hi-lo)/den/old_cost,prediction_agreement_percent=100*float(np.mean(pred==truth_dist.argmin(1))),assignment_certified_percent=100*float(certified.mean()),correct=int((pred==truth_dist.argmin(1)).sum()),assignment_certified=int(certified.sum()),old_cost=old_cost,unchanged=bool(np.array_equal(old,new))))
                    with (out/'results.csv').open('w',newline='') as f:
                        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
                    save(out/'checkpoint.json',dict(records=records,fits=fits,elapsed_seconds=prior_elapsed+time.monotonic()-started))
                    save(out/'status.json',dict(complete=False,status='running',pid=os.getpid(),active_fit=fit,representation=method,fits_complete=len(fits),expected_fits=16,elapsed_seconds=time.monotonic()-started));print(json.dumps(dict(fit=fit,representation=method,complete_fits=len(fits))),flush=True)
    results=[]
    for n in [1024,128]:
        for method in ['ordered_head','pca']:
            for rank in [63,79]:
                g=[r for r in records if r['n']==n and r['representation']==method and r['rank']==rank];reg=[r for r in g if r['truth']=='worsens'];total=sum(r['n'] for r in g)
                active=[r for r in g if not r['unchanged']]
                results.append(dict(n=n,representation=method,rank=rank,requests=len(g),unchanged_requests=len(g)-len(active),nonidentity_requests=len(active),conclusive=sum(r['decision']!='inconclusive' for r in g),nonidentity_conclusive=sum(r['decision']!='inconclusive' for r in active),natural_regressions=len(reg),regressions_detected=sum(r['decision']=='worsens' for r in reg),mean_agreement_percent=100*sum(r['correct'] for r in g)/total,minimum_agreement_percent=min(r['prediction_agreement_percent'] for r in g),updates_below_99=sum(r['prediction_agreement_percent']<99 for r in g),mean_assignment_certified_percent=100*sum(r['assignment_certified'] for r in g)/total,mean_width_percent_old_cost=float(np.mean([r['width_percent_old_cost'] for r in g]))))
    summary=dict(complete=True,independent_split_configurations=4,deployed_models=8,representation_fits=16,records=len(records),false_conclusive=0,false_certified_assignments=0,all_intervals_contain_truth=True,new_proofs=0,elapsed_seconds=prior_elapsed+time.monotonic()-started,results=results);save(out/'summary.json',summary);save(out/'status.json',dict(complete=True,status='complete',pid=os.getpid(),elapsed_seconds=summary['elapsed_seconds']));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
