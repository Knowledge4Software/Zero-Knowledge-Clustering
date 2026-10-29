"""Frozen-reference release batches on sensor and neural text features.

Two row-disjoint fitted configurations per corpus, with six predeclared
new-training-minibatch proposals. Every outcome remains in the evidence.
"""
from pathlib import Path
import argparse,copy,json,time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
from campaigns.local_fast_tradeoff import save,sha
from commitjl.pilot_io import array_sha,sign
from commitjl.enriched_jl import enrich_centroid_basis
from commitjl.protected_jl import centroid_basis
from commitjl.reuse_certificates import integer_stats
from commitjl.relative_certificates import relative_certificate
from commitjl.group_change_bounds import exact_old_correction_bounds
from commitjl.group_moment_reference import prepare_groups
from commitjl.jl_certificate_pilots import hint_order,refinement_work
from experiments.rank79_independent_replication import trace,extend_basis
from experiments.pilot_jl_certificate_ideas import distances,prediction_labels


def minibatch_proposals(old,incoming):
    """Fix the providers' batches/rates without accessing any audit record."""
    result=[];metadata=[]
    for number,batch in enumerate(np.array_split(incoming,2)):
        labels=distances(batch,old).argmin(1)
        target=np.asarray([batch[labels==j].mean(0) if np.any(labels==j) else old[j] for j in range(len(old))])
        for rate in [.25,.5,1.]:
            result.append(np.rint((1-rate)*old+rate*target).astype(np.int64))
            metadata.append(dict(family='new_training_minibatch',batch=number,rate=rate))
    return result,metadata


def proof_parent(root,model,aux,rows,sets,meta,proposal_meta):
    unique=[model['centers']];seen={array_sha(np.asarray(model['centers'],np.int64))};selected=[];duplicates=[]
    for i,c in enumerate(sets):
        digest=array_sha(c)
        if digest in seen:duplicates.append(i);continue
        seen.add(digest);unique.append(c.tolist());selected.append(i)
    inp=dict(model=dict(model,n=128),x=rows[:128].tolist(),salt_x='202610271',salt_z='202610273',
        queries=[list(range(i,i+16)) for i in range(0,128,16)],reuse_centers=unique,comparison_base=model['centers'])
    save(root/'head_r63/input.json',inp)
    full=copy.deepcopy(inp);full['model'].update(method='full',basis=[],seed='');full['queries']=[list(range(i,i+8)) for i in range(0,128,8)]
    save(root/'full/input.json',full)
    save(root/'protocol.json',dict(**meta,n=128,native_n=1024,k=model['k'],fitting_seconds=0.,proof_runtime_workers=8,
        phase_wall_limit_seconds=12*3600,selected_source_sets=selected,duplicate_source_sets=duplicates,
        source_proposals=proposal_meta,selection='All six predeclared minibatch updates; only exact center duplicates cached before auditing.',
        batches=dict(full=8,compact=16),timing='Frozen fitted model supplied consistently. Charge native preparation, keys, all attempts, generation, replay, refinement and newly generated Full fallback.'))
    save(root/'auxiliary_basis.json',dict(source_input_sha256=sha(root/'head_r63/input.json'),training_sha256=meta['training_sha256'],
        audit_sha256=meta['audit_sha256'],basis_scale=16384,certificate_basis=aux.tolist(),rank=79,fitting_seconds=0.,
        excluded_source_fitting_seconds=meta['prediction_fitting_seconds']+meta['auxiliary_fitting_seconds'],
        recipe='Frozen prediction63 prefix; first16 training-residual SVD directions; identical precision14 profile.'))


def run(root):
    plan=json.loads((root/'plan.json').read_text());records=[];ablations=[];fits=[];started=time.time()
    out=root/'learning_native';out.mkdir(exist_ok=False)
    with threadpool_limits(limits=1):
        for dataset,k in [('har561',6),('text_minilm384',5)]:
            source_path=root/'data'/(dataset+'.npz');source=np.load(source_path)
            for split,seed in enumerate(plan['seeds']):
                raw_training=source['training'][split*1000:(split+1)*1000].astype(float)
                raw_audit=source['audit'][split*1024:(split+1)*1024].astype(float)
                scale=4095/np.abs(raw_training[:500]).max()
                training=np.clip(np.rint(raw_training*scale),-4095,4095).astype(np.int64)
                rows=np.clip(np.rint(raw_audit*scale),-4095,4095).astype(np.int64)
                past,incoming=training[:500],training[500:];snapshots=trace(past,k,seed);old=snapshots[8]
                proposals,proposal_meta=minibatch_proposals(old,incoming)
                diagnostic=[snapshots[s] for s in [9,12,16,24]]
                all_proposals=proposals+diagnostic
                all_meta=proposal_meta+[dict(family='lloyd_continuation_diagnostic',step=s) for s in [9,12,16,24]]
                fit=f'{dataset}_k{k}_seed{seed}_release'
                for method in ['ordered_head','pca']:
                    tick=time.time()
                    basis=(enrich_centroid_basis(past.astype(float),old,extra=63-centroid_basis(old).shape[1],random_state=seed,preserve_pca_order=True)[0].T
                        if method=='ordered_head' else PCA(63,svd_solver='randomized',random_state=seed).fit(past.astype(float)).components_)
                    basis=np.rint(basis*16384).astype(np.int64);fit_seconds=time.time()-tick
                    aux,extra=extend_basis(basis,past,old)
                    model=dict(version='bound-audit-v1',method='learned_energy',n=1024,d=rows.shape[1],k=k,input_bits=12,
                        basis_scale=16384,input_scale=str(scale),centers=old.tolist(),basis=basis.tolist(),tail_width=0,tail_rows=[],tail_signs=[],seed=str(seed))
                    archive=out/'fits'/f'{fit}_{method}.npz';archive.parent.mkdir(exist_ok=True)
                    np.savez_compressed(archive,basis=basis,auxiliary_basis=aux,centers=np.asarray([old]+proposals),audit=rows,training=past)
                    meta=dict(dataset=dataset,fit_seed=seed,reference_age=8,source_fit=fit,representation=method,source_archive_sha256=sha(source_path),
                        training_sha256=array_sha(past),incoming_training_sha256=array_sha(incoming),audit_sha256=array_sha(rows),
                        prediction_fitting_seconds=fit_seconds,auxiliary_fitting_seconds=extra,input_scale=str(scale))
                    fits.append(dict(fit=fit,archive_sha256=sha(archive),**meta))
                    if seed==plan['seeds'][0]:proof_parent(root/'learning_inputs'/fit/method,model,aux,rows,proposals,meta,proposal_meta)
                    for n in plan['native_catalog_sizes']:
                        x=rows[:n];m=dict(model,n=n);od=distances(x,old);groups=od.argmin(1);prepared=prepare_groups(x,groups.tolist(),k)
                        old_cost=int(od.min(1).sum());h,_=integer_stats(m,x)
                        for i,new in enumerate(all_proposals):
                            nd=distances(x,new);truth=int(nd.min(1).sum())-old_cost;prediction=prediction_labels(m,h,new)
                            assignment=relative_certificate(m,x,new);flags=np.asarray(assignment['certified_flags'],bool)
                            assert np.array_equal(prediction,np.asarray(assignment['labels'])) and not np.any(flags&(prediction!=nd.argmin(1)))
                            ranks=plan['rank_ablation'] if seed==plan['seeds'][0] and n==128 and i<6 else [79]
                            final=None
                            for rank in ranks:
                                b=exact_old_correction_bounds(dict(m,basis=aux[:rank].tolist()),x,prepared,old,new);den=b['cost_denominator']
                                exact=[int(v)*den for v in nd.min(1)-nd[np.arange(n),groups]];order=hint_order(b,None,'width')
                                for cap in (plan['cap_ablation'] if len(ranks)>1 else [16]):
                                    refined=refinement_work(b,exact,order,cap)
                                    assert refined['lower']<=truth*den<=refined['upper'] and refined['decision'] in ['inconclusive',sign(truth,truth)]
                                    if rank==79 and cap==16:final=(b,refined,order,den)
                                    if len(ranks)>1:ablations.append(dict(fit=fit,representation=method,n=n,set=i,rank=rank,cap=cap,
                                        decision=refined['decision'],exact_rows=refined['exact_rows'],true_change=truth,width_percent=100*(refined['upper']-refined['lower'])/den/old_cost))
                            b,refined,order,den=final
                            records.append(dict(fit=fit,dataset=dataset,representation=method,n=n,set=i,proposal=all_meta[i],true_change=truth,
                                initial_decision=sign(b['change_lower_scaled'],b['change_upper_scaled']),decision=refined['decision'],exact_rows=refined['exact_rows'],
                                agreement_percent=100*float(np.mean(prediction==nd.argmin(1))),assignment_certified_percent=100*float(flags.mean()),
                                width_percent_old_cost=100*(refined['upper']-refined['lower'])/den/old_cost,
                                strict_switch_rows=int((nd.min(1)<nd[np.arange(n),groups]).sum()),lower=str(refined['lower']),upper=str(refined['upper']),denominator=str(den),
                                full_activity_or_topic_ari=float(adjusted_rand_score(source['audit_y'][split*1024:split*1024+n],nd.argmin(1)))))
                    save(out/'checkpoint.json',dict(records=records,ablations=ablations,fits=fits))
                    save(out/'status.json',dict(complete=False,fit=fit,representation=method,completed_fits=len(fits),elapsed_seconds=time.time()-started))
                    print(json.dumps(dict(fit=fit,representation=method,completed_fits=len(fits))),flush=True)
    save(out/'records.json',records);save(out/'ablations.json',ablations);save(out/'fit_records.json',fits)
    save(out/'summary.json',dict(complete=True,native_only=True,new_proofs=0,representation_fits=len(fits),fitted_configurations=4,
        records=len(records),all_intervals_contain_truth=True,false_conclusive=0,elapsed_seconds=time.time()-started,plan_sha256=sha(root/'plan.json')))
    save(out/'status.json',dict(complete=True,status='complete'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root.resolve())
