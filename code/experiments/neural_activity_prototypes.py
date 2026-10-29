"""Post-review diagnostic: a frozen neural activity encoder and prototype head.

This workload is added after observing the text stress result. It is not an
independent confirmation chosen before that observation. Both seeds and all
six incoming-class-mean updates are retained.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.decomposition import PCA
from campaigns.local_fast_tradeoff import save,sha
from commitjl.pilot_io import array_sha,sign
from commitjl.enriched_jl import enrich_centroid_basis
from commitjl.protected_jl import centroid_basis
from commitjl.reuse_certificates import integer_stats
from commitjl.relative_certificates import relative_certificate
from commitjl.group_change_bounds import exact_old_correction_bounds
from commitjl.group_moment_reference import prepare_groups
from commitjl.jl_certificate_pilots import hint_order,refinement_work
from experiments.rank79_independent_replication import extend_basis
from experiments.pilot_jl_certificate_ideas import distances,prediction_labels
from experiments.referee_learning_workflows import proof_parent


def class_updates(old,incoming,labels):
    proposals=[];metadata=[]
    for batch,(x,y) in enumerate(zip(np.array_split(incoming,2),np.array_split(labels,2))):
        target=np.asarray([x[y==j].mean(0) if np.any(y==j) else old[j] for j in range(len(old))])
        for rate in [.25,.5,1.]:
            proposals.append(np.rint((1-rate)*old+rate*target).astype(np.int64))
            metadata.append(dict(family='labeled_training_minibatch',batch=batch,rate=rate))
    return proposals,metadata


def run(root):
    import torch
    from torch import nn
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    source=np.load(root/'data/har561.npz');out=root/'neural_activity';out.mkdir(exist_ok=False)
    amendment=json.loads((root/'neural_activity_amendment.json').read_text());records=[];fits=[];started=time.time()
    with threadpool_limits(limits=1):
        for split,seed in enumerate(amendment['seeds']):
            torch.manual_seed(seed)
            x=np.asarray(source['training'][split*1000:(split+1)*1000],np.float32);y=source['training_y'][split*1000:(split+1)*1000].astype(np.int64)-1
            test=np.asarray(source['audit'][split*1024:(split+1)*1024],np.float32);test_y=source['audit_y'][split*1024:(split+1)*1024].astype(np.int64)-1
            encoder=nn.Sequential(nn.Linear(561,256),nn.ReLU());classifier=nn.Linear(256,6)
            parameters=list(encoder.parameters())+list(classifier.parameters());optimizer=torch.optim.Adam(parameters,lr=.001,weight_decay=.0001)
            generator=torch.Generator().manual_seed(seed)
            for epoch in range(80):
                for ids in torch.randperm(500,generator=generator).split(64):
                    loss=nn.functional.cross_entropy(classifier(encoder(torch.from_numpy(x[:500])[ids])),torch.from_numpy(y[:500])[ids])
                    optimizer.zero_grad();loss.backward();optimizer.step()
            encoder.eval();classifier.eval()
            with torch.inference_mode():
                train_features=encoder(torch.from_numpy(x)).numpy();audit_features=encoder(torch.from_numpy(test)).numpy()
                neural_accuracy=float((classifier(torch.from_numpy(audit_features)).argmax(1).numpy()==test_y).mean())
            train_features/=np.maximum(np.linalg.norm(train_features,axis=1,keepdims=True),1e-12)
            audit_features/=np.maximum(np.linalg.norm(audit_features,axis=1,keepdims=True),1e-12)
            scale=4095/np.abs(train_features[:500]).max();training=np.clip(np.rint(train_features*scale),-4095,4095).astype(np.int64)
            audit=np.clip(np.rint(audit_features*scale),-4095,4095).astype(np.int64);past=training[:500]
            old=np.asarray([np.rint(past[y[:500]==j].mean(0)) for j in range(6)],np.int64)
            proposals,proposal_meta=class_updates(old,training[500:],y[500:]);model_file=out/f'encoder_seed{seed}.npz'
            np.savez_compressed(model_file,**{f'encoder_{k}':v.detach().numpy() for k,v in encoder.state_dict().items()},
                **{f'classifier_{k}':v.detach().numpy() for k,v in classifier.state_dict().items()})
            for method in ['ordered_head','pca']:
                tick=time.time()
                basis=(enrich_centroid_basis(past.astype(float),old,extra=63-centroid_basis(old).shape[1],random_state=seed,preserve_pca_order=True)[0].T
                    if method=='ordered_head' else PCA(63,svd_solver='randomized',random_state=seed).fit(past.astype(float)).components_)
                basis=np.rint(basis*16384).astype(np.int64);seconds=time.time()-tick;aux,extra=extend_basis(basis,past,old)
                m=dict(version='bound-audit-v1',method='learned_energy',n=1024,d=256,k=6,input_bits=12,basis_scale=16384,input_scale=str(scale),
                    centers=old.tolist(),basis=basis.tolist(),tail_width=0,tail_rows=[],tail_signs=[],seed=str(seed))
                fit=f'har_neural256_k6_seed{seed}_classmean';meta=dict(dataset='har_neural256',fit_seed=seed,representation=method,source_fit=fit,
                    training_sha256=array_sha(past),audit_sha256=array_sha(audit),prediction_fitting_seconds=seconds,auxiliary_fitting_seconds=extra,
                    encoder_sha256=sha(model_file),neural_classifier_accuracy=neural_accuracy,input_scale=str(scale))
                fits.append(meta)
                if seed==amendment['seeds'][0]:proof_parent(root/'neural_activity_inputs'/fit/method,m,aux,audit,proposals,meta,proposal_meta)
                for n in [128,1024]:
                    xq=audit[:n];od=distances(xq,old);groups=od.argmin(1);prepared=prepare_groups(xq,groups.tolist(),6);old_cost=int(od.min(1).sum())
                    mm=dict(m,n=n);heads,_=integer_stats(mm,xq)
                    for i,new in enumerate(proposals):
                        nd=distances(xq,new);labels=nd.argmin(1);truth=int(nd.min(1).sum())-old_cost
                        prediction=prediction_labels(mm,heads,new);assignment=relative_certificate(mm,xq,new);flags=np.asarray(assignment['certified_flags'],bool)
                        assert np.array_equal(prediction,assignment['labels']) and not np.any(flags&(prediction!=labels))
                        bounds=exact_old_correction_bounds(dict(mm,basis=aux.tolist()),xq,prepared,old,new);den=bounds['cost_denominator']
                        exact=[int(v)*den for v in nd.min(1)-nd[np.arange(n),groups]];order=hint_order(bounds,None,'width')
                        refined=refinement_work(bounds,exact,order,16);assert refined['lower']<=truth*den<=refined['upper']
                        assert refined['decision'] in ['inconclusive',sign(truth,truth)]
                        records.append(dict(fit=fit,dataset='har_neural256',representation=method,n=n,set=i,proposal=proposal_meta[i],true_change=truth,
                            decision=refined['decision'],exact_rows=refined['exact_rows'],agreement_percent=100*float((prediction==labels).mean()),
                            assignment_certified_percent=100*float(flags.mean()),prototype_classification_accuracy=100*float((labels==test_y[:n]).mean()),
                            predicted_prototype_classification_accuracy=100*float((prediction==test_y[:n]).mean()),
                            lower=str(refined['lower']),upper=str(refined['upper']),denominator=str(den),
                            strict_switch_rows=int((nd.min(1)<nd[np.arange(n),groups]).sum()),width_percent_old_cost=100*(refined['upper']-refined['lower'])/den/old_cost))
                save(out/'checkpoint.json',dict(records=records,fits=fits));print(json.dumps(dict(fit=fit,representation=method)),flush=True)
    save(out/'records.json',records);save(out/'fit_records.json',fits)
    save(out/'summary.json',dict(complete=True,native_only=True,representation_fits=len(fits),records=len(records),all_intervals_contain_truth=True,
        elapsed_seconds=time.time()-started,amendment_sha256=sha(root/'neural_activity_amendment.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root.resolve())
