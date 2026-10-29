"""Native screening for plain reusable head/PCA and JL residual corrections.

Fixed integer models; every declared configuration/map/update retained.
This screen measures fidelity, not cryptographic speed or certified decisions.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from data.loaders import load_dataset
from experiments.review_update_decisions import lloyd_snapshots,update_trace,array_sha
from commitjl.enriched_jl import enrich_centroid_basis
from commitjl.protected_jl import centroid_basis
from commitjl.reuse_certificates import integer_stats
from commitjl.block_certificates import residual_vectors


def metric(scores,common,den,labels,truth):
    predicted=np.asarray(scores.argmin(1),dtype=int)
    selected=scores[np.arange(len(scores)),predicted]+common
    total=sum(int(v) for v in selected)
    return dict(agreement_percent=100*float(np.mean(predicted==labels)),
        absolute_cost_error_percent=100*abs(total/den/truth-1),negative_row_estimates=sum(int(v)<0 for v in selected))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--confirmation',action='store_true')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    seeds=[101,131] if args.confirmation else [11]
    protocol=dict(datasets=['mnist','fashion_mnist'],k=[3,5],seeds=seeds,ages=[1,8],n=1024,
        ranks=[7,15,31,63],jl_head_ranks=[7,15],jl_widths=[8,16],jl_seeds=[211,307,401],jl_weights=['1/4','1/2','1'],
        threshold='Working target minimum agreement99.5% per actual update, not a universal rule.',
        scope='Exploratory native screen; no proof speed, deterministic JL accuracy guarantee or exact-decision claim.',
        map='Dense fixed Rademacher signs on the integer residual; dot/m estimates the omitted residual cross term.',
        basis='Training-only rank63 fit, order-preserving QR for head; PCA has strongest directions first.',threads=2)
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    records=[];start=time.monotonic()
    with threadpool_limits(limits=2):
        for dataset in protocol['datasets']:
            source=load_dataset(dataset,n_max=5000,seed=0);x=np.asarray(source['X'],float)
            x=x[np.linalg.norm(x,axis=1)>0]
            for k in protocol['k']:
                for seed in seeds:
                    idx=np.random.default_rng(741+seed).permutation(len(x));cut=int(.6*len(x));tr=x[idx[:cut]]
                    scale=4095/float(np.abs(tr).max());training=np.rint(tr*scale).astype(np.int64)
                    audit=np.clip(np.rint(x[idx[cut:cut+1024]]*scale),-4096,4096).astype(np.int64)
                    snapshots=lloyd_snapshots(training,k,seed)
                    for age in protocol['ages']:
                        sets,_=update_trace(snapshots,age);old=np.asarray(sets[0],dtype=np.int64)
                        for method in ['learned','pca']:
                            if method=='learned':
                                fitted=enrich_centroid_basis(training.astype(float),old,extra=63-centroid_basis(old).shape[1],random_state=seed,preserve_pca_order=True)[0].T
                            else:
                                fitted=PCA(63,svd_solver='randomized',random_state=seed).fit(training.astype(float)).components_
                            max_basis=np.rint(fitted*16384).astype(np.int64)
                            owner=dict(dataset=dataset,k=k,seed=seed,age=age,method=method,
                                audit_sha256=array_sha(audit),training_sha256=array_sha(training))
                            stem=f'{dataset}_k{k}_seed{seed}_age{age}_{method}'
                            np.savez_compressed(args.out/(stem+'.npz'),basis=max_basis,centers=np.asarray(sets),audit=audit)
                            for rank in protocol['ranks']:
                                basis=max_basis[:rank];model=dict(basis=basis.tolist(),basis_scale=16384,centers=old.tolist())
                                h,e=integer_stats(model,audit);h=h.astype(object);norm=np.asarray([sum(int(v)**2 for v in row) for row in h],dtype=object)
                                common=norm+e;residual=residual_vectors(model,audit) if rank in protocol['jl_head_ranks'] else None
                                maps={}
                                if residual is not None:
                                    for width in protocol['jl_widths']:
                                        for map_seed in protocol['jl_seeds']:
                                            a=np.random.default_rng(map_seed).choice(np.array([-1,1],dtype=np.int64),size=(width,audit.shape[1]))
                                            if int(np.abs(residual).max())*audit.shape[1]>=2**63:
                                                sketch=residual.astype(object)@a.T.astype(object)
                                            else:sketch=residual@a.T
                                            maps[width,map_seed]=(a,sketch.astype(object))
                                for s in range(1,5):
                                    centers=np.asarray(sets[s],dtype=np.int64)
                                    dist=((audit[:,None,:]-centers[None,:,:])**2).sum(2);labels=dist.argmin(1);truth=int(dist.min(1).sum())
                                    hc,ec=integer_stats(model,centers);hc=hc.astype(object)
                                    offsets=np.asarray([sum(int(v)**2 for v in row) for row in hc],dtype=object)+ec
                                    scores=offsets[None,:]-2*(h@hc.T)
                                    base=dict(**owner,rank=rank,set=s,actual_cost=truth,stored_scalars=rank+1)
                                    records.append(dict(**base,variant='plain',jl_width=0,jl_seed=0,jl_weight='0',**metric(scores,common,16384**4,labels,truth)))
                                    if residual is not None:
                                        rc=residual_vectors(model,centers)
                                        for (width,map_seed),(a,sketch) in maps.items():
                                            center_sketch=rc.astype(object)@a.T.astype(object)
                                            cross=sketch@center_sketch.T
                                            for numerator in [1,2,4]:
                                                corrected=4*width*scores-2*numerator*cross
                                                row=dict(**base);row['stored_scalars']=rank+1+width
                                                records.append(dict(**row,variant='jl_correction',jl_width=width,jl_seed=map_seed,jl_weight=f'{numerator}/4',
                                                    **metric(corrected,4*width*common,4*width*16384**4,labels,truth)))
                            pd.DataFrame(records).to_csv(args.out/'results.csv',index=False)
                            print(json.dumps(dict(completed=stem,rows=len(records),elapsed=time.monotonic()-start)),flush=True)
    data=pd.DataFrame(records)
    group=['method','rank','variant','jl_width','jl_weight']
    summary=data.groupby(group).agg(cells=('set','size'),mean_agreement=('agreement_percent','mean'),worst_agreement=('agreement_percent','min'),
        mean_absolute_cost_error=('absolute_cost_error_percent','mean'),worst_absolute_cost_error=('absolute_cost_error_percent','max'),
        negative_rows=('negative_row_estimates','sum'),stored_scalars=('stored_scalars','first')).reset_index()
    summary['passes_working_accuracy']=summary.worst_agreement>=99.5
    summary.to_csv(args.out/'candidates.csv',index=False)
    (args.out/'summary.json').write_text(json.dumps(dict(complete=True,elapsed_seconds=time.monotonic()-start,
        cells=len(data),proof_timings_measured=False,target_percent=99.5,selection_scope=protocol['scope']),indent=2))
    print(summary.sort_values(['passes_working_accuracy','stored_scalars','worst_agreement'],ascending=[False,True,False]).head(15).to_string(index=False),flush=True)


if __name__=='__main__':main()
