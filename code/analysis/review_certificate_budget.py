"""Predeclared native certificate budget/precision controls; no proving claims."""
from pathlib import Path
import argparse,copy,json,sys,time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from data.loaders import load_dataset
from commitjl.enriched_jl import enrich_centroid_basis
from commitjl.protected_jl import centroid_basis
from commitjl.reuse_certificates import summarize
from experiments.bound_audit import save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--owner-inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    ranks=[15,31,63,127];scales=[4096,16384]
    save(a.out/'protocol.json',dict(ranks=ranks,basis_scales=scales,native_catalog=1024,
        proof_pilot=dict(n=256,rank=63,basis_scale=16384,batch=16,methods=['learned_energy','pca_energy']),
        selection='Rank 63/T=16384 pilot specified before native budget results; all ranks/scales retained',
        fit_policy='Training split only; deployed centers retained; no future-center fitting'))
    owner={m:json.loads((a.owner_inputs/f'mnist_n1024_{m}.json').read_text()) for m in ['learned_energy','pca_energy']}
    x=np.asarray(load_dataset('mnist',n_max=5000,seed=0)['X'],float);x=x[np.linalg.norm(x,axis=1)>0]
    order=np.random.default_rng(10000).permutation(len(x));train=order[:int(.6*len(x))];audit=order[int(.6*len(x)):]
    scale=float(owner['learned_energy']['model']['input_scale']);xi=np.rint(x*scale).astype(np.int64)
    if not np.array_equal(xi[audit[:1024]],owner['learned_energy']['x']):raise AssertionError('owner catalog mismatch')
    ci=np.asarray(owner['learned_energy']['model']['centers'],float);rows=[];fits=[]
    with threadpool_limits(limits=4):
        for rank in ranks:
            for method in owner:
                start=time.perf_counter()
                if rank==15:
                    basis=np.asarray(owner[method]['model']['basis'],float)/4096
                    vectors_source='preserved original rounded basis; rescaled for precision control'
                elif method=='learned_energy':
                    basis=enrich_centroid_basis(xi[train].astype(float),ci,extra=rank-centroid_basis(ci).shape[1],random_state=0)[0].T
                    vectors_source='training residual PCA plus deployed-center span'
                else:
                    basis=PCA(rank,svd_solver='randomized',random_state=0).fit(xi[train].astype(float)).components_
                    vectors_source='training PCA'
                fits.append(dict(rank=rank,method=method,fit_seconds=time.perf_counter()-start,basis_source=vectors_source))
                for t in scales:
                    inp=copy.deepcopy(owner[method]);inp['model']['basis_scale']=t;inp['model']['basis']=np.rint(t*basis).astype(np.int64).tolist()
                    for s,centers in enumerate(inp['reuse_centers']):
                        row=summarize(inp['model'],inp['x'],centers);row.update(rank=rank,basis_scale=t,method=method,set=s,mode='native_budget_control');rows.append(row)
                    if rank==63 and t==16384:
                        inp['model']['n']=256;inp['x']=inp['x'][:256];inp['queries']=[list(range(i,i+16)) for i in range(0,256,16)]
                        save(a.out/'owner_inputs'/f'mnist_n256_{method}_r63_t16384.json',inp)
    frame=pd.DataFrame(rows);frame.to_csv(a.out/'certificate_budget_cells.csv',index=False)
    summary=frame[frame['set']>0].groupby(['rank','basis_scale','method'],as_index=False).agg(
        agreement=('agreement_quantized','mean'),certified_fraction=('certified_fraction','mean'),
        worst_certified_fraction=('certified_fraction','min'),interval_relative_width=('interval_relative_width','mean'))
    summary.to_csv(a.out/'certificate_budget_summary.csv',index=False)
    save(a.out/'fit_records.json',fits)
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':main()
