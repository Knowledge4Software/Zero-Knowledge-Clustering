"""Paired validation-only follow-up using the saved text-ablation models."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from commitjl.experimental_norm_correction import residual_norm_corrected_transform
from commitjl.reusable_native import NativeProjection,stable_squared_distances,file_hash
from experiments.comprehensive_native_comparison import load_dataset
from experiments.text_residual_ablation import cluster
from experiments.laptop_comparison import atomic_json,save_arrays
from experiments.paper_compression import bootstrap_crossed


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    args=parser.parse_args();root=args.root
    x,y,_=load_dataset('20ng_tfidf_nonzero')
    with np.load(root/'indices.npz') as ids:validation=x[ids['validation']]
    rng=np.random.default_rng(777);pairs=rng.integers(len(validation),size=(2000,2));pairs=pairs[pairs[:,0]!=pairs[:,1]]
    ref=np.sum((validation[pairs[:,0]]-validation[pairs[:,1]])**2,axis=1);keep=ref>0
    rows=[]
    with threadpool_limits(limits=4):
        for path in sorted((root/'artifacts').glob('s*_protected_m*_seed*_extra12_q*_w1.0_init1.npz')):
            metadata=json.loads((root/'rows'/(path.stem+'.json')).read_text())
            with np.load(path) as a:
                sparse=csr_matrix((a['model_a_data'],a['model_a_indices'],a['model_a_indptr']),shape=tuple(a['model_a_shape']))
                model=NativeProjection('hybrid',x.shape[1],basis=a['model_basis'],sparse_map=sparse,origin=a['model_origin'])
            z=residual_norm_corrected_transform(model,validation)
            c=np.load(root/'models'/f"centers_s{metadata['start']}.npz")['centers']
            cc=residual_norm_corrected_transform(model,c)
            error=float(np.max(np.abs(stable_squared_distances(z,cc)-stable_squared_distances(validation,c))))
            assert error<1e-10
            p=np.sum((z[pairs[:,0]]-z[pairs[:,1]])**2,axis=1)
            q95=float(np.quantile(np.abs(p[keep]/ref[keep]-1),.95))
            for budget in [1,5]:
                name=path.stem.replace('_init1',f'_init{budget}')
                with np.load(root/'artifacts'/(name+'.npz')) as a:initials=a['initial_indices']
                fit,capped=cluster(z,initials)
                with np.load(root/'full'/f"s{metadata['start']}_init{budget}.npz") as a:original=a['labels']
                row=dict(method='protected_norm_corrected',m=metadata['m'],start=metadata['start'],seed=metadata['seed'],
                    q=metadata['q'],extra=12,init_budget=budget,validation_ari=float(adjusted_rand_score(original,fit.labels_)),
                    pair_q95=q95,maximum_original_center_cost_error=error,any_iteration_cap=capped)
                rows.append(row)
                save_arrays(root/'norm_corrected'/f'{name}.npz',labels=fit.labels_,centers=fit.cluster_centers_)
    frame=pd.DataFrame(rows);frame.to_csv(root/'norm_corrected_rows.csv',index=False);summaries=[]
    for values,g in frame.groupby(['m','q','init_budget']):
        r=dict(zip(['m','q','init_budget'],values),cases=len(g))
        for metric in ['validation_ari','pair_q95']:
            mean,lo,hi=bootstrap_crossed(g.pivot(index='seed',columns='start',values=metric).to_numpy(),draws=1000)
            r.update({metric:mean,metric+'_ci_low':lo,metric+'_ci_high':hi})
        summaries.append(r)
    summary=pd.DataFrame(summaries);summary.to_csv(root/'norm_corrected_summary.csv',index=False)
    atomic_json(root/'norm_corrected_manifest.json',dict(complete=True,cases=len(rows),test_evaluations=0,
        module_sha256=file_hash(Path(__file__).resolve().parents[1]/'commitjl/experimental_norm_correction.py'),
        scope='Experimental nonlinear postprocessing; includes original residual norm; no linear JL or ZK claim',
        maximum_original_center_cost_error=max(r['maximum_original_center_cost_error'] for r in rows)))
    print(summary.to_string(index=False))


if __name__=='__main__':main()
