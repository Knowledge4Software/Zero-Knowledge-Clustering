"""Read-only diagnosis of text geometry and saved Full/Protected JL partitions."""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from commitjl.reusable_native import NativeProjection, array_hash, file_hash, stable_squared_distances
from experiments.laptop_comparison import atomic_json
from experiments.protected_jl_study import Archive


def stats(values):
    return dict(zip(['min','q05','median','q95','max'],map(float,np.quantile(values,[0,.05,.5,.95,1]))))


def diagnose(root,out):
    dataset='20ng_tfidf_nonzero'
    archive=Archive();saved=archive.load('results/paper_server_20261006/rq1/20ng_tfidf/artifacts/dataset.npz')
    original=saved['X'];ids=np.flatnonzero(np.any(original!=0,axis=1));x=np.ascontiguousarray(original[ids],dtype=float)
    y=saved['y'][ids]
    metadata=json.loads((root/'datasets'/dataset/'dataset.json').read_text())
    assert array_hash(x)==metadata['used_sha256']
    with np.load(root/'datasets'/dataset/'indices.npz') as folds:
        train=x[folds['train']];test=x[folds['test']]
        pair=folds['pair_indices'];pairref=folds['pair_reference']
    center=train.mean(axis=0);variation=train-center
    energy=float(np.sum(variation**2))
    rows=[json.loads(p.read_text()) for p in (root/'rows').glob(dataset+'*.json')]
    full={}
    full_details=[]
    for row in rows:
        if row['method']!='full':continue
        key=f"{dataset}_s{row['start']}_full_m{x.shape[1]}_seed-1"
        path=root/'artifacts'/(key+'.npz');assert file_hash(path)==row['artifact_sha256']
        with np.load(path) as arrays:full[row['start']]=arrays['fresh_labels']
        counts=np.bincount(full[row['start']],minlength=5)
        centers=np.load(root/'datasets'/dataset/'models'/f"centers_s{row['start']}.npz")['centers']
        distances=np.sort(stable_squared_distances(test,centers),axis=1)
        relative=(distances[:,1]-distances[:,0])/np.maximum(distances[:,1],np.finfo(float).tiny)
        full_details.append(dict(start=row['start'],cluster_sizes=counts.tolist(),
            largest_cluster_fraction=float(counts.max()/len(x)),ground_truth_ari=row['ground_truth_fresh_ari'],
            fixed_center_relative_margin=stats(relative),margin_below_001=float(np.mean(relative<.01))))
    stability=[dict(start_a=a,start_b=b,ari=float(adjusted_rand_score(full[a],full[b]))) for a,b in combinations(sorted(full),2)]
    captured=[]
    for start in sorted(full):
        for name in ['basis','enriched']:
            u=np.load(root/'datasets'/dataset/'models'/f'{name}_s{start}.npz')['basis']
            captured.append(dict(start=start,kind=name,rank=u.shape[1],
                centered_training_energy_fraction=float(np.sum((variation@u)**2)/energy)))
    rng=np.random.default_rng(193)
    p=rng.integers(len(x),size=(10000,2));p=p[p[:,0]!=p[:,1]]
    dots=np.einsum('ij,ij->i',x[p[:,0]],x[p[:,1]])
    result=dict(n=len(x),d=x.shape[1],original_zero_rows_removed=len(original)-len(x),
        source_class_count=len(np.unique(y)),row_l2_norm=stats(np.linalg.norm(x,axis=1)),
        nonzero_features_per_document=stats(np.count_nonzero(x,axis=1)),
        sampled_cosine_similarity=stats(dots),sampled_squared_distances=stats(pairref),
        full_pairwise_start_stability=stability,full_pairwise_ari=stats([a['ari'] for a in stability]),
        full_runs=sorted(full_details,key=lambda a:a['start']),protected_training_energy=captured,
        source_sha256=metadata['used_sha256'],source_run=str(root.resolve()),
        scope='Existing archived nonzero TF-IDF corpus and saved results; no retraining, tuning, or deletion')
    atomic_json(out,result)
    print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    with threadpool_limits(limits=4):diagnose(args.root,args.out)


if __name__=='__main__':main()
