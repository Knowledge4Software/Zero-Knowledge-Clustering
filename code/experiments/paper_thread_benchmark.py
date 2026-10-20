"""Short 4/8-thread planning benchmark; not a paper measurement campaign."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits,threadpool_info

from experiments.paper_compression import create_model,centroid_basis,projection_seed,split_indices,assignment
from experiments.laptop_comparison import atomic_json
from experiments.protected_jl_study import Archive
from commitjl.reusable_native import array_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    archive=Archive();records=[];began=time.monotonic()
    pilot=Path(__file__).resolve().parents[1]/'results/paper_compression_audit_pilot_v2/jobs'
    for position,dataset in enumerate(['mnist','gisette','20ng_tfidf']):
        x=archive.load(f'results/paper_server_20261006/rq1/{dataset}/artifacts/dataset.npz')['X']
        folder=pilot/(dataset+'_primary')
        with np.load(folder/'fits/k5_s0.npz') as p:c=p['centers']
        with np.load(folder/'pca.npz') as p:components,mean=p['components'],p['mean']
        with threadpool_limits(limits=4):
            u=centroid_basis(c);seed=projection_seed(array_hash(x),c,'thread-benchmark',0,128-u.shape[1],1)
            models={name:create_model(name,x.shape[1],128,1,seed,c,u,components,mean)
                for name in ['full','pca','protected_jl']}
            states={name:model.prepare(x) for name,model in models.items()}
        ids=np.random.default_rng(19).choice(len(x),min(len(x),512),replace=False)
        landmarks=np.random.default_rng(20).choice(len(x),64,replace=False)
        initial=np.random.default_rng(900).choice(len(x),5,replace=False)
        functions={}
        for name,model in models.items():
            state=states[name];pc=model.transform(c);pcn=np.einsum('ij,ij->i',pc,pc)
            if name!='full':functions[name+'_compression']=lambda model=model:model.transform(x)
            functions[name+'_distance_query']=lambda state=state:state.squared_distances(ids,landmarks)
            functions[name+'_assignment']=lambda state=state,pc=pc,pcn=pcn:assignment(state.z,pc,state.norms,pcn)
            functions[name+'_fresh_kmeans']=lambda state=state:KMeans(n_clusters=5,n_init=1,
                init=state.z[initial].copy(),max_iter=300,random_state=0,algorithm='lloyd').fit(state.z)
        threads=[4,8] if position%2==0 else [8,4]
        # Alternate thread order for every measured repetition.
        for task,fn in functions.items():
            values={n:[] for n in threads}
            for n in threads:
                with threadpool_limits(limits=n):value=fn()
                del value
            for repetition in range(3):
                for n in threads if repetition%2==0 else threads[::-1]:
                    with threadpool_limits(limits=n):
                        start=time.perf_counter();value=fn();values[n].append(time.perf_counter()-start)
                    del value
            for n in threads:records.append(dict(dataset=dataset,task=task,threads=n,samples=values[n],median=float(np.median(values[n]))))
        train=x[split_indices(len(x))['train']]
        for n in threads:
            with threadpool_limits(limits=n):
                start=time.perf_counter();pca=PCA(n_components=256,svd_solver='full').fit(train);elapsed=time.perf_counter()-start
            records.append(dict(dataset=dataset,task='shared_full_svd_fit',threads=n,samples=[elapsed],median=elapsed))
            del pca
        print(json.dumps(dict(dataset=dataset,benchmark_seconds=time.monotonic()-began)),flush=True)
        atomic_json(args.out,dict(complete=False,scope='Planning microbenchmark; short samples and no guaranteed full-grid speedup',records=records))
    ratios=[];detail=[]
    for dataset in ['mnist','gisette','20ng_tfidf']:
        tasks=sorted(set(r['task'] for r in records if r['dataset']==dataset))
        for task in tasks:
            cell={r['threads']:r['median'] for r in records if r['dataset']==dataset and r['task']==task}
            ratio=cell[4]/cell[8];detail.append(dict(dataset=dataset,task=task,speedup_8_vs_4=ratio))
            if task!='shared_full_svd_fit':ratios.append(ratio)
    score=float(np.exp(np.mean(np.log(ratios))))
    result=dict(complete=True,wall_seconds=time.monotonic()-began,records=records,comparisons=detail,
        native_geomean_speedup_8_vs_4=score,recommended_threads=8 if score>1.10 else 4,
        libraries=threadpool_info(),scope='Planning microbenchmark; equal task weighting, not a campaign runtime forecast. '
        'Three timings per native task; one SVD fit per setting; no final quality data. Retain 4 unless 8 exceeds 10% overall gain.')
    atomic_json(args.out,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','comparisons','libraries')}),flush=True)


if __name__=='__main__':main()
