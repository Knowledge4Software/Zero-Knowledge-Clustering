"""PCA controls at the same total dimension as the full-rank m=16 hybrid."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
from experiments.protected_jl_study import Archive,DATASETS
from commitjl.protected_jl import centroid_basis
from commitjl.clustering import nearest_centroid_assign

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if args.out.exists():ap.error('Output exists')
    args.out.mkdir(parents=True);archive=Archive();rows=[]
    with threadpool_limits(limits=4):
        for name in DATASETS:
            prefix=f'results/paper_server_20261006/rq1/{name}/artifacts/'
            x=archive.load(prefix+'dataset.npz')['X'];components=archive.load(prefix+'pca.npz')['components']
            for k in [5,10]:
                for start in range(5):
                    full=archive.load(prefix+f'full_k{k}_start{start}.npz');c=full['centroids']
                    m=16+centroid_basis(c).shape[1];p=components[:m]
                    labels=nearest_centroid_assign(x@p.T,c@p.T)
                    rows.append(dict(dataset=name,k=k,start=start,total_m=m,
                        ari=float(adjusted_rand_score(full['labels'],labels)),agreement=float(np.mean(full['labels']==labels))))
                    np.savez_compressed(args.out/f'{name}_k{k}_start{start}.npz',labels=labels)
            print(name,flush=True)
    pd.DataFrame(rows).to_csv(args.out/'rows.csv',index=False)
    (args.out/'manifest.json').write_text(json.dumps(dict(complete=True,rows=len(rows),
        scope='Same total output dimension PCA controls, exact full paper subsets; five saved centers, k=5/10; no retraining/timing.'),indent=2))

if __name__=='__main__':main()
