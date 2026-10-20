"""Supplementary native control using BLAS affine scores for every method.

Run only after timed proof jobs finish. Reuses the exact campaign maps/bases.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiments.laptop_comparison import archived, atomic_json
from commitjl.protected_jl import protected_project
from commitjl.sparse_gen import sparse_matrix_gen_csr
from tools.rebuild_laptop_comparison import rebuild


def assign_cost(x,c):
    scores=2*(x@c.T)-np.sum(c*c,axis=1)
    labels=np.argmax(scores,axis=1)
    cost=float(np.sum(x*x)-np.sum(scores[np.arange(len(x)),labels]))
    return labels,cost


def run(root):
    root=Path(root).resolve()
    output=root/'native_affine';output.mkdir(exist_ok=True)
    checked=[];records=[]
    for folder in sorted((root/'jobs').glob('geometry_*')):
        dataset=folder.name.removeprefix('geometry_')
        archive,prefix,data,pca=archived(dataset);x=data['X']
        assert hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()==json.loads(
            next((folder/'rows').glob('*.json')).read_text())['dataset_sha256']
        paths=[]
        for path in sorted((folder/'rows').glob('*.json')):
            row=json.loads(path.read_text())
            if row['k']!=5:continue
            if row['method'] in ['full','protected_only']:paths.append(path)
            elif row['total_m'] in [64,128,256] and (row['method']=='pca' or
                    (row['method'] in ['sparse','hybrid_r4'] and row['seed']==100)):
                paths.append(path)
        # One full-rank hybrid is independently replayed on every dataset.
        replay_path=next(p for p in paths if json.loads(p.read_text())['method']=='hybrid_r4')
        rebuilt=rebuild(replay_path)
        checked.append(dict(dataset=dataset,record=str(replay_path.relative_to(root)),
            shape=list(rebuilt['Z'].shape),replay_ok=True))
        del rebuilt
        for path in paths:
            row=json.loads(path.read_text());destination=output/'rows'/dataset/(path.stem+'.json')
            if destination.exists():records.append(json.loads(destination.read_text()));continue
            with np.load(folder/f'centers_k5_start{row["start"]}.npz') as saved:
                c=saved['centroids'];u=saved['basis'][:,:row['rank']]
            if row['method']=='full':
                projector=lambda:(x,c)
            elif row['method']=='pca':
                matrix=pca[:row['total_m']]
                projector=lambda:(x@matrix.T,c@matrix.T)
            else:
                a=None
                if row['residual_m']:
                    _,a=sparse_matrix_gen_csr(bytes.fromhex(row['map_seed']),row['residual_m'],row['d'],row['q'])
                    a=a.astype(float);a.data/=np.sqrt(row['q'])
                projector=lambda:(protected_project(x,u,a,c[0]),protected_project(c,u,a,c[0]))
            z,cc=projector();assign_cost(z,cc) # full-sized warm-up
            samples=[];project_samples=[];assign_samples=[]
            for _ in range(5):
                begin=time.perf_counter();z,cc=projector();middle=time.perf_counter()
                labels,cost=assign_cost(z,cc);end=time.perf_counter()
                samples.append(end-begin);project_samples.append(middle-begin);assign_samples.append(end-middle)
            with np.load(folder/'arrays'/(path.stem+'.npz')) as saved:
                agreement=float(np.mean(labels==saved['labels']))
                expected_cost=float(saved['projected_centroid_sq'].min(axis=1).sum())
            np.testing.assert_allclose(cost,expected_cost,rtol=1e-9,atol=1e-6)
            item=dict(dataset=dataset,n=len(x),d=x.shape[1],k=5,start=row['start'],method=row['method'],
                total_m=row['total_m'],q=row['q'],record_path=str(path.relative_to(root)),
                seconds_samples=samples,seconds_median=statistics.median(samples),
                projection_seconds_median=statistics.median(project_samples),
                assignment_cost_seconds_median=statistics.median(assign_samples),
                saved_label_agreement=agreement,cost_check_ok=True,
                qualified_geometry=row['qualified_geometry'],ari=row['ari'],pair_q95=row['pair_q95'],
                projected_cost_relative_error=row['projected_cost_relative_error'],
                scope='k5, starts3/4; random map100 only; 5 warm BLAS timings; projection plus labels and objective',
                preparation_included=False)
            atomic_json(destination,item);records.append(item)
        print(f'affine {dataset}: {len(paths)} controls complete',flush=True)
    frame=pd.DataFrame(records)
    for (dataset,start),g in frame.groupby(['dataset','start']):
        full=float(g[g.method=='full'].seconds_median.iloc[0])
        frame.loc[g.index,'speedup_vs_full']=full/g.seconds_median
    frame.to_csv(output/'measurements.csv',index=False)
    atomic_json(output/'validation.json',dict(replays=checked,cost_checks=len(records),
        label_mismatch_rows=int((frame.saved_label_agreement!=1).sum()),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        threads=4,complete=True))
    summary=frame.groupby(['dataset','method','total_m','q'],dropna=False)[
        ['seconds_median','speedup_vs_full','saved_label_agreement']].agg(['mean','min','max'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(output/'summary.csv',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('results/laptop_comparison_20261011'))
    with threadpool_limits(limits=4):run(parser.parse_args().out)
