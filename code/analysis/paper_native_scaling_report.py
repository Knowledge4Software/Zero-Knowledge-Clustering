"""Replay controlled native scaling artifacts and export stage plots."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

from commitjl.reusable_native import NativeProjection,array_hash,file_hash
from experiments.laptop_comparison import atomic_json
from experiments.paper_compression import assignment,sqdist
from experiments.protected_jl_study import Archive


def report(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text());archive=Archive()
    rows=[json.loads(p.read_text()) for p in sorted((root/'rows').glob('*.json'))]
    if not rows:raise ValueError('No scaling rows to validate')
    checks=[]
    for dataset in sorted(set(r['dataset'] for r in rows)):
        original=archive.load(f'results/paper_server_20261006/rq1/{dataset}/artifacts/dataset.npz')['X']
        sha=array_hash(original)
        for row in [r for r in rows if r['dataset']==dataset]:
            assert sha==row['source_sha256']
            for relative,h in row['artifact_sha256'].items():
                path=(root/relative).resolve()
                if not path.is_relative_to(root.resolve()) or file_hash(path)!=h:
                    raise ValueError(f'Unbound/corrupt artifact: {relative}')
            scope=root/dataset/f"d{row['d']}"
            with np.load(scope/'indices.npz') as indices:
                features=indices['feature_indices'];order=indices['source_row_order']
            with np.load(scope/f"{row['method']}_{row['draw']}_model.npz") as p:
                if row['method']=='full':model=NativeProjection('full',row['d'])
                elif row['method'] in ('pca','pca_randomized','protected_only'):
                    model=NativeProjection('pca',row['d'],components=p['components'],origin=p['origin'])
                else:
                    a=csr_matrix((p['a_data'],p['a_indices'],p['a_indptr']),shape=tuple(p['a_shape']))
                    model=NativeProjection('hybrid',row['d'],basis=p['basis'],sparse_map=a,origin=p['origin'])
            assert model.fingerprint==row['model_sha256'] and model.m==row['m']
            with np.load(scope/f"fit{row['draw']}.npz") as fitted:c=fitted['centers']
            with np.load(scope/f"{row['method']}_{row['draw']}_n{row['n']}.npz") as saved:
                np.testing.assert_array_equal(saved['source_indices'],np.sort(order[:row['n']]))
                x=original[saved['source_indices']][:,features];z=model.transform(x);a,b=saved['left_indices'],saved['right_indices']
                np.testing.assert_allclose(sqdist(x[a],x[b]),saved['original_query_sq'],rtol=1e-10,atol=1e-8)
                np.testing.assert_allclose(sqdist(z[a],z[b]),saved['projected_query_sq'],rtol=1e-10,atol=1e-8)
                pc=model.transform(c)
                old,_=assignment(x,c,np.einsum('ij,ij->i',x,x),np.einsum('ij,ij->i',c,c))
                new,_=assignment(z,pc,np.einsum('ij,ij->i',z,z),np.einsum('ij,ij->i',pc,pc))
                np.testing.assert_array_equal(old,saved['original_labels']);np.testing.assert_array_equal(new,saved['projected_labels'])
                assert np.mean(old==new)==row['agreement']
            checks.append(dict(dataset=dataset,n=row['n'],d=row['d'],method=row['method'],draw=row['draw'],passed=True))
    output=root/'report';output.mkdir(exist_ok=True)
    atomic_json(output/'replay_validation.json',dict(complete=manifest['complete'],checked_cases=len(checks),all_passed=True,
        scope=manifest['scope'],paper_grid=manifest['binding']['configuration']['profile']=='paper',checks=checks))
    frame=pd.read_csv(root/'summary.csv')
    for axis,fixed in [('n','d'),('d','n')]:
        fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained')
        for column,dataset in enumerate(sorted(frame.dataset.unique())):
            group=frame[frame.dataset==dataset];group=group[group[fixed]==group[fixed].max()]
            for i,stage in enumerate(['compression_seconds','distance_query_seconds']):
                ax=axs[i,column]
                for method,cells in group.groupby('method'):
                    med=cells.groupby(axis)[stage].median();ax.plot(med.index,med.values,marker='o',label=method)
                ax.set(title=f'{dataset}: {fixed}={int(group[fixed].max())}',xlabel=axis,ylabel=stage.replace('_',' '))
                ax.grid(alpha=.2)
        axs[0,0].legend(fontsize=7)
        fig.suptitle('Conditional model reuse; '+manifest['binding']['configuration']['profile']+' profile (not ZK timing)')
        fig.savefig(output/f'scaling_{axis}.png',dpi=160);fig.savefig(output/f'scaling_{axis}.pdf');plt.close(fig)
    (output/'report.md').write_text(f"# Controlled native scaling\n\n{len(checks)} rows replayed successfully. Complete: {manifest['complete']}.\n\n"
        +manifest['scope']+".\n\nCheck-profile subsets are implementation checks, not paper evidence. Training uses the fixed source training fold per feature count."
        +" Fits, compression, norm preparation, assignment and query costs are separate in summary.csv and raw rows. Timings do not include proof generation.\n",encoding='utf-8')
    return dict(checked_cases=len(checks),all_passed=True,complete=manifest['complete'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    with threadpool_limits(limits=4):print(json.dumps(report(args.out)))


if __name__=='__main__':main()
