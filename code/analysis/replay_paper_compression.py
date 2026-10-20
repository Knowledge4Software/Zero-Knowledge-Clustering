"""Independently replay saved model parameters against SHA-verified inputs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

from commitjl.certificates import pair_squared_distances
from commitjl.reusable_native import NativeProjection, array_hash
from experiments.laptop_comparison import atomic_json
from experiments.paper_compression import sqdist
from experiments.protected_jl_study import Archive
from experiments.paper_artifacts import artifact_issues


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def replay(root, all_cases=False):
    root=Path(root); archive=Archive(); checks=[];digest_cache={}
    version=read(root/'manifest.json')['identity']['configuration'].get('version')
    for folder in sorted((root/'jobs').glob('*')):
        if not (folder/'dataset.json').exists(): continue
        metadata=read(folder/'dataset.json')
        original=archive.load(f"results/paper_server_20261006/rq1/{metadata['dataset']}/artifacts/dataset.npz")['X']
        with np.load(folder/'indices.npz') as ids:
            x=original[np.ix_(ids['source_indices'],ids['feature_indices'])]
            splits={phase:ids[phase] for phase in ('validation','test')}
        if array_hash(x)!=metadata['sha256']: raise ValueError('Saved input binding failed')
        del original
        seen=set()
        for file in sorted((folder/'rows').glob('*.json')):
            row=read(file); representative=(row['phase'],row['method'],row['k'])
            issues=artifact_issues(folder,row,digest_cache,require_hashes=version=='paper-compression-v2')
            if issues:raise ValueError('; '.join(issues))
            if not all_cases and representative in seen: continue
            seen.add(representative)
            with np.load(folder/row['model_artifact']) as p:
                if row['method']=='full': model=NativeProjection('full',row['d'])
                elif row['method'] in ('pca','pca_randomized','protected_only'):
                    model=NativeProjection('pca',row['d'],components=p['components'],origin=p['origin'])
                else:
                    a=csr_matrix((p['a_data'],p['a_indices'],p['a_indptr']),shape=tuple(p['a_shape']))
                    model=NativeProjection('hybrid',row['d'],basis=p['basis'],sparse_map=a,origin=p['origin'])
            assert model.fingerprint==row['model_sha256'] and model.m==row['m']
            with np.load(folder/'fits'/f"k{row['k']}_s{row['start']}.npz") as fitted: c=fitted['centers']
            with np.load(folder/'references'/f"{row['phase']}_k{row['k']}_s{row['start']}.npz") as ref:
                pairs=ref['pairs']; original_sq=ref['original_centroid_sq']
            z=model.transform(x[splits[row['phase']]])
            projected_c=model.transform(c); distances=sqdist(z,projected_c)
            with np.load(folder/'arrays'/f'{file.stem}.npz') as saved:
                np.testing.assert_allclose(projected_c,saved['projected_centroids'],rtol=1e-11,atol=1e-9)
                np.testing.assert_allclose(distances,saved['projected_centroid_sq'],rtol=1e-11,atol=1e-8)
                np.testing.assert_array_equal(distances.argmin(axis=1),saved['labels'])
                np.testing.assert_allclose(pair_squared_distances(z,pairs[:,0],pairs[:,1]),
                    saved['projected_pair_sq'],rtol=1e-11,atol=1e-8)
                assert not np.any(saved['certified'] & (saved['labels']!=original_sq.argmin(axis=1)))
            if row.get('workloads'):
                out=folder/row['workloads'];all_z=model.transform(x)
                for query_file in sorted((out/'queries').glob('*.npz')):
                    with np.load(query_file) as query:
                        a,b=query['left_indices'],query['right_indices']
                        np.testing.assert_allclose(sqdist(x[a],x[b]),query['original'],rtol=1e-11,atol=1e-8)
                        np.testing.assert_allclose(sqdist(all_z[a],all_z[b]),query['projected'],rtol=1e-11,atol=1e-8)
                for task in ('kmeans','kmeans_refine'):
                    if not (out/f'{task}.npz').exists():continue
                    with np.load(out/f'{task}.npz') as fitted:
                        np.testing.assert_array_equal(sqdist(x,fitted['original_centers']).argmin(axis=1),fitted['original_labels'])
                        np.testing.assert_array_equal(sqdist(all_z,fitted['projected_centers']).argmin(axis=1),fitted['projected_labels'])
                from sklearn.cluster import DBSCAN
                w=read(out/'timings.json')
                with np.load(out/'dbscan.npz') as db:
                    for key,matrix in [('original',db['original_sq']),('projected',db['projected_sq'])]:
                        expected=DBSCAN(eps=w['dbscan_radius']**2,min_samples=5,metric='precomputed').fit(matrix).labels_
                        np.testing.assert_array_equal(expected,db[key+'_labels'])
            checks.append(dict(job=folder.name,case=file.stem,model_sha256=model.fingerprint,passed=True))
    if not checks:raise ValueError('No completed cases to validate')
    result=dict(complete=read(root/'manifest.json')['complete'],
        scope='all saved cases' if all_cases else 'first saved case per job/phase/method/k',
        checked_cases=len(checks),all_passed=True,checks=checks)
    (root/'report').mkdir(exist_ok=True);atomic_json(root/'report/replay_validation.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--all-cases',action='store_true');args=parser.parse_args()
    with threadpool_limits(limits=4): result=replay(args.out,args.all_cases)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}))


if __name__=='__main__': main()
