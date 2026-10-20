"""Regenerate every saved pre-optimization hybrid map with the new XOF stream."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from commitjl.reusable_native import NativeProjection, array_hash, file_hash
from commitjl.sparse_gen import sparse_matrix_gen_csr
from experiments.laptop_comparison import atomic_json


def replay(previous, out):
    old_manifest=json.loads((previous/'manifest.json').read_text())
    assert old_manifest['complete']
    q=old_manifest['identity']['configuration']['q']
    checks=[]
    for path in sorted((previous/'rows').glob('*.json')):
        row=json.loads(path.read_text())
        if row['method'] not in ['protected_jl','protected_plus12']:
            continue
        artifact=previous/'artifacts'/(path.stem+'.npz')
        assert file_hash(artifact)==row['artifact_sha256']
        with np.load(artifact,allow_pickle=False) as saved:
            arrays={name:saved[name] for name in saved.files if name.startswith('model_')}
        data=json.loads((previous/'datasets'/row['dataset']/'dataset.json').read_text())
        basis=arrays['model_basis']
        payload=dict(version='enriched-screen-v1',dataset=data['used_sha256'],
            basis=array_hash(basis),m=row['m'],seed=row['seed'],start=row['start'],q=q)
        seed=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).digest()
        _, matrix=sparse_matrix_gen_csr(seed,row['residual_m'],row['d'],q)
        matrix=matrix.astype(float);matrix.data/=np.sqrt(q)
        np.testing.assert_array_equal(np.asarray(matrix.shape),arrays['model_a_shape'])
        for name in ['data','indices','indptr']:
            # NativeProjection serializes index parameters as canonical int64,
            # even when SciPy internally uses int32 for these small matrices.
            expected_array=arrays['model_a_'+name]
            regenerated=np.asarray(getattr(matrix,name),dtype=expected_array.dtype)
            np.testing.assert_array_equal(regenerated,expected_array)
            assert regenerated.tobytes()==expected_array.tobytes()
        model=NativeProjection('hybrid',row['d'],basis=basis,sparse_map=matrix,origin=arrays['model_origin'])
        assert model.fingerprint==row['model_fingerprint']
        checks.append(dict(case=path.stem,csr_arrays_bitwise_equal=True,model_fingerprint_equal=True))
    expected=sum(json.loads(p.read_text())['method'] in ['protected_jl','protected_plus12']
                 for p in (previous/'rows').glob('*.json'))
    assert len(checks)==expected and expected>0
    result=dict(complete=True,previous_run=str(previous.resolve()),
        previous_manifest_sha256=file_hash(previous/'manifest.json'),
        hybrid_maps_checked=len(checks),all_csr_arrays_and_fingerprints_bitwise_equal=True,
        implementation_hashes={str(p.relative_to(Path(__file__).resolve().parents[1])):file_hash(p)
            for p in [Path(__file__),Path(__file__).resolve().parents[1]/'commitjl/hashing.py',
                      Path(__file__).resolve().parents[1]/'commitjl/sparse_gen.py']},
        checks=checks,scope='All saved hybrid maps rebuilt from the original seed payload; '
        'no previous timing or clustering measurements changed')
    atomic_json(out,result)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        parser.error('Choose a new output path')
    with threadpool_limits(limits=4):
        replay(args.previous,args.out)


if __name__=='__main__':
    main()
