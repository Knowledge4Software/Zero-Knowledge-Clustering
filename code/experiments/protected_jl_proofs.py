"""Constraint feasibility and matched small actual proofs of research variants.

Routing specialization authenticates A, but requires new compilation/setup
after every changed row pattern. It is not generic sparse routing in one key.
"""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import pandas as pd
from commitjl import zk_bridge
from commitjl.clustering import nearest_centroid_assign,psi_hat
from commitjl.protected_jl import build_protected_instance,centroid_basis

CODE=Path(__file__).resolve().parents[1]


def cases(shape,scale=64):
    n,d,m,k,q=map(int,shape.split('x'))
    rng=np.random.default_rng(20261011)
    c=rng.integers(-50,51,size=(k,d),dtype=np.int64)
    x=c[np.arange(n)%k]+rng.integers(-3,4,size=(n,d),dtype=np.int64)
    seed=hashlib.sha256(('protected-proof-v1:'+shape).encode()).digest()
    base=zk_bridge.build_sparse_instance(x,c,seed,q,m)
    a_full=nearest_centroid_assign(x,c)
    full=replace(base,variant='full',m=d,q=0,c_a=0,constant_A=np.eye(d,dtype=np.int64),
                 sparse_records=None,assign=a_full.tolist(),psi_hat=int(psi_hat(x,c,a_full)))
    _,_,v=np.linalg.svd(x-x.mean(axis=0),full_matrices=False)
    pca=np.rint(scale*v[:min(k-1,len(v))]).astype(np.int64)
    def constant(matrix):
        z=x@matrix.T; cc=c@matrix.T; labels=nearest_centroid_assign(z,cc)
        return replace(base,variant='pca-public',m=len(matrix),q=0,c_a=0,constant_A=matrix,
                       sparse_records=None,assign=labels.tolist(),psi_hat=int(psi_hat(z,cc,labels)))
    result={
        'full_distances':full,'full_scores':replace(full,score_distances=True),
        'pca_distances':constant(pca),'pca_scores':replace(constant(pca),score_distances=True),
        'sparse_shared_distances':base,'sparse_shared_scores':replace(base,score_distances=True),
        'sparse_specialized_scores':replace(base,score_distances=True,specialize_sparse=True),
        'sparse_lookup_scores':replace(base,score_distances=True,lookup_sparse=True),
    }
    for rank,label in [(1,'partial'),(k-1,'fullrank')]:
        inst=build_protected_instance(x,c,seed,m,q,rank,scale)
        result['hybrid_'+label+'_shared_scores']=inst
        result['hybrid_'+label+'_specialized_scores']=replace(inst,specialize_sparse=True)
        result['hybrid_'+label+'_lookup_scores']=replace(inst,lookup_sparse=True)
    protected=np.rint(scale*centroid_basis(c).T).astype(np.int64)
    result['protected_only_scores']=replace(constant(protected),score_distances=True)
    return x,c,result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--shapes',nargs='+',default=['4x16x8x3x2','2x128x16x5x1'])
    ap.add_argument('--phase',choices=['compile','prove'],required=True)
    ap.add_argument('--repetitions',type=int,default=5)
    ap.add_argument('--warmup',type=int,default=1)
    ap.add_argument('--timeout',type=float,default=240)
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--max-rss-gib',type=float,default=4)
    ap.add_argument('--methods',nargs='+')
    ap.add_argument('--backend',choices=['groth16','plonk'],default='groth16')
    ap.add_argument('--binary',type=Path,default=CODE/'zk/bin/commitjl-experimental-v3.exe')
    ap.add_argument('--basis-scale',type=int,default=64)
    args=ap.parse_args()
    if min(args.repetitions,args.threads,args.timeout,args.max_rss_gib,args.basis_scale)<1 or args.warmup<0: ap.error('Invalid limits')
    if args.out.exists(): ap.error('Choose a new output directory')
    args.out.mkdir(parents=True); records=args.out/'records'; records.mkdir()
    inputs=args.out/'inputs'; inputs.mkdir()
    os.environ['GOMAXPROCS']=str(args.threads)
    zk_bridge.BIN_PATH=args.binary.resolve()
    manifest=dict(phase=args.phase,backend=args.backend,complete=False,scope='Fixed synthetic input engineering study; not corpus speedup evidence',
        configuration={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
        binary_sha256=hashlib.sha256(zk_bridge.BIN_PATH.read_bytes()).hexdigest(),
        sources={str(path.relative_to(CODE)):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                 [CODE/'zk/circuit/circuit.go',CODE/'zk/circuit/sparse_lookup.go',
                  CODE/'zk/cmd/commitjl-prove/main.go',CODE/'commitjl/protected_jl.py',Path(__file__)]},
        specialization='Each new sparse row pattern changes the compiled circuit and needs new setup/key; rows/signs remain authenticated.',
        hybrid='Fixed-point polynomial hybrid, q-weighted protected coordinates; basis fixed in circuit/key. No production protocol setup claimed.')
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    rows=[];begin=time.perf_counter()
    for shape in args.shapes:
        x,c,instances=cases(shape,args.basis_scale)
        for name,inst in instances.items():
            if args.methods and name not in args.methods: continue
            inst=replace(inst,backend=args.backend,solve_only=args.phase=='compile')
            payload=inst.to_json_dict()
            (inputs/f'{shape}_{name}.json').write_text(json.dumps(payload,indent=2))
            result=zk_bridge.run_prove(inst,timeout=args.timeout,build_if_missing=False,
                repetitions=args.repetitions,warmup=args.warmup,max_rss_bytes=int(args.max_rss_gib*2**30))
            (records/f'{shape}_{name}.json').write_text(json.dumps(asdict(result),indent=2))
            passed=bool(result.raw.get('solve_ok')) if args.phase=='compile' else result.ok and result.verify_ok
            row=dict(shape=shape,method=name,phase=args.phase,passed=passed,constraints=result.num_constraints,
                compile_seconds=result.compile_seconds,setup_seconds=result.setup_seconds,
                prove_median=statistics.median(result.raw.get('prove_seconds_samples') or [0.]),
                successful_proofs=result.successful_proofs,peak_rss_gib=result.peak_rss_bytes_psutil/2**30,
                failure_kind=result.failure_kind,error=result.error,
                x_sha256=hashlib.sha256(x.tobytes()).hexdigest(),centers_sha256=hashlib.sha256(c.tobytes()).hexdigest(),
                original_agreement=float(np.mean(np.asarray(inst.assign)==nearest_centroid_assign(x,c))),
                single_run_total=result.compile_seconds+result.setup_seconds+result.witness_seconds+
                                 (statistics.median(result.raw.get('prove_seconds_samples') or [0.]) if args.phase=='prove' else result.raw.get('solve_seconds',0.)))
            rows.append(row);pd.DataFrame(rows).to_csv(args.out/'timings.csv',index=False)
            print(json.dumps(row),flush=True)
            if not passed: raise RuntimeError(f'Failed {shape} {name}: {result.error}')
    manifest.update(complete=True,wall_seconds=time.perf_counter()-begin,configurations=len(rows),
                    successful_proofs=sum(r['successful_proofs'] for r in rows))
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
