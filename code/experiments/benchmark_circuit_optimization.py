"""Matched, bounded local benchmark of legacy and optimized proof circuits.

No server data is overwritten. This is an engineering experiment, not an RQ1
fidelity campaign. Every method/mode uses the same saved input and centers.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import pandas as pd
import psutil

from commitjl import zk_bridge
from commitjl.clustering import nearest_centroid_assign, psi_hat

CODE = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

def write_summary(directory):
    """Summarize saved timings without changing or rerunning measurements."""
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if not manifest['complete']: raise ValueError('Benchmark is incomplete')
    effects=json.loads((directory/'effects.json').read_text(encoding='utf-8'))
    rows=pd.read_csv(directory/'timings.csv')
    text=['# Local circuit optimization benchmark\n',
          'Fixed synthetic integer inputs and centroids; Groth16 on one local host. '
          f"Each configuration has {manifest['warmup']} warm-up and {manifest['repetitions']} measured proofs. "
          'Setup is excluded from prove-only medians. These configurations were not selected for clustering fidelity.\n',
          '| Shape (n,d,m,k,q) | Method | Constraints before / after | Prove median before / after (s) | Constraint reduction | Prove reduction |',
          '| --- | --- | --- | --- | --- | --- |']
    for effect in effects:
        g=rows[(rows['shape']==effect['shape'])&(rows.method==effect['method'])]
        old=g[g['mode']=='legacy'].iloc[0];new=g[g['mode']=='optimized'].iloc[0]
        text.append(f"| {effect['shape']} | {effect['method']} | {old.constraints:,} / {new.constraints:,} | {old.prove_median:.3f} / {new.prove_median:.3f} | {effect['constraint_reduction_pct']:.2f}% | {effect['prove_reduction_pct']:.2f}% |")
    text += ['\nAll recorded proofs verified. Saved input hashes match within each before/after comparison. '
             'The separately built original prover reproduces the retained legacy constraint count where supplied.\n',
             'Constraint counts are deterministic for these circuits. Timing changes are descriptive fixed-witness '
             'measurements, not established statistical improvements or a full-corpus speedup. '
             'The full-dimensional control receives the same numeric-bound optimization. '
             'Routing-only and bounds-only arms, individual timing samples, memory observations and fingerprints '
             'are preserved in timings.csv, records/ and manifest.json. '
             'Regression and smoke tests ran during portions of the study; timing measurements are exploratory.\n']
    (directory/'summary.md').write_text('\n'.join(text),encoding='utf-8')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=CODE/'results/optimization_20261010')
    ap.add_argument('--shapes', nargs='+', default=['8x16x8x3x2','8x32x16x3x1'],
                    help='Each shape is n x d x m x k x q')
    ap.add_argument('--repetitions', type=int, default=5)
    ap.add_argument('--warmup', type=int, default=1)
    ap.add_argument('--timeout', type=float, default=240)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--max-rss-gib', type=float, default=4)
    ap.add_argument('--original-binary', type=Path)
    args = ap.parse_args()
    if args.repetitions < 1 or args.warmup < 0 or args.threads < 1 or args.timeout <= 0 or args.max_rss_gib <= 0:
        ap.error('Invalid resource or repetition limit')
    if (args.out/'manifest.json').exists():
        ap.error('Output already contains a benchmark; choose a new --out directory')
    os.environ['GOMAXPROCS'] = str(args.threads)
    binary = zk_bridge.locate_binary()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = dict(scope='Local fixed-input engineering comparison; no fidelity selection or paper speedup claim',
                    platform=platform.platform(), cpu=platform.processor(), logical_cpus=os.cpu_count(),
                    ram_bytes=psutil.virtual_memory().total, threads=args.threads,
                    repetitions=args.repetitions, warmup=args.warmup,
                    timeout_seconds=args.timeout, max_rss_gib=args.max_rss_gib,
                    binary_sha256=digest(binary),
                    sources={str(p.relative_to(CODE)):digest(p) for p in
                             [CODE/'zk/circuit/circuit.go',CODE/'zk/cmd/commitjl-prove/main.go',Path(__file__)]},
                    complete=False)
    if args.original_binary:
        manifest['original_binary_sha256'] = digest(args.original_binary)
    write(args.out/'manifest.json',manifest)
    rows=[]
    for shape_index, shape in enumerate(args.shapes):
        n,d,m,k,q = map(int,shape.split('x'))
        if min(n,d,m,k,q) < 1 or q > m or k < 2:
            raise ValueError(f'Invalid shape: {shape}')
        rng = np.random.default_rng(20261010+shape_index)
        X=rng.integers(-50,51,size=(n,d),dtype=np.int64)
        C=rng.integers(-50,51,size=(k,d),dtype=np.int64)
        seed=hashlib.sha256(('CommitJL-optimization-v1:'+shape).encode()).digest()
        base=zk_bridge.build_sparse_instance(X,C,seed,q,m)
        full_assign=nearest_centroid_assign(X,C)
        full=replace(base,variant='full',m=d,q=0,c_a=0,
                     constant_A=np.eye(d,dtype=np.int64), sparse_records=None,
                     assign=full_assign.tolist(),psi_hat=int(psi_hat(X,C,full_assign)))
        for method, inst in [('sparse',base),('full',full)]:
            write(args.out/'inputs'/f'{shape}_{method}.json',inst.to_json_dict())
            native_hash=hashlib.sha256(json.dumps(inst.to_json_dict(),sort_keys=True).encode()).hexdigest()
            modes=[('legacy',True,True),('optimized',False,False)]
            if method=='sparse': modes=[('legacy',True,True),('routing_only',False,True),('bounds_only',True,False),('optimized',False,False)]
            if method=='sparse' and shape_index==0 and args.original_binary:
                modes.insert(0,('original_binary',True,True))
            for mode,legacy_projection,legacy_bounds in modes:
                candidate=replace(inst,legacy_projection=legacy_projection,legacy_bounds=legacy_bounds)
                zk_bridge.BIN_PATH=args.original_binary.resolve() if mode=='original_binary' else binary
                start=time.perf_counter()
                result=zk_bridge.run_prove(candidate,timeout=args.timeout,build_if_missing=False,
                    repetitions=args.repetitions,warmup=args.warmup,max_rss_bytes=int(args.max_rss_gib*2**30))
                elapsed=time.perf_counter()-start
                name=f'{shape}_{method}_{mode}'
                write(args.out/'records'/f'{name}.json',asdict(result))
                row=dict(shape=shape,method=method,mode=mode,instance_sha256=native_hash,
                         constraints=result.num_constraints,prove_median=statistics.median(result.raw.get('prove_seconds_samples',[0.])),
                         compile_seconds=result.compile_seconds,setup_seconds=result.setup_seconds,
                         verify_seconds=result.verify_seconds,peak_rss_gib=result.peak_rss_bytes_psutil/2**30,
                         total_process_seconds=elapsed,ok=result.ok,successful_proofs=result.successful_proofs,
                         error=result.error,failure_kind=result.failure_kind)
                rows.append(row)
                pd.DataFrame(rows).to_csv(args.out/'timings.csv',index=False)
                print(json.dumps(row),flush=True)
                if not result.ok:
                    raise RuntimeError(f'Benchmark failed: {name}: {result.error}')
    zk_bridge.BIN_PATH=binary
    frame=pd.DataFrame(rows)
    original=frame[frame['mode']=='original_binary']
    for r in original.itertuples():
        legacy=frame[(frame['shape']==r.shape)&(frame.method==r.method)&(frame['mode']=='legacy')].iloc[0]
        assert r.constraints==legacy.constraints,'Legacy circuit does not reproduce original constraint count'
    effects=[]
    for (shape,method), group in frame.groupby(['shape','method']):
        old=group[group['mode']=='legacy'].iloc[0]; new=group[group['mode']=='optimized'].iloc[0]
        assert old.instance_sha256==new.instance_sha256
        effects.append(dict(shape=shape,method=method,
            constraint_reduction_pct=100*(1-new.constraints/old.constraints),
            prove_reduction_pct=100*(1-new.prove_median/old.prove_median),
            setup_reduction_pct=100*(1-new.setup_seconds/old.setup_seconds),
            peak_rss_reduction_pct=100*(1-new.peak_rss_gib/old.peak_rss_gib)))
    write(args.out/'effects.json',effects)
    manifest.update(complete=True,configurations=len(rows),successful_proofs=int(frame.successful_proofs.sum()),
                    original_constraint_count_reproduced=bool(len(original)),
                    limitations='Single local host, fixed synthetic inputs, sequential timing; not a fidelity-qualified operating point. Setup keys differ by circuit; PLONK production setup is not evaluated.')
    write(args.out/'manifest.json',manifest)
    write_summary(args.out)
    print(json.dumps(effects,indent=2),flush=True)

if __name__=='__main__': main()
