"""Small actual-backend smoke test of the current optimized proof relation."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from commitjl import zk_bridge

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists(): parser.error('Choose a new output file')
    if args.binary: zk_bridge.BIN_PATH=args.binary.resolve()
    binary=zk_bridge.locate_binary(build_if_missing=not bool(args.binary))
    X=np.array([[3,-4,1,2],[-2,1,4,-3]],dtype=np.int64)
    C=np.array([[0,1,-1,2],[2,-1,0,1]],dtype=np.int64)
    records=[]
    for backend in ['groth16','plonk']:
        inst=zk_bridge.build_sparse_instance(X,C,b'optimization-smoke-v1',q=2,m=2,backend=backend)
        res=zk_bridge.run_prove(inst,build_if_missing=False,timeout=120,max_rss_bytes=2*2**30)
        records.append(dict(backend=backend,result=asdict(res)))
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            plonk_setup='unsafe test-only KZG SRS',records=records,complete=len(records)==2),indent=2),encoding='utf-8')
        print(backend,res.ok,res.verify_ok,res.num_constraints,flush=True)
        if not(res.ok and res.verify_ok): raise RuntimeError(res.error)

if __name__=='__main__': main()
