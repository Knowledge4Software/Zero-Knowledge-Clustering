"""Compare canonical dense-to-CSR construction with direct CSR construction."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import numpy as np
from scipy.sparse import csr_matrix

from commitjl.encoding import SparseRecord
from commitjl.sparse_gen import gen_column, sparse_matrix_gen_csr

def reference(seed,m,d,q):
    # The pre-optimization sampler and allocation path, retained as a reference.
    records=[];dense=np.zeros((m,d),dtype=np.int64)
    for t in range(d):
        for ell,(row,sign) in enumerate(gen_column(seed,t,m,q)):
            records.append(SparseRecord(t=t,ell=ell,row=row,sign=sign))
            dense[row,t]=sign
    return records,csr_matrix(dense.astype(float)/np.sqrt(q))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists(): ap.error('Choose a new output file')
    rows=[]
    for m,d,q in [(128,256,2),(1024,10000,2)]:
        before=[];after=[]
        for rep in range(3):
            seed=f'host-optimization-{rep}'.encode()
            gc.collect()
            start=time.perf_counter();old_records,old=reference(seed,m,d,q);before.append(time.perf_counter()-start)
            start=time.perf_counter();new_records,new=sparse_matrix_gen_csr(seed,m,d,q)
            new=new.astype(float);new.data/=np.sqrt(q);after.append(time.perf_counter()-start)
            assert old_records==new_records
            for field in ['indptr','indices','data']:
                np.testing.assert_array_equal(getattr(old,field),getattr(new,field))
            X=np.random.default_rng(rep).normal(size=(8,d))
            np.testing.assert_array_equal(old@X.T,new@X.T)
        rows.append(dict(m=m,d=d,q=q,repetitions=3,
            dense_matrix_bytes=m*d*8,
            csr_matrix_bytes=sum(getattr(new,f).nbytes for f in ['data','indices','indptr']),
            dense_to_csr_seconds=before,direct_csr_seconds=after,
            dense_to_csr_median=statistics.median(before),direct_csr_median=statistics.median(after),
            canonical_records_equal=True,csr_arrays_and_products_bitwise_equal=True))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(dict(complete=True,scope='Matrix construction only; storage excludes records and projected data',rows=rows),indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2))

if __name__=='__main__': main()
