"""Paired old/new canonical construction; no model or RNG changes."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
from threadpoolctl import threadpool_limits

from commitjl import sparse_gen
from commitjl.reusable_native import NativeProjection, file_hash
from experiments.laptop_comparison import atomic_json


def legacy_stream(data):
    """Exact pre-optimization stream implementation (4096-byte refills)."""
    shake = hashlib.shake_256(data)
    block = 4096
    pos = 0
    while True:
        buf = shake.digest(pos + block)
        while pos + 4 <= len(buf):
            yield buf[pos:pos + 4]
            pos += 4


@contextmanager
def stream_backend(stream):
    previous = sparse_gen.shake256_stream
    sparse_gen.shake256_stream = stream
    try:
        yield
    finally:
        sparse_gen.shake256_stream = previous


def compare(m, d, q, repetitions=7):
    rng = np.random.default_rng(349)
    basis = np.linalg.qr(rng.normal(size=(d, 16)))[0]
    origin = rng.normal(size=d)
    points = rng.normal(size=(19, d))
    def make(seed):
        records, matrix = sparse_gen.sparse_matrix_gen_csr(seed, m, d, q)
        matrix = matrix.astype(float); matrix.data /= np.sqrt(q)
        return records, NativeProjection('hybrid', d, basis=basis, sparse_map=matrix, origin=origin)
    backends = dict(before=legacy_stream, after=sparse_gen.shake256_stream)
    samples = {name:[] for name in backends}
    # Both get the same excluded warmup; alternate order to limit timing drift.
    for backend in backends.values():
        with stream_backend(backend):
            make(b'excluded-warmup')
    for rep in range(repetitions):
        models = {}
        seed = hashlib.sha256(f'paired-map-{rep}'.encode()).digest()
        for name in (['before','after'] if rep%2 == 0 else ['after','before']):
            with stream_backend(backends[name]):
                begin = time.perf_counter()
                models[name] = make(seed)
                samples[name].append(time.perf_counter()-begin)
        old_records, old = models['before']; new_records, new = models['after']
        assert old_records == new_records
        assert old.fingerprint == new.fingerprint
        for attr in ['data','indices','indptr']:
            np.testing.assert_array_equal(getattr(old.sparse_map,attr),getattr(new.sparse_map,attr))
        np.testing.assert_array_equal(old.transform(points),new.transform(points))
    before, after = (statistics.median(samples[name]) for name in ['before','after'])
    return dict(residual_m=m,d=d,q=q,protected_rank=16,repetitions=repetitions,
        before_seconds=samples['before'],after_seconds=samples['after'],
        before_median_seconds=before,after_median_seconds=after,speedup=before/after,
        records_csr_fingerprint_and_projection_bitwise_equal=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        parser.error('Choose a new output file')
    with threadpool_limits(limits=4):
        rows=[compare(m,d,q) for m,d,q in [(112,784,1),(240,784,1),(112,5000,1),
             (240,5000,1),(128,5000,2),(257,512,8),(65,100,65)]]
    code=Path(__file__).resolve().parents[1]
    result=dict(complete=True,scope='Canonical map plus NativeProjection construction only; '
        'identical learned basis; no centroid/PCA fitting or whole-corpus conversion included',
        platform=platform.platform(),threads=4,rows=rows,
        source_hashes={str(p.relative_to(code)):file_hash(p) for p in [Path(__file__),
            code/'commitjl/hashing.py',code/'commitjl/sparse_gen.py',code/'commitjl/reusable_native.py']})
    atomic_json(args.out,result)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
