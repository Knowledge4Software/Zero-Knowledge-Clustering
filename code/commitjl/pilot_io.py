"""Small research I/O without importing experiment/ML dependency trees."""
from pathlib import Path
import gzip,hashlib,json,os,time
import numpy as np


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):digest.update(block)
    return digest.hexdigest()


def array_sha(value):return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast('B')).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.{time.time_ns()}.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');started=time.monotonic()
    while True:
        try:os.replace(tmp,path);return
        except PermissionError:
            if time.monotonic()-started>10:raise
            time.sleep(.05)


def sign(lo,hi):return 'worsens' if lo>0 else 'improves' if hi<0 else 'equal' if lo==hi==0 else 'inconclusive'


def frozen_image_sample(repo,dataset):
    """Stream only the same5000 selected rows from existing dense ARFF caches.

    The sample MUST match the canonical loader's already recorded float64
    SHA256, before any fitting/auditing. No download, normalization or new split.
    """
    home=Path(repo)/'code/.cache/commitjl_data';entries=[]
    for path in (home/'manifest_entries').glob(dataset+'_*.json'):
        entry=json.loads(path.read_text())
        if entry.get('n')==5000 and entry.get('seed')==0:entries.append(entry)
    if len({r['sha256'] for r in entries})!=1:raise ValueError('ambiguous/missing canonical5000-row sample pin')
    pin=entries[0];cache=home/'jl_small_samples'/(dataset+'_5000_seed0.npy')
    if cache.exists():
        x=np.load(cache,mmap_mode='r');assert array_sha(x)==pin['sha256'];return x
    names={'mnist':'52667/mnist_784.arff.gz','fashion_mnist':'18238735/Fashion-MNIST.arff.gz'}
    source=home/'openml/openml.org/data/v1/download'/names[dataset];chosen=np.random.default_rng(0).choice(pin['source_n'],size=5000,replace=False);positions={int(i):j for j,i in enumerate(chosen)};x=np.empty((5000,784),dtype=np.float64);found=0;attributes=[]
    with gzip.open(source,'rt') as stream:
        for line in stream:
            if line.lower().startswith('@attribute'):attributes.append(line.split()[1])
            if line.strip().lower()=='@data':break
        if len(attributes)!=785 or attributes[-1].lower()!='class':raise ValueError('unexpected cached feature/target layout')
        for index,line in enumerate(stream):
            position=positions.get(index)
            if position is not None:
                row=np.fromstring(line,sep=',',count=784);assert row.shape==(784,);x[position]=row;found+=1
    assert found==5000 and array_sha(x)==pin['sha256']
    cache.parent.mkdir(parents=True,exist_ok=True);temp=cache.with_name(cache.name+f'.{os.getpid()}.tmp')
    with temp.open('wb') as stream:np.save(stream,x)
    os.replace(temp,cache);save(cache.with_suffix('.json'),dict(canonical_sample_sha256=pin['sha256'],source_gzip_sha256=sha(source),source=str(source),selection='Identical np.default_rng(0).choice(source_n,5000,replace=False), original order; canonical float64 hash matched.'))
    return x
