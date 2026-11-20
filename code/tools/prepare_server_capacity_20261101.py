"""Extend the already pinned extra audit sample to an8192-row catalog."""
from pathlib import Path
import argparse
import gzip
import json

import numpy as np

from commitjl.pilot_io import array_sha, save, sha


def prepare(repo, previous, out):
    old_pin = json.loads((previous/'pin.json').read_text())
    assert sha(previous/'fashion_extra.npz') == old_pin['archive_sha256']
    with np.load(previous/'fashion_extra.npz') as archive:
        old_pixels, old_ids = archive['pixels'], archive['raw_source_ids']
    canonical = np.random.default_rng(0).choice(70000,5000,replace=False)
    available = np.setdiff1d(np.arange(70000),np.concatenate([canonical,old_ids]))
    chosen = np.random.default_rng(20261102).choice(available,4096,replace=False)
    positions = {int(row):i for i,row in enumerate(chosen)}
    pixels = np.empty((4096,784),dtype=np.uint8)
    source = repo/'code/.cache/commitjl_data/openml/openml.org/data/v1/download/18238735/Fashion-MNIST.arff.gz'
    assert sha(source) == old_pin['source_gzip_sha256']
    found = 0
    with gzip.open(source,'rt') as stream:
        for line in stream:
            if line.strip().lower() == '@data':
                break
        for index,line in enumerate(stream):
            position = positions.get(index)
            if position is not None:
                row = np.fromstring(line,sep=',',count=784)
                assert row.shape == (784,) and np.all((row>=0)&(row<=255))
                assert np.all(row == np.rint(row)) and np.any(row)
                pixels[position] = row
                found += 1
    assert index+1 == 70000 and found == 4096
    pixels = np.concatenate([old_pixels,pixels])
    ids = np.concatenate([old_ids,chosen])
    assert len(np.unique(ids)) == 7168 and not np.intersect1d(ids,canonical).size
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'fashion_extra.npz',pixels=pixels,raw_source_ids=ids)
    save(out/'pin.json',dict(archive_sha256=sha(out/'fashion_extra.npz'),
         pixels_sha256=array_sha(pixels),source_rows=70000,source_features=784,
         source_gzip_sha256=old_pin['source_gzip_sha256'],audit_rows=7168,
         raw_source_ids=ids.tolist(),previous_pin_sha256=sha(previous/'pin.json'),
         previous_prefix_rows=3072,additional_rows=4096,additional_seed=20261102,
         selection='Preserve prior3072 extra rows; append4096 fixed random rows excluding all previous extras and canonical5000 IDs. No outcome-based selection.'))
    print(json.dumps(dict(archive=str(out/'fashion_extra.npz'),bytes=(out/'fashion_extra.npz').stat().st_size)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    prepare(Path(__file__).resolve().parents[2],args.previous,args.out)
