"""Freeze additional Fashion audit rows without changing the active server fit."""
from pathlib import Path
import argparse
import gzip
import json

import numpy as np

from commitjl.pilot_io import array_sha, sha, save


def prepare(repo, out):
    source = repo/'code/.cache/commitjl_data/openml/openml.org/data/v1/download/18238735/Fashion-MNIST.arff.gz'
    canonical = np.random.default_rng(0).choice(70000, 5000, replace=False)
    available = np.setdiff1d(np.arange(70000), canonical)
    chosen = np.random.default_rng(20261101).choice(available, 3072, replace=False)
    positions = {int(row): i for i, row in enumerate(chosen)}
    rows = np.empty((3072, 784), dtype=np.uint8)
    found = 0
    attributes = []
    with gzip.open(source, 'rt') as stream:
        for line in stream:
            if line.lower().startswith('@attribute'):
                attributes.append(line.split()[1])
            if line.strip().lower() == '@data':
                break
        assert len(attributes) == 785 and attributes[-1].lower() == 'class'
        count = 0
        for index, line in enumerate(stream):
            count += 1
            position = positions.get(index)
            if position is not None:
                row = np.fromstring(line, sep=',', count=784)
                assert row.shape == (784,) and np.all((row >= 0) & (row <= 255))
                assert np.all(row == np.rint(row)) and np.any(row)
                rows[position] = row
                found += 1
    assert count == 70000 and found == 3072
    assert not set(chosen.tolist()).intersection(canonical.tolist())
    out.mkdir(parents=True, exist_ok=True)
    archive = out/'fashion_extra.npz'
    np.savez_compressed(archive, pixels=rows, raw_source_ids=chosen)
    save(out/'pin.json', dict(archive_sha256=sha(archive), pixels_sha256=array_sha(rows),
         source_gzip_sha256=sha(source), source_rows=count, source_features=784,
         audit_rows=3072, raw_source_ids=chosen.tolist(), seed=20261101,
         selection='Fixed random sample excluding all canonical5000 source IDs; append after the existing1024 audit rows. No outcome-based selection.',
         purpose='Paired scalability extension on the same Fashion seed701 training fit; not independent-fit replication.'))
    print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size,
                          archive_sha256=sha(archive))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    prepare(Path(__file__).resolve().parents[2], args.out)
