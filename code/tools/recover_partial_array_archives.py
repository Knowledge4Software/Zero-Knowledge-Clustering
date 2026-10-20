"""Recover independently hash-verified NPY objects from interrupted transfers.

A partial tar is never reported as a verified archive. Each recovered object
must match its expected SHA256 in the separately verified array index.
"""
import argparse
import gzip
import hashlib
import json
import re
import tarfile
from pathlib import Path


def recover(source):
    source = Path(source).resolve()
    index_archive = source/'priority_compact_index.tar.gz'
    receipt = json.loads(index_archive.with_name(index_archive.name+'.json').read_text())
    if hashlib.sha256(index_archive.read_bytes()).hexdigest() != receipt['sha256']:
        raise ValueError('Unverified array index')
    destination = source/'compact_arrays'
    destination.mkdir(exist_ok=True)
    with tarfile.open(index_archive) as tar:
        index_bytes = tar.extractfile('array_index.json').read()
    index = json.loads(index_bytes)
    (destination/'array_index.json').write_bytes(index_bytes)
    needed = {a['object'] for r in index['files'].values() for a in r['arrays'].values() if 'derive' not in a}
    objects = destination/'objects'
    objects.mkdir(exist_ok=True)
    archives = sorted(set(source.glob('science_arrays_*.tar.gz*')) |
                      set(source.glob('compact_arrays_*.tar.gz*')))
    recovered = 0
    errors = []
    for archive in archives:
        if not (archive.name.endswith('.tar.gz') or archive.name.endswith('.partial')): continue
        try:
            with tarfile.open(archive, 'r|gz') as tar:
                for member in tar:
                    match = re.fullmatch(r'objects/([0-9a-f]{64})\.npy\.gz', member.name)
                    if not match or not member.isfile(): continue
                    digest = match.group(1)
                    if digest not in needed or (objects/(digest+'.npy.gz')).exists(): continue
                    with tar.extractfile(member) as f: compressed = f.read()
                    payload = gzip.decompress(compressed)
                    if hashlib.sha256(payload).hexdigest() != digest:
                        raise ValueError('NPY object hash mismatch: '+digest)
                    target = objects/(digest+'.npy.gz')
                    temporary = target.with_suffix('.tmp')
                    temporary.write_bytes(compressed)
                    temporary.replace(target)
                    recovered += 1
        except (EOFError, tarfile.ReadError, OSError) as e:
            errors.append(dict(archive=archive.name, reason=str(e), classification='incomplete_container'))
    available = {p.name.removesuffix('.npy.gz') for p in objects.glob('*.npy.gz')}
    complete = sum(all(a['object'] in available for a in r['arrays'].values() if 'derive' not in a)
                   for r in index['files'].values())
    result = dict(recovered_new_objects=recovered, available_stored_objects=len(available & needed),
                  expected_stored_objects=len(needed), covered_npz_files=complete,
                  expected_npz_files=len(index['files']), interrupted_containers=errors)
    (source/'recovery_coverage.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'interrupted_containers'}, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-dir', type=Path, required=True)
    args = parser.parse_args()
    recover(args.archive_dir)
