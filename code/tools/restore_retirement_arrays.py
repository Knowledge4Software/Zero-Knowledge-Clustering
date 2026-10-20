"""Verify and restore lossless, deduplicated server result arrays.

Example: python tools/restore_retirement_arrays.py --archive-dir
results/server_retirement_20261007 --verify-only
Omit --verify-only and pass --destination . to restore results/ paths.
"""
import argparse
import gzip
import hashlib
import io
import json
import tarfile
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-dir', type=Path, required=True)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--compact', action='store_true',
                        help='Restore indexed DBSCAN pair distances from preserved X/Z and pair indices')
    parser.add_argument('--prefix', default='results/')
    parser.add_argument('--paper-root-name', default='server_paper_20261006',
                        choices=['server_paper_20261006', 'paper_server_20261006'],
                        help='Use the existing local name of the original campaign')
    args = parser.parse_args()
    if not args.verify_only and args.destination is None:
        parser.error('Restoration requires --destination')
    source = args.archive_dir.resolve()
    unpacked = source/('compact_arrays' if args.compact else 'arrays')
    unpacked.mkdir(exist_ok=True)
    archives = [source/('priority_compact_index.tar.gz' if args.compact else 'priority_science_index.tar.gz')]
    archives += sorted(source.glob('compact_arrays_*.tar.gz' if args.compact else 'science_arrays_*.tar.gz'))
    for archive in archives:
        if not archive.exists():
            raise FileNotFoundError(archive)
        receipt = json.loads(archive.with_name(archive.name+'.json').read_text())
        digest = hashlib.sha256()
        with archive.open('rb') as f:
            while block := f.read(8*1024**2): digest.update(block)
        if digest.hexdigest() != receipt['sha256']:
            raise ValueError('Archive checksum mismatch: '+str(archive))
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                target = (unpacked/member.name).resolve()
                if not target.is_relative_to(unpacked.resolve()) or not member.isfile():
                    raise ValueError('Unsafe archive entry: '+member.name)
                if target.exists() and target.stat().st_size == member.size:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as f, target.open('wb') as g:
                    while block := f.read(1024**2): g.write(block)
    index = json.loads((unpacked/'array_index.json').read_text())
    validated = set()
    missing = set()
    reconstructed = 0
    container_matches = 0
    derived_count = 0
    selected_files = 0
    original_distance_cache = {}
    for relative, record in index['files'].items():
        if not relative.startswith(args.prefix): continue
        selected_files += 1
        arrays = {}
        file_missing = False
        has_derived = any('derive' in item for item in record['arrays'].values())
        for key, item in record['arrays'].items():
            if 'derive' in item: continue
            digest = item['object']
            obj = unpacked/'objects'/(digest+'.npy.gz')
            if not obj.exists():
                missing.add(digest); file_missing = True; continue
            if args.verify_only and digest in validated and not has_derived: continue
            with gzip.open(obj, 'rb') as f: payload = f.read()
            if hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError('Array hash mismatch: '+digest)
            arr = np.load(io.BytesIO(payload), allow_pickle=False)
            if list(arr.shape) != item['shape'] or str(arr.dtype) != item['dtype']:
                raise ValueError('Array schema mismatch: '+relative+'/'+key)
            validated.add(digest)
            if not args.verify_only or has_derived: arrays[key] = arr
        if has_derived and not file_missing:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from commitjl.certificates import pair_squared_distances
            for key, item in record['arrays'].items():
                if 'derive' not in item: continue
                if args.verify_only and item['object'] in validated: continue
                d = item['derive']
                if d['function'] != 'commitjl.certificates.pair_squared_distances':
                    raise ValueError('Unrecognized derived function')
                digest = item['object']
                if key == 'original_sq' and digest in original_distance_cache:
                    arr = original_distance_cache[digest]
                else:
                    arr = pair_squared_distances(arrays[d['input']], arrays[d['i']], arrays[d['j']])
                buf = io.BytesIO(); np.save(buf, arr, allow_pickle=False)
                if hashlib.sha256(buf.getvalue()).hexdigest() != digest:
                    raise ValueError('Derived distance hash mismatch; use the recorded numerical environment: '+relative+'/'+key)
                if list(arr.shape) != item['shape'] or str(arr.dtype) != item['dtype']:
                    raise ValueError('Derived array schema mismatch')
                if key == 'original_sq': original_distance_cache[digest] = arr
                validated.add(digest); derived_count += 1
                if not args.verify_only: arrays[key] = arr
        if not args.verify_only and not file_missing:
            local_relative = relative.replace('results/paper_server_20261006/',
                                               'results/'+args.paper_root_name+'/', 1)
            target = (args.destination.resolve()/local_relative).resolve()
            if not target.is_relative_to(args.destination.resolve()):
                raise ValueError('Unsafe restoration path: '+relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest() == record['sha256']:
                    reconstructed += 1; container_matches += 1
                    continue
                raise ValueError('Existing file differs; preserve it before restoration: '+str(target))
            np.savez_compressed(target, **{key: arrays[key] for key in record['arrays']})
            reconstructed += 1
            container_matches += hashlib.sha256(target.read_bytes()).hexdigest() == record['sha256']
    result = dict(status='passed' if not missing else 'incomplete',
                  validated_unique_arrays=len(validated), missing_objects=sorted(missing),
                  verified_derived_distance_arrays=derived_count,
                  selected_npz_files=selected_files, scope_prefix=args.prefix,
                  included_npz_files=len(index['files']),
                  omitted_recomputable_maps=len(index['omitted_recomputable_maps']),
                  reconstructed_npz_files=reconstructed, original_container_hash_matches=container_matches)
    report_name = ('array_verification.json' if args.prefix == 'results/' else
                   'array_verification_subset_'+hashlib.sha256(args.prefix.encode()).hexdigest()[:8]+'.json')
    (source/report_name).write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'missing_objects'}, indent=2))
    raise SystemExit(0 if not missing else 1)


if __name__ == '__main__':
    main()
