"""Checkpoint integrity checks shared by the runner and report exporter."""
import hashlib
import json
from pathlib import Path


def digest(path,cache=None):
    path=Path(path);stat=path.stat();key=(str(path.resolve()),stat.st_size,stat.st_mtime_ns)
    if cache is not None and key in cache: return cache[key]
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    value=h.hexdigest()
    if cache is not None:cache[key]=value
    return value


def artifact_issues(folder,row,cache=None,require_hashes=True):
    folder=Path(folder);errors=[];hashes=row.get('artifact_sha256',{})
    if require_hashes and not hashes:return ['Missing checkpoint artifact hashes']
    key=f"{row['phase']}_k{row['k']}_s{row['start']}_{row['method']}_m{row['m']}_q{row['q']}_seed{row['seed']}"
    required={f'arrays/{key}.npz',str(row['model_artifact']).replace('\\','/')}
    if row.get('workloads'):
        prefix=str(row['workloads']).replace('\\','/')
        required|={prefix+'/'+name for name in ('timings.json','knn.npz','kmeans.npz','dbscan.npz')}
        if require_hashes:required|={prefix+'/'+name for name in ('query_indices.npz','kmeans_refine.npz')}
    if require_hashes:
        missing=required-set(hashes)
        errors.extend('Unbound required artifact: '+p for p in sorted(missing))
    for name in sorted(set(hashes)|required):
        path=(folder/name).resolve()
        if not path.is_relative_to(folder.resolve()):errors.append('Artifact escapes job directory: '+name);continue
        if not path.is_file():errors.append('Missing artifact: '+name)
        elif name in hashes and digest(path,cache)!=hashes[name]:errors.append('Corrupt artifact: '+name)
    return errors


def job_complete(folder,cache=None,require_hashes=True):
    folder=Path(folder)
    try:
        done=json.loads((folder/'done.json').read_text());inventory=json.loads((folder/'case_inventory.json').read_text())
        expected={c['key'] for c in inventory['cases']};actual={p.stem for p in (folder/'rows').glob('*.json')}
        if not done.get('complete') or expected!=actual:return False
        for key in expected:
            row=json.loads((folder/'rows'/f'{key}.json').read_text())
            if artifact_issues(folder,row,cache,require_hashes):return False
        return True
    except (OSError,ValueError,KeyError):return False
