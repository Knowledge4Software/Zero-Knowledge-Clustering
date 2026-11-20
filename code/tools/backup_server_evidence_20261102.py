"""Portable private backups, excluding rebuildable proving keys and caches."""
from pathlib import Path
import fcntl
import os
import tarfile
import time

from experiments.pca_advantage_20261102 import read,save,sha

REPO=Path(__file__).resolve().parents[2]
OUT=REPO/'code/validation/outputs/server_final_method_20261102/backups'
COHORTS=['server_campaign_20261101','server_diversity_20261101','server_all_sources_20261101',
         'server_pca_advantage_20261102','server_calibrated_routing_20261102']


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT.parents[1]/'server_campaign_20261101/coordinator.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    index=read(OUT/'index.json') if(OUT/'index.json').exists() else dict(files=[],created_unix=time.time())
    try:
        for name in COHORTS:
            dest=OUT/(name+'.tar.gz')
            if any(r['file']==dest.name for r in index['files']):continue
            root=REPO/'code/validation/outputs'/name
            if not root.exists():continue
            selected=[p for p in root.rglob('*') if p.is_file() and not p.is_symlink()
               and (p.suffix in ['.json','.proof','.vk','.npz','.md','.txt','.log'] or p.name in ['params','vk','proof'])
               and p.name!='pk']
            manifest={str(p.relative_to(root)):sha(p) for p in selected}
            with tarfile.open(dest,'w:gz',compresslevel=1,dereference=False) as a:
                for p in selected:a.add(p,arcname=str(Path(name)/p.relative_to(root)),recursive=False)
            save(OUT/(name+'.members.json'),manifest)
            index['files'].append(dict(file=dest.name,sha256=sha(dest),bytes=dest.stat().st_size))
            save(OUT/'index.json',index)
        # Portable source tree, including tests, method definitions and pins.
        dest=OUT/'final_source_and_plans.tar.gz'
        if not any(r['file']==dest.name for r in index['files']):
            selected=[p for p in (REPO/'code').rglob('*') if p.is_file() and not p.is_symlink()
               and p.suffix in ['.py','.go','.rs','.toml','.md','.json','.txt']
               and not any(s in p.relative_to(REPO/'code').parts for s in ['outputs','.cache','__pycache__','target'])]
            selected += [p for p in REPO.glob('*.json') if p.stat().st_size<1000000]
            selected += [p for p in REPO.glob('*.txt') if p.stat().st_size<1000000]
            with tarfile.open(dest,'w:gz',compresslevel=1,dereference=False) as a:
                for p in selected:a.add(p,arcname=str(p.relative_to(REPO)),recursive=False)
            index['files'].append(dict(file=dest.name,sha256=sha(dest),bytes=dest.stat().st_size))
        index.update(complete=True,finished_unix=time.time());save(OUT/'index.json',index)
    finally:lock.close()


if __name__=='__main__':run()
