"""Seal all final evidence after computation, including failures and receipts."""
from pathlib import Path
import os
import tarfile
import time

import psutil

from experiments.pca_advantage_20261102 import read,save,sha

ROOT=Path(__file__).resolve().parents[2]/'code/validation/outputs/server_final_method_20261102'


def active_workers():
    prefix=str(ROOT.resolve())
    found=[]
    for p in psutil.process_iter(['pid','cmdline','status']):
        if p.pid==os.getpid() or p.info['status']==psutil.STATUS_ZOMBIE:continue
        args=p.info['cmdline'] or []
        if any(a==prefix or a.startswith(prefix+'/') for a in args):found.append(p.pid)
    return found


def run():
    while not(ROOT/'plan.json').exists():time.sleep(5)
    deadline=read(ROOT/'plan.json')['compute_deadline_unix']
    while not(ROOT/'finished.json').exists() and time.time()<deadline+90:time.sleep(10)
    while active_workers() and time.time()<deadline+300:time.sleep(5)
    if active_workers():
        save(ROOT/'seal_failure.json',dict(reason='Owned workers still live',pids=active_workers(),unix=time.time()))
        return
    time.sleep(35)  # Give the independent descriptive-table reporter its last pass.
    save(ROOT/'seal_status.json',dict(started_unix=time.time(),source_sha256=sha(Path(__file__)),
        includes='All final inputs, public parameters/keys/proofs, validation/test records, resources, failures and exclusions.',
        excludes='Rebuildable private proving keys, serialized circuit caches and duplicate archive folders.'))
    selected=[p for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink()
        and p.relative_to(ROOT).parts[0] not in ['archives','backups']
        and p.name!='pk' and p.suffix not in ['.pk','.r1cs','.gob']]
    target=ROOT/'archives/final_complete_evidence.tar.gz';assert not target.exists()
    tmp=target.with_suffix('.tmp')
    with tarfile.open(tmp,'w:gz',compresslevel=1,dereference=False) as a:
        for p in selected:a.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    tmp.replace(target)
    save(ROOT/'archives/final_complete_evidence.sha256.json',dict(file=target.name,bytes=target.stat().st_size,sha256=sha(target)))
    state=read(ROOT/'seal_status.json');state.update(complete=True,finished_unix=time.time(),archive_sha256=sha(target))
    save(ROOT/'seal_status.json',state)


if __name__=='__main__':run()
