"""Freeze the predeclared sensitivity protocol and launch serially on Mobin."""
from pathlib import Path
import datetime
import hashlib
import io
import json
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from experiments.review_sensitivity import prepare, sha


def main():
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    label = 'review_sensitivity_'+stamp
    remote = '/root/commitjl/results/'+label
    local = ROOT/'tmp'/label
    local.mkdir(parents=True,exist_ok=False)
    binary = ROOT/'code/zk/bin/commitjl-bound-review-verifier-20261020-linux'
    # Reuse only the previously checked verifier and its exact matching Go source.
    release = json.loads((ROOT/'code/validation/outputs/review_verifier_release_20261020/manifest.json').read_text())
    for entry in release['files']:
        path = ROOT/entry['path'].replace('\\','/')
        if sha(path) != entry['sha256']:
            raise ValueError('Go source differs from checked verifier: '+entry['path'])
    if sha(binary) != release['binaries']['code\\zk\\bin\\commitjl-bound-review-verifier-20261020-linux']:
        raise ValueError('checked verifier binary hash mismatch')
    protocol = prepare(
        ROOT/'code/validation/outputs/review_certificate_budget_20261020/owner_inputs',
        ROOT/'code/validation/outputs/review_certificate_preflight_20261020/owner_inputs',
        local/'prepared')
    jobs = protocol['jobs']
    commands = []
    for job in jobs:
        mode = '--full' if job['full'] else '--certify'
        commands.append(f"step {job['name']} {job['input']} {job['batch']} {mode}")
    queue = f'''#!/usr/bin/env bash
set -u
R={remote}
export PYTHONPATH="$R/source"
export GOMAXPROCS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
P=/root/commitjl/.venv/bin/python
B="$R/bin/commitjl-bound-review-linux"
exec 9>/root/commitjl/results/review-proof-campaign.lock
echo "$(date -Is) WAIT_FOR_CAMPAIGN_LOCK" >> "$R/queue.log"
flock 9
errors=0
step() {{
 name="$1"; input="$2"; batch="$3"; mode="$4"
 echo "$(date -Is) START $name" >> "$R/queue.log"
 "$P" -m experiments.review_controls --bin "$B" --input "$R/prepared/owner_inputs/$input" --out "$R/$name" --label "$name" --batch "$batch" --reps 3 --warmup 1 --phase-hours 12 $mode > "$R/$name.log" 2>&1
 rc=$?
 echo "$(date -Is) END $name rc=$rc" >> "$R/queue.log"
 if [ "$rc" -ne 0 ]; then errors=$((errors+1)); fi
}}
step smoke_reuse smoke.json 2 --certify
step smoke_full smoke.json 2 --full
if [ "$errors" -ne 0 ]; then echo "$(date -Is) FAILED_SMOKE=$errors" >> "$R/queue.log"; exit 1; fi
echo "$(date -Is) START native" >> "$R/queue.log"
"$P" -m experiments.review_sensitivity native --prepared "$R/prepared" --out "$R/native" > "$R/native.log" 2>&1
rc=$?
echo "$(date -Is) END native rc=$rc" >> "$R/queue.log"
if [ "$rc" -ne 0 ]; then echo "$(date -Is) FAILED_NATIVE" >> "$R/queue.log"; exit "$rc"; fi
{chr(10).join(commands)}
if [ "$errors" -eq 0 ]; then
 "$P" -m experiments.review_sensitivity finalize --prepared "$R/prepared" --runs "$R" --out "$R/completed_summary.json" > "$R/finalize.log" 2>&1
 rc=$?
 if [ "$rc" -ne 0 ]; then errors=$((errors+1)); fi
fi
if [ "$errors" -eq 0 ]; then echo "$(date -Is) ALL_DONE" >> "$R/queue.log"; else echo "$(date -Is) FAILED_JOBS=$errors" >> "$R/queue.log"; fi
'''
    head = subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    manifest = dict(remote=remote,session='review_sensitivity_'+stamp,created_utc=stamp,
                    base_git_commit=head,working_source_frozen_by_file_hash=True,
                    jobs=jobs,reps=3,warmup=1,protocol_sha256=sha(local/'prepared/protocol.json'),
                    binary_sha256=sha(binary),files=[])
    archive = local/'release.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        def add(path,name):
            tar.add(path,arcname=name)
            manifest['files'].append(dict(path=name,sha256=sha(path)))
        for folder in ['experiments','commitjl','data']:
            for path in sorted((ROOT/'code'/folder).rglob('*.py')):
                add(path,'source/'+path.relative_to(ROOT/'code').as_posix())
        for path in sorted((ROOT/'code/zk').rglob('*')):
            if path.is_file() and (path.suffix=='.go' or path.name in ('go.mod','go.sum')):
                add(path,'source/zk/'+path.relative_to(ROOT/'code/zk').as_posix())
        for path in sorted((local/'prepared').rglob('*.json')):
            add(path,path.relative_to(local).as_posix())
        add(ROOT/'code/validation/outputs/review_driver_smoke_20261020/input.json','prepared/owner_inputs/smoke.json')
        add(binary,'bin/commitjl-bound-review-linux')
        add(Path(__file__).resolve(),'source/campaigns/launch_review_sensitivity.py')
        for name,data in [('queue.sh',queue.encode()),('source_manifest.json',json.dumps(manifest,indent=2).encode())]:
            info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o600
            tar.addfile(info,io.BytesIO(data))
    (local/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    wsl=['wsl','-d','Ubuntu','--']
    ssh=wsl+['ssh','-C','-o','BatchMode=yes','-o','ConnectTimeout=15','cloudfpsmobin']
    def remote_python(code):
        for _ in range(3):
            p=subprocess.run(ssh+['python3','-'],input=code.encode(),capture_output=True,timeout=90)
            if p.returncode==0:
                return p.stdout.decode(errors='replace')
        raise RuntimeError(p.stderr.decode(errors='replace')[-1200:])
    # Check for existing proof workers; this campaign must not compete with them.
    remote_python(f'''from pathlib import Path
import os
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try: args=p.read_bytes().split(bytes([0]))
 except (FileNotFoundError,PermissionError,ProcessLookupError): continue
 if b'experiments.review_controls' in args:
  raise RuntimeError('Existing proof worker is active; launch refused')
r=Path({remote!r})
r.mkdir(parents=True,exist_ok=True)
assert not (r/'source').exists() and not (r/'queue.log').exists()
''')
    linux = '/mnt/'+archive.drive[0].lower()+'/'+archive.as_posix().split(':',1)[1].lstrip('/')
    for attempt in range(3):
        upload=f'release_{attempt}.tar.gz'
        p=subprocess.run(wsl+['scp','-C','-o','BatchMode=yes','-o','ConnectTimeout=15',linux,
                              'cloudfpsmobin:'+remote+'/'+upload],capture_output=True,timeout=300)
        if p.returncode==0:
            break
    else:
        raise RuntimeError(p.stderr.decode(errors='replace')[-1200:])
    result = remote_python(f'''import pathlib,tarfile,json,hashlib,subprocess,os
r=pathlib.Path({remote!r});a=r/{upload!r}
assert hashlib.sha256(a.read_bytes()).hexdigest()=={sha(archive)!r}
if not (r/'source').exists():
 with tarfile.open(a) as tar:tar.extractall(r,filter='data')
m=json.loads((r/'source_manifest.json').read_text())
for e in m['files']:
 assert hashlib.sha256((r/e['path']).read_bytes()).hexdigest()==e['sha256'],e['path']
os.chmod(r/'bin/commitjl-bound-review-linux',0o700)
s=subprocess.run(['tmux','has-session','-t',m['session']],capture_output=True)
if s.returncode!=0 and not (r/'queue.log').exists():
 subprocess.run(['tmux','new-session','-d','-s',m['session'],'bash',str(r/'queue.sh')],check=True)
print(json.dumps(dict(remote=str(r),session=m['session'],files_checked=len(m['files']),status='queued_serially')))
''')
    print(result,flush=True)
    manifest['archive_sha256']=sha(archive)
    receipt=ROOT/'code/validation/outputs'/(label+'.json')
    with receipt.open('x',encoding='utf-8') as f:
        json.dump(manifest,f,indent=2)
    print(receipt,flush=True)


if __name__=='__main__':
    main()
