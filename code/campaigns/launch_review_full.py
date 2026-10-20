"""Freeze tested sources and launch fresh public-center Full controls on Mobin."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[2]


def main():
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    label='review_full_controls_'+stamp
    remote='/root/commitjl/results/'+label
    local=ROOT/'tmp'/label;local.mkdir(parents=True,exist_ok=False)
    binary=ROOT/'code/zk/bin/commitjl-bound-review-full-20261020-linux'
    entries=[]
    archive=local/'release.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for folder in ['experiments','commitjl','data']:
            for p in sorted((ROOT/'code'/folder).rglob('*.py')):
                name='source/'+p.relative_to(ROOT/'code').as_posix()
                tar.add(p,arcname=name);entries.append(dict(path=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        for p in sorted((ROOT/'code/zk').rglob('*')):
            if p.is_file() and (p.suffix=='.go' or p.name in ['go.mod','go.sum']):
                name='source/zk/'+p.relative_to(ROOT/'code/zk').as_posix()
                tar.add(p,arcname=name);entries.append(dict(path=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        tar.add(binary,arcname='bin/commitjl-bound-review-full-linux')
    queue=f'''#!/usr/bin/env bash
set -u
R={remote}
export PYTHONPATH="$R/source"
export COMMITJL_DATA=/root/commitjl/.cache/commitjl_data
export GOMAXPROCS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
P=/root/commitjl/.venv/bin/python
B="$R/bin/commitjl-bound-review-full-linux"
errors=0
step() {{
  name="$1"; input="$2"; batch="$3"
  echo "$(date -Is) START $name" >> "$R/queue.log"
  "$P" -m experiments.review_controls --bin "$B" --input "$input" --out "$R/$name" --full --batch "$batch" --reps 1 --warmup 1 > "$R/$name.log" 2>&1
  rc=$?
  echo "$(date -Is) END $name rc=$rc" >> "$R/queue.log"
  if [ "$rc" -ne 0 ]; then errors=$((errors+1)); fi
}}
step mnist_n64 /root/commitjl/results/paper_runs_20261019/reuse_mnist_n64/cases/mnist_n64_learned_energy/input.json 8
step mnist_n256 /root/commitjl/results/paper_runs_20261019/reuse_mnist_n256/cases/mnist_n256_learned_energy/input.json 8
step cifar_n64 /root/commitjl/results/paper_runs_20261019/reuse_cifar_n64/cases/cifar10_resnet50_n64_learned_energy/input.json 2
if [ "$errors" -eq 0 ]; then echo "$(date -Is) ALL_DONE" >> "$R/queue.log"; else echo "$(date -Is) FAILED_JOBS=$errors" >> "$R/queue.log"; fi
'''
    (local/'queue.sh').write_text(queue,encoding='utf-8',newline='\n')
    manifest=dict(remote=remote,session='review_full_'+stamp,source_files=entries,
                  binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),created_utc=stamp)
    (local/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    wsl=['wsl','-d','Ubuntu','--']
    ssh=wsl+['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','cloudfpsmobin']
    bootstrap=f"from pathlib import Path\np=Path({remote!r});p.mkdir(parents=True,exist_ok=False)\nprint(p)\n"
    p=subprocess.run(ssh+['python3','-'],input=bootstrap.encode(),capture_output=True,timeout=60)
    if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
    for path in [archive,local/'queue.sh',local/'source_manifest.json']:
        linux='/mnt/'+path.drive[0].lower()+'/'+path.as_posix().split(':',1)[1].lstrip('/')
        subprocess.run(wsl+['scp','-o','BatchMode=yes','-o','ConnectTimeout=15',linux,'cloudfpsmobin:'+remote+'/'+path.name],check=True,capture_output=True,timeout=300)
    command=f"tar -xzf {remote}/release.tar.gz -C {remote} && chmod +x {remote}/bin/commitjl-bound-review-full-linux && tmux new-session -d -s {manifest['session']} bash {remote}/queue.sh"
    p=subprocess.run(ssh+[command],capture_output=True,timeout=60)
    if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
    output=ROOT/'code/validation/outputs'/(label+'.json')
    with output.open('x',encoding='utf-8') as f:json.dump(manifest,f,indent=2)
    print(json.dumps(dict(remote=remote,session=manifest['session'],manifest=str(output)),indent=2))


if __name__=='__main__':main()
