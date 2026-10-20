"""Freeze validated certified-audit sources and actual 1k-row controls on Mobin."""
from pathlib import Path
import datetime,hashlib,io,json,subprocess,tarfile

ROOT=Path(__file__).resolve().parents[2]


def main():
    prior=json.loads((ROOT/'code/validation/outputs/review_full_controls_20261019T224939Z.json').read_text())
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    label='review_certified_controls_'+stamp;remote='/root/commitjl/results/'+label
    local=ROOT/'tmp'/label;local.mkdir(parents=True,exist_ok=False)
    binary=ROOT/'code/zk/bin/commitjl-bound-review-certified-20261020-linux'
    refs=ROOT/'code/validation/outputs/review_owner_inputs_20261020'
    owner=ROOT/'code/validation/outputs/review_certificate_preflight_20261020/owner_inputs'
    budget=ROOT/'code/validation/outputs/review_certificate_budget_20261020/owner_inputs'
    inputs={}
    for name,source_run in [('mnist_n64','reuse_mnist_n64'),('mnist_n256','reuse_mnist_n256'),('cifar_n64','reuse_cifar_n64')]:
        for method in ['learned_energy','pca_energy']:
            inputs[name+'_'+method+'.json']=refs/(source_run+'_'+method+'.json')
    for method in ['learned_energy','pca_energy']:
        name=f'mnist_n1024_{method}.json';inputs[name]=owner/name
        name=f'mnist_n256_{method}_r63_t16384.json';inputs[name]=budget/name
    for path in inputs.values():
        if not path.exists():raise FileNotFoundError(path)
    jobs=[]
    for name in ['mnist_n64','mnist_n1024','mnist_n256','cifar_n64']:
        for method in ['learned_energy','pca_energy']:
            jobs.append((name+'_'+method,name+'_'+method+'.json',64,False,True))
    for method in ['learned_energy','pca_energy']:
        name='mnist_n256_'+method+'_r63_t16384';jobs.append((name,name+'.json',16,False,True))
    jobs.append(('mnist_n1024_full_public','mnist_n1024_learned_energy.json',8,True,False))
    commands=[]
    for name,inp,batch,full,certify in jobs:
        extra='--full' if full else '--certify'
        commands.append(f'step {name} {inp} {batch} "{extra}"')
    queue=f'''#!/usr/bin/env bash
set -u
R={remote}
export PYTHONPATH="$R/source"
export COMMITJL_DATA=/root/commitjl/.cache/commitjl_data
export GOMAXPROCS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
P=/root/commitjl/.venv/bin/python
B="$R/bin/commitjl-bound-review-certified-linux"
echo "$(date -Is) WAIT_FOR_PRIOR_QUEUE" >> "$R/queue.log"
"$P" -m experiments.review_campaign_gate --queue {prior['remote']}/queue.log --state "$R/gate.json" > "$R/gate.log" 2>&1
gate_rc=$?
if [ "$gate_rc" -ne 0 ]; then echo "$(date -Is) FAILED_JOBS=gate" >> "$R/queue.log"; exit "$gate_rc"; fi
errors=0
step() {{
 name="$1"; input="$2"; batch="$3"; mode="$4"
 echo "$(date -Is) START $name" >> "$R/queue.log"
 "$P" -m experiments.review_controls --bin "$B" --input "$R/owner_inputs/$input" --out "$R/$name" --label "$name" --batch "$batch" --reps 1 --warmup 1 $mode > "$R/$name.log" 2>&1
 rc=$?
 echo "$(date -Is) END $name rc=$rc" >> "$R/queue.log"
 if [ "$rc" -ne 0 ]; then errors=$((errors+1)); fi
}}
{chr(10).join(commands)}
if [ "$errors" -eq 0 ]; then echo "$(date -Is) ALL_DONE" >> "$R/queue.log"; else echo "$(date -Is) FAILED_JOBS=$errors" >> "$R/queue.log"; fi
'''
    manifest=dict(remote=remote,session='review_certified_'+stamp,previous_queue=prior['remote'],
        created_utc=stamp,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        jobs=[dict(name=n,input=i,batch=b,full=f,certified=c) for n,i,b,f,c in jobs],files=[])
    archive=local/'release.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        def add(path,name):
            data=path.read_bytes();tar.add(path,arcname=name)
            manifest['files'].append(dict(path=name,sha256=hashlib.sha256(data).hexdigest()))
        for folder in ['experiments','commitjl','data']:
            for path in sorted((ROOT/'code'/folder).rglob('*.py')):add(path,'source/'+path.relative_to(ROOT/'code').as_posix())
        for path in sorted((ROOT/'code/zk').rglob('*')):
            if path.is_file() and (path.suffix=='.go' or path.name in ['go.mod','go.sum']):add(path,'source/zk/'+path.relative_to(ROOT/'code/zk').as_posix())
        for path in sorted((ROOT/'code/analysis').glob('review_certificate*.py')):add(path,'source/analysis/'+path.name)
        for name,path in inputs.items():add(path,'owner_inputs/'+name)
        add(binary,'bin/commitjl-bound-review-certified-linux')
        for name,data in [('queue.sh',queue.encode()),('source_manifest.json',json.dumps(manifest,indent=2).encode())]:
            info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o600;tar.addfile(info,io.BytesIO(data))
    (local/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    wsl=['wsl','-d','Ubuntu','--'];ssh=wsl+['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','cloudfpsmobin']
    def remote_python(code):
        for attempt in range(3):
            p=subprocess.run(ssh+['python3','-'],input=code.encode(),capture_output=True,timeout=90)
            if p.returncode==0:return p.stdout.decode(errors='replace')
        raise RuntimeError(p.stderr.decode(errors='replace')[-1200:])
    remote_python(f"from pathlib import Path\nr=Path({remote!r});r.mkdir(parents=True,exist_ok=True)\nassert not (r/'queue.log').exists()\n")
    linux='/mnt/'+archive.drive[0].lower()+'/'+archive.as_posix().split(':',1)[1].lstrip('/')
    for attempt in range(3):
        upload='release_'+str(attempt)+'.tar.gz'
        p=subprocess.run(wsl+['scp','-o','BatchMode=yes','-o','ConnectTimeout=15',linux,'cloudfpsmobin:'+remote+'/'+upload],capture_output=True,timeout=300)
        if p.returncode==0:break
    else:raise RuntimeError(p.stderr.decode(errors='replace')[-1200:])
    code=f'''import pathlib,tarfile,hashlib,json,subprocess,os
r=pathlib.Path({remote!r})
a=r/{upload!r}
assert hashlib.sha256(a.read_bytes()).hexdigest()=={hashlib.sha256(archive.read_bytes()).hexdigest()!r}
if not (r/'source').exists():
 with tarfile.open(a) as tar:tar.extractall(r,filter='data')
m=json.loads((r/'source_manifest.json').read_text())
for e in m['files']:assert hashlib.sha256((r/e['path']).read_bytes()).hexdigest()==e['sha256'],e['path']
os.chmod(r/'bin/commitjl-bound-review-certified-linux',0o700)
if not (r/'queue.log').exists():subprocess.run(['tmux','new-session','-d','-s',m['session'],'bash',str(r/'queue.sh')],check=True)
print(json.dumps({{'remote':str(r),'session':m['session'],'files_checked':len(m['files']),'status':'queued_serially'}}))
'''
    print(remote_python(code),flush=True)
    out=ROOT/'code/validation/outputs'/(label+'.json')
    with out.open('x',encoding='utf-8') as f:json.dump(manifest,f,indent=2)
    print(out,flush=True)


if __name__=='__main__':main()
