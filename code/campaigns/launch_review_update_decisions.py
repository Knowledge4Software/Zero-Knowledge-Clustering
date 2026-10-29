"""Freeze and queue paired-update decision experiments on Mobin."""
from pathlib import Path
import base64,datetime,hashlib,io,json,subprocess,sys,tarfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from experiments.review_update_decisions import protocol,DATASETS,SEEDS,KS


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    label='review_update_decisions_'+stamp;remote='/root/commitjl/results/'+label
    local=ROOT/'tmp'/label;local.mkdir(parents=True,exist_ok=False)
    binary=ROOT/'code/zk/bin/commitjl-bound-paired-20261021-linux'
    if not binary.is_file():raise FileNotFoundError(binary)
    prior='/root/commitjl/results/review_sensitivity_20261021T061346Z'
    repair='review_report_fix_20261021T103551Z'
    declared=protocol(DATASETS,SEEDS,KS)
    declared.update(proof_case_count=33,proof_catalog_size=128,
        proof_cases='6 predeclared independent fits x (learned/PCA x rank15/63 + exact Full), plus 2 single-repetition deployment controls and 1 same-input portability control.',
        portability='Frozen locally prepared MNIST/k3/seed11/early/r63 input, same bytes on Windows and Mobin Linux, 4-CPU caps, paired-only one-repetition workflow; not an additional independent fitted model.',
        query_timing='First MNIST k3/seed11 family uses 3 query repetitions and 1 warmup; other families use 1 repetition and no warmup. All alternatives within each fit matched. Stage1 once per row.',
        batches='Common within each fit: MNIST/Fashion4; text k5 batch2; text k10 batch1 for exact Full memory/constraint feasibility.',
        workflow='Every predeclared update and all original rows proved; absolute and paired alternatives independently publicly replayed. Single-repetition deployment controls measure paired-only and exact Full workflows.',
        publication_status='Private repository; public release not authorized. Author details delegated to supervisor.')
    queue=f'''#!/usr/bin/env bash
set -u
R={remote}
P=/root/commitjl/.venv/bin/python
export PYTHONPATH="$R/source" COMMITJL_DATA=/root/commitjl/.cache/commitjl_data
export GOMAXPROCS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
echo "$(date -Is) WAIT_FOR_PRIOR_CAMPAIGN" >> "$R/queue.log"
"$P" -m experiments.review_campaign_gate --queue {prior}/queue.log --state "$R/previous_gate.json" > "$R/previous_gate.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then echo "$(date -Is) FAILED_GATE" >> "$R/queue.log"; exit "$rc"; fi
while tmux has-session -t {repair} 2>/dev/null; do sleep 10; done
exec 9>/root/commitjl/results/review-proof-campaign.lock
flock 9
echo "$(date -Is) START server_smoke" >> "$R/queue.log"
"$P" -m experiments.review_decision_controls --input "$R/smoke_input.json" --out "$R/server_smoke" --bin "$R/bin/commitjl-bound-paired-linux" --batch 2 --reps 3 --warmup 1 --also-absolute > "$R/server_smoke.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then echo "$(date -Is) FAILED_SMOKE" >> "$R/queue.log"; exit "$rc"; fi
echo "$(date -Is) END server_smoke rc=0" >> "$R/queue.log"
echo "$(date -Is) START portability_paired" >> "$R/queue.log"
"$P" -m experiments.review_decision_controls --input "$R/portability_owner.json" --out "$R/portability_paired" --bin "$R/bin/commitjl-bound-paired-linux" --batch 4 --reps 1 --warmup 0 > "$R/portability_paired.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then echo "$(date -Is) FAILED_PORTABILITY" >> "$R/queue.log"; exit "$rc"; fi
echo "$(date -Is) END portability_paired rc=0" >> "$R/queue.log"
echo "$(date -Is) START native_independent_fits" >> "$R/queue.log"
"$P" -m experiments.review_update_decisions --out "$R/native" > "$R/native.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then echo "$(date -Is) FAILED_NATIVE" >> "$R/queue.log"; exit "$rc"; fi
echo "$(date -Is) END native_independent_fits rc=0" >> "$R/queue.log"
"$P" -m experiments.review_decision_campaign --root "$R" --bin "$R/bin/commitjl-bound-paired-linux" > "$R/proof_campaign.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then echo "$(date -Is) FAILED_PROOF_CAMPAIGN rc=$rc" >> "$R/queue.log"; fi
'''
    manifest=dict(remote=remote,session='review_update_'+stamp,created_utc=stamp,previous_campaign=prior,
        source_policy='All new sources/binary independently frozen; older running sources untouched.',
        protocol=declared,binary_sha256=sha(binary),files=[])
    archive=local/'release.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        def add(path,name):
            tar.add(path,arcname=name);manifest['files'].append(dict(path=name,sha256=sha(path)))
        for folder in ['experiments','commitjl','data']:
            for path in sorted((ROOT/'code'/folder).rglob('*.py')):add(path,'source/'+path.relative_to(ROOT/'code').as_posix())
        for path in sorted((ROOT/'code/zk').rglob('*')):
            if path.is_file() and (path.suffix=='.go' or path.name in ['go.mod','go.sum']):add(path,'source/zk/'+path.relative_to(ROOT/'code/zk').as_posix())
        for name in ['test_paired_certificates.py','test_update_decision_campaign.py']:
            add(ROOT/'code/tests'/name,'source/tests/'+name)
        add(Path(__file__).resolve(),'source/campaigns/launch_review_update_decisions.py')
        add(binary,'bin/commitjl-bound-paired-linux')
        add(ROOT/'code/validation/outputs/review_paired_release_20261021/decision_smoke_input.json','smoke_input.json')
        add(ROOT/'tmp/review_decision_native_smoke_20261021/owner_inputs/mnist_k3_seed11_age1_learned_r63.json','portability_owner.json')
        for name,data in [('queue.sh',queue.encode()),('predeclared_protocol.json',json.dumps(declared,indent=2).encode()),
                          ('source_manifest.json',json.dumps(manifest,indent=2).encode())]:
            info=tarfile.TarInfo(name);info.mode=0o600;info.size=len(data);tar.addfile(info,io.BytesIO(data))
    (local/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    wsl=['wsl','-d','Ubuntu','--'];ssh=wsl+['ssh','-C','-o','BatchMode=yes','-o','ConnectTimeout=15','cloudfpsmobin']
    def remote_python(code):
        for _ in range(3):
            p=subprocess.run(ssh+['python3','-'],input=code.encode(),capture_output=True,timeout=90)
            if not p.returncode:return p.stdout.decode(errors='replace')
        raise RuntimeError(p.stderr.decode(errors='replace')[-1200:])
    remote_python(f"from pathlib import Path\nr=Path({remote!r});r.mkdir(parents=True,exist_ok=True)\nassert not (r/'source').exists() and not (r/'queue.log').exists()\n")
    linux='/mnt/'+archive.drive[0].lower()+'/'+archive.as_posix().split(':',1)[1].lstrip('/')
    upload='release.tar.gz'
    for attempt in range(3):
        try:
            p=subprocess.run(wsl+['rsync','-a','--partial','--append-verify','--timeout=120',
                '-e','ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=4',
                linux,'cloudfpsmobin:'+remote+'/'+upload],capture_output=True,timeout=1200)
        except subprocess.TimeoutExpired:
            if attempt==2:raise
            continue
        if not p.returncode:break
    else:raise RuntimeError(p.stderr.decode(errors='replace')[-1000:])
    code=f'''import pathlib,tarfile,hashlib,json,subprocess,os
r=pathlib.Path({remote!r});a=r/{upload!r}
assert hashlib.sha256(a.read_bytes()).hexdigest()=={sha(archive)!r}
if not (r/'source').exists():
 with tarfile.open(a) as tar:tar.extractall(r,filter='data')
m=json.loads((r/'source_manifest.json').read_text())
for e in m['files']:assert hashlib.sha256((r/e['path']).read_bytes()).hexdigest()==e['sha256'],e['path']
os.chmod(r/'bin/commitjl-bound-paired-linux',0o700)
if not (r/'queue.log').exists():subprocess.run(['tmux','new-session','-d','-s',m['session'],'bash',str(r/'queue.sh')],check=True)
print(json.dumps(dict(remote=str(r),session=m['session'],files_checked=len(m['files']),status='queued_after_previous_work')))
'''
    print(remote_python(code),flush=True)
    manifest['archive_sha256']=sha(archive)
    out=ROOT/'code/validation/outputs'/(label+'.json')
    with out.open('x',encoding='utf-8') as f:json.dump(manifest,f,indent=2)
    print(out,flush=True)


if __name__=='__main__':main()
