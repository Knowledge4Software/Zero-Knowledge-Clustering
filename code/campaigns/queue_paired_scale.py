"""Queue a same-model N128->N1024 decision proof scale control, without refits."""
from pathlib import Path
import datetime,hashlib,io,json,subprocess,sys,tarfile
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from data.loaders import load_dataset


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    label='paired_scale_'+stamp
    parent=json.loads((ROOT/'code/validation/outputs/review_update_decisions_20261021T184553Z.json').read_text())
    remote=parent['remote']+'/'+label
    local=ROOT/'tmp'/label;local.mkdir(parents=True,exist_ok=False)
    x=load_dataset('mnist',n_max=5000,seed=0)['X'];x=x[np.linalg.norm(x,axis=1)>0]
    order=np.random.default_rng(752).permutation(len(x));cut=int(.6*len(x))
    refs=ROOT/'tmp/review_decision_native_smoke_20261021/owner_inputs'
    files=[]
    for method in ['learned','pca']:
        owner=json.loads((refs/f'mnist_k3_seed11_age1_{method}_r63.json').read_text())
        rows=np.clip(np.rint(x[order[cut:cut+1024]]*float(owner['model']['input_scale'])),-4096,4096).astype(np.int64)
        if owner['x']!=rows[:128].tolist():raise ValueError('old catalog prefix differs')
        if hashlib.sha256(rows.tobytes()).hexdigest()!=owner['experiment_metadata']['audit_sha256']:raise ValueError('full native audit hash differs')
        owner['model']['n']=1024;owner['x']=rows.tolist();owner['queries']=[list(range(i,i+4)) for i in range(0,1024,4)]
        owner['experiment_metadata'].update(proof_scope='Actual N1024 scale control of frozen N128 portability model; no refit',source_n=128)
        path=local/(method+'.json');path.write_text(json.dumps(owner,indent=2));files.append(path)
    protocol=dict(remote=remote,parent_campaign=parent['remote'],parent_session=parent['session'],
        input_scope='Same frozen MNIST/k3/seed11/early/rank63 learned and PCA fits; actual N1024, no new independent model.',
        selection='Scale check selected as the first fixed portability workload; broad grid negatives and all original jobs retained.',
        jobs=['learned','pca','full'],batch=4,reps=1,warmup=0,refit=False,
        frozen_backend_sha256=parent['binary_sha256'])
    queue=f'''#!/usr/bin/env bash
set -u
R={remote}
PARENT={parent['remote']}
P=/root/commitjl/.venv/bin/python
export PYTHONPATH="$PARENT/source" GOMAXPROCS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
echo "$(date -Is) WAIT_FOR_PARENT_PROOF_QUEUE" >> "$R/queue.log"
while tmux has-session -t {parent['session']} 2>/dev/null; do sleep 30; done
exec 9>/root/commitjl/results/review-proof-campaign.lock
flock 9
errors=0
for name in learned pca full; do
 input="$name"; mode=""
 if [ "$name" = full ]; then input=learned; mode=--full; fi
 echo "$(date -Is) START $name" >> "$R/queue.log"
 "$P" -m experiments.review_decision_controls --input "$R/$input.json" --out "$R/$name" --bin "$PARENT/bin/commitjl-bound-paired-linux" --batch 4 --reps 1 --warmup 0 --phase-hours 18 $mode > "$R/$name.log" 2>&1
 rc=$?
 echo "$(date -Is) END $name rc=$rc" >> "$R/queue.log"
 if [ "$rc" -ne 0 ]; then errors=$((errors+1)); fi
done
if [ "$errors" -eq 0 ]; then echo "$(date -Is) ALL_DONE" >> "$R/queue.log"; else echo "$(date -Is) FAILED_JOBS=$errors" >> "$R/queue.log"; fi
'''
    archive=local/'release.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for path in files:tar.add(path,arcname=path.name)
        for name,data in [('queue.sh',queue.encode()),('protocol.json',json.dumps(protocol,indent=2).encode())]:
            info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
    wsl=['wsl','-d','Ubuntu','--'];ssh=wsl+['ssh','-C','-o','BatchMode=yes','-o','ConnectTimeout=15','cloudfpsmobin']
    def remote_python(code):
        p=subprocess.run(ssh+['python3','-'],input=code.encode(),capture_output=True,timeout=90)
        if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-1200:])
        return p.stdout.decode()
    remote_python(f"from pathlib import Path\nr=Path({remote!r});r.mkdir(parents=True,exist_ok=False)\n")
    linux='/mnt/'+archive.drive[0].lower()+'/'+archive.as_posix().split(':',1)[1].lstrip('/')
    subprocess.run(wsl+['rsync','-a','--partial','--append-verify','--timeout=120','-e','ssh -o BatchMode=yes -o ConnectTimeout=15',linux,
                        'cloudfpsmobin:'+remote+'/release.tar.gz'],check=True,timeout=1200)
    session='review_scale_'+stamp
    print(remote_python(f'''from pathlib import Path
import tarfile,hashlib,subprocess
r=Path({remote!r});a=r/'release.tar.gz'
assert hashlib.sha256(a.read_bytes()).hexdigest()=={sha(archive)!r}
assert hashlib.sha256((Path({parent['remote']!r})/'bin/commitjl-bound-paired-linux').read_bytes()).hexdigest()=={parent['binary_sha256']!r}
with tarfile.open(a) as tar:tar.extractall(r,filter='data')
subprocess.run(['tmux','new-session','-d','-s',{session!r},'bash',str(r/'queue.sh')],check=True)
print('queued '+str(r))
'''),flush=True)
    protocol.update(session=session,archive_sha256=sha(archive),inputs=[dict(name=p.name,sha256=sha(p)) for p in files])
    out=ROOT/'code/validation/outputs/review_update_decisions_20261021T184553Z'/('scale_'+stamp+'.json')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(protocol,f,indent=2)
    print(out,flush=True)


if __name__=='__main__':main()
