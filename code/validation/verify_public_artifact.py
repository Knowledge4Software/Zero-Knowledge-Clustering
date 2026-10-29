"""Verify staged artifact hashes and freshly replay its public proof bundles."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,shutil,subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--bin',type=Path);p.add_argument('--hashes-only',action='store_true');a=p.parse_args()
    root=a.root.resolve();manifest=json.loads((root/'artifact_manifest.json').read_text())
    def inside(name):
        path=(root/name).resolve()
        if not path.is_relative_to(root):raise ValueError('artifact path escaped root')
        return path
    for e in manifest['files']:
        if hashlib.sha256(inside(e['path']).read_bytes()).hexdigest()!=e['sha256']:raise ValueError('hash mismatch: '+e['path'])
    if a.hashes_only:
        print(json.dumps(dict(files_checked=len(manifest['files']),status='hashes_pass')));return
    binary=a.bin.resolve() if a.bin else root/'bin'/('commitjl-bound.exe' if os.name=='nt' else 'commitjl-bound-linux')
    if os.name!='nt':binary.chmod(0o700)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output=root/'verification_runs'/stamp;output.mkdir(parents=True,exist_ok=False);records=[]
    for case in manifest['public_cases']:
        target=output/case['name'];shutil.copytree(inside(case['bundle']),target)
        cmd=[str(binary),'-phase','verify','-out',str(target),'-trust',str(inside(case['pin'])),'-public-task',str(inside(case['task']))]
        with (output/(case['name']+'.log')).open('w') as f:
            r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'GOMAXPROCS':'4'})
        data=json.loads((target/'verify.json').read_text())
        if r.returncode or not data.get('complete'):raise RuntimeError('public verification failed: '+case['name'])
        records.append(dict(case=case['name'],verified=data['verified'],complete=True))
        print(json.dumps(records[-1]),flush=True)
    receipt=dict(status='PASS',files_checked=len(manifest['files']),public_cases=len(records),
        freshly_verified_proofs=sum(r['verified'] for r in records),records=records,
        trust='Artifact pins represent the experimental data owner/setup. Real deployments obtain authentic pins/tasks independently of the prover.')
    (output/'verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
