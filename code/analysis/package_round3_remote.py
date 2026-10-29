"""Create a PRIVATE, unexecuted Mobin research packet; no network writes."""
from pathlib import Path
import hashlib,json,zipfile
from campaigns.local_fast_tradeoff import save,sha

REPO=Path(__file__).resolve().parents[2]


def main():
    policy=json.loads((REPO/'code/validation/outputs/referee_round2_20261026/execution_policy.json').read_text())
    if policy.get('mobin_server_retired',False):
        raise RuntimeError('Mobin is retired; preserve the archived packet and do not create a new Mobin deployment package')
    files=set()
    for folder in ['analysis','campaigns','commitjl','data','experiments']:
        files.update((REPO/'code'/folder).rglob('*.py'))
    files.update((REPO/'code/validation').glob('*.py'))
    for folder in ['cmd','circuit']:files.update((REPO/'code/zk'/folder).rglob('*.go'))
    for name in ['go.mod','go.sum']:files.add(REPO/'code/zk'/name)
    for name in ['requirements.txt','code/requirements.txt']:
        if (REPO/name).exists():files.add(REPO/name)
    root=REPO/'code/validation/outputs/referee_round3_20261026'
    files.update((root/'remote_controls_v2').rglob('*.json'))
    files.add(root/'extended_study_plan.json')
    files.add(REPO/'code/validation/outputs/referee_round2_20261026/execution_policy.json')
    files.add(REPO/'output/referee_round3_remote_instructions.md')
    data={p.relative_to(REPO).as_posix():p.read_bytes() for p in sorted(files)}
    assert not any(name.endswith(('.pk','.proof','.gob')) for name in data)
    manifest={name:hashlib.sha256(value).hexdigest() for name,value in data.items()}
    data['round3_packet_manifest.json']=json.dumps(manifest,indent=2).encode()
    target=REPO/'output/referee_round3_mobin_private_20261026.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for name,value in data.items():z.writestr(name,value)
    with zipfile.ZipFile(target) as z:assert all(hashlib.sha256(z.read(name)).hexdigest()==digest for name,digest in manifest.items())
    receipt=dict(complete=True,packet=target.relative_to(REPO).as_posix(),sha256=sha(target),bytes=target.stat().st_size,manifest_files=len(manifest),uploaded=False,remote_jobs_started=False,private_material=True)
    save(root/'packet_receipt.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
