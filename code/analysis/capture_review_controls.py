"""Read completed Mobin review controls into a fresh, hashed local snapshot.

Never writes to the server. This is metadata/timing evidence, not a substitute
for the independent cryptographic replay already required by the worker.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--remote', required=True)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--include-public', action='store_true', help='also transfer public verification keys and proofs')
    a = p.parse_args()
    if not a.remote.startswith('/root/commitjl/results/review_'):
        raise ValueError('unexpected evidence root')
    script = '''
import base64,hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); records=[]; cases=[]
for summary in sorted(root.glob('*/summary.json')):
 case=summary.parent
 status=json.loads((case/'status.json').read_text())
 if not status.get('complete'):continue
 cases.append(case.name)
 paths=[case/x for x in ['summary.json','status.json','protocol.json','workload.csv','approved_public_task.json','owner_setup_pin.json']]
 paths+=list((case/'bundle').glob('*.json'))
 paths+=[case/'public_bundle'/'verify.json']
 if len(sys.argv)>2 and sys.argv[2]=='public':
  paths += [p for p in (case/'public_bundle').rglob('*') if p.is_file() and p.suffix in {'.json','.vk','.proof'}]
 # input/row witness files are deliberately omitted; phase samples can contain
 # private diagnostic assignments, so keep this evidence separate from exports.
 paths=[x for x in paths if x.is_file() and x.name!='input.json']
 for path in sorted(set(paths)):
  data=path.read_bytes()
  records.append(dict(path=path.relative_to(root).as_posix(),sha256=hashlib.sha256(data).hexdigest(),data=base64.b64encode(data).decode()))
print(json.dumps(dict(remote=str(root),completed_cases=cases,files=records)))
'''
    command = 'python3 -c ' + shlex.quote(script) + ' ' + shlex.quote(a.remote) + (' public' if a.include_public else '')
    for attempt in range(3):
        result = subprocess.run(['wsl','-d','Ubuntu','--','ssh','-C','-o','BatchMode=yes',
                                 '-o','ConnectTimeout=15','cloudfpsmobin',command], capture_output=True)
        if result.returncode == 0:
            break
        if attempt == 2:
            raise RuntimeError(result.stderr.decode(errors='replace')[-1500:])
        time.sleep(2)
    snapshot = json.loads(result.stdout)
    a.out.mkdir(parents=True, exist_ok=False)
    for record in snapshot['files']:
        data = base64.b64decode(record.pop('data'))
        if hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError('transfer hash mismatch')
        target = a.out / record['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    snapshot['captured_unix'] = time.time()
    snapshot['all_transferred_hashes_valid'] = True
    (a.out/'capture.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
    print(json.dumps(dict(out=str(a.out),cases=snapshot['completed_cases'],files=len(snapshot['files']))))


if __name__ == '__main__':
    main()
