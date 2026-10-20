"""Read-only server snapshot for the 19 October reuse campaign review."""
import base64
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

REMOTE = r'''
import base64, datetime, hashlib, json, pathlib, subprocess
root = pathlib.Path('/root/commitjl')
files = []
for run in ['paper_runs_20261019', 'shift_owner_reuse_20261019',
            'new_center_reuse_20261018', 'new_center_hybrid_20261019']:
    for p in sorted((root/'results'/run).rglob('*')):
        if not p.is_file() or p.stat().st_size > 2_000_000:
            continue
        if p.suffix not in ('.json', '.csv', '.log', '.sh', '.txt'):
            continue
        if p.name in ('input.json', 'prepared.json'):
            continue
        # Existing native jobs already have a preserved byte-checked local copy.
        # Capture their small protocols/summaries without retransferring all arrays.
        if run != 'paper_runs_20261019' and run != 'shift_owner_reuse_20261019' and 'jobs' in p.parts:
            continue
        data = p.read_bytes()
        files.append(dict(path=str(p.relative_to(root)), sha256=hashlib.sha256(data).hexdigest(),
                          data=base64.b64encode(data).decode()))
for name in ['experiments/reuse_workload.py', 'experiments/shift_owner_reuse.py',
             'experiments/new_center_hybrid.py', 'experiments/protected_fixed_center.py',
             'commitjl/enriched_jl.py', 'zk/circuit/bound_reuse.go',
             'zk/circuit/bound_reuse_test.go', 'zk/cmd/commitjl-bound/reuse.go',
             'zk/cmd/commitjl-bound/main.go', 'zk/circuit/bound_model.go']:
    p = root/name
    if not p.exists() and name.startswith('zk/'):
        p = root/'code'/name
    if p.exists():
        data = p.read_bytes()
        files.append(dict(path='source/'+name, sha256=hashlib.sha256(data).hexdigest(),
                          data=base64.b64encode(data).decode()))
commands = {}
for name, cmd in [('processes', ['ps', '-eo', 'pid,etime,pcpu,pmem,args', '--sort=-pcpu']),
                  ('sessions', ['tmux', 'ls']), ('machine', ['uname', '-a']),
                  ('memory', ['free', '-h'])]:
    r = subprocess.run(cmd, capture_output=True, text=True)
    commands[name] = '\n'.join(r.stdout.splitlines()[:12])
binary = root/'zk/bin/commitjl-bound-reuse-linux'
print(json.dumps(dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      commands=commands, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                      files=files)))
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path(__file__).parent/'outputs'/
                    ('server_review_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    args = ap.parse_args()
    dest = args.out
    # Each observation is immutable evidence. Never replace a previous capture.
    dest.mkdir(parents=True, exist_ok=False)
    r = subprocess.run(['wsl', '-d', 'Ubuntu', '--', 'ssh', '-o', 'BatchMode=yes',
                        '-o', 'ConnectTimeout=15', 'cloudfpsmobin', 'python3', '-'],
                       input=REMOTE.encode(), capture_output=True, timeout=300)
    if r.returncode:
        raise RuntimeError(r.stderr.decode(errors='replace'))
    snapshot = json.loads(r.stdout)
    for item in snapshot['files']:
        data = base64.b64decode(item.pop('data'))
        assert hashlib.sha256(data).hexdigest() == item['sha256']
        path = dest/item['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        item['bytes'] = len(data)
    (dest/'manifest.json').write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
    print(json.dumps(dict(dest=str(dest), files=len(snapshot['files']),
                          observed_utc=snapshot['observed_utc'], commands=snapshot['commands']), indent=2))


if __name__ == '__main__':
    main()
