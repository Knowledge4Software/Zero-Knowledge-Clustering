"""Local automatic downloads; SSH password is read once from stdin, never saved."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'code'))
from tools.server_connection_20261101 import connect,execute,REMOTE
import paramiko

DEST=REPO/'output/server_run_20261101/final_delivery'
SNAPSHOT=r'''
from pathlib import Path
import json,time
r=Path('/home/user01/zk-audit-20261101/code/validation/outputs/server_final_method_20261102')
x={'now':time.time(),'metadata':{},'downloads':[]}
for n in ['plan.json','plan_pin.json','cost_profiles.json','summary.json','results.md','matched_queue.json','failure.json','finished.json','smoke_complete.json','waiting.json','watchdog_recovery.json']:
 p=r/n
 if p.exists():x['metadata'][n]=p.read_text()
for p in (r/'cases').glob('*/result.json'):
 x['metadata'][str(p.relative_to(r))]=p.read_text()
for p in (r/'archives').glob('*.sha256.json'):
 v=json.loads(p.read_text());x['downloads'].append(dict(v,category='archives',remote=str(p.parent/v['file'])))
if (r/'backups/index.json').exists():
 idx=json.loads((r/'backups/index.json').read_text());x['metadata']['backups/index.json']=json.dumps(idx)
 for v in idx['files']:x['downloads'].append(dict(v,category='backups',remote=str(r/'backups'/v['file'])))
print(json.dumps(x))
'''


def file_sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def run(deadline):
    password=sys.stdin.readline().rstrip('\r\n')
    if not password:raise RuntimeError('Password must be supplied in memory on stdin')
    DEST.mkdir(parents=True,exist_ok=True)
    assert DEST.resolve().is_relative_to(REPO.resolve())
    verified={};last=None
    while time.time()<deadline:
        client=None
        try:
            client=connect(password)
            x=json.loads(execute(client,REMOTE+'/.venv/bin/python -',SNAPSHOT,timeout=55))
            for name,content in x['metadata'].items():
                p=DEST/'metadata'/name
                assert p.resolve().is_relative_to((DEST/'metadata').resolve())
                p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content,encoding='utf-8')
            sftp=paramiko.SFTPClient.from_transport(client.get_transport(),window_size=64*2**20,max_packet_size=32768)
            for item in sorted(x['downloads'],key=lambda v:(v['category']!='archives',v['file']!='final_source_and_plans.tar.gz',v['file'])):
                name=item['file'];assert Path(name).name==name and name.endswith('.tar.gz')
                out=DEST/item['category']/name;out.parent.mkdir(parents=True,exist_ok=True)
                if verified.get(str(out))==item['sha256']:continue
                if out.exists() and file_sha(out)==item['sha256']:
                    verified[str(out)]=item['sha256'];continue
                tmp=out.with_suffix(out.suffix+'.part')
                print(json.dumps(dict(event='download_started',file=name,bytes=item['bytes'],unix=time.time())),flush=True)
                sftp.get(item['remote'],str(tmp),prefetch=True,max_concurrent_prefetch_requests=64)
                if file_sha(tmp)!=item['sha256']:raise RuntimeError('Downloaded hash mismatch: '+name)
                tmp.replace(out);verified[str(out)]=item['sha256']
                print(json.dumps(dict(event='download_verified',file=name,bytes=item['bytes'],unix=time.time())),flush=True)
                # Refresh final results between potentially large legacy archives.
                current=json.loads(execute(client,REMOTE+'/.venv/bin/python -',SNAPSHOT,timeout=55))
                for m,c in current['metadata'].items():
                    q=DEST/'metadata'/m;assert q.resolve().is_relative_to((DEST/'metadata').resolve())
                    q.parent.mkdir(parents=True,exist_ok=True);q.write_text(c,encoding='utf-8')
            sftp.close();client.close();client=None
            state=dict(updated_unix=time.time(),verified=verified,download_deadline_unix=deadline)
            (DEST/'verified_downloads.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
            if last!=len(verified):
                print(json.dumps(dict(event='snapshot_saved',archives=len(verified),unix=time.time())),flush=True);last=len(verified)
        except Exception as error:
            print(json.dumps(dict(event='retry',error=str(error),unix=time.time())),flush=True)
        finally:
            if client:client.close()
        time.sleep(30)
    print(json.dumps(dict(event='download_window_finished',unix=time.time())),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--deadline',type=float,required=True);a=p.parse_args();run(a.deadline)
