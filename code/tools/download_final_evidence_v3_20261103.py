"""Bound SSH/SFTP waits and compress metadata; resume the existing downloads."""
from pathlib import Path
import argparse
import base64
import gzip
import json
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.server_connection_20261101 import connect,execute,REMOTE
from tools.download_final_evidence_20261102 import DEST,file_sha
from tools.download_final_evidence_v2_20261102 import download_segment,write_metadata
import paramiko


SNAPSHOT = r'''
from pathlib import Path
import base64,gzip,json,time
r=Path('/home/user01/zk-audit-20261101/code/validation/outputs/server_final_method_20261102')
x={'now':time.time(),'metadata':{},'downloads':[]}
for n in ['summary.json','results.md','matched_queue.json','failure.json','finished.json','seal_status.json']:
 p=r/n
 if p.exists():x['metadata'][n]=p.read_text()
for p in (r/'cases').glob('*/result.json'):
 x['metadata'][str(p.relative_to(r))]=p.read_text()
s=r.parent/'paper_completion_20261102'
for n in ['waiting.json','matched_queue.json','failure.json','finished.json','sealed.json','reports/coverage.json','reports/coverage.md']:
 p=s/n
 if p.exists():x['metadata']['paper_completion/'+n]=p.read_text()
for p in (r/'archives').glob('*.sha256.json'):
 v=json.loads(p.read_text());x['downloads'].append(dict(v,category='archives',remote=str(p.parent/v['file'])))
if (r/'backups/index.json').exists():
 idx=json.loads((r/'backups/index.json').read_text());x['metadata']['backups/index.json']=json.dumps(idx)
 for v in idx['files']:x['downloads'].append(dict(v,category='backups',remote=str(r/'backups'/v['file'])))
print(base64.b64encode(gzip.compress(json.dumps(x).encode())).decode())
'''


def decode_snapshot(value):
    result=json.loads(gzip.decompress(base64.b64decode(value.strip(),validate=True)))
    if not isinstance(result.get('metadata'),dict) or not isinstance(result.get('downloads'),list):
        raise ValueError('Incomplete metadata snapshot')
    return result


def configure_transport(client):
    transport=client.get_transport()
    transport.channel_timeout=45
    transport.set_keepalive(15)
    return transport


def run(deadline):
    password=sys.stdin.readline().rstrip('\r\n')
    if not password:raise RuntimeError('Password must arrive on stdin')
    DEST.mkdir(parents=True,exist_ok=True);verified={}
    while time.time()<deadline:
        client=None;selected=None;sftp=None
        try:
            client=connect(password);transport=configure_transport(client)
            snapshot=decode_snapshot(execute(client,REMOTE+'/.venv/bin/python -',SNAPSHOT,timeout=50))
            write_metadata(snapshot)
            for item in sorted(snapshot['downloads'],key=lambda x:(x['category']!='archives',x['bytes'],x['file'])):
                name=item['file']
                assert Path(name).name==name and name.endswith('.tar.gz')
                assert item['category'] in ['archives','backups']
                out=DEST/item['category']/name;out.parent.mkdir(parents=True,exist_ok=True)
                assert out.resolve().is_relative_to(DEST.resolve())
                if verified.get(str(out))==item['sha256']:continue
                if out.exists() and file_sha(out)==item['sha256']:
                    verified[str(out)]=item['sha256'];continue
                selected=(item,out);break
            if selected:
                item,out=selected;partial=out.with_suffix(out.suffix+'.part')
                if partial.exists() and partial.stat().st_size>item['bytes']:
                    partial.replace(partial.with_name(partial.name+'.oversize.'+str(int(time.time()))))
                sftp=paramiko.SFTPClient.from_transport(transport,window_size=64*2**20,max_packet_size=32768)
                sftp.get_channel().settimeout(60)
                size=download_segment(sftp,item['remote'],partial,item['bytes']);sftp.close();sftp=None
                if size==item['bytes']:
                    if file_sha(partial)!=item['sha256']:
                        partial.replace(partial.with_name(partial.name+'.bad.'+str(int(time.time()))))
                        raise RuntimeError('Archive SHA256 mismatch; partial quarantined')
                    partial.replace(out);verified[str(out)]=item['sha256']
                    print(json.dumps(dict(event='download_verified',file=item['file'],bytes=size,unix=time.time())),flush=True)
                else:print(json.dumps(dict(event='partial_saved',file=item['file'],bytes=size,total=item['bytes'],unix=time.time())),flush=True)
            state=dict(updated_unix=time.time(),verified=verified,download_deadline_unix=deadline,
                       downloader_version=3,remote_snapshot_unix=snapshot['now'])
            temp=DEST/'verified_downloads.json.new';temp.write_text(json.dumps(state,indent=2),encoding='utf-8');temp.replace(DEST/'verified_downloads.json')
        except Exception as error:
            print(json.dumps(dict(event='retry_with_partial_preserved',error_type=type(error).__name__,error=str(error)[:240],unix=time.time())),flush=True)
        finally:
            if sftp:sftp.close()
            if client:client.close()
        time.sleep(2 if selected else 30)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--deadline',type=float,required=True)
    run(p.parse_args().deadline)
