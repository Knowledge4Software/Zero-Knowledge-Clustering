"""Resumable, prioritized private downloads; password stays in process memory."""
from pathlib import Path
import argparse
import json
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'code'))
from tools.download_final_evidence_20261102 import DEST,SNAPSHOT,file_sha
from tools.server_connection_20261101 import connect,execute,REMOTE
import paramiko


def download_segment(sftp,remote,partial,total,limit=16*2**20):
    start=partial.stat().st_size if partial.exists() else 0
    if start>total:raise ValueError('Partial file exceeds immutable archive length')
    length=min(limit,total-start)
    if not length:return start
    chunks=[(i,min(32768,start+length-i)) for i in range(start,start+length,32768)]
    with sftp.open(remote,'rb') as src,partial.open('ab') as dest:
        for data,(_,expected) in zip(src.readv(chunks,max_concurrent_prefetch_requests=64),chunks):
            if len(data)!=expected:raise IOError('Short read from immutable archive')
            dest.write(data)
    return partial.stat().st_size


def write_metadata(snapshot):
    for name,content in snapshot['metadata'].items():
        out=DEST/'metadata'/name
        assert out.resolve().is_relative_to((DEST/'metadata').resolve())
        out.parent.mkdir(parents=True,exist_ok=True)
        temporary=out.with_suffix(out.suffix+'.new');temporary.write_text(content,encoding='utf-8');temporary.replace(out)


def run(deadline):
    password=sys.stdin.readline().rstrip('\r\n')
    if not password:raise RuntimeError('Password must arrive on stdin, not in arguments or files')
    DEST.mkdir(parents=True,exist_ok=True);verified={}
    while time.time()<deadline:
        client=None;selected=None
        try:
            client=connect(password)
            snapshot=json.loads(execute(client,REMOTE+'/.venv/bin/python -',SNAPSHOT,timeout=55))
            write_metadata(snapshot)
            selected=None
            for item in sorted(snapshot['downloads'],key=lambda x:(x['category']!='archives',x['bytes'],x['file'])):
                name=item['file'];assert Path(name).name==name and name.endswith('.tar.gz')
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
                sftp=paramiko.SFTPClient.from_transport(client.get_transport(),window_size=64*2**20,max_packet_size=32768)
                size=download_segment(sftp,item['remote'],partial,item['bytes']);sftp.close()
                if size==item['bytes']:
                    if file_sha(partial)!=item['sha256']:
                        partial.replace(partial.with_name(partial.name+'.bad.'+str(int(time.time()))))
                        raise RuntimeError('Archive hash mismatch; quarantined corrupt partial: '+item['file'])
                    partial.replace(out);verified[str(out)]=item['sha256']
                    print(json.dumps(dict(event='download_verified',file=item['file'],bytes=size,unix=time.time())),flush=True)
                else:print(json.dumps(dict(event='partial_saved',file=item['file'],bytes=size,total=item['bytes'],unix=time.time())),flush=True)
            (DEST/'verified_downloads.json').write_text(json.dumps(dict(updated_unix=time.time(),verified=verified,
                download_deadline_unix=deadline),indent=2),encoding='utf-8')
        except Exception as error:
            print(json.dumps(dict(event='retry_with_partial_preserved',error=str(error),unix=time.time())),flush=True)
        finally:
            if client:client.close()
        time.sleep(2 if selected else 30)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--deadline',type=float,required=True)
    args=parser.parse_args();run(args.deadline)
