"""An interrupted transfer must resume at the written prefix, not byte zero."""
from pathlib import Path
import os
import subprocess
import sys
import pytest

from tools.download_final_evidence_v2_20261102 import download_segment


class RemoteFile:
    def __init__(self,data,fail=False):self.data,self.fail=data,fail
    def __enter__(self):return self
    def __exit__(self,*_):pass
    def readv(self,chunks,max_concurrent_prefetch_requests):
        for i,(offset,length) in enumerate(chunks):
            if self.fail and i==1:raise IOError('simulated network drop')
            yield self.data[offset:offset+length]


class SFTP:
    def __init__(self,data,fail=False):self.data,self.fail=data,fail
    def open(self,*_):return RemoteFile(self.data,self.fail)


def test_resume_after_network_drop_preserves_every_byte(tmp_path):
    data=bytes(range(251))*1000;p=tmp_path/'archive.part'
    p.write_bytes(data[:12345])
    with pytest.raises(IOError):download_segment(SFTP(data,True),'remote',p,len(data))
    assert p.read_bytes()==data[:p.stat().st_size]
    assert p.stat().st_size==12345+32768
    while p.stat().st_size<len(data):download_segment(SFTP(data),'remote',p,len(data),limit=65536)
    assert p.read_bytes()==data


def test_reject_oversize_and_short_remote_reads(tmp_path):
    p=tmp_path/'archive.part';p.write_bytes(b'12345')
    with pytest.raises(ValueError):download_segment(SFTP(b'123'),'remote',p,3)
    p.unlink()
    with pytest.raises(IOError):download_segment(SFTP(b'12'),'remote',p,4)


def test_standalone_launcher_resolves_imports_without_pythonpath(tmp_path):
    script=Path(__file__).resolve().parents[1]/'tools/download_final_evidence_v2_20261102.py'
    env={k:v for k,v in os.environ.items() if k.upper()!='PYTHONPATH'}
    result=subprocess.run([sys.executable,str(script),'--help'],cwd=tmp_path,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
