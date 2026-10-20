"""Flush checkpoint contents before replacement; retain the previous manifest.

This reduces loss from interrupted writes. It cannot guarantee survival of
hardware failure or storage that ignores flush requests.
"""
import json
import os
from pathlib import Path


def _replace_bytes(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.durable.tmp')
    with temporary.open('wb') as stream:
        stream.write(payload);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def atomic_json(path,value):
    path=Path(path)
    payload=json.dumps(value,indent=2,allow_nan=False).encode('utf-8')
    if path.name=='manifest.json' and path.exists():
        previous=path.read_bytes()
        try:json.loads(previous)
        except (ValueError,UnicodeError):pass
        else:_replace_bytes(path.with_name('manifest.previous.json'),previous)
    _replace_bytes(path,payload)


def save_arrays(path,**arrays):
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.durable.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
