"""Record non-secret benchmark hardware/software metadata without changing it."""
import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    versions={}
    for name in ['numpy','scipy','scikit-learn','pandas','psutil','threadpoolctl','pytest']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    models=[]
    cpu=Path('/proc/cpuinfo')
    if cpu.exists():
        models=sorted({line.split(':',1)[1].strip() for line in cpu.read_text().splitlines()
                       if line.startswith('model name')})
    result=dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        python=sys.version,executable=sys.executable,platform=platform.platform(),
        machine=platform.machine(),logical_cpu_count=os.cpu_count(),cpu_models=models,
        allowed_cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        package_versions=versions,
        note='Read-only metadata. Per-attempt receipts hold actual proof worker/affinity settings and process-tree RSS.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(recorded=True,output=str(args.output))))


if __name__=='__main__':main()
