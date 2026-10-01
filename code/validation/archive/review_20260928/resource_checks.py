"""Measured memory and setup-reuse checks; no analytic estimates as results."""
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[2]
sys.path.insert(0, str(CODE))

if "--worker" in sys.argv:
    from experiments.rq4_certificates import run_dbscan_cert
    d = int(sys.argv[-1])
    row = {"dataset":{"loader":"gaussian_mixture","kwargs":{"n":1000,"d":d,"k":5}},
           "projection":"sparse_jl","m":32,"q":4,"seed":0}
    start = time.perf_counter()
    result = run_dbscan_cert(row,{"eps":0.15})
    print(json.dumps({"n":1000,"d":d,"pairs_evaluated":result["pairs_evaluated"],
                      "seconds":time.perf_counter()-start}))
    raise SystemExit(0)

import psutil
records=[]
for d in (1024,10000):
    process=subprocess.Popen([sys.executable,__file__,"--worker",str(d)], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,text=True)
    observer=psutil.Process(process.pid)
    peak=0
    while process.poll() is None:
        try:
            peak=max(peak,observer.memory_info().rss)
        except psutil.Error:
            pass
        time.sleep(0.01)
    stdout,stderr=process.communicate()
    if process.returncode:
        raise RuntimeError(stderr)
    record=json.loads(stdout)
    record.update({"peak_rss_bytes":peak,"under_3GB":peak<3_000_000_000})
    records.append(record)
    (HERE/"rq4_memory.json").write_text(json.dumps(records,indent=2))
    print(record,flush=True)

import numpy as np
from commitjl.zk_bridge import build_sparse_instance,run_prove
rng=np.random.default_rng(43)
instance=build_sparse_instance(rng.integers(-8,9,(4,8)),rng.integers(-8,9,(2,8)),
                               seed=b"setup-reuse-review",q=2,m=4)
start=time.perf_counter()
reused=run_prove(instance,repetitions=5,timeout=180)
reused_wall=time.perf_counter()-start
separate=[]
start=time.perf_counter()
for _ in range(5):
    separate.append(run_prove(instance,timeout=180))
separate_wall=time.perf_counter()-start
record={"shape":{"n":4,"d":8,"m":4,"k":2,"q":2},"backend":"groth16",
        "reused_wall_seconds":reused_wall,"separate_wall_seconds":separate_wall,
        "measured_speedup":separate_wall/reused_wall,"reused":reused.raw,
        "separate":[result.raw for result in separate],
        "all_verified":reused.ok and reused.successful_proofs==5 and all(result.ok for result in separate)}
(HERE/"setup_reuse.json").write_text(json.dumps(record,indent=2))
print({k:v for k,v in record.items() if k not in {"reused","separate"}},flush=True)
raise SystemExit(not record["all_verified"] or any(not r["under_3GB"] for r in records))
