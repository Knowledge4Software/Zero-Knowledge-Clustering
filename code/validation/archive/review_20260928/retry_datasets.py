"""Retry only failed network loads, preserving the original check results."""
import json
from pathlib import Path
import socket
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
socket.setdefaulttimeout(30)
from experiments.pipeline import load_row_dataset

records=[]
for previous in json.loads((HERE/"datasets.json").read_text()):
    if previous["status"]=="PASS":
        continue
    start=time.perf_counter()
    result={"specification":previous["specification"]}
    try:
        data=load_row_dataset({"dataset":previous["specification"],"seed":0})
        result.update({"status":"PASS","shape":list(data["X"].shape),"metadata":data["meta"]})
    except Exception as exc:
        result.update({"status":"FAIL","error":repr(exc)})
    result["seconds"]=time.perf_counter()-start
    records.append(result)
    (HERE/"dataset_retries.json").write_text(json.dumps(records,indent=2))
    print(json.dumps(result),flush=True)
raise SystemExit(any(r["status"]!="PASS" for r in records))
