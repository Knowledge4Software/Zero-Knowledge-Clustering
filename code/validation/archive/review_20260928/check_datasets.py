"""Load configured data and record actual dimensions, finite values and hashes."""
import gc
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[2]
sys.path.insert(0, str(CODE))
import numpy as np
from experiments.common import load_config
from experiments.pipeline import load_row_dataset

records = []
for spec in load_config(CODE / "experiments/configs/native.yaml")["grid"]["dataset"]:
    start = time.perf_counter()
    record = {"specification": spec}
    try:
        data = load_row_dataset({"dataset":spec,"seed":0})
        X = data["X"]
        record.update({"status":"PASS", "shape":list(X.shape), "finite":bool(np.isfinite(X).all()),
                       "metadata":data["meta"], "seconds":time.perf_counter()-start})
        if not record["finite"] or X.shape != (data["meta"]["n"],data["meta"]["d"]):
            record["status"] = "FAIL"
        del X, data
        gc.collect()
    except Exception as exc:
        record.update({"status":"FAIL", "error":repr(exc)})
    records.append(record)
    (HERE / "datasets.json").write_text(json.dumps(records,indent=2))
    print(spec["loader"],record["status"],record.get("shape"),flush=True)
raise SystemExit(any(r["status"]!="PASS" for r in records))
