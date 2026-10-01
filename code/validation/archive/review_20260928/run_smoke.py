"""Run an isolated regression campaign; these outputs are not paper results."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[2]
OUTPUT = HERE / "smoke_v2"
OUTPUT.mkdir(exist_ok=True)
env = {**os.environ, "COMMITJL_RESULTS_DIR": str(OUTPUT), "OMP_NUM_THREADS": "1",
       "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
commands = [
    ("rq1_fidelity", []), ("rq2_proof_cost", []), ("rq3_overhead", []),
    ("rq3_grinding", ["--n-trials", "8", "--Q-list", "1,2,4,8", "--workers", "2"]),
    ("rq4_certificates", ["--phase-diagram"]), ("rq5_scaling", []), ("ablation", []),
]
records = []
for name, extra in commands:
    command = [sys.executable, "-m", "experiments." + name, "--config", "experiments/configs/smoke.yaml", *extra]
    start = time.perf_counter()
    result = subprocess.run(command, cwd=CODE, env=env, capture_output=True, text=True)
    (OUTPUT / (name + ".log")).write_text(result.stdout + result.stderr, encoding="utf-8")
    records.append({"module": name, "returncode": result.returncode, "seconds": time.perf_counter()-start})
    (OUTPUT / "smoke_commands.json").write_text(json.dumps(records, indent=2))
    print(records[-1], flush=True)
raise SystemExit(any(record["returncode"] for record in records))
