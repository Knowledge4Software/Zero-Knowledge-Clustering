"""Launch supplements into their respective RQ run directories."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

OUTPUTS = {
    "RQ3": "rq3/runs/paired",
    "RQ4": "rq4/runs/real",
    "RQ5Gamma": "rq5/runs/gamma",
    "RQ5Precision": "rq5/runs/precision_24ref",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("All", *OUTPUTS), default="All")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, help="development smoke-test rows")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    if args.stage == "All" and args.out_dir is not None:
        parser.error("--out-dir requires one stage; All uses per-RQ directories")

    # Set limits before NumPy imports and before Windows workers spawn.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    from .engine import load_prereg, run_stage

    results = Path(__file__).resolve().parents[2] / "results"
    prereg = load_prereg()
    stages = OUTPUTS if args.stage == "All" else (args.stage,)
    for stage in stages:
        output = args.out_dir if args.out_dir is not None else results / OUTPUTS[stage]
        run_stage(stage, output, prereg, args.workers, args.limit)


if __name__ == "__main__":
    main()
