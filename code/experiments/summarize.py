#!/usr/bin/env python
"""summarize.py -- aggregates results/*.parquet into headline numbers for
the abstract/contributions section: results/headline.json.

Usage: python -m experiments.summarize
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from experiments.common import RESULTS_DIR, capture_env
from experiments.make_tables import _read


def _safe_read(name: str):
    path = RESULTS_DIR / f"{name}.parquet"
    if not path.exists():
        return None
    return _read(name)


def main():
    headline = {"env": capture_env()}

    rq1 = _safe_read("rq1_fidelity")
    if rq1 is not None and len(rq1):
        ok = rq1[rq1.get("_status", "ok") == "ok"] if "_status" in rq1 else rq1
        headline["rq1_fidelity"] = {
            "n_rows": int(len(rq1)),
            "median_ari": float(ok["ari"].median()) if "ari" in ok else None,
            "median_r_phi": float(ok["r_phi"].median()) if "r_phi" in ok else None,
            "fraction_fidelity_ok": float(ok["fidelity_ok"].mean()) if "fidelity_ok" in ok else None,
        }

    rq2 = _safe_read("rq2_proof_cost")
    if rq2 is not None and len(rq2):
        ok = rq2[rq2.get("_status") == "ok"] if "_status" in rq2 else rq2
        headline["rq2_proof_cost"] = {
            "n_rows": int(len(rq2)),
            "n_proved_ok": int(len(ok)),
            "median_prove_seconds": float(ok["prove_seconds_mean"].median()) if "prove_seconds_mean" in ok and len(ok) else None,
            "median_reduction_pct": float(rq2["reduction_pct"].median()) if "reduction_pct" in rq2 else None,
        }

    rq3 = _safe_read("rq3_overhead")
    if rq3 is not None and len(rq3):
        ok = rq3[rq3.get("_status") == "ok"] if "_status" in rq3 else rq3
        headline["rq3_overhead"] = {
            "n_rows": int(len(rq3)),
            "median_total_time_s": float(ok["total_time_s"].median()) if "total_time_s" in ok and len(ok) else None,
        }

    rq3g = _safe_read("rq3_grinding")
    if rq3g is not None and len(rq3g):
        headline["rq3_grinding"] = {
            "n_rows": int(len(rq3g)),
            "fraction_bound_holds": float(rq3g["bound_holds"].mean()) if "bound_holds" in rq3g else None,
        }

    rq4 = _safe_read("rq4_certificates")
    if rq4 is not None and len(rq4):
        ok = rq4[rq4.get("_status") == "ok"] if "_status" in rq4 else rq4
        headline["rq4_certificates"] = {
            "n_rows": int(len(rq4)),
            "median_coverage": float(ok["coverage"].median()) if "coverage" in ok and len(ok) else None,
            "median_violation_rate": float(ok["violation_rate"].median()) if "violation_rate" in ok and len(ok) else None,
            "n_certified_violations": int((ok.get("violation_rate", pd.Series(dtype=float)) > 0).sum()) if len(ok) else 0,
        }

    rq5 = _safe_read("rq5_scaling")
    if rq5 is not None and len(rq5):
        headline["rq5_scaling"] = {"n_rows": int(len(rq5))}

    ablation = _safe_read("ablation")
    if ablation is not None and len(ablation):
        headline["ablation"] = {"n_rows": int(len(ablation))}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "headline.json"
    out_path.write_text(json.dumps(headline, indent=2, default=str))
    print(f"summarize: wrote {out_path}")


if __name__ == "__main__":
    main()
