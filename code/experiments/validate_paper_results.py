"""Reject incomplete, failed, or mixed-source paper campaign inputs."""
from __future__ import annotations

import pandas as pd

from experiments.common import RESULTS_DIR, source_fingerprint


EXPECTED = {
    "rq1_fidelity": 7180,
    "rq3_overhead": 90,
    "rq3_grinding": 540,
    "rq4_certificates": 265,
    "rq5_scaling": 200,
    "rq5_precision": 120,
}


def main():
    fingerprint = source_fingerprint()
    for name in (*EXPECTED, "rq2_proof_cost", "ablation"):
        path = RESULTS_DIR / f"{name}.parquet"
        df = pd.read_parquet(path)
        if df.empty:
            raise ValueError(f"{name}: no measured rows")
        if name in EXPECTED and len(df) != EXPECTED[name]:
            raise ValueError(f"{name}: expected {EXPECTED[name]} rows, found {len(df)}")
        allowed = {"ok", "native_only"} if name == "ablation" else {"ok"}
        if not df["_status"].isin(allowed).all():
            raise ValueError(f"{name}: contains unsuccessful rows: {df['_status'].value_counts().to_dict()}")
        if not df["_source_sha256"].eq(fingerprint).all():
            raise ValueError(f"{name}: mixed or stale source fingerprints")
        if name in EXPECTED:
            if set(df["seed"].dropna().astype(int)) != set(range(10)):
                raise ValueError(f"{name}: ten-seed coverage is incomplete")
        print(f"PASS {name}: {len(df)} measured rows")


if __name__ == "__main__":
    main()
