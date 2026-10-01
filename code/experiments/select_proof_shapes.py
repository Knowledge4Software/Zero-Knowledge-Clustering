"""Select genuine reduction shapes from completed RQ1 rows for RQ2/ablation.

Selection is based only on preregistered native fidelity, before proof cost is
observed. The selected source dataset, n, d, k, m and q are retained exactly.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import yaml

from experiments.common import load_prereg, source_fingerprint


def clean_json_value(value):
    if isinstance(value, dict):
        return {k: clean_json_value(v) for k, v in value.items()
                if v is not None and not (isinstance(v, float) and math.isnan(v))}
    if isinstance(value, list):
        return [clean_json_value(v) for v in value]
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return value


def select_shapes(df: pd.DataFrame, prereg: dict, max_shapes: int = 2) -> list[dict]:
    if max_shapes < 1:
        raise ValueError("max_shapes must be positive")
    ok = df[df["_status"].eq("ok") & df["projection"].eq("sparse_jl") & (df["m"] < df["d"])].copy()
    groups = ["dataset_name", "dataset_sha256", "n", "d", "k", "m", "q"]
    summary = ok.groupby(groups, dropna=False).agg(
        repetitions=("seed", "nunique"), ari_median=("ari", "median"),
        r_phi_median=("r_phi", "median"), fidelity_rate=("fidelity_ok", "mean"),
    ).reset_index()
    required_reps = int(prereg.get("n_reps_native", 10))
    eligible = summary[(summary.repetitions == required_reps) &
                       (summary.ari_median >= float(prereg["tau_ARI"])) &
                       ((summary.r_phi_median - 1).abs() <= float(prereg["tau_Phi"]))]
    eligible = eligible.sort_values(["dataset_name", "d", "k", "m", "q"])
    selected = eligible.groupby(["dataset_name", "dataset_sha256", "d", "k"], as_index=False).first()
    selected = selected.assign(input_elements=selected.n * selected.d)
    selected = selected.sort_values(["input_elements", "m", "q"]).head(max_shapes)
    rows = []
    for _, choice in selected.iterrows():
        matched = ok[(ok.dataset_name == choice.dataset_name) &
                     (ok.dataset_sha256 == choice.dataset_sha256) &
                     (ok.d == choice.d) & (ok.k == choice.k) &
                     (ok.m == choice.m) & (ok.q == choice.q)]
        source = matched.sort_values("seed").iloc[0]
        rows.append({
            "dataset": clean_json_value(source["dataset"]),
            "k": int(choice.k), "m": int(choice.m), "q": int(choice.q),
            "seed": 0, "precision_bits": int(source.precision_bits),
            "B": int(source.B), "backend": "groth16",
            "selection_ari_median": float(choice.ari_median),
            "selection_r_phi_median": float(choice.r_phi_median),
            "selection_fidelity_rate": float(choice.fidelity_rate),
            "selection_dataset_sha256": str(choice.dataset_sha256),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rq1", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-shapes", type=int, default=2)
    args = ap.parse_args()
    df = pd.read_parquet(args.rq1)
    if len(df) != 7180 or not df["_status"].eq("ok").all():
        raise ValueError("RQ1 must be complete and successful before selecting proof shapes")
    if not df["_source_sha256"].eq(source_fingerprint()).all():
        raise ValueError("RQ1 source fingerprint differs from current code")
    prereg = load_prereg()
    selected = select_shapes(df, prereg, args.max_shapes)
    if not selected:
        raise ValueError("no fidelity-eligible SparseJL shape with m<d in RQ1")
    # One fixed proof witness per selected shape, with ten repeated proofs and
    # one warmup on a single compiled/setup circuit. There is no shape cut.
    config = {"fixed": {"n_reps_proof": int(prereg.get("n_reps_proof", 10)),
                         "n_warmup_proof": int(prereg.get("n_warmup_proof", 1))},
              "blocks": [{"fixed": {k: v for k, v in row.items()
                                     if not k.startswith("selection_") or k == "selection_dataset_sha256"},
                          "grid": {"variant": ["sparse"]}} for row in selected]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest = {"selection_rule": "median ARI>=tau_ARI and |median r_phi-1|<=tau_Phi; smallest m then q per dataset/k; m<d",
                "n_shapes": len(selected), "shapes": selected,
                "warning": "Full RQ1 n and d are retained. These gnark shapes can exceed laptop RAM; proof watchdogs record resource failures."}
    args.out.with_suffix(".selection.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"selected {len(selected)} proof shapes: {args.out}")


if __name__ == "__main__":
    main()
