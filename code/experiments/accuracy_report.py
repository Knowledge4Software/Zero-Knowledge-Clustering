"""Summarize a SparseJL RQ1 pilot into paper-reviewable fidelity evidence."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import load_prereg, source_fingerprint
from experiments.statistics import summarize_repetitions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--prereg", default=None)
    args = parser.parse_args()
    prereg = load_prereg(args.prereg)
    df = pd.read_parquet(args.results)
    if df.empty or not df.get("_status", pd.Series(dtype=str)).eq("ok").all():
        raise ValueError("accuracy report requires nonempty, entirely successful RQ1 results")
    if "_source_sha256" not in df or not df["_source_sha256"].eq(source_fingerprint()).all():
        raise ValueError("results were produced by different or unrecorded source code")

    keys = ["dataset_name", "dataset_sha256", "n", "d", "k", "m", "q"]
    summary = df.groupby(keys, dropna=False).agg(
        repetitions=("seed", "nunique"),
        ari_vs_full_median=("ari", "median"),
        ari_vs_full_q25=("ari", lambda x: x.quantile(.25)),
        ari_vs_full_q75=("ari", lambda x: x.quantile(.75)),
        r_phi_median=("r_phi", "median"),
        distortion_q95_median=("delta_q95", "median"),
        distortion_exceedance_median=("delta_exceedance", "median"),
        ground_truth_ari_full=("ari_ground_truth_full", "median"),
        ground_truth_ari_projected=("ari_ground_truth_projected", "median"),
    ).reset_index()
    ci = summarize_repetitions(df, keys, metrics=("ari", "delta_q95", "r_phi"))
    summary = summary.merge(ci.drop(columns=["attempted_rows", "repetitions"]), on=keys,
                            how="left", validate="one_to_one")
    summary["fidelity_eligible"] = (
        (summary.ari_vs_full_median >= float(prereg["tau_ARI"])) &
        ((summary.r_phi_median - 1.0).abs() <= float(prereg["tau_Phi"]))
    )

    selected = (summary[summary.fidelity_eligible]
                .sort_values(["dataset_name", "d", "k", "m", "q"])
                .groupby(["dataset_name", "dataset_sha256", "d", "k"], as_index=False)
                .first())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "accuracy_all_configurations.csv", index=False)
    selected.to_csv(args.out_dir / "accuracy_smallest_eligible_m.csv", index=False)
    ci.to_csv(args.out_dir / "accuracy_mean_std_ci95.csv", index=False)

    columns = ["dataset_name", "n", "d", "k", "m", "q", "repetitions",
               "ari_mean", "ari_std", "ari_ci95_low", "ari_ci95_high",
               "delta_q95_mean", "delta_q95_std", "delta_q95_ci95_low", "delta_q95_ci95_high",
               "r_phi_mean", "r_phi_std", "r_phi_ci95_low", "r_phi_ci95_high",
               "ground_truth_ari_full", "ground_truth_ari_projected"]
    if selected.empty:
        markdown = "No configuration met both preregistered fidelity thresholds.\n"
    else:
        shown = selected[columns]
        markdown = "| " + " | ".join(columns) + " |\n"
        markdown += "| " + " | ".join("---" for _ in columns) + " |\n"
        for _, row in shown.iterrows():
            values = []
            for column in columns:
                value = row[column]
                values.append(f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value))
            markdown += "| " + " | ".join(values) + " |\n"
    note = ("\nMeans, sample standard deviations, and two-sided 95% Student-t confidence intervals "
            "are computed over paired seeds for each configuration. "
            "ARI vs full measures whether SparseJL preserves the full-space clustering. "
            "Ground-truth ARI measures clustering quality where labels exist; it is not available "
            "for the DBSCAN-margin generator. This pilot uses five seeds and is preliminary.\n")
    (args.out_dir / "ACCURACY_PILOT.md").write_text(markdown + note, encoding="utf-8")
    print(markdown)
    print(f"wrote {args.out_dir / 'ACCURACY_PILOT.md'}")


if __name__ == "__main__":
    main()
