"""Generate reproducible statistics, charts, and an evidence report for paper runs.

Missing experiments remain explicitly missing. This script never substitutes
symbolic estimates for observed proof metrics or pools seeds across datasets.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import load_prereg
from experiments.statistics import summarize_repetitions

ROOT = Path(__file__).resolve().parents[1] / "results"


class AnalysisOutput:
    """Place each RQ's statistics and figures beside that RQ's run folder."""

    def __init__(self, root: Path):
        self.root = root

    def mkdir(self, **kwargs):
        self.root.mkdir(**kwargs)

    def __truediv__(self, name: str) -> Path:
        if name.startswith(("rq1_", "rq2_", "rq3_", "rq4_", "rq5_")):
            group = name[:3]
        elif name.startswith("precision_"):
            group = "rq5"
        else:
            group = "overview"
        path = self.root / group / "analysis" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def __str__(self) -> str:
        return str(self.root)


def load(folder: Path, filename: str):
    path = folder / filename
    return pd.read_parquet(path) if path.exists() else None


def markdown_table(df, columns=None, max_rows=30):
    if df is None or df.empty:
        return "No measured rows available."
    shown = df[columns].head(max_rows) if columns else df.head(max_rows)
    def cell(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:.4g}"
        return str(value).replace("|", "\\|")
    names = list(shown.columns)
    lines = ["| " + " | ".join(names) + " |", "| " + " | ".join(["---"] * len(names)) + " |"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in shown.itertuples(index=False, name=None)]
    return "\n".join(lines)


def save_chart(fig, out: Path, name: str):
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=170)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def dataset_shape_label(spec):
    kwargs = spec.get("kwargs", {})
    def choose(*keys):
        for key in keys:
            value = kwargs.get(key)
            if value is not None and pd.notna(value):
                return str(int(value))
        return "?"
    known_d = {"mnist": 784, "fashion_mnist": 784, "gisette": 5000,
               "arcene": 10000, "20ng_tfidf": 2000}
    d = choose("d")
    if d == "?" and spec["loader"] in known_d:
        d = str(known_d[spec["loader"]])
    return f"{spec['loader']} {choose('n', 'n_max')}x{d}"


def summarize(df, groups, metrics, out, name):
    summary = summarize_repetitions(df, groups, metrics)
    if not summary.empty:
        summary.to_csv(out / f"{name}.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, default=None,
                        help="legacy single-directory native campaign override")
    parser.add_argument("--precision", type=Path, default=ROOT / "rq5/runs/precision_20ref")
    parser.add_argument("--proof", type=Path, default=ROOT / "rq2/runs/representative")
    parser.add_argument("--supplement", type=Path, default=None,
                        help="legacy single-directory supplement override")
    parser.add_argument("--out", type=Path, default=ROOT)
    args = parser.parse_args()
    out = AnalysisOutput(args.out)
    out.mkdir(parents=True, exist_ok=True)
    native = lambda rq: args.native or ROOT / rq / "runs/native"
    supplement = lambda rq, kind: args.supplement or ROOT / rq / "runs" / kind
    paths = {
        "RQ1 fidelity": (native("rq1"), "rq1_fidelity.parquet"),
        "RQ3 overhead": (native("rq3"), "rq3_overhead.parquet"),
        "RQ3 grinding": (native("rq3"), "rq3_grinding.parquet"),
        "RQ4 certificates": (native("rq4"), "rq4_certificates.parquet"),
        "RQ5 scaling": (native("rq5"), "rq5_scaling.parquet"),
        "Precision 20-bit reference": (args.precision, "rq5_precision.parquet"),
        "RQ2 proof": (args.proof, "rq2_proof_cost.parquet"),
        "RQ3 paired supplement": (supplement("rq3", "paired"), "rq3_paired.parquet"),
        "RQ4 real supplement": (supplement("rq4", "real"), "rq4_real_certificates.parquet"),
        "RQ5 separation supplement": (supplement("rq5", "gamma"), "rq5_gamma.parquet"),
        "Precision 24-bit reference": (supplement("rq5", "precision_24ref"), "rq5_precision_24_reference.parquet"),
    }
    data = {name: load(*path) for name, path in paths.items()}
    inventory = pd.DataFrame([{
        "experiment": name, "rows": 0 if df is None else len(df),
        "ok": 0 if df is None else int(df["_status"].eq("ok").sum()),
        "failed": 0 if df is None else int(df["_status"].ne("ok").sum()),
        "file": str(paths[name][0] / paths[name][1]),
    } for name, df in data.items()])
    inventory.to_csv(out / "run_inventory.csv", index=False)
    report = ["# Paper experiment analysis", "",
              "All numerical entries below come from saved rows. Confidence intervals are 95% Student-t intervals over the ten recorded seeds within each configuration; they are descriptive and may extend outside bounded metric ranges. No CI pools datasets or treats repeated rows as independent datasets.",
              "", "## Run inventory", "", markdown_table(inventory, ["experiment", "rows", "ok", "failed"]), ""]

    rq1 = data["RQ1 fidelity"]
    if rq1 is not None:
        report += ["## RQ1: fidelity", "",
                   f"Measured {len(rq1)} rows on {rq1['dataset_name'].nunique()} dataset families. Each configuration has ten seed rows. ARI compares projected assignments with the full-space reference; it is not ground-truth label accuracy.", ""]
        all_summary = summarize(rq1[rq1._status.eq("ok")],
                                ["dataset_name", "dataset_sha256", "n", "d", "projection", "k", "m", "q"],
                                ["ari", "r_phi", "delta_q95", "t_total_s"], out, "rq1_all_configurations")
        prereg = load_prereg()
        eligible = all_summary[(all_summary.ari_ci95_low >= float(prereg["tau_ARI"])) &
                               (all_summary.r_phi_ci95_low >= 1.0 - float(prereg["tau_Phi"])) &
                               (all_summary.r_phi_ci95_high <= 1.0 + float(prereg["tau_Phi"])) &
                               (all_summary.m < all_summary.d)]
        selected = eligible.sort_values(["m", "q"]).groupby(
            ["dataset_name", "dataset_sha256", "n", "d", "projection", "k"], as_index=False).first()
        selected.to_csv(out / "rq1_smallest_ci_eligible_m.csv", index=False)
        report += [f"[All {len(all_summary)} configuration-level summaries](rq1_all_configurations.csv) and [smallest CI-eligible dimensions](rq1_smallest_ci_eligible_m.csv) are available. Eligibility here requires lower ARI CI ≥ {prereg['tau_ARI']} and the entire cost-ratio CI within 1 ± {prereg['tau_Phi']}.", ""]
        subset = rq1[(rq1._status == "ok") & (rq1.projection == "sparse_jl") &
                     (rq1.q == 2) & (rq1.k == 5) & (rq1.m < rq1.d)]
        summary = summarize(subset, ["dataset_name", "dataset_sha256", "n", "d", "m"],
                            ["ari", "r_phi", "delta_q95"], out, "rq1_sparse_q2_k5")
        fig, ax = plt.subplots(figsize=(9, 5))
        for (dataset, n, d), group in summary.groupby(["dataset_name", "n", "d"]):
            group = group.sort_values("m")
            ax.plot(group.m, group.ari_mean, marker="o", label=f"{dataset} ({n}x{d})")
        ax.axhline(0.8, color="black", linestyle="--", linewidth=.8, label="ARI threshold")
        ax.set(xlabel="Projected dimension m", ylabel="Mean assignment ARI", title="RQ1: sparse JL, q=2, k=5")
        ax.set_ylim(-.05, 1.05)
        ax.legend(fontsize=7, ncol=2)
        save_chart(fig, out, "rq1_ari_vs_m")
        report += ["[ARI versus dimension](rq1_ari_vs_m.png); [configuration statistics](rq1_sparse_q2_k5.csv).", "",
                   markdown_table(summary, ["dataset_name", "n", "d", "m", "repetitions", "ari_mean", "ari_ci95_low", "ari_ci95_high"], 30), ""]

    overhead = data["RQ3 overhead"]
    grinding = data["RQ3 grinding"]
    report += ["## RQ3: protocol overhead and grinding", ""]
    if overhead is not None:
        stage_cols = ["time_commit_X", "time_record_theta", "time_issue_nonce",
                      "time_expand_sparse_and_CA", "time_project_and_cluster"]
        oh = overhead[overhead._status.eq("ok")].copy()
        oh["shape"] = oh["dataset"].map(dataset_shape_label)
        stage = oh.groupby(["shape", "seed_source"])[stage_cols].median().reset_index()
        stage.to_csv(out / "rq3_stage_medians.csv", index=False)
        report += ["The existing overhead run used six concurrent processes. Its per-job timings include CPU contention. The beacon is a **simulated local trace**, not an external beacon measurement.", "",
                   markdown_table(stage), ""]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        labels = [f"{r['shape']} / {r['seed_source']}" for _, r in stage.iterrows()]
        axes[0].bar(labels, stage.time_commit_X, color="tab:blue")
        axes[1].bar(labels, stage[stage_cols[1:]].sum(axis=1), color="tab:orange")
        axes[0].set(ylabel="Median commitment seconds", title="Input commitment")
        axes[1].set(ylabel="Median other-stage seconds", title="Challenge, expansion, assignment")
        for ax in axes:
            ax.tick_params(axis="x", rotation=40, labelsize=7)
        save_chart(fig, out, "rq3_stage_time")
    if grinding is not None:
        g = summarize(grinding[grinding._status.eq("ok")], ["m", "q", "Q"],
                      ["empirical_success_rate", "union_bound", "delta_geom"], out, "rq3_grinding_summary")
        representative = g[(g.m == 32) & (g.q == 2)].sort_values("Q")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(representative.Q, representative.empirical_success_rate_mean, "o-", label="empirical")
        ax.plot(representative.Q, representative.union_bound_mean, "s--", label="plug-in union bound")
        ax.set(xlabel="Attempt budget Q", ylabel="Mean rate", title="RQ3: grinding sensitivity (m=32, q=2)")
        ax.legend()
        save_chart(fig, out, "rq3_grinding")
        report += ["[Grinding statistics](rq3_grinding_summary.csv) and [chart](rq3_grinding.png). The plotted union bound is an empirical plug-in quantity, not a proved population guarantee.", ""]
    paired = data["RQ3 paired supplement"]
    if paired is not None:
        ps = summarize(paired[paired._status.eq("ok")], ["dataset_name", "n", "d"],
                       ["incremental_postcommit_s", "incremental_after_commit_s",
                        "prepublished_transaction_s", "postcommit_transaction_s"], out, "rq3_paired_summary")
        report += ["Paired pre-published baseline: map generation/authentication occurs before the timed transaction and is recorded separately. This is native timing only; it does not establish ZK proof-cost savings.", "",
                   markdown_table(ps, ["dataset_name", "n", "d", "repetitions", "incremental_postcommit_s_mean",
                                      "incremental_postcommit_s_ci95_low", "incremental_postcommit_s_ci95_high",
                                      "incremental_after_commit_s_mean"]), ""]
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = [f"{r.dataset_name} {r.n}x{r.d}" for r in ps.itertuples()]
        ax.errorbar(labels, ps.incremental_after_commit_s_mean,
                    yerr=[ps.incremental_after_commit_s_mean - ps.incremental_after_commit_s_ci95_low,
                          ps.incremental_after_commit_s_ci95_high - ps.incremental_after_commit_s_mean],
                    fmt="o", capsize=4)
        ax.axhline(0, color="black", linewidth=.8)
        ax.tick_params(axis="x", rotation=20)
        ax.set(ylabel="Post-commit minus pre-published seconds", title="RQ3: paired incremental overhead")
        save_chart(fig, out, "rq3_paired_overhead")
    else:
        report += ["The direct pre-published comparator has not been run yet.", ""]

    rq4 = data["RQ4 certificates"]
    report += ["## RQ4: certificate behavior", ""]
    if rq4 is not None:
        core = rq4[rq4._status.eq("ok") & rq4.cert_fn.ne("phase")].copy()
        c = summarize(core, ["cert_type", "m", "q"],
                      ["coverage", "violation_rate", "uncertified_stability"], out, "rq4_certificate_summary")
        report += [markdown_table(c, ["cert_type", "m", "repetitions", "coverage_mean",
                                      "q", "violation_rate_mean", "uncertified_stability_mean"]), ""]
        fig, ax = plt.subplots(figsize=(8, 4))
        for (cert_type, q), group in c.groupby(["cert_type", "q"]):
            group = group.sort_values("m")
            ax.plot(group.m, group.coverage_mean, "o-", label=f"{cert_type}, q={q}")
        ax.set(xlabel="Projected dimension m", ylabel="Mean certified fraction", title="RQ4: certificate coverage")
        ax.legend()
        save_chart(fig, out, "rq4_coverage")
        phase = rq4[rq4._status.eq("ok") & rq4.cert_fn.eq("phase")].copy()
        if len(phase):
            phase[["gamma", "m", "coverage", "violation_rate", "uncertified_stability"]].to_csv(
                out / "rq4_phase_diagram.csv", index=False)
            grid = phase.pivot_table(index="gamma", columns="m", values="coverage", aggfunc="mean")
            fig, ax = plt.subplots(figsize=(7, 4))
            view = ax.imshow(grid.to_numpy(), origin="lower", aspect="auto", vmin=0, vmax=1)
            ax.set_xticks(range(len(grid.columns)), labels=[str(v) for v in grid.columns])
            ax.set_yticks(range(len(grid.index)), labels=[str(v) for v in grid.index])
            ax.set(xlabel="Projected dimension m", ylabel="Separation gamma", title="RQ4: centroid certificate phase diagram")
            fig.colorbar(view, ax=ax, label="Certified fraction")
            save_chart(fig, out, "rq4_phase_diagram")
            report += [f"The [phase diagram](rq4_phase_diagram.png) contains {len(phase)} single-seed cells; it has no seed-level uncertainty interval.", ""]
        dbscan = core[core.cert_type.eq("dbscan_band")].copy()
        if len(dbscan):
            dbscan["uncertified_pairs"] = dbscan.n_total - dbscan.n_certified
            near = summarize(dbscan, ["m", "q"], ["uncertified_pairs", "uncertified_stability"],
                             out, "rq4_near_band_summary")
            report += ["The DBSCAN generator concentrates difficulty in paired near-radius points. Overall pair coverage is dominated by far-apart pairs; use the uncertified near-band denominator when judging this stress test.", "",
                       markdown_table(near, ["m", "q", "repetitions", "uncertified_pairs_mean",
                                             "uncertified_stability_mean"]), ""]
    real = data["RQ4 real supplement"]
    if real is not None:
        rs = summarize(real[real._status.eq("ok")], ["dataset_name", "m"],
                       ["coverage", "violation_rate", "uncertified_stability"], out, "rq4_real_summary")
        event = real[real._status.eq("ok")].groupby(["dataset_name", "m"])["embedding_event_holds"].mean().reset_index(name="embedding_event_rate")
        rs = rs.merge(event, on=["dataset_name", "m"], how="left")
        conditional = real[real._status.eq("ok")].groupby(["dataset_name", "m"])[
            ["n_certified_with_pointwise_event", "n_certified_pointwise_event_violations"]].sum().reset_index()
        rs = rs.merge(conditional, on=["dataset_name", "m"], how="left")
        rs.to_csv(out / "rq4_real_summary.csv", index=False)
        report += ["Real-data centroid certificate supplement:", "", markdown_table(rs,
                   ["dataset_name", "m", "repetitions", "coverage_mean", "violation_rate_mean",
                    "embedding_event_rate", "n_certified_with_pointwise_event", "n_certified_pointwise_event_violations"]),
                   "", "A certified assignment is guaranteed stable only when the point-to-centroid embedding event holds. The final two columns count certified points satisfying their pointwise event and violations among those points. Unconditional violations on failed-event rows do not contradict the conditional theorem.", ""]
    else:
        report += ["Real-data centroid certificate supplement has not been run yet.", ""]

    rq5 = data["RQ5 scaling"]
    report += ["## RQ5: scaling and sensitivity", ""]
    if rq5 is not None:
        s = summarize(rq5[rq5._status.eq("ok")], ["sweep_factor", "sweep_value"],
                      ["ari", "r_phi", "t_total_s"], out, "rq5_scaling_summary_analysis")
        report += ["The reported `t_total_s` is a sum of cached component timings and assignment timing, not an end-to-end wall-clock stopwatch or ZK proving time.", "",
                   markdown_table(s, ["sweep_factor", "sweep_value", "repetitions", "ari_mean", "r_phi_mean", "t_total_s_mean"]), ""]
        for metric, ylabel, filename in (("ari_mean", "Mean ARI", "rq5_scaling_ari"),
                                          ("t_total_s_mean", "Mean component-time sum (s)", "rq5_scaling_time")):
            fig, axes = plt.subplots(2, 3, figsize=(12, 7))
            for ax, (factor, group) in zip(axes.flat, s.groupby("sweep_factor")):
                group = group.sort_values("sweep_value")
                ax.plot(group.sweep_value, group[metric], "o-")
                ax.set(xlabel=factor, ylabel=ylabel, title=f"Vary {factor}")
            for ax in list(axes.flat)[s.sweep_factor.nunique():]:
                ax.axis("off")
            save_chart(fig, out, filename)
        report += ["[Scaling ARI chart](rq5_scaling_ari.png) and [reported component-time chart](rq5_scaling_time.png).", ""]
    gamma = data["RQ5 separation supplement"]
    if gamma is not None:
        gs = summarize(gamma[gamma._status.eq("ok")], ["sweep_value"],
                       ["ari", "r_phi", "certificate_coverage", "certificate_violation_rate"],
                       out, "rq5_gamma_summary")
        event = gamma[gamma._status.eq("ok")].groupby("sweep_value")["embedding_event_holds"].mean().reset_index(name="embedding_event_rate")
        gs = gs.merge(event, on="sweep_value", how="left")
        conditional = gamma[gamma._status.eq("ok")].groupby("sweep_value")[
            ["n_certified_with_pointwise_event", "n_certified_pointwise_event_violations"]].sum().reset_index()
        gs = gs.merge(conditional, on="sweep_value", how="left")
        gs.to_csv(out / "rq5_gamma_summary.csv", index=False)
        report += ["Controlled separation supplement:", "", markdown_table(gs,
                   ["sweep_value", "repetitions", "ari_mean", "ari_ci95_low", "ari_ci95_high",
                    "certificate_coverage_mean", "certificate_violation_rate_mean", "embedding_event_rate",
                    "n_certified_with_pointwise_event", "n_certified_pointwise_event_violations"]), ""]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(gs.sweep_value, gs.ari_mean, "o-", label="assignment ARI")
        ax.plot(gs.sweep_value, gs.certificate_coverage_mean, "s-", label="certificate coverage")
        ax.set(xlabel="Gaussian separation gamma", ylabel="Mean metric", title="RQ5: controlled separation")
        ax.legend()
        save_chart(fig, out, "rq5_gamma")
    else:
        report += ["Controlled separation sweep has not been run yet.", ""]

    precision = data["Precision 24-bit reference"]
    precision_name = "24-bit common reference" if precision is not None else "20-bit common reference"
    if precision is None:
        precision = data["Precision 20-bit reference"]
    if precision is not None:
        pr = summarize(precision[precision._status.eq("ok")], ["dataset_name", "precision_bits"],
                       ["ari", "r_phi", "delta_q95", "mean_abs_quantization_error"],
                       out, "precision_summary_analysis")
        report += [f"Precision sweep shown with the {precision_name}. Do not combine rows from different normalization references as one precision curve.", "",
                   markdown_table(pr, ["dataset_name", "precision_bits", "repetitions", "ari_mean",
                                       "ari_ci95_low", "ari_ci95_high", "mean_abs_quantization_error_mean"]), ""]
        fig, ax = plt.subplots(figsize=(8, 4))
        for dataset, group in pr.groupby("dataset_name"):
            group = group.sort_values("precision_bits")
            ax.plot(group.precision_bits, group.ari_mean, "o-", label=dataset)
        ax.set(xlabel="Quantization bits", ylabel="Mean assignment ARI", title="RQ5: fixed-reference precision")
        ax.legend(fontsize=8)
        save_chart(fig, out, "rq5_precision")
        if "nonzero_quantized_fraction" in precision:
            collapse = precision[(precision.precision_bits == 8) &
                                 (precision.nonzero_quantized_fraction == 0)]
            if len(collapse):
                report += [f"At the 24-bit common physical reference, all {len(collapse)} observed 8-bit rows quantized every coordinate to zero. These are measured precision-collapse cases, not valid evidence of useful 8-bit clustering.", ""]

    proof = data["RQ2 proof"]
    report += ["## Proof and remaining evidence", ""]
    if proof is not None:
        failed = proof[[c for c in ("dataset_name", "n", "d", "m", "_status", "failure_kind",
                                     "peak_rss_bytes_psutil_max", "_error") if c in proof.columns]]
        failed.to_csv(out / "rq2_proof_status.csv", index=False)
        report += [markdown_table(failed), "",
                   "These failed proof rows provide no measured constraint count or proving time. The memory watchdog threshold is not the RAM needed for a successful proof.", ""]
    report += ["Backend/hardware proof sensitivity and proof-based ablation remain unmeasured. Symbolic constraint units are never presented here as measured gnark counts.", ""]
    report_text = "\n".join(report)
    report_path = out / "analysis.md"
    report_text = re.sub(r"\]\(([^/()]+)\)",
                         lambda match: "](" + Path(os.path.relpath(
                             out / match.group(1), report_path.parent)).as_posix() + ")",
                         report_text)
    report_path.write_text(report_text, encoding="utf-8")
    print(f"Wrote {out / 'analysis.md'} and figures/statistics in {out}")


if __name__ == "__main__":
    main()
