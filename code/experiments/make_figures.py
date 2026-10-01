#!/usr/bin/env python
"""make_figures.py --fig <label> -- renders manuscript figures (PDF+PNG)
from results/*.parquet: fidelity-vs-m, pareto, cost-breakdown, phase-diagram,
scaling-dashboard, dbscan-band, margin-geometry. Writes to
results/figures/<label>.{pdf,png}.

Usage: python -m experiments.make_figures --fig fidelity-vs-m
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import RESULTS_DIR
from experiments.make_tables import _read

FIG_DIR = RESULTS_DIR / "figures"
LABELS = ["fidelity-vs-m", "pareto", "cost-breakdown", "phase-diagram",
          "scaling-dashboard", "precision-sensitivity", "dbscan-band", "margin-geometry"]


def _save(fig, label: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{label}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{label}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG_DIR / (label + '.pdf')} (+.png)")


def _empty_fig(label: str):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.text(0.5, 0.5, f"no data for {label}\n(run the corresponding rq*.py first)",
            ha="center", va="center", wrap=True)
    ax.axis("off")
    return fig


def fig_fidelity_vs_m():
    df = _read("rq1_fidelity")
    if not len(df) or "m" not in df.columns:
        return _save(_empty_fig("fidelity-vs-m"), "fidelity-vs-m")
    ok = df[df.get("_status", "ok") == "ok"] if "_status" in df else df
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for proj, g in ok.groupby("projection") if "projection" in ok.columns else []:
        gg = g.groupby("m")["ari"].median().sort_index()
        ax.plot(gg.index, gg.values, marker="o", label=str(proj))
    ax.set_xlabel("projected dimension $m$")
    ax.set_ylabel("ARI vs. full")
    ax.legend(fontsize=7)
    _save(fig, "fidelity-vs-m")


def fig_pareto():
    df = _read("rq2_proof_cost")
    if not len(df) or "prove_seconds_mean" not in df.columns:
        return _save(_empty_fig("pareto"), "pareto")
    ok = df[df.get("_status") == "ok"] if "_status" in df else df
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for variant, g in ok.groupby("variant") if "variant" in ok.columns else []:
        ax.scatter(g["prove_seconds_mean"], g.get("ari", np.nan), label=str(variant))
    ax.set_xlabel("proving time (s)")
    ax.set_ylabel("ARI vs. full")
    ax.legend(fontsize=7)
    _save(fig, "pareto")


def fig_cost_breakdown():
    df = _read("rq3_overhead")
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    if len(df):
        ok = df[df.get("_status") == "ok"] if "_status" in df else df
        cols = [c for c in ok.columns if c.startswith("time_")]
        if cols:
            means = ok[cols].mean()
            ax.bar([c.replace("time_", "") for c in cols], means.values)
            ax.set_ylabel("mean stage time (s)")
            plt.xticks(rotation=30, ha="right")
        else:
            plt.close(fig)
            return _save(_empty_fig("cost-breakdown"), "cost-breakdown")
    else:
        plt.close(fig)
        return _save(_empty_fig("cost-breakdown"), "cost-breakdown")
    _save(fig, "cost-breakdown")


def fig_phase_diagram():
    df = _read("rq4_certificates")
    if not len(df) or "cert_fn" not in df.columns:
        return _save(_empty_fig("phase-diagram"), "phase-diagram")
    phase = df[df["cert_fn"] == "phase"] if "cert_fn" in df.columns else pd.DataFrame()
    if not len(phase) or "gamma" not in phase.columns:
        return _save(_empty_fig("phase-diagram"), "phase-diagram")
    pivot = phase.pivot_table(index="gamma", columns="m", values="coverage", aggfunc="median")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{g:.1f}" for g in pivot.index])
    ax.set_xlabel("$m$")
    ax.set_ylabel("$\\gamma$")
    fig.colorbar(im, ax=ax, label="certificate coverage")
    _save(fig, "phase-diagram")


def fig_scaling_dashboard():
    df = _read("rq5_scaling")
    if not len(df):
        return _save(_empty_fig("scaling-dashboard"), "scaling-dashboard")
    ok = df[df.get("_status") == "ok"] if "_status" in df else df
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    n_rows = ok[ok["sweep_factor"] == "n"] if "sweep_factor" in ok else ok
    d_rows = ok[ok["sweep_factor"] == "d"] if "sweep_factor" in ok else ok
    q_rows = ok[ok["sweep_factor"] == "q"] if "sweep_factor" in ok else ok
    m_rows = ok[ok["sweep_factor"] == "m"] if "sweep_factor" in ok else ok
    if "n" in n_rows.columns and "t_total_s" in n_rows.columns:
        g = n_rows.groupby("n")["t_total_s"].median().sort_index()
        axes[0, 0].loglog(g.index, g.values, marker="o")
    axes[0, 0].set_title("native component time vs n")
    if "d" in d_rows.columns:
        g = d_rows.groupby("d")["t_total_s"].median().sort_index()
        axes[0, 1].loglog(g.index, g.values, marker="o")
    axes[0, 1].set_title("native component time vs d")
    if "q" in q_rows.columns and "t_total_s" in q_rows.columns:
        g = q_rows.groupby("q")["t_total_s"].median().sort_index()
        axes[1, 0].plot(g.index, g.values, marker="o")
    axes[1, 0].set_title("native component time vs q")
    if "m" in m_rows.columns:
        g = m_rows.groupby("m")["t_total_s"].median().sort_index()
        axes[1, 1].plot(g.index, g.values, marker="o")
    axes[1, 1].set_title("time vs m")
    fig.tight_layout()
    _save(fig, "scaling-dashboard")


def fig_precision_sensitivity():
    df = _read("rq5_precision")
    if df.empty:
        return _save(_empty_fig("precision-sensitivity"), "precision-sensitivity")
    ok = df[df["_status"].eq("ok")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for dataset, g in ok.groupby("dataset_name"):
        agg = g.groupby("precision_bits").agg(
            ari=("ari", "mean"),
            distortion=("delta_q95", "mean"),
            error=("mean_abs_quantization_error", "mean"),
        ).sort_index()
        for ax, col in zip(axes, ["ari", "distortion", "error"]):
            ax.plot(agg.index, agg[col], marker="o", label=dataset)
    for ax in axes:
        ax.set_xticks([8, 12, 16, 20])
        ax.set_xlabel("Quantization precision (bits)")
    axes[0].set_ylabel("ARI vs full")
    axes[1].set_ylabel("Q95 absolute distortion")
    axes[2].set_ylabel("Mean absolute quantization error")
    axes[0].legend(fontsize=7)
    fig.suptitle("Fixed-input-scale precision sweep; circuit range bound B=4096")
    fig.tight_layout()
    _save(fig, "precision-sensitivity")


def fig_dbscan_band():
    df = _read("rq4_certificates")
    if not len(df) or "cert_type" not in df.columns:
        return _save(_empty_fig("dbscan-band"), "dbscan-band")
    dd = df[df["cert_type"] == "dbscan_band"] if "cert_type" in df.columns else pd.DataFrame()
    if not len(dd):
        return _save(_empty_fig("dbscan-band"), "dbscan-band")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(["coverage", "1-violation_rate", "uncertified_stability"],
           [dd["coverage"].median(), 1 - dd["violation_rate"].median(), dd["uncertified_stability"].median()])
    ax.set_ylim(0, 1.05)
    _save(fig, "dbscan-band")


def fig_margin_geometry():
    from commitjl.certificates import dbscan_ambiguity_band
    fig, ax = plt.subplots(figsize=(5, 3.5))
    rho = 1.0
    eps_vals = np.linspace(0.01, 0.5, 50)
    lo = [dbscan_ambiguity_band(rho, e)[0] for e in eps_vals]
    hi = [dbscan_ambiguity_band(rho, e)[1] for e in eps_vals]
    ax.plot(eps_vals, lo, label="band lower")
    ax.plot(eps_vals, hi, label="band upper")
    ax.fill_between(eps_vals, lo, hi, alpha=0.2)
    ax.set_xlabel("$\\epsilon$")
    ax.set_ylabel("squared distance")
    ax.legend()
    _save(fig, "margin-geometry")


DISPATCH = {
    "fidelity-vs-m": fig_fidelity_vs_m, "pareto": fig_pareto, "cost-breakdown": fig_cost_breakdown,
    "phase-diagram": fig_phase_diagram, "scaling-dashboard": fig_scaling_dashboard,
    "precision-sensitivity": fig_precision_sensitivity,
    "dbscan-band": fig_dbscan_band, "margin-geometry": fig_margin_geometry,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", required=True, choices=LABELS + ["all"])
    args = ap.parse_args()
    labels = LABELS if args.fig == "all" else [args.fig]
    for label in labels:
        DISPATCH[label]()


if __name__ == "__main__":
    main()
