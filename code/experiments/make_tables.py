#!/usr/bin/env python
"""make_tables.py --table <label> -- emits LaTeX tables from results/*.parquet
matching manuscript.tex's tables (fidelity-a l.563, zkperf-a l.587,
ordering-a l.614, cert-a l.634, sensitivity-a l.654, ablation-a l.672,
datasets-a l.447, circuit-cost l.314). Writes to results/tables/<label>.tex.

Usage: python -m experiments.make_tables --table fidelity-a
       python -m experiments.make_tables --table all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from experiments.common import RESULTS_DIR

TABLES_DIR = RESULTS_DIR / "tables"

LABELS = ["fidelity-a", "zkperf-a", "ordering-a", "cert-a", "sensitivity-a",
          "ablation-a", "datasets-a", "circuit-cost"]


def _read(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _tex_escape(s: str) -> str:
    return str(s).replace("_", "\\_")


def _write(label: str, body: str):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / f"{label}.tex"
    path.write_text(body)
    print(f"wrote {path}")


def _empty_note(label: str) -> str:
    return f"% table {label}: no data available (run experiments/rq*.py first)\n"


def table_fidelity_a():
    df = _read("rq1_fidelity")
    lines = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Dataset & Projection $(m,q)$ & $r_\\Phi$ & ARI vs. full & $Q_{0.95}(|\\delta|)$ & "
        "$\\Pr(|\\delta|>\\epsilon)$ & certified coverage \\\\",
        "\\midrule",
    ]
    if len(df):
        ok = df[df.get("_status", "ok") == "ok"] if "_status" in df else df
        grp_cols = [c for c in ("dataset_name", "projection", "m", "q") if c in ok.columns]
        if grp_cols:
            g = ok.groupby(grp_cols, dropna=False).agg(
                r_phi=("r_phi", "median"), ari=("ari", "median"),
                q95=("delta_q95", "median"), exceed=("delta_exceedance", "median"),
            ).reset_index()
            for _, row in g.iterrows():
                name = _tex_escape(row.get("dataset_name", ""))
                mq = f"({int(row.get('m', 0))},{int(row.get('q', 0))})" if "m" in row else ""
                lines.append(f"{name} & {mq} & {row['r_phi']:.3f} & {row['ari']:.3f} & "
                              f"{row['q95']:.4f} & {row['exceed']:.3f} & -- \\\\")
    else:
        lines.append(f"\\multicolumn{{7}}{{c}}{{{_empty_note('fidelity-a').strip('% ')}}} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write("fidelity-a", "\n".join(lines) + "\n")


def table_zkperf_a():
    df = _read("rq2_proof_cost")
    lines = [
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Method & constraints & witness/compile (s) & proving (s) & peak RAM (GiB) & "
        "verify (ms) & proof (KiB) & ARI \\\\",
        "\\midrule",
    ]
    if len(df) and "variant" in df.columns:
        ok = df[df.get("_status") == "ok"] if "_status" in df else df
        for variant, g in ok.groupby("variant"):
            constraints = g["num_constraints"].median() if "num_constraints" in g else float("nan")
            wc = g["witness_seconds_mean"].median() + g["compile_seconds_mean"].median() \
                if {"witness_seconds_mean", "compile_seconds_mean"} <= set(g.columns) else float("nan")
            prove = g["prove_seconds_mean"].median() if "prove_seconds_mean" in g else float("nan")
            ram = g["peak_rss_bytes_psutil_max"].median() / (1024 ** 3) if "peak_rss_bytes_psutil_max" in g else float("nan")
            verify = g["verify_seconds_mean"].median() * 1000 if "verify_seconds_mean" in g else float("nan")
            proof_kib = g["proof_bytes"].median() / 1024 if "proof_bytes" in g else float("nan")
            ari = g["ari"].median() if "ari" in g else float("nan")
            lines.append(f"{_tex_escape(variant)} & {constraints:.0f} & {wc:.3f} & {prove:.3f} & "
                          f"{ram:.3f} & {verify:.1f} & {proof_kib:.1f} & {ari if not pd.isna(ari) else '--'} \\\\")
    else:
        lines.append(f"\\multicolumn{{8}}{{c}}{{{_empty_note('zkperf-a').strip('% ')}}} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write("zkperf-a", "\n".join(lines) + "\n")


def table_ordering_a():
    df = _read("rq3_overhead")
    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Component & host latency (ms) & circuit constraints & transcript bytes & "
        "invocations/attempt & Included in end-to-end total \\\\",
        "\\midrule",
    ]
    if len(df):
        ok = df[df.get("_status") == "ok"] if "_status" in df else df
        stage_map = {
            "data/parameter commitment": "time_commit_X",
            "challenge acquisition": "time_issue_nonce",
            "sparse-matrix expansion": "time_expand_sparse_and_CA",
            "matrix authentication": "time_expand_sparse_and_CA",
            "project and cluster": "time_project_and_cluster",
        }
        byte_map = {
            "data/parameter commitment": "bytes_commit_X",
            "challenge acquisition": "bytes_issue_nonce",
            "sparse-matrix expansion": "bytes_expand_sparse_and_CA",
        }
        for label, col in stage_map.items():
            if col in ok.columns:
                lat = ok[col].median() * 1000
                byte_col = byte_map.get(label)
                nbytes = ok[byte_col].median() if byte_col and byte_col in ok.columns else 0
                lines.append(f"{label} & {lat:.3f} & -- & {nbytes:.0f} & 1 & yes \\\\")
        n_retries = ok["attempts"].sub(1).sum() if "attempts" in ok.columns else 0
        lines.append(f"rejected/retried commitments & -- & 0 & -- & {n_retries:.0f} & reported separately \\\\")
    else:
        lines.append(f"\\multicolumn{{5}}{{c}}{{{_empty_note('ordering-a').strip('% ')}}} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write("ordering-a", "\n".join(lines) + "\n")


def table_cert_a():
    df = _read("rq4_certificates")
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Dataset/configuration & certified points/pairs & observed stable & uncertified stable & violations \\\\",
        "\\midrule",
    ]
    if len(df):
        ok = df[df.get("_status") == "ok"] if "_status" in df else df
        for cert_type, g in ok.groupby(ok.get("cert_type", pd.Series(dtype=str))):
            n_cert = int(g["n_certified"].sum()) if "n_certified" in g else 0
            n_total = int(g["n_total"].sum()) if "n_total" in g else 0
            viol = g["violation_rate"].median() if "violation_rate" in g else float("nan")
            uncert = g["uncertified_stability"].median() if "uncertified_stability" in g else float("nan")
            n_violations = int((g["violation_rate"] * g["n_certified"]).sum()) if "violation_rate" in g else 0
            lines.append(f"{_tex_escape(cert_type)} & {n_cert}/{n_total} & "
                          f"{1 - viol if not pd.isna(viol) else float('nan'):.3f} & {uncert:.3f} & {n_violations} \\\\")
    else:
        lines.append(f"\\multicolumn{{5}}{{c}}{{{_empty_note('cert-a').strip('% ')}}} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write("cert-a", "\n".join(lines) + "\n")


def table_sensitivity_a():
    df = _read("rq5_scaling")
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Factor & n rows & median total time (s) & median constraints \\\\",
        "\\midrule",
    ]
    if len(df):
        ok = df[df.get("_status") == "ok"] if "_status" in df else df
        lines.append(f"n,d,k,m,q sweep & {len(ok)} & "
                      f"{ok['t_total_s'].median() if 't_total_s' in ok else float('nan'):.4f} & "
                      f"{ok['symbolic_total_constraints'].median() if 'symbolic_total_constraints' in ok else float('nan'):.0f} \\\\")
    else:
        lines.append(f"\\multicolumn{{3}}{{c}}{{{_empty_note('sensitivity-a').strip('% ')}}} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write("sensitivity-a", "\n".join(lines) + "\n")


def table_ablation_a():
    df = _read("ablation")
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Variant & proving (s) & RAM (GiB) & constraints & status \\\\",
        "\\midrule",
    ]
    if len(df):
        for variant, g in df.groupby("variant"):
            prove = g["prove_seconds"].median() if "prove_seconds" in g and g["prove_seconds"].notna().any() else float("nan")
            constraints = g["symbolic_total_constraints"].median() if "symbolic_total_constraints" in g else float("nan")
            status = g["_status"].mode().iat[0] if "_status" in g and len(g) else "--"
            lines.append(f"{_tex_escape(variant)} & {prove if not pd.isna(prove) else float('nan'):.4f} & "
                          f"-- & {constraints:.0f} & {_tex_escape(status)} \\\\")
    else:
        lines.append(f"\\multicolumn{{4}}{{c}}{{{_empty_note('ablation-a').strip('% ')}}} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write("ablation-a", "\n".join(lines) + "\n")


def table_datasets_a():
    import json
    manifest_path = Path(__file__).resolve().parent.parent / ".cache" / "commitjl_data" / "manifest.json"
    lines = [
        "\\begin{tabular}{lcccl}",
        "\\toprule",
        "Dataset/family & $n$ & $d$ & classes/$k$ & source \\\\",
        "\\midrule",
    ]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, meta in manifest.items():
            lines.append(f"{_tex_escape(name)} & {meta.get('n', '--')} & {meta.get('d', '--')} & "
                          f"{meta.get('k', '--')} & {_tex_escape(meta.get('source', ''))} \\\\")
    else:
        lines.append(f"\\multicolumn{{5}}{{c}}{{{_empty_note('datasets-a').strip('% ')}}} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write("datasets-a", "\n".join(lines) + "\n")


def table_circuit_cost():
    from commitjl.cost_model import total_circuit_cost
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "$(n,d,m,k,q)$ & hash inv. & range inv. & cmp inv. & total (symbolic) \\\\",
        "\\midrule",
    ]
    for (n, d, m, k, q) in [(1000, 1024, 64, 10, 4), (10000, 4096, 128, 10, 4)]:
        c = total_circuit_cost(n, d, m, k, q)
        hashes = c["commitment_auth"]["hash_invocations"] + c["matrix_auth"]["hash_invocations"]
        ranges = (c["commitment_auth"]["range_invocations"] + c["matrix_auth"]["range_invocations"]
                  + c["sparse_projection"]["range_invocations"] + c["projected_distance"]["range_invocations"])
        cmps = c.get("argmin", {}).get("cmp_invocations", 0)
        lines.append(f"({n},{d},{m},{k},{q}) & {hashes:.0f} & {ranges:.0f} & {cmps:.0f} & "
                      f"{c['total_constraints']:.0f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write("circuit-cost", "\n".join(lines) + "\n")


DISPATCH = {
    "fidelity-a": table_fidelity_a, "zkperf-a": table_zkperf_a, "ordering-a": table_ordering_a,
    "cert-a": table_cert_a, "sensitivity-a": table_sensitivity_a, "ablation-a": table_ablation_a,
    "datasets-a": table_datasets_a, "circuit-cost": table_circuit_cost,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, choices=LABELS + ["all"])
    args = ap.parse_args()
    labels = LABELS if args.table == "all" else [args.table]
    for label in labels:
        DISPATCH[label]()


if __name__ == "__main__":
    main()
