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

from experiments.common import RESULTS_DIR, source_fingerprint
from experiments.statistics import mean_std_ci95

TABLES_DIR = RESULTS_DIR / "tables"

LABELS = ["fidelity-a", "zkperf-a", "ordering-a", "cert-a", "sensitivity-a",
          "precision-a", "ablation-a", "datasets-a", "circuit-cost"]


def _read(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "_source_sha256" not in df or not df["_source_sha256"].eq(source_fingerprint()).all():
        raise ValueError(f"{path}: stale or unversioned results; rerun before generating tables")
    return df


def configuration_groups(df, extra=()):
    """Aggregate repetitions/seeds, never mix different experimental shapes."""
    import json
    df = df.copy()
    if "dataset" in df:
        df["dataset_config"] = df["dataset"].map(lambda v: json.dumps(v, sort_keys=True, default=str))
    keys = [c for c in dict.fromkeys((*extra, "dataset_config", "n", "d", "k", "m", "q",
                                   "precision_bits", "B", "backend", "no_range_batch")) if c in df]
    for _, group in df.groupby(keys, dropna=False):
        first = group.iloc[0]
        label = "; ".join(f"{key}={first[key]}" for key in keys)
        yield label, group


def _tex_escape(s: str) -> str:
    return str(s).replace("_", "\\_")


def _write(label: str, body: str):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / f"{label}.tex"
    path.write_text(body)
    print(f"wrote {path}")


def _empty_note(label: str) -> str:
    return f"% table {label}: no data available (run experiments/rq*.py first)\n"


def _mean_ci_text(values, digits=3):
    stats = mean_std_ci95(values)
    if not np.isfinite(stats["mean"]):
        return "--"
    if not np.isfinite(stats["ci95_low"]):
        return f"{stats['mean']:.{digits}f}"
    return f"{stats['mean']:.{digits}f} [{stats['ci95_low']:.{digits}f}, {stats['ci95_high']:.{digits}f}]"


def table_fidelity_a():
    df = _read("rq1_fidelity")
    lines = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Dataset & Projection $(m,q)$ & $r_\\Phi$ mean [95\\% CI] & ARI mean [95\\% CI] & "
        "$Q_{0.95}(|\\delta|)$ mean [95\\% CI] & "
        "$\\Pr(|\\delta|>\\epsilon)$ & fidelity pass rate \\\\",
        "\\midrule",
    ]
    if len(df):
        ok = df[df.get("_status", "ok") == "ok"] if "_status" in df else df
        grp_cols = [c for c in ("dataset_name", "projection", "n", "d", "k", "m", "q", "precision_bits") if c in ok.columns]
        if grp_cols:
            for keys, group in ok.groupby(grp_cols, dropna=False):
                keys = keys if isinstance(keys, tuple) else (keys,)
                row = dict(zip(grp_cols, keys))
                name = _tex_escape(f"{row.get('dataset_name', '')}/{row.get('projection', '')} "
                                   f"(n={row.get('n')},d={row.get('d')},k={row.get('k')})")
                mq = f"({int(row.get('m', 0))},{int(row.get('q', 0))})" if "m" in row else ""
                lines.append(f"{name} & {mq} & {_mean_ci_text(group['r_phi'])} & "
                              f"{_mean_ci_text(group['ari'])} & {_mean_ci_text(group['delta_q95'], 4)} & "
                              f"{group['delta_exceedance'].mean():.3f} & {group['fidelity_ok'].mean():.3f} \\\\")
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
        for variant, g in configuration_groups(ok, ("variant",)):
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
    lines += ["\\midrule",
              "\\multicolumn{8}{l}{\\footnotesize Input commitment/authentication scales with original $d$; projected distance operations scale with reduced $m$.} \\\\",
              "\\bottomrule", "\\end{tabular}"]
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
            "data commitment": "time_commit_X",
            "parameter commitment": "time_record_theta",
            "challenge acquisition": "time_issue_nonce",
            "sparse-matrix expansion and authentication": "time_expand_sparse_and_CA",
            "project and cluster": "time_project_and_cluster",
        }
        byte_map = {
            "data commitment": "bytes_commit_X",
            "parameter commitment": "bytes_record_theta",
            "challenge acquisition": "bytes_issue_nonce",
            "sparse-matrix expansion and authentication": "bytes_expand_sparse_and_CA",
        }
        sources = ok.groupby("seed_source", dropna=False) if "seed_source" in ok else [("unknown", ok)]
        for source, group in sources:
            source_label = "beacon (Simulated local trace)" if source == "beacon" else str(source).replace("_", "-")
            for label, col in stage_map.items():
                if col in group.columns:
                    lat = group[col].median() * 1000
                    byte_col = byte_map.get(label)
                    nbytes = group[byte_col].median() if byte_col and byte_col in group.columns else 0
                    lines.append(f"{_tex_escape(source_label)} / {label} & {lat:.3f} & -- & {nbytes:.0f} & 1 & yes \\\\")
        n_retries = ok["attempts"].sub(1).sum() if "attempts" in ok.columns else 0
        lines.append(f"rejected/retried commitments & -- & 0 & -- & {n_retries:.0f} & reported separately \\\\")
    else:
        lines.append(f"\\multicolumn{{5}}{{c}}{{{_empty_note('ordering-a').strip('% ')}}} \\\\")
    lines += ["\\midrule",
              "\\multicolumn{6}{l}{\\footnotesize Beacon source: Simulated local trace; no external public beacon latency is claimed.} \\\\",
              "\\bottomrule", "\\end{tabular}"]
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
        for cert_type, g in configuration_groups(ok, ("cert_type", "projection", "rho", "gamma")):
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
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Factor & value & repetitions & median time (s) & time IQR (s) & median ARI \\\\",
        "\\midrule",
    ]
    if len(df):
        ok = df[df.get("_status") == "ok"] if "_status" in df else df
        if {"sweep_factor", "sweep_value"} <= set(ok.columns):
            for (factor, value), group in ok.groupby(["sweep_factor", "sweep_value"], sort=False):
                q1, q3 = group["t_total_s"].quantile([0.25, 0.75])
                lines.append(f"{_tex_escape(factor)} & {value:g} & {len(group)} & "
                             f"{group['t_total_s'].median():.4f} & {q3 - q1:.4f} & "
                             f"{group['ari'].median():.3f} \\\\")
        else:
            lines.append(f"legacy mixed sweep & -- & {len(ok)} & "
                         f"{ok['t_total_s'].median() if 't_total_s' in ok else float('nan'):.4f} & -- & "
                         f"{ok['ari'].median() if 'ari' in ok else float('nan'):.3f} \\\\")
    else:
        lines.append(f"\\multicolumn{{6}}{{c}}{{{_empty_note('sensitivity-a').strip('% ')}}} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write("sensitivity-a", "\n".join(lines) + "\n")


def table_precision_a():
    df = _read("rq5_precision")
    lines = [
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Dataset & precision bits & SparseJL $q$ & seeds & ARI mean [95\\% CI] & "
        "$Q_{0.95}(|\\delta|)$ mean [95\\% CI] & $r_\\Phi$ mean [95\\% CI] & "
        "symbolic constraint units \\\\",
        "\\midrule",
    ]
    if len(df):
        ok = df[df["_status"].eq("ok")]
        for keys, group in ok.groupby(["dataset_name", "d", "m", "q", "precision_bits"], dropna=False):
            dataset, d, m, q, bits = keys
            estimates = group["estimated_constraints"].unique()
            if len(estimates) != 1:
                raise ValueError("precision sweep has inconsistent constraint estimates")
            lines.append(
                f"{_tex_escape(dataset)} ($d={int(d)},m={int(m)}$) & {int(bits)} & {int(q)} & "
                f"{group['seed'].nunique()} & {_mean_ci_text(group['ari'])} & "
                f"{_mean_ci_text(group['delta_q95'])} & {_mean_ci_text(group['r_phi'])} & "
                f"{int(estimates[0])} \\\\")
    else:
        lines.append("\\multicolumn{8}{c}{No precision sweep results available.} \\\\")
    lines += [
        "\\midrule",
        "\\multicolumn{8}{l}{\\footnotesize Constraint counts are symbolic unit estimates, not measured gnark counts. At fixed $B=4096$, precision bits do not change circuit range constraints.} \\\\",
        "\\bottomrule", "\\end{tabular}",
    ]
    _write("precision-a", "\n".join(lines) + "\n")


def table_ablation_a():
    df = _read("ablation")
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Variant & proving (s) & RAM (GiB) & constraints & status \\\\",
        "\\midrule",
    ]
    if len(df):
        df = df[df["_status"].isin(["ok", "native_only"])]
        for variant, g in configuration_groups(df, ("variant",)):
            prove = g["prove_seconds"].median() if "prove_seconds" in g and g["prove_seconds"].notna().any() else float("nan")
            constraints = g["num_constraints"].median() if "num_constraints" in g and g["num_constraints"].notna().any() else float("nan")
            ram = g["peak_rss_bytes"].median() / (1024 ** 3) if "peak_rss_bytes" in g and g["peak_rss_bytes"].notna().any() else float("nan")
            status = g["_status"].mode().iat[0] if "_status" in g and len(g) else "--"
            display_variant = ("incircuit-gen (Evaluated as native-only baseline)"
                               if str(g["variant"].iloc[0]) == "incircuit-gen" else variant)
            prove_text = f"{prove:.4f}" if np.isfinite(prove) else "--"
            ram_text = f"{ram:.3f}" if np.isfinite(ram) else "--"
            constraint_text = f"{constraints:.0f}" if np.isfinite(constraints) else "--"
            lines.append(f"{_tex_escape(display_variant)} & {prove_text} & "
                          f"{ram_text} & {constraint_text} & {_tex_escape(status)} \\\\")
    else:
        lines.append(f"\\multicolumn{{4}}{{c}}{{{_empty_note('ablation-a').strip('% ')}}} \\\\")
    lines += ["\\midrule",
              "\\multicolumn{5}{l}{\\footnotesize incircuit-gen is not implemented in the circuit and is evaluated as a native-only baseline.} \\\\",
              "\\bottomrule", "\\end{tabular}"]
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
    "cert-a": table_cert_a, "sensitivity-a": table_sensitivity_a,
    "precision-a": table_precision_a, "ablation-a": table_ablation_a,
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
