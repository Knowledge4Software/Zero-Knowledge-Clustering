"""Shared repetition summaries for paper-facing native measurements."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t


DEFAULT_METRICS = ("ari", "delta_q95", "r_phi")


def mean_std_ci95(values: Iterable[float]) -> dict[str, float | int]:
    sample = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(float)
    n = int(sample.size)
    if n == 0:
        return {"n": 0, "mean": math.nan, "std": math.nan, "ci95_low": math.nan, "ci95_high": math.nan}
    mean = float(np.mean(sample))
    if n == 1:
        return {"n": 1, "mean": mean, "std": math.nan, "ci95_low": math.nan, "ci95_high": math.nan}
    std = float(np.std(sample, ddof=1))
    half = float(t.ppf(0.975, n - 1) * std / math.sqrt(n))
    return {"n": n, "mean": mean, "std": std, "ci95_low": mean - half, "ci95_high": mean + half}


def summarize_repetitions(df: pd.DataFrame, group_columns: Sequence[str],
                          metrics: Sequence[str] = DEFAULT_METRICS) -> pd.DataFrame:
    ok = df[df["_status"].isin(["ok", "native_only"])].copy() if "_status" in df else df.copy()
    groups = [c for c in group_columns if c in ok.columns]
    available = [m for m in metrics if m in ok.columns]
    rows = []
    iterator = ok.groupby(groups, dropna=False, sort=False) if groups else [((), ok)]
    for key, group in iterator:
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, key))
        row["repetitions"] = int(group["seed"].nunique()) if "seed" in group else int(len(group))
        row["attempted_rows"] = int(len(group))
        for metric in available:
            stats = mean_std_ci95(group[metric])
            for name in ("mean", "std", "ci95_low", "ci95_high"):
                row[f"{metric}_{name}"] = stats[name]
            row[f"{metric}_n"] = stats["n"]
        rows.append(row)
    return pd.DataFrame(rows)


def write_native_summary(result_path: str | Path, group_columns: Sequence[str],
                         metrics: Sequence[str] = DEFAULT_METRICS,
                         output_path: str | Path | None = None) -> Path:
    result_path = Path(result_path)
    output = Path(output_path) if output_path else result_path.with_name(result_path.stem + "_summary.parquet")
    raw = pd.read_parquet(result_path)
    summary = summarize_repetitions(raw, group_columns, metrics)
    if summary.empty:
        raise ValueError(f"cannot summarize empty native results: {result_path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(output, index=False)
    summary.to_csv(output.with_suffix(".csv"), index=False)
    print(f"wrote native repetition summary: {output} (+.csv)")
    return output
