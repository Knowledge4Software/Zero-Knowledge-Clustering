"""Statistics: hierarchical bootstrap (dataset -> seed -> run), paired
ratio/difference bootstrap, Holm-Bonferroni correction, failure-inclusive
denominators, log-log slope fits.

Manuscript reference: "Statistics paragraph" -- hierarchical bootstrap
resampling at the dataset/seed/run levels; medians, IQR, 95% CIs, paired
absolute differences and ratios; Holm correction within each RQ; timeouts /
failures remain in the denominator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class BootstrapResult:
    point_estimate: float
    ci_low: float
    ci_high: float
    n_resamples: int


def hierarchical_bootstrap(
    data: Dict[str, Dict[int, List[float]]],
    statistic: Callable[[np.ndarray], float] = np.median,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapResult:
    """Three-level hierarchical bootstrap: dataset -> seed -> run.

    `data` maps dataset_name -> {seed: [run_value, run_value, ...]}.
    Each bootstrap replicate resamples datasets with replacement, then within
    each resampled dataset resamples seeds with replacement, then within each
    resampled seed resamples runs with replacement, then applies `statistic`
    to the pooled resampled values.
    """
    rng = np.random.default_rng(seed)
    dataset_names = list(data.keys())
    D = len(dataset_names)
    replicates = []
    all_values = [v for ds in data.values() for runs in ds.values() for v in runs]
    point = float(statistic(np.array(all_values))) if all_values else float("nan")

    for _ in range(n_resamples):
        pooled = []
        chosen_datasets = rng.choice(dataset_names, size=D, replace=True) if D else []
        for ds_name in chosen_datasets:
            seeds = list(data[ds_name].keys())
            S = len(seeds)
            if S == 0:
                continue
            chosen_seeds = rng.choice(seeds, size=S, replace=True)
            for sd in chosen_seeds:
                runs = data[ds_name][sd]
                R = len(runs)
                if R == 0:
                    continue
                idx = rng.integers(0, R, size=R)
                pooled.extend([runs[i] for i in idx])
        if pooled:
            replicates.append(statistic(np.array(pooled)))

    if not replicates:
        return BootstrapResult(point, float("nan"), float("nan"), 0)
    lo = float(np.quantile(replicates, alpha / 2))
    hi = float(np.quantile(replicates, 1 - alpha / 2))
    return BootstrapResult(point, lo, hi, len(replicates))


def paired_bootstrap(a: Sequence[float], b: Sequence[float], mode: str = "ratio",
                      n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0) -> BootstrapResult:
    """Paired ratio or difference bootstrap between two equal-length paired
    sequences (e.g. per-seed measurements for method A vs method B)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert len(a) == len(b) and len(a) > 0
    rng = np.random.default_rng(seed)
    n = len(a)

    def stat(x, y):
        if mode == "ratio":
            return float(np.mean(x) / np.mean(y)) if np.mean(y) != 0 else float("nan")
        elif mode == "difference":
            return float(np.mean(x) - np.mean(y))
        else:
            raise ValueError("mode must be 'ratio' or 'difference'")

    point = stat(a, b)
    replicates = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        replicates.append(stat(a[idx], b[idx]))
    lo = float(np.quantile(replicates, alpha / 2))
    hi = float(np.quantile(replicates, 1 - alpha / 2))
    return BootstrapResult(point, lo, hi, n_resamples)


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> List[bool]:
    """Holm-Bonferroni step-down correction. Returns a boolean list (same
    order as input) indicating which hypotheses are rejected at level alpha."""
    p_values = np.asarray(p_values, dtype=np.float64)
    m = len(p_values)
    order = np.argsort(p_values)
    reject = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if p_values[idx] <= threshold:
            reject[idx] = True
        else:
            break  # Holm: stop at first non-rejection
    return reject.tolist()


def failure_inclusive_rate(n_success: int, n_failure: int) -> float:
    """Failure-inclusive denominator: rate over ALL attempts, not just
    completed ones."""
    total = n_success + n_failure
    return n_failure / total if total else float("nan")


def loglog_slope(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """Fit y ~ C * x^alpha via linear regression in log-log space:
    log(y) = alpha*log(x) + log(C). Returns slope (alpha), intercept
    (log C), and R^2."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = (x > 0) & (y > 0)
    lx, ly = np.log(x[mask]), np.log(y[mask])
    if len(lx) < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    A = np.vstack([lx, np.ones_like(lx)]).T
    coef, residuals, rank, sv = np.linalg.lstsq(A, ly, rcond=None)
    slope, intercept = coef
    y_pred = A @ coef
    ss_res = np.sum((ly - y_pred) ** 2)
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2)}
