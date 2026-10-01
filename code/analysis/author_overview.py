"""Build a plain-English overview and static charts from completed paper runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "overview" / "analysis"
RQ1_ANALYSIS = RESULTS / "rq1" / "analysis"
RQ3_ANALYSIS = RESULTS / "rq3" / "analysis"
RQ5_ANALYSIS = RESULTS / "rq5" / "analysis"
COLORS = {"SparseJL": "#087e8b", "PCA": "#e07a38", "Raw reference": "#44536a"}
NAMES = {
    "gaussian_mixture": "Gaussian mixture",
    "dbscan_margin": "DBSCAN margin",
    "mnist": "MNIST",
    "fashion_mnist": "Fashion-MNIST",
    "gisette": "Gisette",
    "arcene": "Arcene",
    "20ng_tfidf": "20 Newsgroups",
}
ORDER = list(NAMES)


def theme() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.labelcolor": "#243247",
        "text.color": "#243247", "axes.edgecolor": "#bdc7d2",
        "grid.color": "#e2e8ef", "figure.facecolor": "white",
        "savefig.facecolor": "white", "savefig.dpi": 220,
    })


def save(fig: plt.Figure, folder: Path, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / name, bbox_inches="tight", dpi=220)
    plt.close(fig)


def rq1_selection() -> pd.DataFrame:
    data = pd.read_csv(RQ1_ANALYSIS / "rq1_all_configurations.csv")
    data = data[(data.k == 5) & (data.dataset_name.isin(ORDER))]
    data = data[(data.dataset_name != "gaussian_mixture") | (data.d == 256)]
    return data[
        ((data.projection == "sparse_jl") & (data.q == 2))
        | ((data.projection == "pca") & (data.q == 1))
        | ((data.projection == "identity") & (data.q == 1))
    ].copy()


def chart_a(data: pd.DataFrame) -> pd.DataFrame:
    selection = data[(data.m == 64) & data.projection.isin(["sparse_jl", "pca"])]
    if len(selection) != 14:
        raise ValueError(f"Expected 7 SparseJL/PCA pairs at m=64; found {len(selection)}")
    plot = selection.pivot(index="dataset_name", columns="projection", values="ari_mean").loc[ORDER]
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(11.8, 5.6), layout="constrained")
    ax.barh(y + .18, plot.sparse_jl, height=.33, color=COLORS["SparseJL"], label="SparseJL")
    ax.barh(y - .18, plot.pca, height=.33, color=COLORS["PCA"], label="PCA")
    ax.axvline(1, color=COLORS["Raw reference"], lw=2, linestyle="--", label="Raw reference (ARI = 1)")
    ax.set_yticks(y, [NAMES[name] for name in ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Assignment agreement with raw-space K-means (ARI; higher is better)")
    fig.suptitle("RQ1  |  Quality after reduction to 64 dimensions", fontsize=15, weight="bold")
    ax.set_title("Same dataset, k = 5; SparseJL q = 2; 10 seeds per configuration",
                 loc="left", fontsize=9, color="#617087", pad=8)
    ax.grid(axis="x", alpha=.7)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, ncol=3, fontsize=8)
    save(fig, RQ1_ANALYSIS, "method_quality.png")
    return plot


def chart_b(data: pd.DataFrame) -> None:
    selection = data[data.projection.isin(["sparse_jl", "pca"])]
    fig, axs = plt.subplots(2, 4, figsize=(14.5, 7.3), sharey=True, layout="constrained")
    for ax, dataset in zip(axs.flat, ORDER):
        part = selection[selection.dataset_name == dataset]
        for key, label in (("sparse_jl", "SparseJL"), ("pca", "PCA")):
            values = part[part.projection == key].sort_values("m")
            ax.plot(values.m, values.r_phi_mean, marker="o", lw=2, ms=4,
                    color=COLORS[label], label=label)
        ax.axhline(1, color=COLORS["Raw reference"], linestyle="--", lw=1.5)
        ax.set_title(NAMES[dataset], fontsize=10)
        ax.set_xticks(sorted(part.m.unique()))
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.grid(axis="y", alpha=.8)
        ax.set_axisbelow(True)
        ax.set_ylim(.992, 1.025)
        ax.set_xlabel("Reduced dimensions m")
    axs.flat[-1].axis("off")
    legend = [Line2D([0], [0], color=COLORS[label], lw=2, marker="o", label=label)
              for label in ("SparseJL", "PCA")]
    legend.append(Line2D([0], [0], color=COLORS["Raw reference"],
                         lw=1.5, linestyle="--", label="Raw reference = 1"))
    axs.flat[-1].legend(handles=legend, loc="upper left", frameon=False)
    axs.flat[-1].text(0, .70, "Closer to 1 is better.\n\nThis is clustering objective,\nnot runtime or proof cost.",
                      transform=axs.flat[-1].transAxes, fontsize=10, color="#617087", va="top")
    fig.suptitle("RQ1  |  Clustering objective ratio vs. reduced dimensions", fontsize=15, weight="bold")
    save(fig, RQ1_ANALYSIS, "objective_ratio.png")


def chart_c() -> tuple[pd.DataFrame, pd.DataFrame]:
    gamma = pd.read_csv(RQ5_ANALYSIS / "rq5_gamma_summary.csv")
    precision = pd.read_csv(RQ5_ANALYSIS / "precision_summary_analysis.csv")
    precision = precision[precision.precision_bits.isin([8, 12, 16, 24])]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.2, 4.8), layout="constrained")
    left.plot(gamma.sweep_value, gamma.ari_mean, marker="o", lw=2.5,
              color=COLORS["SparseJL"])
    left.fill_between(gamma.sweep_value, gamma.ari_ci95_low.clip(0, 1),
                      gamma.ari_ci95_high.clip(0, 1), color=COLORS["SparseJL"], alpha=.15)
    left.set(xlabel="Cluster separation gamma", ylabel="ARI vs. full-space reference",
             title="A. More separated clusters are easier", ylim=(-.04, 1.05))
    left.grid(alpha=.7)
    for dataset, color in (("gaussian_mixture", "#087e8b"),
                           ("dbscan_margin", "#e07a38"), ("mnist", "#7058a3")):
        part = precision[precision.dataset_name == dataset].sort_values("precision_bits")
        right.plot(part.precision_bits, part.ari_mean, marker="o", lw=2.4,
                   color=color, label=NAMES[dataset])
    right.set(xlabel="Quantization bit width", title="B. Precision changes fidelity",
              ylim=(-.04, 1.05), xticks=[8, 12, 16, 24])
    right.legend(frameon=False, loc="lower right")
    right.grid(alpha=.7)
    fig.suptitle("RQ5  |  Sensitivity to data difficulty and precision", fontsize=15, weight="bold")
    fig.text(.5, -.035, "Separate native experiments; curves do not represent a joint gamma-by-precision sweep. Precision uses one shared 24-bit reference.",
             ha="center", fontsize=9, color="#617087")
    save(fig, RQ5_ANALYSIS, "separation_precision.png")
    return gamma, precision


def chart_d() -> pd.DataFrame:
    data = pd.read_csv(RQ3_ANALYSIS / "rq3_paired_summary.csv")
    labels = ["Gaussian\n500 x 256", "Gaussian\n1,000 x 1,024", "MNIST\n500 x 784"]
    x = np.arange(len(data))
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.8, 4.7), layout="constrained")
    width = .34
    left.bar(x - width / 2, data.prepublished_transaction_s_mean, width,
             color="#a5b4c4", label="Map ready before transaction")
    left.bar(x + width / 2, data.postcommit_transaction_s_mean, width,
             color=COLORS["SparseJL"], label="Post-commit map generation")
    left.set(xticks=x, xticklabels=labels, ylabel="Native transaction time (seconds)",
             title="A. Whole transaction")
    left.legend(frameon=False, fontsize=8)
    left.grid(axis="y", alpha=.7)
    right.bar(x, data.incremental_after_commit_s_mean, color=COLORS["PCA"])
    for i, row in data.iterrows():
        pct = 100 * row.incremental_after_commit_s_mean / row.postcommit_transaction_s_mean
        right.text(i, row.incremental_after_commit_s_mean + .025,
                   f"{pct:.2f}% of total", ha="center", fontsize=9)
    right.set(xticks=x, xticklabels=labels, ylabel="After-commit stage (seconds)",
              title="B. Incremental seed/map stage", ylim=(0, .78))
    right.grid(axis="y", alpha=.7)
    for ax in (left, right):
        ax.set_axisbelow(True)
    fig.suptitle("RQ3  |  Post-commit work is a small part of native time", fontsize=15, weight="bold")
    fig.text(.5, -.04, "10 paired seeds. Prepublished map setup is excluded from its transaction; full-transaction differences are not statistically resolved.",
             ha="center", fontsize=9, color="#617087")
    save(fig, RQ3_ANALYSIS, "transaction_overhead.png")
    return data


def dataset_table() -> str:
    rq1 = pd.read_csv(RQ1_ANALYSIS / "rq1_all_configurations.csv")
    entries = ROOT / ".cache" / "commitjl_data" / "manifest_entries"
    meta = {}
    for path in entries.glob("*.json"):
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("source_n") and item.get("name") in NAMES:
            meta[item["name"]] = (item["source_n"], item["source_d"])
    labels = {
        "gaussian_mixture": "Synthetic; generated for each shape",
        "dbscan_margin": "Synthetic; generated for each shape",
        "mnist": "70,000 x 784",
        "fashion_mnist": "70,000 x 784",
        "gisette": "6,000 x 5,000 training set",
        "arcene": "100 x 10,000 training set",
        "20ng_tfidf": "11,314 training texts; 2,000 TF-IDF features",
    }
    lines = ["| Dataset | Type | Source pool / features | RQ1 tested `n x d` |",
             "| --- | --- | --- | --- |"]
    for dataset in ORDER:
        shapes = rq1[rq1.dataset_name == dataset][["n", "d"]].drop_duplicates()
        tested = ", ".join(f"{int(row.n):,} x {int(row.d):,}" for row in shapes.itertuples())
        pool = labels[dataset]
        if dataset in meta:
            src_n, src_d = meta[dataset]
            if dataset != "20ng_tfidf":
                pool = f"{src_n:,} x {src_d:,}" + (" training set" if dataset in {"gisette", "arcene"} else "")
        kind = "Synthetic" if dataset in {"gaussian_mixture", "dbscan_margin"} else "Real"
        lines.append(f"| {NAMES[dataset]} | {kind} | {pool} | {tested} |")
    return "\n".join(lines)


def write_report(rq1: pd.DataFrame, quality: pd.DataFrame,
                 gamma: pd.DataFrame, precision: pd.DataFrame, overhead: pd.DataFrame) -> None:
    inventory = pd.read_csv(OUT / "run_inventory.csv")
    counts = {row.experiment: (int(row.ok), int(row.failed)) for row in inventory.itertuples()}
    def ar(dataset: str, method: str) -> float:
        return float(quality.loc[dataset, method])
    def pr(dataset: str, bits: int) -> float:
        return float(precision[(precision.dataset_name == dataset)
                               & (precision.precision_bits == bits)].ari_mean.iloc[0])
    native_total = sum(ok for label, (ok, _) in counts.items() if label != "RQ2 proof")
    report = f"""# Project overview for authors

This project asks whether a **sparse, randomly chosen dimension-reduction map** can preserve useful clustering results while fitting a commit-then-challenge zero-knowledge protocol. The laptop runs show where clustering quality survives reduction. They **do not yet show a successful full-size ZK proof**: both selected RQ2 proof attempts hit the memory watchdog.

**How to read the numbers.** `n` is the number of analyzed points, `d` the original number of features, and `m` the number after reduction. ARI measures how closely the reduced-data cluster assignments match K-means assignments on the original data: 1 means identical assignments, 0 means chance-level agreement. It is **not** accuracy against the dataset's true labels. The raw full-space reference has ARI 1 by definition. Each main configuration has ten recorded seeds; the plots show means, with a seed-level 95% interval on the separation chart.

## Which data actually ran?

{dataset_table()}

The real-data rows use the displayed samples, not whole source datasets. Arcene's 100 training rows are the entire training pool used by its loader; its nominal `n_max=200` could not increase that pool. The 20 Newsgroups representation is a fitted 2,000-feature TF-IDF matrix, not raw text. The two synthetic families have no larger source pool. RQ1 applies K-means comparisons even to the DBSCAN-margin generator; RQ4 separately checks DBSCAN neighbor-pair stability. RQ3-RQ5 use targeted, generally smaller samples: RQ3 uses Gaussian `500 x 256`, Gaussian `1,000 x 1,024`, and MNIST `500 x 784`; the RQ4 real supplement uses 1,000 rows each from MNIST, Fashion-MNIST, Gisette and 20 Newsgroups, plus all 100 Arcene training rows; RQ5 precision uses Gaussian `1,000 x 256`, DBSCAN margin `1,000 x 128`, and MNIST `1,000 x 784`.

## What did each research question test?

| Question | Plain-language test | What completed |
| --- | --- | --- |
| **RQ1 — clustering quality and reduction** | After compressing from `d` to `m` features, do points still join the same clusters as in full space? How does SparseJL compare with PCA and other native baselines? | {counts['RQ1 fidelity'][0]:,} successful **native** rows, seven dataset families, ten seeds per configuration. |
| **RQ2 — ZK proving cost and memory** | Can the protocol prove its commitment, projection and assignments, and how much time, memory and circuit size does that take? | Two representative full-shape proof attempts, **zero successes**. Gaussian `2,000 x 256 -> 16` and Arcene `100 x 10,000 -> 16` stopped at the memory watchdog. No measured full-shape prover cost. |
| **RQ3 — security and post-commit overhead** | What host work is added when the map is fixed only after the data commitment? How does a prepublished map compare, and what does grinding show? | {counts['RQ3 overhead'][0]:,} overhead, {counts['RQ3 grinding'][0]:,} grinding, and {counts['RQ3 paired supplement'][0]:,} paired **native** rows. Beacon is a **simulated local trace**, not an external beacon. Grinding is simulation, not a proof of adversarial security. |
| **RQ4 — certificates and boundary cases** | Can margin checks identify stable cluster assignments or stable DBSCAN neighbor relations? What happens near the boundary? | {counts['RQ4 certificates'][0]:,} synthetic and {counts['RQ4 real supplement'][0]:,} real-data **native** rows. The certificate theorem's embedding condition must hold before its guarantee applies. |
| **RQ5 — sensitivity and quantization** | How do cluster separation, shape, sparsity and precision affect clustering quality? | {counts['RQ5 scaling'][0]:,} scaling, {counts['RQ5 separation supplement'][0]:,} separation, and {counts['Precision 24-bit reference'][0]:,} comparable 24-bit-reference precision **native** rows. The separate older 20-bit-reference sweep has {counts['Precision 20-bit reference'][0]:,} rows and should not be mixed into the same precision curve. |

Across these named campaigns, {native_total:,} native rows succeeded and the two selected full-shape ZK rows failed. Smaller earlier ZK smoke proofs do not replace the missing RQ2 measurements. Proof-based ablation is also unavailable at paper shapes. RQ2 input commitment covers `n x d` coordinates even when projected distance work uses only `m` dimensions, so a small `m` alone does not imply a small prover memory footprint.

## Four pictures of the measured results

### A. Quality at the same reduced dimension

![RQ1 ARI comparison](../../rq1/analysis/method_quality.png)

The plot fixes `m=64` and `k=5` in every dataset, with SparseJL `q=2`; PCA has no active `q`. Gaussian uses the `d=256` shape. The dashed raw line is the definition of the reference, not an independent win. SparseJL matches the raw Gaussian clustering (ARI {ar('gaussian_mixture','sparse_jl'):.3f}) and stays close on Arcene ({ar('arcene','sparse_jl'):.3f}); it falls behind PCA on MNIST ({ar('mnist','sparse_jl'):.3f} vs {ar('mnist','pca'):.3f}), Gisette ({ar('gisette','sparse_jl'):.3f} vs {ar('gisette','pca'):.3f}), and 20 Newsgroups ({ar('20ng_tfidf','sparse_jl'):.3f} vs {ar('20ng_tfidf','pca'):.3f}). These are assignment-fidelity comparisons at the same `m`, **not** ground-truth-label or ZK-cost comparisons.

### B. Clustering objective ratio across dimensions

![RQ1 objective ratio](../../rq1/analysis/objective_ratio.png)

This ratio compares the full-space clustering objective obtained from reduced-data assignments with the raw reference cost. A value near 1 means a similar objective, even when individual assignments differ. **It is not a time, RAM, or circuit-cost ratio.** The near-1 lines alongside lower ARI on some real datasets show why both quality metrics matter. More dimensions generally help assignment fidelity, but the objective ratio by itself can hide moved boundary points.

### C. Cluster separation and precision

![RQ5 sensitivity](../../rq5/analysis/separation_precision.png)

On the controlled Gaussian generator, mean ARI rises from {float(gamma.loc[gamma.sweep_value.eq(0), 'ari_mean'].iloc[0]):.3f} at `gamma=0` to {float(gamma.loc[gamma.sweep_value.eq(.5), 'ari_mean'].iloc[0]):.3f} at `gamma=0.5` and {float(gamma.loc[gamma.sweep_value.eq(1), 'ari_mean'].iloc[0]):.3f} at `gamma=1`. At a shared 24-bit physical reference, 8-bit quantization collapses every tested matrix to zero. Increasing precision from 16 to 24 bits improves mean ARI on the difficult DBSCAN-margin data ({pr('dbscan_margin',16):.3f} to {pr('dbscan_margin',24):.3f}) and MNIST ({pr('mnist',16):.3f} to {pr('mnist',24):.3f}). Sixteen bits is **not established as optimal**. The precision plot deliberately uses 8, 12, 16 and 24 bits from one normalization regime; it does not combine the older 20-bit-reference run. No real gnark constraint-vs-precision curve was measured.

### D. Native post-commit overhead

![RQ3 native overhead](../../rq3/analysis/transaction_overhead.png)

The added after-commit seed/map stage averaged {overhead.incremental_after_commit_s_mean.iloc[0]:.3f}, {overhead.incremental_after_commit_s_mean.iloc[1]:.3f}, and {overhead.incremental_after_commit_s_mean.iloc[2]:.3f} seconds across the three shapes, around {100*overhead.incremental_after_commit_s_mean.iloc[0]/overhead.postcommit_transaction_s_mean.iloc[0]:.2f}%, {100*overhead.incremental_after_commit_s_mean.iloc[1]/overhead.postcommit_transaction_s_mean.iloc[1]:.2f}%, and {100*overhead.incremental_after_commit_s_mean.iloc[2]/overhead.postcommit_transaction_s_mean.iloc[2]:.2f}% of their total native transactions. The full-transaction paired difference has a 95% interval crossing zero in every shape, so the experiment does **not** resolve an overall speed difference. Prepublished map preparation is measured separately and excluded from that baseline transaction. The run used competing local processes, and no ZK proving time or external beacon latency is included.

## Where SparseJL works, and where it falls short

- **Best observed behavior:** It preserves the easy, well-separated Gaussian assignments even at low `m` and performs well on Arcene's small training set. Its native after-commit map work is a small fraction of the whole transaction. Its sparse entries and transcript-derived map are structurally suited to a circuit that authenticates a challenged projection.
- **Data-dependent losses:** On MNIST, Gisette and 20 Newsgroups, SparseJL has much lower native assignment agreement than PCA at `m=64`. At `m=256, q=2, k=5`, SparseJL still reaches only about 0.693 ARI on Gisette and 0.552 on 20 Newsgroups. This is a strong limit on any blanket quality claim.
- **Certificate limits:** In all 100 real-data RQ4 rows, the required *global* embedding condition fails. Among 2,457 certified points satisfying the checked pointwise condition, none changed assignment, but unconditional certified assignment changes occurred when that premise failed. The approximately 99.95% DBSCAN pair coverage is dominated by far pairs; only 250 of 499,500 pairs per synthetic row are in the difficult near-radius band.
- **Precision and prover limits:** Eight-bit quantization destroys the tested signal. On difficult data, 24 bits beats 16 bits under the common reference. Both selected full-shape ZK proofs exceeded the laptop watchdog, so no measured prover advantage, memory curve, verification rate, or proof-based ablation can be claimed.
- **Why choose SparseJL over PCA?** PCA is often the better *native* fidelity baseline here. SparseJL remains interesting because its challenged sparse map can be reconstructed and checked inside the intended proof protocol without first fitting a data-dependent PCA basis. That is a **structural motivation**, not a measured full-size proof-cost victory; RQ2 must succeed before claiming that advantage empirically.

For paper use, describe these as ten-seed native results on the shown samples. The current files use seed-level Student-t intervals; the manuscript's described 30 projection seeds, ten clustering starts, hierarchical bootstrap and Holm adjustment were not executed here. The complete statistical design or the manuscript methods must be reconciled before final publication. See [the detailed analysis](analysis.md) and [paper-readiness review](paper_readiness_review.md) for supporting tables and caveats.
"""
    (OUT / "project_overview_for_authors.md").write_text(report, encoding="utf-8")


def main() -> None:
    theme()
    OUT.mkdir(parents=True, exist_ok=True)
    rq1 = rq1_selection()
    quality = chart_a(rq1)
    chart_b(rq1)
    gamma, precision = chart_c()
    overhead = chart_d()
    write_report(rq1, quality, gamma, precision, overhead)
    print(f"Wrote four charts and {OUT / 'project_overview_for_authors.md'}")


if __name__ == "__main__":
    main()
