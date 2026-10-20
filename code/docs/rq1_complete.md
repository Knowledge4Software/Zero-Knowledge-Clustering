# RQ1 comparison campaign and reusable plotting data (v3)

Run from `code/`. This new entry point leaves the older runner/campaign intact.
By default it evaluates canonical sparse JL, full-space controls, Gaussian,
Rademacher, Achlioptas, coordinate sampling and PCA on all eight cases.
An explicitly unverified Pisot surrogate is available only when requested
with `--methods`. It does not run ZK proofs or projected KMeans
training. The Pisot surrogate is not the cited authors' verified algorithm;
that manuscript row cannot be called a reproduced baseline until the actual
artifact is verified. Surrogate rows are excluded from operating-point selection.

## Data scope (explicit subsets)

| Case | Default analyzed rows | Features |
| --- | ---: | ---: |
| Gaussian, gamma 3 | 2,000 | 256 |
| Gaussian, gamma 2, imbalance 2 | 2,000 | 512 |
| DBSCAN margin generator | 1,000 | 128 |
| MNIST | 5,000 random rows | 784 |
| Fashion-MNIST | 5,000 random rows | 784 |
| Gisette | 3,000 random rows | 5,000 |
| Arcene | 100 available training rows | 10,000 |
| 20 Newsgroups train TF-IDF | 3,000 random rows | up to 2,000 |

Real-data subsets use fixed seed 12345, independently of projection and
clustering seeds. Loaders record actual dimensions, source counts and processed
hashes in each run manifest. TF-IDF is fitted on the loader's full training
corpus before subsampling, as in the established loader. Dataset labels do not
determine the chosen clustering k.

`--scope full` removes `n_max` for real datasets: MNIST/Fashion-MNIST use all
70,000 loader rows, Gisette its available training rows, text all 11,314 train
documents, and Arcene still only the supported training split. It does not
implement the paper's larger combined Arcene dataset or change synthetic sizes.
The optional embedding corpus is excluded because the paper has not selected
a concrete embedding workload. Full-loader scope needs its own resource profile.

## Commands

```bash
# Plan only; no dataset loading or experiments.
python -m experiments.rq1_complete

# All datasets, full m/q/k grid, but only 2 projection seeds x 2 starts.
# Measures process RSS and extrapolates compute stages to the requested 10x5.
python -m experiments.rq1_complete --profile --out results/rq1/runs/my_resource_profile

# Actual paper-size repetition design on the explicitly listed subsets.
python -m experiments.rq1_complete --run

# Profile full-loader scope before attempting that larger run.
python -m experiments.rq1_complete --profile --scope full --out results/rq1/runs/full_scope_profile

# Focused debugging or a chosen dataset; retain distinct output folders/configs.
python -m experiments.rq1_complete --run --datasets arcene --seeds 2 --starts 2 --m 32 --q 2 --k 5 --out results/rq1/runs/arcene_debug

# Your Ubuntu server: 4 dataset jobs x 4 threads = 16 numerical threads.
# First profile the expanded grid on that server; these commands run from code/.
python -m experiments.rq1_complete --profile --jobs 4 --threads 4 --max-rss-gib 8 --out results/rq1/runs/server_profile_v3
python -m experiments.rq1_complete --run --jobs 4 --threads 4 --max-rss-gib 8 --out results/rq1/runs/server_rq1_v3

# Export tables and figures after the campaign, without rerunning calculations.
python -m analysis.rq1_export --runs results/rq1/runs/server_rq1_v3

# To run only our sparse method and the implicit full control:
python -m experiments.rq1_complete --run --methods sparse_jl --out results/rq1/runs/sparse_only_v3
```

RQ1 holds an exclusive lock on its output directory while running. The exporter
uses the same lock and refuses to rewrite an active run. An interrupted process
releases the OS lock; the ignored `.campaign.lock` file may remain and does not
prevent resuming. Use the same source/configuration to resume and a fresh
output directory for any changed design.

Resume by repeating the exact same command/output directory. Source, config,
loader metadata and data hashes must match; otherwise use a new run directory.
Recorded rows are not repeated. KMeans baselines and summaries are recomputed
on resume; those costs are included in resumed wall time. A memory/timeout/error
attempt is recorded in `resources.json`, its log remains available, and the
command exits nonzero after processing the selected datasets. Increasing the
RSS/timeout limit does not change mathematical config and permits resume.

## Scientific design and outputs

- Two cluster counts: k=5 and 10.
- m=16,32,64,128,256,512,1024, omitting m>d. q=1,2,4, omitting q>m.
- q applies only to sparse JL. Other methods use q=0 (inactive). PCA dimensions
  above min(n-1,d) are omitted, not relabeled: Arcene PCA supports 16/32/64 in
  this grid. PCA fits one deterministic randomized decomposition per dataset;
  its saved components are reused. This differs from the old exact-SVD runner.
- 10 projection seeds crossed with 5 independent KMeans starts. Each start's
  full-space centers are shared across all projections. Synthetic data seed is
  fixed, so uncertainty is conditional on that generated instance.
- PCA and full controls have only five start rows per operating point, with
  projection_seed=-1. Randomized methods have 10x5 rows. Default scope/grid
  produces 34,120 rows, including 14,400 sparse JL observations. A fresh version
  directory is required; do not mix these outputs with the old v2 profiles.
- Canonical `SparseMatrixGen` generates integer A; sparse execution uses
  P=A/sqrt(q), normalized exactly once. No dense n-by-n distance matrix.
- One fixed 2,000-pair sample per dataset, shared across all configurations.
  Identical-coordinate pairs are counted and excluded from distortion ratios.
- ARI/NMI/raw assignment agreement; original-space objective ratio for centers
  derived from projected assignments; fixed-center assignment-cost ratio;
  geometric quantiles/exceedance; margin qualification, realized pointwise
  embedding premise, and qualified assignment violations.
- Ground-truth ARI/NMI are secondary where labels exist. Projection-stage times
  are shared by all starts, not independent per-start latency observations.
- Bootstrap resamples projection seeds and clustering starts independently as
  crossed axes. CIs are conditional on one fixed dataset, not uncertainty over
  source populations. Eligibility requires lower ARI CI >=0.80 and upper
  objective-ratio CI <=1.10 and complete rows. Random methods require at least
  two seeds; all methods require at least two starts. PCA CIs reflect starts
  only. Selected points are per dataset/method/k, with smallest m then q.
  Selection CIs are pointwise, not simultaneous guarantees over the frontier;
  no unperformed Holm-adjusted hypothesis tests are claimed.

Per dataset: `manifest.json`, resumable `rows.jsonl`, exported `rows.parquet`,
`summary.csv`, `selected_operating_points.csv`, `completion.json`, and `run.log`.
The root `resources.json` records all dataset attempt statuses, peak process-tree
RSS, wall time and profile extrapolations. No eligible point is represented by
an empty selection CSV; the runner does not silently promote a failing point.
`grid_inventory.json` declares every requested group, expected rows and dimension
exclusions before computing. The export inventory shows wholly uncompleted
groups rather than silently dropping missing methods or dimensions.

Saved arrays (compressed NPZ): original X/ground-truth labels/pair indices;
full-space centers, labels and point-to-center squared distances per k/start;
projected Z once per method/m/q/seed; projected assignment labels, pointwise
maximum center-distance distortion, assigned original/projected costs and signed
pair distortions per k/configuration. These permit new histograms, ECDFs, point
scatter plots, confusion matrices and margin diagnostics without rerunning the
campaign. Source/config/data provenance stays attached to the measurements.

The exporter produces:

- `fidelity_table.csv` and `.tex`: selected operating points with objective,
  ARI, distortion q95, exceedance and certified coverage CIs. Failed methods
  show an explicitly marked largest-m/smallest-q fallback, not an eligible point.
- `all_configurations.csv/.parquet`, `operating_points.csv`,
  `completion_inventory.csv`, and `raw_files.json` for custom analysis.
- PDF/PNG fidelity curves (objective, ARI, q95), distortion ECDF and sparse
  m/q ARI heatmap, separately for each dataset/k. The q95 plot uses a symmetric
  log y-axis to keep very large surrogate errors from hiding the other methods.
- Dataset dimensions, fixed-center scope, CI interpretation and conditional
  certificate meaning in the output README. Raw full conditional coverage is
  1; PCA table coverage is n/a to match the draft. No conditional certificate
  coverage is described as an unconditional global guarantee.

Reserve roughly 100 GiB free disk for the default campaign, arrays and downloads;
this is a conservative storage allowance, not a measured compressed output size.
The uncompressed projected/pointwise arrays alone are tens of GiB. Your 200 GB
disk is suitable if sufficient space remains free. Larger full-loader scope
can be substantially larger.

## RAM and time interpretation

**The following measurements are historical v2 sparse-only measurements, not
runtime or RAM measurements for the expanded v3 baseline campaign.** In
particular the previous ~2-hour four-job server estimate cannot be reused for
the expanded workload. Use the v3 server profile command above. A focused
all-methods implementation check on Gaussian and Arcene succeeded, but does not
measure the full dimension grid or its full repetition cost.

Measured on this machine on 2026-10-05: Windows, Python 3.12.10, eight physical
CPU cores, 13.86 GiB total system RAM, four numerical threads, cached sources.
All eight profiles completed, totaling 1,280 observations at 2 seeds x 2 starts.
No qualified certificate violations were recorded. These pilots are resource
measurements, not the final 30x10 statistical evidence.

| Dataset | Measured pilot peak process RSS (GiB) | Estimated 30x10 time (minutes) |
| --- | ---: | ---: |
| Gaussian easy | 0.17 | 9.7 |
| Gaussian hard | 0.18 | 15.5 |
| DBSCAN margin | 0.16 | 3.1 |
| MNIST | 0.99 | 39.5 |
| Fashion-MNIST | 1.01 | 44.2 |
| Gisette | 0.49 | 122.0 |
| Arcene | 0.22 | 26.1 |
| 20 Newsgroups TF-IDF | 0.43 | 52.9 |

Total extrapolated runtime: **5.2 hours**. A **6-8 hour planning allowance** is
reasonable on this machine; it is not a measured full runtime or confidence
interval. Reserve **2-3 GiB available RAM for the campaign**, in addition to OS
and other applications. Current 13.86 GiB system RAM is sufficient for this
sequential subset campaign based on the pilots. The full-repetition peak has
not been measured, so retain the 3 GiB watchdog. Full-loader scope and additional
projection baseline campaigns need separate measurements.

Raw resource records and per-dataset logs are in
[`resource_profile_20261005`](../results/rq1/runs/resource_profile_20261005/).

Four dataset processes run by default; use `--jobs 1` on a smaller machine. Each uses
four numerical threads by default. Set jobs x threads to your CPU budget.
The parent monitors its process tree every 100 ms; default ceiling is 8 GiB per job.
This is a stop threshold, not a guarantee against brief unsampled peaks or an
operating-system OOM. Loader transient memory is monitored too: selecting 5,000
MNIST rows does not mean the loader never holds its full source array.

Seed repetition count mainly increases time/output, not peak projection memory;
the full run retains ten baseline centroid/distance sets instead of two in the
profile. Raw X alone occupies `8*n*d` bytes, but loading, KMeans, projections,
distance workspaces, imports and result export add memory. Do not equate raw
array size with required machine RAM.

Runtime estimates use measured baseline time scaled by starts, projection time
scaled by seeds, and assignment/objective/certificate evaluation time scaled by
both. Loading and summary costs are added once. They are extrapolations for a
fresh run on identical scope/grid/threads; interruption/resume, cold downloads,
thermal throttling, other processes and full-run bootstrap/export overhead can
change elapsed time. Use a fresh profile directory for resource estimates.
