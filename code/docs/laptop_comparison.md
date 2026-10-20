# Laptop diagnosis, 11 October 2026

The target machine has an AMD Ryzen 7 5800H, 8 physical/16 logical cores and
approximately 13.86 GiB visible RAM. Jobs run sequentially with four threads,
a maximum 3 GiB Go RSS budget, process-tree monitoring, a system free-memory
floor, per-job deadlines and a 45-minute campaign deadline. These limits are
watchdogs, not hard OS reservations. The same command resumes checkpoints;
sources, archive index, binary and job plan must match. Close heavy applications
for more reproducible timing. Keep the computer awake for the run.

From CMD, run these lines:

```bat
cd /d C:\PyProjects\zero-knowledge\code
python -u -m campaigns.laptop_comparison --out results\laptop_comparison_next
```

The experimental v4 binary is already built locally. No server, installation,
network access, or additional datasets are required. A controlled stop or Ctrl+C
saves completed measurements. Running the same command continues unfinished
jobs; completed resource-limited rows remain explicit, not silently retried.
A bug failure stops the queue for review. Other processes are never terminated.

## Experiments

1. Geometry on all eight exact preserved server subsets, every original row:
   k=5/10; saved center starts 3/4; fresh maps 100/101/102; total output dimension
   20/64/128/256/512 where <=d; q=1/2. Full, cached PCA, plain Sparse JL,
   two-direction protection, full centroid-span protection, and protected-only
   ablation. All random/PCA comparisons use the same total dimension. PCA is
   skipped when the archived component count is insufficient (Arcene rank99).
   At m=d the row is a noncompressive control and cannot qualify as compression.
2. Direct normalized projected nearest-center and fixed-assignment cost ratios,
   original-space cost of projected labels, full point-to-center relative
   errors, fixed sampled pair errors, and five-neighbor recall on a saved sample
   of min(256,n) points. No sampled statistic is a global JL guarantee.
3. Native project/assign timing: three measurements after a small warm-up.
   Basis and map construction costs are separately recorded; cached PCA fitting
   is excluded. Matched output size does not imply matched quality.
4. Exact integer rounding screen on first min(128,n) rows of each same dataset,
   full saved centers, k=5/10, start3, three new maps, m=20/64/128 and
   T=1024/4096. Both integer Full and original float Full references are used.
   Labels are selected with exact integers, never float-converted distances.
5. Circuit prefix profiling for n8,d32,k5,total_m16,q1: signed input checks,
   center digest, data Merkle authentication, sparse authentication/validity,
   projection and projected range checks, complete assignment/cost relation.
   Prefixes are solver-only and prohibited from generating proofs. Deferred
   batching makes prefix differences an approximate engineering decomposition.
6. Compile/solve growth controls: n=2/8/32, d=32/128, k=5, plus k10 and q2
   controls; eight methods including specialization and protected-only. One
   MNIST prefix of two points at the original d784 tests real-shape feasibility.
   Synthetic shapes are not substitutes for full-corpus proof timings.
7. Actual Groth16 comparisons on (n,d,total_m,k,q) = (2,32,16,5,1),
   (8,32,16,5,1), (2,128,32,5,1). Full, PCA, plain sparse and shared/lookup
   hybrids get three measured proofs and one warm-up, only after successful
   solve and <=750,000 constraints. All authentication and range checks stay.
   A new specialized map requires new setup; no free specialization is claimed.

The whole paper subsets are: Gaussian easy 2000x256; hard 2000x512;
DBSCAN-margin 1000x128; MNIST and Fashion 5000x784 each; Gisette 3000x5000;
Arcene 100x10000; text 3000x2000. The paper subsets themselves remain subsets
of the original benchmark corpora. No real KMeans or PCA is retrained. Seeds
and starts are additional confirmation on fixed data, not an independent test
split. Native challenges and proof fixtures are experimental simulations; the
complete production randomness/key-validation protocol is not implemented.

## Outputs

`results/laptop_comparison_20261011/report.md` is generated even after a bounded
stop. CSVs include `geometry_raw`, `geometry_summary`, `qualified_candidates`,
`precision_raw`, `precision_summary`, `circuits_raw`, `constraint_breakdown`,
`proof_speedups`, and `all_results`. Per-job JSON checkpoints, logs, bases,
original distance references, lossless labels and projected distance arrays,
complete proof inputs and raw backend records remain available. Source copies
and hashes prevent mixing different code versions. Nonfinite/zero-reference
cases cannot silently qualify. Qualification needs a complete config across
starts/maps and every observed row to meet the declared exploratory thresholds.

Replay a geometry row or recover Z for additional plots:

```bat
python -m tools.rebuild_laptop_comparison --record results\laptop_comparison_20261011\jobs\geometry_mnist\rows\k5_s3_m20_r4_q1_seed100.json --out results\recovered_laptop_row.npz
```

Backend `single_run_total` sums compile/setup/witness plus median prove/verify
for one proof. It excludes Python preparation and cached PCA training. Both
prove-only and this backend total are exported. A speed win on synthetic data
and a quality win elsewhere do not establish a joint win on a real workload.
The planned shared-routing graph optimization and commit-and-prove integration
are not in these runs; this campaign measures the baseline to guide that work.

## Supplementary controls and final validation

After timed proving ends, the supplementary affine control reuses archived
maps/bases for k5, starts3/4, dimensions64/128/256 (where available) and map100,
q1/2. Full and all projections use BLAS affine classification plus objective
evaluation, with five warm measurements and four threads. This avoids treating
the earlier difference-based Full kernel as the optimal native baseline.
It also independently reconstructs a full-rank hybrid on all eight datasets.

```bat
python -u -m experiments.laptop_affine_control --out results\laptop_comparison_next
python -u -m experiments.laptop_pair_workload --out results\laptop_comparison_next
python -m analysis.finalize_laptop_comparison --out results\laptop_comparison_next
```

The finalizer checks completion, fingerprints, cell counts, matched proof
inputs, three verified proofs per actual control, reconstructed geometry and
native objective checks. It adds diagnosis.png and concise decision tables to
report.md; all original raw records are preserved. The timing control excludes
map/basis preparation and cached PCA fitting. It is a fixed-center workload,
not an end-to-end clustering benchmark.

The pair workload tests compression plus all pair distances on a deterministic
sample of min(1024,n), using the smallest tested qualified k5 hybrid. It reuses
start3/map100 and matches PCA dimension where archived rank permits. Three
warm measurements include projection and BLAS pair computation. Every unordered
pair in this sample is checked; coordinates, condensed distances and point
indices are saved. This is a separate native downstream use case, not an RQ2
proof or a formal full-corpus distance guarantee.

During the original run, the bridge's memory_limit outcome was initially
misclassified as a worker bug. A narrowly verified scheduling-only checkpoint
migration corrected that status without changing numerical computations or
the binary. The original manifest, source snapshots and raw failure remain
preserved in source_revisions/memory_limit_fix. General resume fingerprint
checks remain enforced. The migration utility applies only to that original
checkpoint and must not be rerun on an already migrated campaign.

The measured 11 October campaign is complete and preserved with its original
source snapshots. A later audit found that the interrupted d128 proof group
regenerated its commitment salt on resume. Its actual inputs/models match and
all proofs verify, but its speed ratios are excluded: adaptive RAM/GC budgets
also changed. The current worker authenticates and reuses the saved commitment
opening; regression tests reject corrupted openings. Since the code fingerprint
changed after measurement, use the fresh output folder in the commands above
for a new campaign. Do not replace or relabel the archived measurements.
