# Complete comparison of the current optimized method

The experiment is `experiments.comprehensive_native_comparison`. It preserves
the current protected centroid span +12 training-residual PCA directions + q=1
canonical Sparse JL. Experimental text modifications remain in a separate study.
Every PCA model is fitted independently at its requested feasible width; no
512-component prefix is used for smaller PCA comparisons.

## Coverage

All saved coordinates and rows are used, with no extra dataset subsampling:

| Dataset | Saved rows x features |
|---|---:|
| Gaussian easy | 2000 x 256 |
| Gaussian hard | 2000 x 512 |
| DBSCAN margin | 1000 x 128 |
| MNIST | 5000 x 784 |
| Fashion-MNIST | 5000 x 784 |
| Gisette | 3000 x 5000 |
| Arcene | 100 x 10000 |
| 20 Newsgroups | 3000 x 2000 |
| Nonzero-text sensitivity | 2886 x 2000; same 114 original zero rows removed |

The deterministic 60/20/20 numerical split is shared. These archived subsets
have already been explored; a new random draw is not independent new test data.
The text vocabulary was learned before this numerical split.

k=5 and 10; requested total output widths 32, 64, 128, 256, 512, restricted to
widths smaller than the input feature count. Arcene additionally receives width
59, the maximum centered-PCA rank with 60 training rows. Wider Arcene JL outputs
are explicitly marked as lacking matched-rank PCA controls. Total hybrid width
includes the protected directions. Three new draws (30--32) crossed with three
new centroid/clustering starts (30--32), q=1, four CPU threads. Each learning and
timing stage has three repetitions after an excluded warmup.

Methods: Full, full-SVD PCA, default randomized PCA, randomized PCA with two power
iterations, plain Sparse JL, original Protected JL, Protected JL +12, and a
protected-only ablation at its own actual rank. The latter is not an equal-width
PCA comparison. `plan.json` records exact cases and feasible widths.

## Workloads and cost

Every case records separate model learning, construction, corpus conversion,
norm preparation, fixed-center assignment, 32 distinct distance queries, 5-NN
retrieval, fresh whole-corpus KMeans, centroid-initialized refinement, DBSCAN
pair-matrix construction, and DBSCAN fitting. Optimized Full controls are
interleaved in the downstream workload measurements. Assignment/fresh/refinement
are distinct tasks. Shared fitting checkpoints never erase their measured cost.

All those tasks use the full saved corpus except DBSCAN, which uses explicitly
saved indices for at most 256 corpus rows to bound quadratic work. Its radius
and sensitivity radii come only from original validation geometry. Query sets
use up to 512 points by 64 distinct landmarks; 5-NN uses up to 128 query points
against every corpus point. This restriction affects query counts, not corpus
compression or clustering. The same indices are used in Full and compressed
controls. Query results are not cached to advantage a method.

Cold costs include only the setup needed by the task: Full/plain-JL fresh
clustering or distance queries do not need trained centroids; protected variants
do. All fixed-center and centroid-initialized tasks include center learning for
every method. Fresh clustering needs no externally prepared norms or projected
fixed centers. DBSCAN counts norm preparation once, matrix building once and
clustering once. Sum-of-stage-medians totals exclude common archive I/O, process
startup and evidence writes; they are not direct process wall-clock benchmarks.

Saved evidence supports fresh/held-out/fixed labels and ARI/NMI, source-label ARI,
pair and center Q50/Q95/Q99/max distortion, zero-distance checks, partition SSE
and reconstructed-center objective, retrieval recall, DBSCAN neighborhood/core
events, iteration caps, cumulative reuse analysis and task-specific timings.
Pointwise crossed bootstrap intervals remain exploratory, conditional on fixed
corpora and without multiplicity correction. All-noise DBSCAN is flagged.

## Running

From `code/`, start a hidden independent Windows process:

```powershell
python -m campaigns.run_comprehensive_native --out results/comprehensive_native_current_20261016
```

`launcher.json` identifies the supervisor; `background_status.json` records the
worker PID and stage. The worker writes `manifest.json` and `run.log`; completed
cases are atomic checkpoints. The supervisor automatically audits and exports
`report.md`, `index.html`, `rows.csv`, `summary.csv`, `comparisons.csv`, and
`stage_means.csv` after completion. VS Code can close; the computer must remain
powered on. Idle system sleep is inhibited during the campaign and restored on
exit. An 8-hour per-invocation budget and 2.5 GiB between-stage worker RSS guard
bound execution. If interrupted or a guard triggers, results remain incomplete
and resumable with unchanged sources/configuration.

The launcher refuses to duplicate a live supervisor for the same output folder;
the experiment also uses an OS file lock. Source snapshots, checksums, per-case
artifacts and deduplicated model parameters are preserved. This native study
does not establish a ZK performance result or privacy of training-derived bases.

## Recovery after an unexpected shutdown

The 16 October restart keeps the numerical experiment and its pinned source
identity unchanged. `tools.recover_comprehensive_run` checks every committed
case artifact and fit-cache checksum, verifies the archived source copies, and
reconstructs the damaged manifest from the original CLI configuration. Invalid
rows/artifacts are quarantined, never silently accepted. `recovery_latest.json`
records the checks and retained count; the recovery directory preserves damaged
files and the former supervisor state. Cases with invalid evidence are rerun.

The supervisor now invokes `campaigns.comprehensive_durable_worker`. It replaces
only evidence I/O aliases with `commitjl.durable_io`: contents are flushed with
fsync before replacement, and the prior valid manifest is retained as
`manifest.previous.json`. Source copies and hashes for this I/O wrapper are in
`execution_io_sources/` and `execution_io.json`. Numerical routines, fitting,
random states and timed compute boundaries are unchanged. These changes reduce
checkpoint loss; they cannot prevent hardware failure or power loss.

The resumed run still executes independently in the background and automatically
runs the final audit/report. It must remain on a powered-on laptop.
