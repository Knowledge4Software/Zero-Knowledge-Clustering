# Paper compression and downstream experiment protocol

This is the new **native numerical** study. It compares Full, full-SVD PCA,
randomized PCA, ordinary Sparse JL, protected centroid-span + residual Sparse JL, and the
protected-only ablation. It does not measure ZK proof generation. Its coverage
export explicitly identifies the remaining proof/theory requirements.

## Data and fitting

Use the SHA-verified server archive, entirely offline. The preserved corpus is
the experimental input; no new random subset is silently substituted.

| Corpus | Preserved rows × features | Train / validation / test | Source subset |
|---|---:|---:|---|
| Gaussian easy | 2000 × 256 | 1200 / 400 / 400 | Entire saved synthetic fixture |
| Gaussian hard | 2000 × 512 | 1200 / 400 / 400 | Entire saved synthetic fixture |
| DBSCAN margin | 1000 × 128 | 600 / 200 / 200 | Entire saved synthetic fixture |
| MNIST | 5000 × 784 | 3000 / 1000 / 1000 | 5000 of the 70000 combined source rows |
| Fashion-MNIST | 5000 × 784 | 3000 / 1000 / 1000 | 5000 of the 70000 combined source rows |
| Gisette | 3000 × 5000 | 1800 / 600 / 600 | 3000 of the 6000 training source rows |
| Arcene | 100 × 10000 | 60 / 20 / 20 | All 100 labeled training source rows |
| 20 Newsgroups TF-IDF | 3000 × 2000 | 1800 / 600 / 600 | 3000 of the 11314 training documents |
| Text nonzero sensitivity | 2886 × 2000 | 1731 / 577 / 578 | Remove exactly 114 all-zero feature rows |

The split is deterministic, shuffled, and disjoint. All original image/UCI
coordinates are preserved without new scaling. Text features use the archived
L2-normalized TF-IDF representation and vocabulary. That vocabulary was fitted
on the original training corpus before this numerical split, so this is a
representation compression study, not a raw-text generalization experiment.
The same datasets have been explored earlier: a new held-out split is not a new
independent study population. Arcene has only 20 test points: report that limit.

Train KMeans with k=5 and 10, n_init=1 for each explicit start, Lloyd's algorithm,
tol=1e-4 and max_iter=300. Use those SAME original-space centers for fixed-center
Full/PCA/JL comparisons. Learn the protected basis only from the trained centers.
Train PCA only on the training fold, using one full-SVD fit to the largest
requested feasible dimension, then take nested prefixes. Its fit is a shared
sweep cost; do not call it a measured independent fit at every m. Randomized PCA
fits that same largest feasible dimension separately per draw on the training
fold and reuses its nested prefixes on validation/test. Include this optimized
solver baseline before making a broad claim about beating PCA.

## Configurations and selection

`paper`: 10 independent projection draws × 5 centroid starts; k ∈ {5,10},
m ∈ {16,32,64,128,256,512}, q ∈ {1,2}. PCA and Full have one deterministic
projection per start, not ten duplicated rows. Geometric screening runs all
valid cells on validation AND test. Timing runs m=64,128,256 plus each
validation-selected operating point, Full, and protected-only.

Total m includes the protected head. For example, k=5 and m=128 usually means
4 protected + 124 residual coordinates. Require m<d and m>protected rank.
PCA also needs m≤min(n_train−1,d). Add that maximum feasible dimension when
the sample rank is below the requested dimension range; in Arcene this adds m=59. Larger JL dimensions
still run and have no matched-rank PCA counterpart; never imply otherwise.
Every invalid dimension is listed in dataset metadata/inventory.

Choose the smallest m, then smallest q, using ONLY validation quality:
ARI CI lower bound ≥0.99; training-reconstructed original objective ratio CI
upper bound ≤1.10; pair AND centroid squared-distance Q95 distortion CI upper
bounds ≤0.20; projected objective relative error CI upper bound ≤0.10; no
zero-distance preservation violations. Require the complete crossed grid.
If nothing qualifies, save an explicit failed selection, not a fabricated m.
No test result, timing or proof performance is used in selection.

Protected geometric maps bind the archived data hash, trained center hash, phase, draw,
residual dimension and q. Validation and test use distinct fresh maps to assess
out-of-draw performance. These deterministic research fixtures are not a
deployment commit-challenge transcript. A new challenge/map requires new
compression; caching across different maps is not a valid optimization. Plain
Sparse JL seeds do not depend on trained centers; its cold distance cost therefore
does not include centroid training. Both seed bindings include the start axis.
Undefined objective/distance ratios remain null and cannot qualify a selection.
Missing or duplicate crossed cells, test rows and capped training fits are rejected.

## Timing boundaries

Keep the following columns separate, with every raw measurement retained:

1. Original-space centroid training, iterations and cap hits.
2. Shared PCA training and protected-basis extraction.
3. Projection/model construction, including sparse generation, validation and AU.
4. Whole-corpus compression; Full is exactly zero because it has no transform.
5. Centroid compression; Full is exactly zero.
6. Numeric norm preparation, measured for BOTH compressed and Full inputs.
7. Fixed-center assignment on the whole corpus, and a separate held-out timing;
   extract held-out arrays before starting its timer.
8. Thirty-two different distance queries, with matched indices and optimized Full.
9. kNN query-to-corpus distances and partial top-k selection, with matched self exclusions.
10. Fresh KMeans clustering on the whole representation, initialized with the
    SAME point indices in Full and compressed spaces; report iterations and caps.
11. KMeans refinement from the SAME learned original/projected centers, distinct
    from discovering clusters from fresh point-index initialization.
12. DBSCAN pair-matrix construction and DBSCAN clustering as TWO distinct stages;
    the combined task includes preparation once, not twice.

Main downstream tasks use the whole preserved corpus, not just the test fold.
Quality selection and primary geometric fidelity use only held-out folds.
DBSCAN uses at most 256 corpus rows selected by fixed indices to bound O(n²)
memory; its radius is selected from original validation geometry and reused
unchanged, min_samples=5. Precomputed SQUARED distances use eps=radius².
For the synthetic margin corpus retain radius=1 as the certificate fixture;
it produces an all-noise reference and is not useful clustering evidence.
Additional original-validation radius quartiles give sensitivity profiles for
all corpora. Export graph precision/recall/F1, core fractions, cluster counts
and degenerate-reference flags; retain failed or uninformative partitions.
Report exact evaluated counts.
Distance queries use up to 512 query points × 64 different landmarks; small
corpora reduce those counts explicitly. kNN uses up to 128 query points and
5 neighbors over the entire corpus, excluding each point itself.

Five timing repetitions follow one excluded warmup. Centroid/PCA training has
three measured fits and one excluded warmup. Model construction is a single
observed time per independently generated model. Pair queries and assignment
interleave randomized method/Full order, as do kNN, both KMeans tasks and DBSCAN;
very small differences should not be presented as established speed wins.
Use four BLAS/OpenMP threads and sequential jobs on this laptop.

Report conditional reuse separately from cold execution. Cold hybrid distances
require training centers plus basis extraction, whereas Full distances do not.
Fixed-center Full and compressed assignments share the center-learning cost.
PCA's full-SVD fit is a shared sweep cost. Summed stage medians are descriptive
composites, not directly measured total wall-clock times. Each task includes only
the stages it consumes: pair queries use point norms, fresh KMeans needs neither
centroid projection nor external norms, and refinement projects its initial centers.
Separate stage-only DBSCAN rows have no additive cold/reuse totals.
Saving artifacts and
hashing inputs are outside native task timings; their cost remains in wall time.
An unchanged query's final result can be cached by every method. The experiment
uses DIFFERENT queries and does not cache final query answers to favor one side.

## Quality and statistics

Export pair and point-to-centroid squared-distance distortion Q50/Q95/Q99/max,
signed mean, epsilon=.15 exceedance, zero reference counts and zero violations.
Export labels, assignment agreement, ARI/NMI against Full, secondary ARI/NMI
against source classes, projected cost error, original fixed-assignment cost,
and both within-fold and training-reconstructed original-space objective ratios.
The latter drives validation selection; do not mix these two objectives.
Also export conditional margin/embedding events, certificate coverage and
violations, kNN recall, fresh-clustering and refinement ARI/objective, DBSCAN neighborhood/core
agreement and noise fractions. Perfect fixed-center labels alone do not establish
distance preservation or successful fresh clustering. Either method hitting its
iteration cap excludes that cell from convergence-qualified speed comparisons.

The 95% exploratory pointwise bootstrap independently resamples projection
draws and centroid starts. It is conditional on each fixed corpus. Arithmetic
timing repetitions are not independent projection seeds. No across-dataset
population claim or multiple-comparison-adjusted superiority test is implied.
Full's ARI=1 means self-agreement, not perfect agreement with ground truth.
Zero-distance pairs are excluded from relative-error division and reported
separately. Conditional certificates include floating-roundoff guards.

Raw parameters, rows, labels, original/projected query matrices, split/source/
feature indices, references and pointwise error/certificate masks support new
figures without rerunning training. Models are deduplicated by fingerprint.
Representation storage uses exact array bytes. Worker RSS includes resident
inputs and models and cannot be called an isolated per-method allocation peak.
Every query index set and timing is saved. Two original/projected query-answer
matrices per case are retained; regenerate other answers from saved inputs/models
instead of duplicating many gigabytes of arrays.

## Run and resume from Windows CMD

The default command first writes a reviewable plan; `--run` executes it.
Output directories bind source/configuration/archive hashes. Use the same
command to resume after a time or memory limit. If sources change, choose a
new directory. Completed cases are preserved; partial cases are recomputed.
Artifact hashes and the logical case inventory are checked before skipping a
checkpoint. A damaged case is repaired; damaged shared training artifacts stop
the run instead of silently changing the model. An OS lock blocks simultaneous
runners targeting the same output directory.

```cmd
cd /d C:\PyProjects\zero-knowledge\code
python -u -m campaigns.paper_compression --profile paper --out results\paper_compression_paper_audited --run --max-minutes 480 --max-rss-gib 3
```

`--max-minutes 480` is an attempt budget, NOT a promise of eight-hour completion.
Each worker has a four-hour attempt budget; resume can finish its remaining
cases. The watchdog only terminates its own worker tree and keeps checkpoints.
Insufficient free memory or an unexpected worker error stops further jobs and
produces a partial coverage report. Do not use the laptop's theoretical 14 GiB
as free memory while other applications occupy it.

For a smaller full-size study use `--profile balanced`: five projection draws
and three centroid starts, all datasets/dimensions/k/q, five timing repetitions.
Its stochastic grid is 30% of the paper profile; it still needs measured time
and must be described with its actual sample counts in the manuscript.
`pilot` is one draw/start, k=5, m=64/128/256, q=1, three timing repetitions;
use it for cost estimates, not confidence or robustness claims. It measures only
two distinct queries per case; the main profiles use 32.
`smoke` uses up to 128 rows and 32 varying columns, two draws/two starts,
m=8/16, one timing repetition, at most 32 DBSCAN points, and max_iter=30.
Its results MUST NOT enter paper numerical tables.

```cmd
python -u -m campaigns.paper_compression --profile pilot --out results\paper_compression_audit_pilot_v2 --run --max-minutes 20 --max-rss-gib 3
python -m analysis.paper_compression_report --out results\paper_compression_balanced_audited
python -m analysis.replay_paper_compression --out results\paper_compression_balanced_audited --all-cases
```

## Audit verification and planning cost (11 October)

The corrected v2 [all-dataset pilot](../results/paper_compression_audit_pilot_v2/report/report.md)
completed all nine jobs: 224 geometry rows, 3868 stage samples and 684 query
samples. All 224 saved cases replayed successfully. The full-size pilot took
370 worker seconds (about 6.2 minutes), excluding reporting. Maximum observed
worker-tree RSS was 0.77 GiB. There were no preservation/certificate violations,
Full control errors, source-snapshot errors or unconverged measured KMeans cases.
Two primary DBSCAN cases had uninformative references and remain flagged.
The numerical regression suite passed 28 focused tests; the broader Python suite
excluding proof integration passed 176 tests with two skips. A damaged copied Arcene
checkpoint was detected and repaired without changing an unrelated row or any
original result. A completed-run resume skipped all workers.

These are one-draw/one-start pilot measurements, not final paper results.
The earlier v1 pilots/smokes remain separately preserved. Their held-out timer
included array extraction and their composite setup totals should not support
new performance claims. Use the corrected v2 results and fresh output plans.

`analysis.estimate_paper_compression` gives rough laptop planning estimates:

| Profile | Geometry rows | Estimated hours | Broad planning range |
|---|---:|---:|---:|
| balanced: 5 draws, 3 starts | 15216 | 12.8 | 6.4-25.5 hours |
| paper: 10 draws, 5 starts | 49460 | 41.2 | 20.6-82.5 hours |

These ranges are not confidence intervals or completion guarantees. Selection,
iterations, m/q/k scaling, thermal throttling, background load and I/O can change
them. Pilot RSS is not a guaranteed maximum for the complete grid. The previous
7.7/24.6-hour forecasts applied to the smaller v1 protocol and are superseded.

```cmd
python -m analysis.estimate_paper_compression --pilot results\paper_compression_audit_pilot_v2 --profile balanced
python -u -m campaigns.paper_compression --profile balanced --out results\paper_compression_balanced_audited --run --max-minutes 480 --max-rss-gib 3
```

The balanced run completed on 12 October: nine jobs and all 15216 cases.
Integrity validation reports a complete replicated native grid, zero certificate/
zero-distance violations, no Full-control/source-snapshot errors and no capped
measured KMeans cases. The resumed background session took about ten hours,
including report generation and automatic resumption after its attempt limit.
The larger ten-draw/five-start paper profile has not been started.

The [readable table bundle](../results/paper_compression_balanced_audited/paper_tables/index.html)
contains eleven families of CSV/LaTeX tables, selected-point fidelity/performance
and explicit infeasible cells. The [findings](../results/paper_compression_balanced_audited/paper_tables/findings.md)
distinguish validation qualification, held-out confirmation, query-only speed,
reuse cost and cold execution. Original detailed exports and figures remain in
`results/paper_compression_balanced_audited/report/`.

```cmd
python -m analysis.export_native_paper_tables --out results\paper_compression_balanced_audited
```

This command exports completed tables and does not launch new experiments.

For a detached laptop run, `campaigns.paper_laptop_background` supervises
sequential checkpointed attempts. It allows at most 30 hours and ten attempts,
resumes time-limited attempts, and stops on errors or two attempts without new
checkpoints. It starts no concurrent benchmark jobs. Keep Windows awake and
the laptop plugged in; a powered-off or sleeping laptop cannot compute.

```cmd
python -u -m campaigns.paper_laptop_background --profile balanced --out results\paper_compression_balanced_audited --max-hours 30 --max-attempts 10
```

Do not start this command again while that output already has an active runner.
The background supervisor status is in `background/status.json`; it updates at
attempt boundaries. Worker checkpoints under `jobs/*/rows/` and worker logs show
progress during an attempt. A hidden detached run was started on 11 October;
stdout/stderr logs are under this output's `background/` directory.

After closing other apps, available RAM rose to about 6 GiB. A 63-second
four/eight-thread planning benchmark on MNIST, Gisette and text observed a
1.094 geometric-mean eight-thread speedup across equally weighted native tasks,
with some regressions. This is not a full-campaign forecast. The runner retained
four threads under the benchmark's predeclared 10% improvement rule and resumed
the same frozen configuration and 769 saved cases. Raw measurements are in
`results/paper_thread_benchmark_20261011.json`. Additional RAM provides headroom;
it does not establish a large speed improvement.

## Controlled native row/feature scaling supplement

`experiments.paper_native_scaling` measures Full, both PCA solvers, Sparse JL,
protected JL and protected-only on nested row/feature subsets of MNIST, Gisette
and text. It preserves every subset index, model, training fit, raw timing,
assignment label and query matrix. This is conditional model reuse engineering:
training uses the fixed source training fold per feature count and models are
not refitted at each n. Its three paired draw/start replicates are not an
independently crossed confidence-interval grid. It measures no proof costs.

The implementation check uses n=128/512, d=64/128 and m=32. All 72 cases completed
and replayed. These small subset checks must not enter paper tables. The `paper`
profile uses m=128, n=128/256/512/1000/2000/4000 plus each preserved corpus size,
and d=256/512/1024/2000/4000 plus each original d (only feasible values). It uses
three draws/starts and three timing repetitions. It is not yet measured.

```cmd
python -u -m experiments.paper_native_scaling --profile paper --out results\paper_native_scaling_paper --max-minutes 60
python -m analysis.paper_native_scaling_report --out results\paper_native_scaling_paper
```

Do not run timing studies concurrently. This supplement checkpoints between
finite cases and uses four threads; its time budget is checked between fits and
cases, not a hard interrupt inside a solver. Report exact subset sizes. The main
campaign has the stronger process/RAM watchdog.

## Evidence needed beyond native compression

`report/paper_coverage.json` maps every manuscript table category to evidence.
Native fidelity, certificates, ablation and dimension/q/k sensitivity are
covered by this suite when complete. ZK constraint, witness, setup, prove,
verify, proof bytes, nonce ordering and precision/backend scaling must come
from verified proof records with their original scopes.

Existing server RQ2–RQ5 and `laptop_comparison_20261011` evidence is preserved.
These earlier records do not become a new fidelity-matched protected+JL proof
grid merely because a native map runs faster. The public, data-dependent
protected basis and its statement/security/theory must be reconciled with the
manuscript before claiming a completed new hybrid ZK paper. Large real-data
proofs already hit laptop resource limits. Compile/setup costs for specialized
maps must remain separate and recur when the circuit map changes.
