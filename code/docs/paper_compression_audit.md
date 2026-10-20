# Experiment correctness and evidence audit

11 October 2026. Scope: the new offline native compression/downstream suite,
its reusable numerical kernel, reports, replay and resume behavior. Existing
server ZK campaigns are preserved with their original scope. This audit does
not establish a completed paper or a joint quality/speed win over PCA.

## Bugs and misleading measurements corrected

| Finding | Correction |
|---|---|
| Held-out assignment included NumPy row extraction/copying | Extract arrays before the timed computation; flag legacy v1 timers |
| Composite task totals charged unused centroid/norm setup; DBSCAN setup could be counted twice | Task-specific cold/reuse totals; one combined DBSCAN task; separate stage timings |
| Norm-dot cancellation could collapse nearby translated points to zero | Optimized matrix product with direct-difference fallback for cancellation; regression against explicit distances |
| Zero objective denominators and no positive reference pairs could appear perfect | Undefined ratios remain null and cannot qualify selection; zero pairs counted separately |
| Full/method timing order differed for several workloads | Warm both and interleave randomized paired repetitions for kNN, both KMeans tasks and DBSCAN |
| kNN sorted every corpus distance | Partial top-k selection with deterministic tie handling |
| KMeans could hit 100 iterations; Full convergence was not checked | Cap 300; retain both controls' iterations/cap flags; exclude capped cases from qualified speed claims |
| Selector could accept test rows or duplicate Cartesian cells | Validation-only, complete independent-axis inventory, undefined/cap rejection |
| Completed marker alone could hide missing/corrupt arrays | Logical inventory and artifact SHA checks; damaged case repair; strict shared-fit binding |
| Concurrent runners could write the same output | Portable OS output lock, released on process exit |
| Empty exports could look like passing validation | Empty results explicitly fail readiness/replay |
| Plain JL map seed unnecessarily bound centers while its distance cost excluded training | Plain JL seed independent of centers; protected JL retains center binding |

## Experiment-design gaps addressed

Full-SVD PCA alone did not support a broad comparison with optimized PCA.
Randomized PCA is now an explicit baseline, with training-fold fit cost recorded
per draw and shared prefix maps at equal output dimension.

Fresh clustering and refinement answer different questions. The former starts
from the same selected point indices in each representation. The latter starts
from the shared learned centers. Both results are retained. Fixed-center label
preservation does not guarantee either task's final labels.

The synthetic DBSCAN radius=1 produced an all-noise reference. Its original
certificate fixture remains, with a degenerate-reference flag. Validation-radius
quartiles add useful sensitivity measurements. Edge precision/recall/F1, core
fractions, cluster counts and noise fractions prevent a high all-pairs agreement
or trivial ARI from masquerading as useful clustering.

The main profiles now measure 32 distinct queries, with common Full controls.
Every query index/timing is retained; two answer matrices per case plus saved
inputs/maps support replay without duplicating huge arrays. Task setup and fit
costs remain separate from query-only timing and amortized composites.

A controlled native n/d supplement was added for MNIST, Gisette and text.
Nested subsets and all source indices are saved. Models are trained once per d
on the fixed training fold, then reused across n. This is conditional reuse
scaling, not training complexity as n varies or new ZK scaling.

## Verification performed

The focused numerical tests passed **28/28**. The broader Python suite excluding
proof integration passed **176 tests**, with **2 skipped**, in 41.5 seconds.
Proof integration was not rerun in this native-suite audit. The corrected full-size pilot
completed **9/9 jobs**, **224 geometry rows**, **3868 stage samples** and
**684 query samples**. All **224 cases replayed**, including saved maps,
geometry, sampled query answers, clustering labels and primary DBSCAN labels.
The reporter found zero Full-control errors, zero certificate/zero-distance
violations, zero source-snapshot errors and no capped measured clustering cases.
Two primary DBSCAN cases remain explicitly uninformative.
Additional numerical replay passed for all **57 kNN cases** and **102 extra
DBSCAN radius profiles**, including saved neighbors, graph matrices, labels,
core masks and certificate consistency.

Pilot worker time was **370 seconds** and the largest observed worker-tree RSS
was **0.77 GiB**. These measurements are not guaranteed full-grid resource limits.

A copied Arcene checkpoint was deliberately corrupted, detected and repaired;
an unrelated row and all original results remained unchanged. A completed
campaign resume skipped all workers. The small native scaling check completed
and replayed **72/72 cases** across three datasets.

Evidence: [pilot validation](../results/paper_compression_audit_pilot_v2/report/validation.json),
[case replay](../results/paper_compression_audit_pilot_v2/report/replay_validation.json),
[additional workloads](../results/paper_compression_audit_pilot_v2/report/additional_workload_replay.json),
[native scaling replay](../results/paper_native_scaling_audit_check/report/replay_validation.json).

## Results that remain insufficient or weak

The pilot has one projection draw and one centroid start. It cannot establish
robust superiority or replace the complete balanced/paper grid. Fresh audited
plans are ready, but those long grids have not been run. Balanced has five
draws and three starts; paper has ten draws and five starts. The rough laptop
forecasts are 12.8 and 41.2 hours respectively, with wide planning ranges.

There are genuine quality limitations. At m=128, this pilot's protected-JL fresh
clustering ARI against Full is approximately 0.571 on MNIST, 0.272 on Gisette
and 0.047 on text. Refinement improves MNIST to 0.987 and Gisette to 0.713,
but text remains about 0.049. These are single-draw observations, not confidence
claims. Near-one original objective ratios do not imply identical partitions.
Text kNN recall is approximately 0.895 with archived zero rows, versus 0.166
in the nonzero sensitivity corpus. These are different corpora/maps, not a
paired causal estimate, but show why the sensitivity result must be reported.

The controlled scaling paper profile also remains unrun; its 72 small check
cases validate code, not final scaling claims. Arcene has only 100 training-source
rows and 20 held-out test rows, with PCA rank limited to 59. DBSCAN uses at most
256 selected rows; do not describe it as a full-corpus graph experiment.

Native timing cannot fill witness/prove/verify/proof-byte columns. A new hybrid
claim needs fidelity-matched quantized models and verified proof records;
existing server/laptop proof results retain their original method, precision,
backend and dataset scope. Public data-dependent basis/security/theory claims
also need manuscript alignment. Those gaps are explicit in paper_coverage.json.

Earlier v1 pilot raw data is preserved, but its affected held-out/composite
timings and superseded forecasts should not support new performance claims.
Use the fresh audited output directories and commands in the
[experiment protocol](paper_compression.md). Do not pool differing snapshots.
