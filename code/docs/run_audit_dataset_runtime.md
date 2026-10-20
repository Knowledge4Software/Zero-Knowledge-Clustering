# Run audit and runtime budget — 2026-10-05

The balanced design is 10 projection seeds and 5 clustering starts. RQ1 uses
six methods (sparse JL, Gaussian, Rademacher, Achlioptas, coordinate, PCA),
with a full-space control. PCA is deterministic and has five starts, not ten
independent projection draws. Runs preserve raw results, rather than only
plot summaries. Estimates below are planning ranges, not measurements on the
Ubuntu server.

Current validation: 123 Python tests passed (one optional live test skipped),
`go test ./...` passed, and a separate live test verified the three matched
proof arms. All five Ubuntu launcher modes were checked with a logging stub;
a failed RQ1 prevents export. The earlier tiny run exercised all 13 stages and
verified 14 proof arms. Full campaigns have not been executed; these checks
do not establish that every large workload fits the server.

The layout check passed for 11 historical raw files and 116 local links. It
validates the original tracked manifest's source fingerprint and file hashes,
and separately reports whether a result matches the current code. Historical
results are preserved evidence, not reruns of the balanced implementation.

Output locks prevent concurrent RQ1 runs or exports from rewriting the same
campaign. Missing scaling levels cannot produce a complete-grid CI; failed
children cannot retain a success status. Proof CSVs identify each arm's actual
method, dimensions, precision, range-batching flag and fidelity. Saved witnesses
support correct metadata when re-exporting earlier raw runs. Horizontal proof
plots preserve long labels. The proof plan contains 45 tasks and 61 arms across
pilot/final campaigns. Temporary cache and launcher-probe files are archived
under `.cache/housekeeping/`; source, datasets and saved evidence are preserved.

Completed proof arms are checkpointed individually. If a later arm fails or
the parent is interrupted, completed evidence remains exportable and the task
keeps its failed status. After a hard stop, an unfinished checkpoint is retained
until an explicit `--retry-failed` archives and restarts the task; it is never
silently overwritten on resume. The CSV includes the enclosing `task_status`.

## Fixed datasets

| Case | RQ1 / RQ4 centroid / RQ5 precision: rows × features | Scope | RQ4 DBSCAN rows |
|---|---:|---|---:|
| GaussianEasy | 2,000 × 256 | Generated, true k=5, gamma=3 | 1,000 |
| GaussianHard | 2,000 × 512 | Generated, true k=8, gamma=2, imbalance=2 | 1,000 |
| DBSCAN-margin | 1,000 × 128 | Generated, radius=1, epsilon=.15 | 1,000 |
| MNIST | 5,000 × 784 | Random subset of 70,000 loader rows | 1,000 |
| FashionMNIST | 5,000 × 784 | Random subset of 70,000 loader rows | 1,000 |
| Gisette | 3,000 × 5,000 | Random subset of 6,000 training rows | 1,000 |
| Arcene | 100 × 10,000 | All available training rows; not the combined corpus | 100 |
| 20 Newsgroups TF-IDF | 3,000 × 2,000 | Subset of 11,314 training documents; vocabulary fitted on training corpus | 1,000 |

Corpus seed is 12345. Additional DBSCAN subsets use seed 20261005 and retain
all features. Their indices are saved. DBSCAN examines all 499,500 pairs for
1,000 rows, or all 4,950 pairs for Arcene. These graph results describe the
subset, not the original corpus-wide DBSCAN graph.

## Workloads by research question

| RQ | Workload |
|---|---|
| RQ1 | Eight cases above; k=5,10; valid m in 16,32,64,128,256,512,1024; sparse q=1,2,4. Default total 34,120 rows, including full-space controls. |
| RQ2 pilot | Generated Gaussian (n,d)=(8,16),(16,32),(32,64),(64,128); five variants each: 20 compile/setup arms, three measured proofs plus one warm-up each. These are engineering workloads. |
| RQ2 selected | Exact frozen RQ1 arrays and eligible operating points, requalified with integer arithmetic. Default 8,192 input-element feasibility guard excludes all eight original shapes. Exclusions cannot populate a successful full-dataset proof comparison table. |
| RQ3 | Generated Gaussian 16×32, k=5,m=16,q=2; three randomness sources, ten matrices and ten paired timings. Grinding: 256 map trials. Three real proof counterfactual arms. |
| RQ4 centroid | All 14,400 sparse RQ1 rows; no additional subset. |
| RQ4 DBSCAN | Eight subsets above; m=32,64,128; q=1,2,4; ten seeds: 720 rows. |
| RQ4 phase | Generated Gaussian 1,000×256; five gamma levels and four m levels: 1,000 seed/start rows. |
| RQ5 native | Generated Gaussian one-factor n/d/k/m/q/gamma sweeps: 24 blocks, 1,200 rows. Default base 1,000×256, true k=5, fitted k=5,m=64,q=2. |
| RQ5 precision | Eight fixed cases above; bits=8,12,16,24; m=32,q=2,k=5: 1,600 rows. Fixed physical normalization across precision levels. |
| RQ5 proofs | Small Gaussian scaling/precision blocks: 22 arms; ablations at 16×32 and 32×64: 16 arms. Ten measured proofs plus one warm-up each. |

`--proof-points N` explicitly selects and archives a fixed row subset for
RQ2; it never drops features. For N=32 only GaussianEasy fits the default
8,192-element guard. MNIST still has 25,088 elements; Gisette has 160,000.
The guard is an engineering policy, not a measured OOM threshold. Increasing
it does not guarantee feasibility. Wide real-data proofs need a separately
declared feature-reduced workload or circuit improvements.

## What figures the outputs support

Raw CSV/Parquet/JSON rows and NPZ inputs, projections, labels, centroids,
point/pair distortions and margins support fidelity/objective curves,
distortion histograms/ECDFs, certificate coverage and conditional violations,
DBSCAN band/graph/core/noise comparisons, gamma–m heatmaps, precision plots,
paired ordering overhead, empirical grinding rates, scaling slopes, and
proof cost/size/memory/failure plots. `artifact_index.csv` identifies saved
arrays. Raw proof repetitions allow distributions rather than just means.

The audit corrected mixed-stage DBSCAN grouping, connected k/q sensitivity
curves, expanded metric exports, and saved grinding trial matrices and pair
distortions. RQ4 centroid rows identify their original RQ1 artifact directory.

This does not support arbitrary unmeasured figures: no successful excluded
full-size proofs, no proof of KMeans training or DBSCAN, no isolated verifier
memory, no per-gadget compiler constraint breakdown, and no second-machine
comparison. Certificates are conditional on their recorded premises.

## Runtime and RAM on the proposed server

Use four native workers × four numerical threads; proof and timing tasks run
serially with 16 Go threads. Native watchdogs allow 8 GiB per worker; proofs
allow 40 GiB for Go and 48 GiB for the enclosing process tree. These are
limits, not expected allocations. More RAM prevents memory pressure but does
not make hashing or arithmetic automatically faster.

Measured local checks (8 physical / 16 logical cores, approximately 13.86 GiB
RAM, cached data) all succeeded:

| Check | Wall time | Peak process-tree RSS |
|---|---:|---:|
| Gaussian 16×32 sparse proof, setup + warm-up + two proofs | 100.35 s | 2.24 GiB |
| Same proof: compile / setup / mean proving | 4.18 / 69.86 / 6.73 s | — |
| RQ3 nonce, one matrix, ten paired repetitions + warm-up | 21.74 s | .15 GiB |
| Gisette precision, two seeds × two starts, one precision | 25.28 s | .83 GiB |
| Gisette DBSCAN, 1,000×5,000, one configuration | 38.81 s | .50 GiB |
| Native Gaussian reference, two seeds × two starts | .70 s | .17 GiB |

The eight-case RQ1 profile measured m=64,q=2,k=5, two seeds/two starts,
all six methods: approximately 12–39 seconds per case. A component-level
extrapolation to the full grid gives 2.27–4.07 hours of serial worker time,
or .83–1.27 hours with ideal four-worker scheduling. Larger m, bootstrap
cost, shared CPU contention and server CPU differences justify a wider budget.

| Campaign | Planning wall-time range |
|---|---:|
| RQ1 balanced | 1–3 hours |
| Native RQ3/RQ4/RQ5 combined | .5–1.5 hours |
| RQ2 engineering pilot | 1–4 hours |
| Final small RQ3/RQ5 proofs and ablations | 2–6 hours |
| RQ2 full-size selected default | Exclusions only; no successful proof-time estimate |
| Total usable native + small-proof campaign | Approximately 5–15 hours |

These ranges exclude downloads, environment installation and failed-task
retries. They are extrapolations, not statistical confidence intervals or
guarantees. The largest pilot shapes are not benchmarked locally; setup can
grow nonlinearly. A proof-arm timeout of two hours is a failure limit, not an
expected duration. Re-estimate after the server pilot using `inventory.csv`.
At 117,150 rial/hour, 5–15 hours is 585,750–1,757,250 rial before idle time
and provider billing rounding. The complete original full-dataset RQ2 table
is outside this estimate.

Raw benchmark evidence is in `.cache/rq_audit_profile_v1/` locally; compact
copies are retained under `results/diagnostics/rq_audit_20261005/`.
