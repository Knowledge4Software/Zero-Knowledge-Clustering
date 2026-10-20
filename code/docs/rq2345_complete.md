# RQ2–RQ5: balanced server campaign

Run from `code/`. The entry point is `python -m experiments.rq2345_complete`.
It creates a plan without executing unless `--run` is supplied. The Ubuntu
shortcut is `bash campaigns/paper_server.sh <stage>`.

The defaults match the recommended RQ1 design: **10 projection draws × 5 paired
clustering starts**, `epsilon=0.15`, 2,000 sampled distortion pairs and 2,000
bootstrap draws. Confidence intervals are conditional on each fixed dataset.
They are pointwise intervals, with no Holm or simultaneous-inference claim.
Update the manuscript's old 30×10 description to the design actually run.

## What runs and what it establishes

| Stage | Dataset/workload | Evidence and purpose |
|---|---|---|
| `rq2_pilot` | Gaussian engineering ladder `(n,d)=(8,16),(16,32),(32,64),(64,128)`; full, sparse, dense Rademacher, coordinate, PCA-public | Real proofs, compiler constraints, setup/witness/prove/verify times, raw repetitions, proof bytes, process memory, quality and failures. Establish feasible circuit sizes before large proofs. |
| `rq2_quality` | RQ1-selected implemented method/k points; exact saved RQ1 arrays by default | Requalify the actual integer preprocessing and integer projection maps using lower ARI CI ≥0.80 and upper objective-ratio CI ≤1.10. Save quantized inputs, centroids, maps, labels, pairs, commitments and challenge seeds. |
| `rq2_selected` | Same selected shapes and frozen arrays, plus full controls for each k | Run only eligible reduced configurations; independently check the fresh witness's observed fidelity. Each configuration has one warm-up and 10 measured proofs. Failed qualification, unsupported methods and resource failures remain visible. |
| `rq3_native` | Fixed small Gaussian shape, 10 matrices, nonce/local beacon fixture/commit-reveal | Ten paired timing repetitions per matrix, with one warm-up, alternating execution order. Both arms use the **same X, centers, blinding, challenge and A**. Record wall time, commitment/log/challenge/expansion/hash stages, serialized audit transcript size and exact witness arrays. |
| `rq3_proof` | Small reference shape | Actual sparse proof, same-matrix prepublished proof and same-witness unbatched-range proof. Shows authentication/arithmetic costs at a feasible shape. |
| `rq3_grinding` | Small fixed native Gaussian shape; 256 independent map trials; Q=1,2,4,8,16 | Every trial flag and grouped empirical success rates. The reported union bound is an empirical plug-in quantity, not a proof of the formal security bound. |
| `rq4_centroid` | All sparse RQ1 configurations on all eight saved cases | Reuse exact RQ1 labels, margins and point-to-centroid distortion arrays. Report margin and qualified coverage, conditional violations and uncertified stability, preserving both seed/start axes. |
| `rq4_dbscan` | All eight cases; fixed subset up to 1,000 points; m=32,64,128 and q=1,2,4; 10 seeds | Evaluate **every pair in the subset**. Save both distance graphs, pair premises, band membership, labels and core decisions. Report near-band stability, ARI/NMI, noise/core changes and graph stability. Original geometry is cached once per dataset. |
| `rq4_phase` | Gaussian n=1000,d=256,k=5; gamma=.5,1,2,3,4; m=16,32,64,128; q=2 | Full 10×5 grid, conditional certificate diagnostics and gamma–m heatmaps. |
| `rq5_native` | One-factor Gaussian blocks: n=500,1000,2000,4000; d=128,256,512,1024; fitted k=5,10,20; m=16,32,64,128,256; q=1,2,4; gamma=.5,1,2,3,4 | Full 10×5 grid. Save raw metrics, data, maps, projected coordinates, labels and centroids. Fit log–log slopes for measured native components. Fitted-k sweeps preserve the dataset's true k=5. |
| `rq5_precision` | All eight original RQ1 case definitions; m=32,q=2,k=5; bits=8,12,16,24 | One physical normalization fixed at the 24-bit reference. Report quantization error, objective/fidelity/distortion and collapsed coordinates. Save physical/quantized arrays, maps and labels. Baselines/maps are cached within a task. |
| `rq5_proof` | Feasible small one-factor n/d/m/k/q blocks and fixed-reference precision sweep | Actual compiler/proof costs and resource failures. Engineering-shape slopes are separately identified from native dataset-scale slopes. |
| `rq5_ablation` | Reference 16×32 and stress 32×64 Gaussian cases | Sparse reference, identical-matrix prepublished, identical-witness unbatched ranges, full/dense/coordinate/PCA-public, and lower precision at the same physical reference. Save each arm's witness, quality and proof repetitions. |

The synthetic corpus seed is fixed at 12345. RQ5 native sweeps use the existing
native runner's sparse seed encoding; RQ1/DBSCAN use the RQ1 canonical seed
encoding. These have the same implemented sparse-map distribution, but their
numeric seed labels do not identify identical matrices across the two runners.
Actual matrices/seed material are archived. RQ3 matched contrasts reuse exact
matrices rather than relying on equal numeric seeds.

## Your Ubuntu 22 / 16-core / 56 GB / 200 GB server

- Certificate and precision tasks: four processes with four numerical threads
  each, monitored at **8 GiB per process tree**, at most 32 GiB collectively.
- Native timing and all proof tasks: isolated, one task at a time.
- Go prover: **16 threads**, monitored at **40 GiB**; the containing task tree
  is capped at **48 GiB**, leaving RAM for the OS. Limits are watchdog budgets,
  not predictions of actual memory use.
- Proof timeout: 7,200 seconds per arm, including compilation/setup/warm-up and
  10 repetitions. Entire task timeout: 14,400 seconds. A multi-arm task can hit
  the enclosing timeout; the failure is recorded.
- Keep substantial disk space free for RQ1 and RQ2–RQ5 arrays. Tasks stop with a
  recorded failure if free space is below 5 GiB. Compressed size depends on the
  data; 200 GB is not permission to assume every full-size circuit fits RAM.
- Launch one campaign at a time for clean timing. A filesystem lock prevents
  two writers/exporters using the same output directory.

Install `requirements.txt` in a virtual environment. Use the Go toolchain
declared by `zk/go.mod` (currently `go 1.27.1`), rather than assuming Ubuntu's
packaged Go version is sufficient. Build/check before starting the server run:

```bash
python -m pip install -r requirements.txt
python -m pytest tests --ignore=tests/test_zk.py -q
python -c "from commitjl.zk_bridge import locate_binary; print(locate_binary())"
```

## Recommended execution order

1. Run `bash campaigns/paper_server.sh rq1` to finish and export balanced RQ1.
   Freeze source/configs after that
   point; later source changes require fresh campaign directories.
2. Run the proof pilot first, using three measured proofs per cell as engineering
   evidence. Inspect `inventory.csv` and `proof_summary.csv` for feasibility.
3. Run the native certificate/sensitivity campaign.
4. Run final proof scaling/ablations at feasible sizes. Selected full-dataset
   RQ2 shapes are **not established as feasible** on this server; the default
   8,192-input-coordinate budget excludes all eight original RQ1 cases before
   commitment/proving. This is a declared exclusion, not a measured OOM.

```bash
# From code/: balanced RQ1, followed automatically by its export.
bash campaigns/paper_server.sh rq1

# Engineering pilot, no RQ1 dependency.
bash campaigns/paper_server.sh pilot

# Native RQ3/RQ4/RQ5, after RQ1 is complete.
bash campaigns/paper_server.sh native

# RQ2 inventory/requalification: full RQ1 shapes exceed the default proof gate.
bash campaigns/paper_server.sh selected

# Final small-shape proof scaling and matched ablations.
bash campaigns/paper_server.sh proof
```

RQ2 quality runs precede selected proofs even when task-plan order interleaves
them. Sparse quality uses fresh challenges within each clustering start and a
nested bootstrap; it reuses one input opening to avoid repeatedly hashing a
large dataset. This is a geometric ensemble, not an authenticated-Q security
experiment. Dense/coordinate use crossed seeded maps; deterministic PCA uses
one map and five starts. Native eligibility alone never qualifies a quantized
proof. A fresh proof witness that fails observed eligibility is retained as an
exclusion, rather than selecting challenges until one passes.

**Subsets:** centroid certificates reuse every RQ1 row. DBSCAN intentionally
uses at most 1,000 points, selected with seed 20261005, preserving all features;
Arcene uses its available 100 rows. Synthetic DBSCAN uses its generator radius
1.0. Other cases use the median original fifth-neighbor distance (including
self) with `min_samples=5`, chosen before projection. Graph/label claims apply
only to this declared subset. Precision keeps all rows of the fixed RQ1 case
definitions, not loader-wide datasets outside RQ1's scope.

RQ2 does **not** silently take a subset. The optional subset command below
is meaningful only where its requested shape fits the declared feasibility
budget. For example, 32 MNIST rows still mean 25,088 input coordinates and
exceed the default 8,192-coordinate gate. Wide Gisette/Arcene cases cannot
support a meaningful k=5/10 proof within that budget without a separately
declared reduced-feature workload or a circuit optimization.

```bash
python -m experiments.rq2345_complete --run \
  --stages rq2_quality rq2_selected --proof-points 32 \
  --out results/complete_rq2345/selected_subset32_v1
```

That option uses the same fixed 32-point subset for every method/k comparison,
preserves every feature, saves original indices and a new array hash, and
requalifies it. It is **subset evidence**, not a proof of the full RQ1 dataset.
PCA points above the subset rank are excluded explicitly; selecting a new PCA
operating point would require another declared native selection run.

The guard can be changed explicitly with `--max-proof-input-elements`, but
raising it does not create RAM or certify feasibility. Original dimensions are
never silently reduced. A subset must also retain at least k rows.

## Outputs, plots and time

Each campaign writes `settings.json`, `plan.json`, `environment.json`, task
logs and raw JSON/Parquet/NPZ files. Exports include:

- `inventory.csv`: every planned task and status, elapsed seconds, peak RSS,
  limits/exclusions, dataset/shape identity and artifact directory.
- `all_rows.csv`: native seed/start rows and certificate counts.
- `artifact_index.csv`: relative file paths, byte sizes and NPZ array names;
  centroid rows also identify their original RQ1 artifact directory.
- `summary.csv` / `summary.tex`: complete-grid means and bootstrap intervals.
- `proof_summary.csv`: each proof arm, constraints, raw timings, repetition
  counts, proof size, memory and failures. Per-arm JSON retains every prove/
  verify sample and compile/setup/witness timing; per-arm witness JSON makes
  matched comparisons reviewable.
  Method, dimensions and precision identify each arm's actual witness; fidelity
  and fixed-reference quality fields are available directly in the CSV.
  `task_status` retains the enclosing failure when only some arms completed.
- `ordering_pairs.csv`: paired postcommit/prepublished wall times and effects.
- `scaling_slopes.csv`, `failure_counts.csv`: native component slopes with
  crossed CIs, descriptive engineering-proof slopes and failure inventory.
- `figures/`: PDF/PNG fidelity/sensitivity/certificate curves, phase heatmaps
  and proof-time distributions. Raw pair/point arrays support further plots.
- `run_resource_summary.json`: measured active invocation time, observed task
  RAM and the arithmetic cost at your quoted 117,150 rial/hour. It excludes
  idle time and provider billing increments.

There is no measured full-campaign runtime for your server yet. RQ1 laptop
timings do not predict Go proof setup or large-circuit RAM. First measure the
pilot and small native grid on the server; `inventory.csv` gives actual task
times. Do not interpret the two-hour proof timeout as an expected runtime.

For a fast engineering check (different design, fresh output):

```bash
python -m experiments.rq2345_complete --run \
  --stages rq3_proof rq4_phase rq5_native rq5_precision \
  --datasets gaussian_easy --seeds 2 --starts 2 --proof-reps 2 \
  --bootstrap 100 --out results/complete_rq2345/server_check_v1
```

Rerun the **identical** command to resume completed tasks. A failed task remains
recorded by default; append `--retry-failed` to archive its previous attempt and
retry. Resume is at task level: interrupted groups may recompute their partial
work. Source/config or RQ1 selection changes are refused; use a fresh `--out`.
An interrupted child process is killed with its descendants. Proof arms are
checkpointed after completion. Failed/interrupted proof tasks retain those
arms and require `--retry-failed` to archive the attempt before restarting;
their successful arms do not turn the enclosing task into a successful run.
Export without running experiments:

```bash
python -m experiments.rq2345_complete --export \
  --out results/complete_rq2345/native_balanced_v1
```

## Supported evidence and remaining paper scope

These runners execute the existing fixed-centroid assignment circuits, not
KMeans training or a DBSCAN proof. Gaussian/Achlioptas native baselines have no
matching implemented proof circuits and are listed as exclusions. The Pisot
surrogate and in-circuit generator are excluded from verified proof claims.
Constraints are actual **total compiler counts**; symbolic component estimates
must not be presented as measured compiler stage counts. Process-tree peak
memory is not a separately measured verification-stage peak.

RQ3's beacon is a local fixture; paired challenge replay is a controlled timing
counterfactual. Neither measures public beacon networking or deployment
security. Prepublished authentication scope differs by design. Host challenge
binding, secret assignment visibility, deployment task authentication and the
remaining statement limitations must be described as implemented. Commit-reveal
now refuses early reveal, and default TaskLog deadlines use the wall clock.

`--backends groth16 plonk` expands supported proof comparisons if required;
the existing PLONK runner uses **unsafe test KZG setup**, so those results are
engineering benchmarks. Hardware replication needs an actual second machine
and a distinct output directory. Neither a second machine nor a production
setup is fabricated by the campaign.

Low precision can collapse coordinates under the fixed 24-bit physical
reference; that is reported as an outcome. The current fixed integer bound
B=4096 means precision alone does not imply fewer range-check constraints.
Native component timings use shared caches and are labelled accordingly;
their sum is not cold end-to-end latency. The proof scaling fits are descriptive
fits on one witness per level, not CIs over independent datasets/challenges.

Validation includes exact paired witness tests, premature-reveal/deadline
tests, data-hash checks, conditional-certificate checks, cache equivalence,
export/phase figures, memory-watchdog/resume tests, and a real gnark test of
three matched proof arms on n=6,d=4,k=2,m=2,q=1. Full paper campaigns have not
been run during implementation.
