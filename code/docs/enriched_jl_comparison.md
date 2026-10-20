# Extra 12 residual directions: fair native PCA screen

This is an opt-in experiment. Existing runs, projection definitions and manuscript
are preserved. `commitjl.enriched_jl.enrich_centroid_basis` appends up to 12
randomized-PCA directions of the **training-only orthogonal residual** to the
complete centroid-difference span. Reorthogonalization retains the original span.
At k=5 with rank-four centers, total m=128 means 16 protected and 112 JL outputs;
m=256 means 16 protected and 240 JL outputs. Degenerate data may have lower rank.

The canonical normalized Sparse JL generator and reusable native transform are
unchanged. The training-residual PCA uses 12 components, two power iterations,
and ten oversamples. Extra basis training includes constructing/centering the
residual; it is charged in full to every cold use of the upgraded method.

## Comparators and timing

Every PCA model is fitted **separately at its actual output dimension**:

- `pca_full`: sklearn full SVD with `n_components=m`.
- `pca_randomized`: sklearn randomized SVD with its default power iterations.
- `pca_randomized_fast`: randomized SVD with two power iterations, ten oversamples.
- `protected_jl`: original complete centroid span plus residual Sparse JL.
- `protected_plus12`: that span plus the additional training-residual directions.
- `full`: uncompressed reference.

No 512-dimensional prefix supplies the PCA128 or PCA256 controls. Full SVD can
internally compute a complete decomposition as part of its chosen algorithm;
randomized controls directly compute the requested low rank. Fit specification,
actual fitted rank, seed, all model parameters and source hashes are retained.

Three timing repeats and three fitting repeats follow excluded warmups. Cold
compression includes all necessary model learning, map construction and whole
corpus transformation. Cold fresh clustering adds a new KMeans run. Cold distance
queries add norm preparation and all 32 distinct query blocks. Fixed-center
assignment includes training centers for **all** methods that need them. Reused
fit checkpoints never amortize away their measured cost in a cold total. Common
archive I/O and artifact writing are excluded consistently. Totals are sums of
stage medians, not directly measured process startup wall times.

## Scope and endpoints

Defaults: MNIST 5000x784, Fashion-MNIST 5000x784, Gisette 3000x5000, and text
2886x2000 after removing the 114 archived zero rows. There is no additional row
or feature truncation. Deterministic disjoint 60/20/20 folds match the previous
native protocol. Centers and both kinds of PCA see only training rows. Total
m=128/256, q=1, three JL/randomized draws crossed with three center/initialization
starts, and four CPU threads give 324 cases including controls.

`fresh_ari` compares new whole-corpus clustering with Full under the same
point-index initialization. `test_fresh_ari` independently clusters held-out test
rows and compares with Full on those rows; validation-fold clustering is also
saved. Fixed-center test ARI is separate and cannot by itself count as a quality
win. Pairwise squared-distance Q95, nearest-neighbor recall, ground-truth ARI,
original-space partition SSE, iteration counts, labels and geometric samples are
retained. A capped KMeans cannot count as a win. The new 12-direction recipe is
fixed in advance; these test results do not select a new direction count.

Crossed bootstrap intervals condition on fixed corpora and are pointwise,
exploratory, with no multiple-comparison correction. A screening win is not a
publication-level confirmation. Train-derived extra directions depend on the
private points; no public-basis privacy claim or new ZK proof benchmark is made.

## Run and inspect

From `code/`:

```powershell
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
python -m experiments.enriched_jl_comparison --out results/enriched_jl_dedicated_pca_20261015
```

The output contains `manifest.json`, dataset/fit checkpoints, every case JSON,
NPZ model/label/geometry evidence, `rows.csv`, `summary.csv`, `comparisons.csv`,
and `report.md`. Repeating the identical command resumes complete checkpoints;
changed source/configuration requires a fresh output folder. Each case is saved
atomically. There is an 1800-second per-invocation budget and low-memory checks;
partial results are retained and labeled incomplete on interruption.

After completion, audit and generate the focused comparison and figures:

```powershell
python -m analysis.review_enriched_jl results/enriched_jl_dedicated_pca_20261015
```

Use `findings.md`, `index.html`, and `reviewed_comparisons.csv` for interpretation.
The review explicitly labels comparisons that combine held-out quality with
whole-corpus speed as different-scope diagnostics. Only whole-corpus fresh ARI
and whole-corpus cold fresh-clustering time form the primary joint win endpoint.
The audit hashes all artifacts and snapshots the exact sources; numeric replay
checks representative projections, assignments, labels and geometry. It does
not rerun every KMeans optimization.

For synthetic correctness only, choose another output folder and add `--smoke`.
Smoke results are explicitly synthetic and must not be pooled into real data.

## 16 October: identical-map host optimization and new-draw replication

The SHAKE word stream now starts with a 64-byte prefix and doubles refill size
up to 4096 bytes. This avoids expanding 4096 bytes per domain when q=1 consumes
only four. It produces the exact same unbounded XOF word stream, rejection
decisions, signs, and projection matrices. Tests cover refill boundaries,
high-rejection row domains, duplicate rejection, and bitwise protected outputs.
No RNG, sparse matrix distribution, or protected-direction algorithm changed.

The dedicated runner now accepts `--seed-offset` and `--start-offset` and records
these in its source/configuration identity. Its manifest includes the hashing
dependency. Nonzero offsets change all relevant draw and initialization IDs;
they are not additional repetitions of the original 0--2 configurations.

```powershell
python -m tools.benchmark_shake_expansion --out results/map_construction_optimization_20261016/benchmark.json
python -u -m experiments.enriched_jl_comparison --out results/enriched_jl_optimized_replication_20261016 --seeds 3 --starts 5 --seed-offset 10 --start-offset 10 --max-seconds 2400
python -m analysis.review_enriched_jl results/enriched_jl_optimized_replication_20261016
```

The replication plan is saved in that output directory's `protocol.md` before
real-data collection. All four original corpus subsets and both target widths
are retained, with no further sampling. Five new starts crossed with three new
draw IDs yield 540 cases. These remain exploratory intervals conditional on the
same corpora; changing random starts does not create independent new test data.
