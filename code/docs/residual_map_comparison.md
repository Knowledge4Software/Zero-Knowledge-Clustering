# Residual-map experiment, 16 October 2026

This campaign was requested after stopping the previous native comparison at
2,372 saved cases. Previous results and numerical source files are preserved.
The new experiment changes only the residual sketch; it is not a prototype-head
or cheap-centroid-learning experiment.

## Fixed protocol

- Same training-derived centroid span plus up to 12 residual PCA directions
  for every sketch variant in a dataset/k/start block.
- Total compressed sizes: **64, 128, 256 only**. No 512-dimensional cases.
  Sizes greater than or equal to the input dimension are excluded. PCA sizes
  above `min(training_rows - 1, d)` are unavailable, with no substituted size.
- All nine saved dataset variants, k=5 and 10, three draw IDs (70–72) crossed
  with three start IDs (70–72). New pilot IDs are 60 and are not pooled.
- Complete saved subsets: Gaussian easy 2000x256; Gaussian hard 2000x512;
  DBSCAN margin 1000x128; MNIST and Fashion-MNIST each 5000x784; Gisette
  3000x5000; Arcene 100x10000; 20 Newsgroups 3000x2000; nonzero text
  2886x2000 (114 all-zero rows removed). No additional subsampling.
- PCA is unavailable at all three requested widths for Arcene because only
  60 rows enter training. Those cells compare sketches and Full only and must
  never count as PCA victories. DBSCAN-margin compresses only to 64; Gaussian
  easy compresses only to 64/128.
- Three timing repetitions, three fit repetitions, excluded warmups, float64,
  four compute threads. Model order randomized within blocks. Downstream Full
  controls are measured in interleaved order.
- Train/validation/test split 60/20/20. All representation learning uses train
  only. Whole-corpus fresh clustering is explicitly distinct from held-out
  diagnostic clustering and assignment to the original training centers.
- All candidates are retained. This is an exploratory comparison, with no
  outcome-based promotion or test-selected map seeds.

## Map variants

| Name | Residual operation |
| --- | --- |
| protected_plus12 | Canonical q=1 sparse map; current-method control |
| jl_q2, jl_q4, jl_q8 | Canonical sparse generator with distinct rows per column and entries +/-1/sqrt(q) |
| block_q4 | Four independent row blocks, one signed entry per column per block, scaled by 1/2 |
| mixed_q1, mixed_q4 | Zero-pad to a power of two, random input signs, normalized fast Hadamard transform, canonical sparse sketch |
| dense_gaussian | Dense independent N(0,1/tail) entries; diagnostic control, all costs included |

All methods compute `[U.T w, S R w - (S R U) U.T w]`, with `w=x-c0`.
R is identity except for the mixed maps. The cached `S R U` product is charged
to construction. Mixing is charged on every data conversion, including padding.
The entire effective mixed transformation is not itself a sparse matrix.
For block widths not divisible by four, block sizes differ by at most one.
Actual matrices, mixing signs, basis and origin are saved, not just seeds.

Full, full-SVD PCA, default randomized PCA and fast randomized PCA (two power
iterations) are included. Every PCA is fitted at its actual target width.

## Saved measurements

Learning, map construction, corpus conversion, norm preparation, assignment,
32 distance queries, 5-NN search, fresh K-means, centroid-initialized refinement,
DBSCAN pair construction and DBSCAN fitting are recorded separately. DBSCAN uses
at most 256 saved rows; kNN uses 128 query rows against the complete saved corpus.
All tasks use bounded K-means iterations (300) and record limit flags.

Per-case labels, centers, selected distance answers, query/pair indices, model
parameters, timing samples, distortions and original-space objectives are saved.
Full costs for each task include only stages that task needs. Fit caches retain
their complete measured cost in cold totals. Common archive I/O and evidence
writes are outside numerical timings.

Bootstrap intervals resample projection and start axes. Joint pointwise wins
require complete grids with at least three draws and three starts and no limit
flags. Pilots never count as statistical wins. No multiplicity correction or
independent-data confirmation is implied.

## Execution and recovery

From the `code` directory:

```cmd
python -m campaigns.run_residual_maps --out results/residual_maps_full_20261016
```

This launches a detached background supervisor. It prevents idle sleep on Windows
while active and generates an audited report after the worker completes.
Completed cases use flushed atomic checkpoints and source/config identity checks.
Low-RAM exits can retry at most three times, after available RAM stays above 3 GiB
for three observations; each recovery wait is bounded to one hour. Other failures
are reported without automatic retries. The worker has a 2.5 GiB RSS guard and
a 12-hour invocation budget. The laptop must remain powered on.

To stop at the next checkpoint, create an empty `STOP_REQUESTED` file in the
output directory. Delete only that marker before requesting a resume. The
launcher refuses to create another recognized campaign while one is active.

## Scope

This is native numerical evidence. The canonical commitment generator and ZK
circuit are not modified, and new families require separate proof analysis and
implementation. Better pair-distance fidelity does not guarantee better fresh
clustering ARI. Fixed-center ARI=1 is structural and cannot establish a general
clustering improvement. Full-data costs remain essential controls.
