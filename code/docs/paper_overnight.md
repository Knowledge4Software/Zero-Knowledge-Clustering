# Queued paper follow-up, 17 October

The follow-up runs after the existing `bound_audit_20261017_v2_1792247977_main`
workflow reaches a terminal state. It never interrupts that campaign. A prior
failure is recorded; only independently verified source bundles can be reused.
Sources and the new Linux executable are frozen in a separate deployment.
`tmp/paper_overnight_deployment.json` records the exact deployed paths/hashes.

The runner is `experiments/paper_overnight.py`. The additional geometry relation
is `zk/circuit/bound_geometry.go`; the new `geometry` CLI phase is implemented in
`zk/cmd/commitjl-bound/geometry.go`. The old running binary is unchanged.

## Predeclared experiments

1. **48 geometric-utility jobs:** all eight capped native corpora from the
   preceding study; k=5/10; three clustering starts; total storage budgets
   16/32/64; ten independently seeded maps per start/budget. Measure pair-distance
   distortion on about 2,000 sampled pairs and nearest-neighbor recall at 5/10
   for up to 128 held-out query points against the held-out database. Compare
   PCA, protected head alone, plain sparse JL and protected head+JL. The energy
   value is part of our storage budget but is not a Euclidean coordinate.
   PCA settings exceeding the training rank are explicitly excluded.
2. **32 authenticated geometry-proof jobs:** Gaussian-hard at k=5/10, MNIST at
   k=5 and text at k=5. Full and head-only controls are each run once per case;
   PCA and protected+JL are compared at widths 16/32/64. Prove eight distinct
   point pairs with three timing repetitions after a warmup. Existing verified
   preprocessing is reused for existing representations; changed widths receive
   new preprocessing proofs under the same owner original-data root. Save actual
   proofs, independent verification, exact constraint counts, compilation/setup
   time and pairwise original-space error diagnostics.
3. **30 full-cached-data native jobs:** MNIST, Fashion-MNIST and the available
   20 Newsgroups training corpus; k=5/10; five starts. Use all cached loader rows,
   followed by zero-row removal, keeping the current 2,000-feature text
   vocabulary. Record actual dataset sizes/hashes. Compare Full, PCA-16,
   head+energy, head+energy+JL-16 and exact affine. Save assignment arrays, split
   indices, distances to floating centers and fitted integer models.

Models/centers use the same 60%/40% fit/evaluation split convention. Full corpus
here means native computation over all available rows; cryptographic catalogs
remain explicitly small. Nearest-neighbor recall is a geometric retrieval
metric, not classification accuracy. Reproducible native seeds and post-commit
proof challenges are separate cohorts. No test-selected best result is hidden:
all predeclared widths/maps and failures remain in the exports.

## Meaning of the geometry proof

The new circuit opens both points under the global head root and, when present,
the global tail root. It proves the sum of squared differences of head and tail
coordinates. Joint head/energy leaves authenticate energy, but energy is not
included in that Euclidean sum. Full opens original rows. Public indices must
be distinct, ordered and in range. The context binds the owner root and public
model. Full Stage-1 coverage is independently checked before accepting a
compressed-data geometry proof.

The proof establishes the exact **stored embedding distance**. It does not
establish that the distance approximates the original space within a particular
error, or that the embedding's nearest neighbors match Full. Those are measured
separately. An actual geometry application can lose accuracy; this campaign is
designed to detect that outcome as well as improvements.

The original trusted owner/setup manifest and verification keys remain required.
The benchmark's public-model approval and challenge derivation are host duties;
there is no in-circuit matrix-generating XOF or formal external security audit.

## Timing, accounting and limits

Full-native timing preserves the preceding runner's decomposition. Shared
KMeans training, parameter construction, compression application and downstream
answering are separate. Native geometric map application timing is incremental
after the head/residual has been computed; it must not be presented as complete
compression time. Width-16 reused preprocessing is not zero-cost preprocessing:
its original costs remain in the source campaign and must be included in an
end-to-end comparison.

Jobs run sequentially using four threads. Each job has a 45-minute deadline,
5.3 GiB own-process-tree RSS limit and 768 MiB host reserve. Go phases also have
a 1.3M-constraint ceiling and 40-minute deadline. The follow-up campaign has an
eight-hour maximum after it starts. Resource failures/exclusions are retained;
three repeated logic failures stop further jobs. The queue gate has a finite
ten-hour wait. Only new job descendants are terminated by these guards.

The detached queue survives closing SSH, VS Code and the laptop. It first runs
Linux circuit regressions and an end-to-end smoke campaign. Long jobs start only
after those pass. An eight-job local smoke campaign passed, including fresh
width-32 preprocessing and geometry proofs; serialized-proof tests also reject
changed geometry distances and substituted pairs without requiring private data
at verification.

## Outputs and status

Under the new server result directory:

- `queue_status.json`: waiting, regressions, smoke, running, complete or failed.
- `campaign/status.json`: current job, completed count, failures and exclusions.
- `campaign/full_native.csv`, `geometry_native.csv`, `geometry_proofs.csv`.
- `campaign/jobs/*/result.json`: complete individual receipts and raw NPZ/proof
  files. Incomplete jobs are never represented as zero-time successes.
- `campaign/report.md`: scope and export interpretation, regenerated as jobs end.

These experiments address missing evidence; their execution does not guarantee
that JL helps or that the manuscript is ready. The exact affine comparator and
the geometric utility results remain central to that assessment.
