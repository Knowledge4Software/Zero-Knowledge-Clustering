# Paper completion campaign — 17 October 2026

This is a separately frozen, detached supplement. It never changes or stops the
completed bound-audit and geometric-utility campaigns. Deployment paths, SHA256
hashes and the successful seven-job local smoke are recorded in
`tmp/paper_supplement_deployment.json` at the repository root.

## Fixed plan: 96 jobs

* **24 actual ZK controls:** Gaussian-hard at k=5/10, MNIST at k=5, and nonzero
  text at k=5. Each uses six variants: protected head alone; head+JL without
  energy; plain sparse JL; normalized signed dense Rademacher projection; PCA
  with h+1 components to match the protected head/energy statistic count; and
  cached exact nearest-center labels and costs. The cached baseline proves
  these task-specific answers once and reuses authenticated records; it is not
  a Euclidean embedding and does not answer arbitrary point-pair distances. The
  original catalogs, original root and public centers match the preceding
  campaign. All original rows get Stage-1 proofs. Query sizes are 1/8 where
  valid, with distinct subsets, a warmup and three repetitions. These controls
  complement existing Full, PCA-16, head+energy, head+energy+JL and exact-affine
  measurements; they do not replace or pool the older MiMC cohort.
* **48 native controls:** eight capped datasets, k=5/10, three starts. Budgets
  16/32/64 plus PCA at h+1; five random maps; sparse q=1/2/4; signed dense
  projections; head-only and head+energy controls. Energy removal is tested
  with the exact same head/tail and also at equal total storage. Input and
  protected-basis quantization independently sweep 8/12/16 bits. All observed
  assignment arrays and cost estimates are saved. Rank/budget exclusions are
  explicit. Reference assignments and costs come from floating original data;
  quantized-Full agreement is recorded separately.
* **18 center-count sensitivity jobs:** Gaussian-hard and capped MNIST,
  k=2/20/50, three starts. These test the dependence on protected rank; unsupported
  budgets are excluded rather than filled with invented results.
* **Six larger actual-proof catalogs:** 64 MNIST held-out records, k=5, one
  shared new owner root: Full, PCA-16, head+energy, head+energy+JL-16, affine and
  cached exact answers.
  Compressed queries use sizes 1/8/32/64, with genuinely distinct subsets where
  possible. Full is proved at size 1; larger Full shapes already exceed the
  existing circuit budget. Catalog size 64 is not a full-corpus proof.

The full-native 70,000-image runs and 10,944-nonzero-text runs are already in the
preceding completed follow-up; they are not rerun by this supplement. Its text
representation remains the cached 2,000-feature training corpus.

## Exact ablation relation

`protected_linear` uses the identical rounded protected basis and sparse tail
as the energy representation, computes no energy, authenticates the head and
tail rows, and proves the nearest-center cost **in that stored embedding**.
Rounded centers can have nonzero residual tails; these are included in every
distance and comparison. Energy-based fixed-center queries use head plus energy
and intentionally do not also add the tail norm.

Head-only, sparse and Rademacher controls use the general public linear-map
relation internally named `pca`. Their experiment labels identify the actual
map; no PCA-training or orthonormality assertion follows from that internal name.
Rademacher coefficients are quantized after normalization by sqrt(m). Public-map
approval, sparse-generator checks and entropy policy remain verifier host duties.

Native baselines and compression accounting are comparative measurements, not a
claim that all methods answer an identical original-space objective exactly.
Affine does answer the quantized original-space fixed-center objective exactly.

## Privacy and public exports

Each new completed proof job produces an explicitly whitelisted `public_bundle`:
public statements/model/root/key pins, verification keys and serialized proofs.
Private inputs, salts, head/energy/tail arrays, authentication paths, assignment
metadata, progress files and proving keys are excluded. Independent verification
is actually run using only this bundle. Roots and key pins must still come from
a trusted owner/setup channel; accepting a prover's self-selected trust manifest
would change the security claim.

Centers, dimension/scale metadata, requested indices and claimed costs/distances
are public outputs. Such outputs can reveal information, especially through
single-record or differenced subset queries. This is not differential privacy,
a production security audit, or a claim of zero information leakage.

## Automatic handoff and finite execution

The workflow first runs Linux circuit regressions and the seven-job proof/export
smoke. Long jobs start only after those pass. It then runs 96 jobs sequentially
with four threads, a 5.3 GiB own-process-tree guard, 768 MiB host reserve,
45-minute job limits, 40-minute Go phase limits, a 1.3M-constraint limit and an
overall six-hour campaign deadline. Three job failures stop further work.
Exclusions are retained. Existing processes are never signaled by these guards.

At the end, `analysis/paper_readiness.py` generates `handoff/report.md`, numerical
tables, exploratory within-corpus bootstrap intervals over clustering starts,
and PNG/PDF figures for component attribution and geometric tradeoffs. These
intervals are not cross-dataset generalization or multiplicity-adjusted tests.
All timing phases remain separate; setup, preprocessing and query reuse must
be included in any complete workload comparison. An identical query can cache
its prior proof for every comparator.

## Submission decision

Two additional Full q=8 controls (MNIST/Fashion, two existing distinct subsets)
are queued AFTER the 96-job workflow. Only the constraint ceiling rises from
1.3M to 1.5M; the same RAM guard remains. This tests whether the earlier near-cap
exclusions can be replaced with actual measurements. They have their own output
and receipts under `full_budget_control`, a seven-hour finite queue gate and a
two-hour active deadline. Their source is `experiments/full_query_budget.py`;
`tmp/full_budget_deployment.json` pins the detached process. The small actual-proof
and public-only verification path was tested locally. Excluded cases remain
excluded; there is no repeated resource escalation.

Completion of this plan should allow writing from a fixed, inspectable evidence
set. It cannot guarantee that every broad claim in the older manuscript is
supported or that a Q1 journal will accept the paper. Main unresolved scientific
decisions are the contribution relative to exact affine statistics and cached
fixed-task answers, the specific
geometric use case justifying JL, formal statement/quantization guarantees, and
comparison with verified prior ZK-PCA work. Negative comparisons must remain in
the paper. The older manuscript's unimplemented-Stage-1 wording must be updated
with the new cohort's actual scope rather than mechanically replaced.
