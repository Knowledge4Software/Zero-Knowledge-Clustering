# Original-data binding and authenticated subset audit

This is a new, separate experiment cohort (`bound-audit-v1`). The manuscript's
older query-only MiMC timings are preserved. They do not include this protocol.
The new Go implementation is in `zk/circuit/bound_*.go` and
`zk/cmd/commitjl-bound/`; the finite Linux campaign is `experiments/bound_audit.py`.

## The three missing controls

1. **Preprocessing proofs:** one Groth16 proof for each catalog index links a
   committed original integer row to its compressed statistics. The independent
   verifier requires every index, all under the same original and output roots.
2. **Exact baseline and JL ablation:** compare Full, PCA-16, protected head plus
   energy, that same head/energy plus a signed sparse residual tail, and an
   optimized exact affine sufficient-statistics baseline. The latter stores
   exactly k numbers, including energy, rather than a redundant zero score.
3. **Actual subset workload:** queries authenticate public sorted distinct row
   indices under one fixed catalog root. Different subsets reuse a compiled
   circuit and key. A whole-catalog query has only one possible subset; repeating
   it is a timing repetition, not a new query or an amortization argument.

## Exact proved relation

The owner commits **quantized integer rows**, not the original floating-point
files. The benchmark uses input scale S=4096/max(abs(data),abs(centers)), rounds
data and centers, and takes T=4096. These public centers are fitted on 60% of the
data, with evaluation on the remaining 40%. Model fitting is outside the proof
for every comparator. Center indices break assignment ties in ascending order.

Let w=x-c0 and let B/T approximate an orthonormal basis of the public center
differences. The protected relation is integer-exact:

```
dot = B w
H   = T dot
R   = T^2 w - B^T dot
E   = sum(R_i^2)
tail = A R                 # protected_tail only
```

Decoded coordinates are H/T^2, residual R/T^2, and energy E/T^4. Dividing costs
by S^2 then expresses them in the original input units. B is rounded, so the
decoded distance identity is approximate. Neither exact real-valued cost nor
universal perfect agreement is claimed. Quality exports compare against both
floating Full and quantized Full. Bounds prevent integer/field ambiguity.

For the affine baseline, store k-1 exact scores
`s_j=2<w,c_j-c0>-||c_j-c0||^2` and `E=||w||^2`. With implicit s0=0,
assignment is argmax(s) and minimum distance is E-max(s). Thus affine is exact
relative to quantized Full, uses k statistics, and needs no JL tail.

PCA uses the same integer input, scale and commitment rules, with 16 fitted
principal directions. Protected-tail stores at most 16 values total: k-1 head
coordinates, one energy, and the remaining tail coordinates. The no-tail and
affine controls use k values. Full stores d coordinates.
PCA soundly proves its own projected-space cost. Its difference from the
original-space cost measures the error of using that objective as a proxy; it
does not mean that a verified PCA proof is mathematically invalid. The exact
affine baseline answers the same original-space cost question as Full.

## Commitments, trust and verifier

All comparators use the same new Poseidon2/BN254 row Merkle implementation.
An original leaf binds domain, N, d, index, original salt, and coordinates.
Compressed head **and energy share one row leaf/path**, also binding the public
model hash. Tail rows have a separate root. The reserved energy-root field CE
must be zero. A public context hash binds all roots and the model.

The benchmark constructs one original catalog root and asserts identical roots
across comparators. Stage 1 proves every row's original and compressed openings
plus the arithmetic above. Stage 2 opens the requested compressed rows and
checks the fixed-center assignment/cost. Full opens original rows directly and
requires no Stage 1. Keys are reused within each model and query-size shape.

`-phase verify -trust trusted.json` reads saved proofs and public statements;
it does not need original rows, compressed arrays, or the prepared Gob cache.
The trusted manifest pins the owner root, public model and verification-key
digests. **It must be obtained from the trusted owner/setup, not accepted from
an untrusted prover.** This experimental trusted setup is not a production
ceremony. A verifier must also approve the public model's construction: circuit
validation bounds coefficients but does not prove model training, orthonormality
or center-span coverage.

For the tail, the benchmark samples a local verifier nonce after fixing the
original root and derives a canonical signed sparse matrix from root, model
parameters and nonce. It is one signed bucket per input coordinate. The matrix
is pinned in the circuit, but its generation is not an in-circuit XOF and the
Go bundle verifier does not independently regenerate it. There is no distributed
beacon or formal security/novelty claim. A deployment verifier must check the
challenge derivation and canonical matrix externally.

## What the tail experiment establishes

Fixed-center cost/assignment use head and energy; the tail is unused there.
Tail and no-tail must have identical head/energy answers. Stage 1 does bind the
tail, so its extra preprocessing cost is measured. Separate native pair-distance
measurements evaluate head-only versus head plus tail, **excluding energy as a
Euclidean feature**. No authenticated pair-distance query proof is implemented.
The study uses one map per clustering start, not a broad JL failure-probability
study. A geometric benefit alone does not establish a downstream application
benefit or an advantage over the exact affine baseline.
Native geometry maps use reproducible dataset/start seeds. Actual proof bundles
use separate post-commit challenges; the geometry table does not measure those
specific challenged matrices. With energy excluded, the Euclidean tail/head
representation has 15 coordinates when the storage budget is 16 values.

## Campaign, timing and resource limits

Smoke: synthetic n=100,d=32; catalog N=4; all five methods.
Pilot: Gaussian-hard n=2000,d=512; k=5; catalog N=8; all five methods.
Main native: eight datasets, k in {5,10}, three clustering starts (240 rows).
Real-data caps: MNIST/Fashion 5000, Gisette 3000, Arcene 100, nonzero text at
most 3000 before zero-row removal; synthetic sizes appear in datasets.json.

Main actual proofs: Gaussian-hard N=32 at k=5/10; MNIST, Fashion and text N=16
at k=5; Gisette N=8 at k=5. There are 30 paired bundles and 480 Stage-1 row
proofs. These are small full-feature catalogs, not full-corpus proofs. Query
sizes are 1,8,16,32 when they fit. For q<N, main tests four distinct subsets
(eight for q=8); q=N tests one. Query timing uses three measured repetitions
after one warmup; setup is reused, and only one serialized proof is retained
per distinct query. Stage-1 rows are each measured once after a first-row warmup.

Exports separate KMeans training, model fit, native compression, downstream
answering, native commitments/preparation, circuit compilation, setup, witness
construction, proving, verification, and observed phase wall time. Native
compression and Go preparation are different pipelines, so do not add both
as if they were consecutive measured steps. Query-only speedups do not include
preprocessing; end-to-end figures must include it and clearly state the query
workload. Repeated identical queries can cache a previous proof.

The server runs four threads. Each child has a 2400-second deadline, 5.3 GiB RSS
guard, 768 MiB host reserve and a 1.3M-constraint budget. Full may be excluded
before compilation by a conservative estimate; such values are **not measured
constraint counts**. All loops are finite and failures stop the new campaign.
Runs use fresh directories, immutable deployed sources and detached processes.
They survive closing the laptop. Previous experiments are not stopped.

Regression coverage includes native/circuit conformance, altered data, resealed
false head/energy/tail, invalid membership/index/cost/context, model serialization,
missing preprocessing proofs and substituted roots/keys. These checks are
engineering evidence and do not replace an independent cryptographic audit.
