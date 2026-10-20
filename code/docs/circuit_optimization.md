# Circuit optimization, 10 October 2026

The default Go prover now shares Sparse JL routing coefficients and derives
numeric bounds from the matrix representation. The projection distribution,
commitment formats, private inputs, public centroids, assignment rule and
objective definition are unchanged. This is a new compiled circuit: proving
and verification keys from the previous circuit cannot be reused.

## Shared routing

For each output row r and input column t, the circuit constructs

`a[r,t] = sum_l [SparseRows[t,l] == r] * SparseSigns[t,l]`.

These constrained coefficients are computed once and reused for all n points
and k centroids. Each vector is then projected as `y[r] = sum_t a[r,t] * x[t]`.
Row bounds, sign checks, column-wise distinctness and matrix authentication
remain enforced. No matrix is converted into an unauthenticated constant.

At the source level, coefficient construction scans m*d*q slots once, followed
by (n+k)*m*d coefficient products. The previous projection path scanned
(n+k)*m*d*q slots. These are operation counts, not exact compiled constraint
counts; backend optimizations and linear-expression compression also matter.
The new path still performs dense coefficient products and does not achieve
the native sparse algorithm's ideal (n+k)*d*q operation count.

## Derived bounds

The input and centroid bound remains inclusive at B=4096. The bound on a
projected coordinate is the smaller of the configured RangeBits limit and:

| Matrix representation | Derived absolute coordinate bound |
| --- | --- |
| Witnessed Sparse JL | d*B, using distinct rows per input column |
| Witnessed dense signs | d*B |
| Witnessed coordinate selection | B |
| Constant matrix, including public PCA and identity | B*max_r sum_t abs(A[r,t]) |

The bound is computed from constrained distributions or public constants,
never from the observed private input or an untrusted witness hint. A zero
constant matrix uses a conservative positive bound of one for comparator
compatibility; its projection equality still forces zero outputs.

The circuit continues to check projected coordinates, differences, distances
and the aggregate objective. Difference, distance and aggregate bounds are
respectively 2*P, m*(2*P)^2 and n*m*(2*P)^2. The existing field-size guard
rejects unsafe aggregate bounds. Restrictive user-specified RangeBits limits
are preserved. Both Sparse JL and comparison methods receive the same bound
optimization; the full-dimensional baseline is not intentionally left slow.

## Native Sparse matrix construction

The RQ1 runner now constructs the canonical matrix directly as SciPy CSR,
avoiding an intermediate dense m-by-d allocation. The column sampler,
domain separation, rejection rules and sign order are unchanged. Independent
tests compare canonical records, CSR data/indices/row pointers and projected
values exactly, including q=1 and q=m.

For m=1024, d=10000 and q=2, the dense int64 intermediate alone uses 81,920,000
bytes, while the resulting float64 CSR matrix uses 244,100 bytes. This
comparison concerns matrix storage, not total process RAM: records, input
data and projected outputs also occupy memory. In a three-repetition local
construction benchmark, the median decreased from 0.713 s to 0.519 s.
This host-side optimization does not contribute to the Go prove-only timer.

## Diagnostic controls

Instance JSON accepts `legacy_projection` and `legacy_bounds`, both false by
default. Enabling both reproduces the previous paths on supported inputs.
The Python ZKInstance bridge and RQ2 builder propagate these controls.
Prover output records `circuit_revision` and both flags to identify the circuit.

The retained implementation is for engineering comparisons. Historical server
reports and their scientific records remain results of the previous circuit.
They are not relabeled as optimized measurements. Campaign source fingerprints
also change, so an old campaign must not be resumed under the new source.

## Matched benchmark

From `code/`:

```powershell
python -m experiments.benchmark_circuit_optimization --out results/optimization_new --shapes 8x16x8x3x2 8x32x16x3x1 --repetitions 5 --warmup 1 --threads 4 --timeout 240 --max-rss-gib 4
```

Shapes encode n, d, m, k and q. The runner uses bounded, fixed synthetic integer
inputs and centroids, and saves every input JSON, proof timing sample, peak
process RSS, setup/compile time, binary hash and source fingerprint. Each
Sparse configuration has legacy, routing-only, bounds-only and combined arms;
the full-dimensional control has legacy and combined arms. It refuses to
overwrite an existing benchmark manifest and stops on a failed proof.

An optional `--original-binary` adds a run of a separately preserved original
prover and checks that its constraint count matches the retained legacy path.
This option is intended for independently validating the comparison.

The benchmark has per-process timeout and RAM limits. Total duration depends
on the host and number of shapes; the per-process timeout is not a total study
deadline. Synthetic inputs are not fidelity-selected RQ1 operating points.
Timing repetitions on one witness are engineering measurements, and any
speedup remains local until tested on the preserved paper subsets.

## Verification

Run the Go relation tests from `code/zk/` with `go test ./...`. They compare
legacy and optimized relations using both Groth16/R1CS and PLONK/SCS solvers,
with batched and unbatched range checks. Cases include signed boundary inputs,
maximum-distance configurations, mixed signs, exact ties and malformed rows,
signs, commitments, objectives and inputs. Mutated input/matrix cases recompute
their commitments so rejection cannot be attributed solely to stale hashes.

The Python regression suite checks bridge serialization, diagnostic control
propagation and the existing protocol, projection, hashing and campaign code.
Real local Groth16 proofs complement the compiled-solver regression tests.
Full-size server remeasurement and production PLONK setup are outside this
local optimization study.

## Completed local measurements

The saved [summary](../results/optimization_20261010/summary.md) and
[timings](../results/optimization_20261010/timings.csv) contain 13 configurations
and 65 measured Groth16 proofs, all verified. Two additional tiny actual-backend
controls verified with Groth16 and PLONK. PLONK uses a test-only KZG SRS.

| Sparse shape (n,d,m,k,q) | Constraints before / after | Prove median before / after |
| --- | --- | --- |
| 8,16,8,3,2 | 241,677 / 220,532 | 4.468 / 4.169 s |
| 8,32,16,3,1 | 457,862 / 421,759 | 7.734 / 7.603 s |

These are approximately 8.7% and 7.9% fewer constraints, and descriptive
prove-time improvements of 6.7% and 1.7%. The optimized full-dimensional controls
take 3.887 s and 7.107 s respectively, so Sparse is still slower on these shapes.
Peak Sparse process RAM is approximately 0.52 GiB and 0.94 GiB; RAM reduction
is not consistent across both shapes. Local setup medians are not recorded:
each configuration has one setup measurement, reduced by roughly 10% for Sparse.

The two timed shapes use an initial build of the optimization. The final build
adds a guard for an all-zero public constant matrix. This guard does not alter
any bound in the timed configurations, whose matrices are nonzero. The measured
binary/source fingerprints and the final-build validation receipt are retained
separately. Regression and smoke tests ran during parts of this local study;
timing changes should therefore be treated as exploratory, not as a controlled
paper benchmark. An isolated remeasurement on fidelity-qualified paper subsets
is required before replacing the manuscript's performance numbers.

Final checks passed: the Go test suite, `go vet ./...`, and 136 Python tests
with two skips (excluding the expensive full-size `test_zk.py` integration
module). After the final integer-domain validation change, all 24 targeted
sampler, RQ1 and bridge tests passed. The validation receipt verifies both
backend smoke results, every timed proof record, the current executable hash
and the exact Go source snapshot used for the timing study.
