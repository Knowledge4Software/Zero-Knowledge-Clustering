# Completed local group-moment experiments, 23 October 2026

The exact group-moment identity works on every existing natural regression.
A bounded authenticated Groth16 implementation also works, but its current
cost is higher than reusable Full. This is a correctness result and a useful
negative cost result, not a demonstrated efficient full-scale replacement.

## Expanded numerical experiment

The development cohort contains all eight historical fits with natural
regressions, at both early and mature baseline ages. Every companion actual
update is retained. Four additional fits use preselected new seeds 67/79 for
MNIST k5 and Fashion-MNIST k3. All catalogs contain 1,024 audit records.

| Cohort | Unique update requests | Improvements | Regressions | Exact changes matching Full |
| --- | ---: | ---: | ---: | ---: |
| Historical development | 64 | 40 | 24 | 64/64 |
| New split/fit seeds | 32 | 32 | 0 | 32/32 |

Both head and PCA are evaluated on each request: 192 method/request checks,
zero incorrect aggregate changes. The 24 development regressions comprise
the exact ten MNIST and fourteen Fashion-MNIST requests identified in the
referee reanalysis; the summarizer checks their identities explicitly.
The new fits contain no naturally worsening updates, so they do not provide
independent confirmation of regression sensitivity. Repeated updates within
each fit share data and are dependent observations.

| On the 24 historical regressions | Head | PCA |
| --- | ---: | ---: |
| Exact changes recovered | 24/24 | 24/24 |
| Median rows requiring exact nearest-center work | 2.44% | 5.47% |
| Minimum | 0.59% | 0.98% |
| Maximum | 12.70% | 17.58% |

Numerical execution completed in 560.48 seconds with two compute threads.
These row fractions do not measure proof cost. Training/audit hashes and true
changes match historical captures. Historical PCA basis hashes match exactly;
head bases are locally refitted and differ from the historical ones. Actual
local bases, group labels, centers and moments are persisted in the native
output folder, with pins and SHA256 manifest. The confirmation seeds have no
historical basis to compare against.

## Authenticated implementation

The new `code/zk/circuit/bound_group_moments.go` implements:

1. Complete original-row coverage, with fixed catalog indices, original
   Merkle authentication, and checked head/residual-energy derivation.
2. A salted commitment to private group memberships, counts and coordinate
   sums, checked against all original rows. A self-consistent false sum and
   matching commitment are rejected, not merely changes against a fixed hash.
3. Query authentication of every head/energy row and the same group catalog.
   For each uncorrected row, the circuit proves strict old/new nearest-group
   assignments using the existing rounded-basis error bounds.
4. Authenticated original rows and exact old/new nearest-center costs for
   correction slots. Active indices are sorted and unique; every uncovered
   row must satisfy both assignment certificates. Padded slots cannot hide
   a missing correction. Signed group sums and changes are range checked.
5. Public-only verification rebuilt from approved full centers, root/model
   pins and pinned verifier keys. Output changes agree with independent Full
   proofs, not just an owner-side numerical oracle.

The command `commitjl-group` supports checkpoint reuse only after verifying
completed phase proofs against reconstructed witnesses and matching pins/tasks.
Public replay requires no row inputs. Its trusted pins/tasks must come from
the authorized owner/auditor; prover-supplied keys or roots alone are not trust.

The implemented contract is a fixed full audit catalog, fixed k and center
index correspondence. Arbitrary subset audits need corresponding authenticated
moments. The prototype makes correction indices and active slots public; it
therefore has a different leakage contract from a protocol with private flags.
Group moments, memberships and record/head values remain private witnesses.
The exact aggregate change is public.

## Proof costs

Each workload uses one timing repetition, no warmup, one reusable Full setup,
the exact baseline plus four requested center sets for Full, and complete
group preparation plus four group queries. Neither fitting nor network,
artifact serialization/I/O or deployment concurrency is included. Query
verification is charged once in the following reconstructed component totals;
additional independent replay is a validation check recorded separately.

| Authenticated workload | Group pipeline | Reusable Full | Group / Full |
| --- | ---: | ---: | ---: |
| Synthetic N4, d4, rank1, k2 | 12.59 s | 4.76 s | 2.65 |
| Fashion-MNIST N4, d784, rank63, k3 | 733.84 s | 299.39 s | 2.45 |

The real-data pilot uses the first four rows of the pinned local Fashion
audit fit, at its full 784-dimensional input. This is not an N1024 proof run.
Its correction capacity is zero because all four rows have stable certified
assignments; the synthetic pilot exercises actual corrections. Thus even
the zero-correction real pilot does not demonstrate a net cost benefit.
The real full-feature run was resumed from completed verified phases; its
reported totals reconstruct successful components across phases. Its final
invocation wall time is not an end-to-end deployment latency measurement.
Repeated quantized center sets are retained as requested tasks and proved
again in both pipelines; an exact-result cache could remove duplicate work.
These are bounded prototype measurements, not optimized-service comparisons.

| Real-data phase | Constraints | Compile | Setup | Total proving |
| --- | ---: | ---: | ---: | ---: |
| Group preparation, once | 1,223,922 | 20.69 s | 401.83 s | 30.86 s |
| Group query, four proofs | 625,197 | 15.76 s | 200.65 s | 57.95 s |
| Full, five proofs | 703,315 | 3.50 s | 216.37 s | 79.43 s |

Reopening high-dimensional group sums and binding the whole catalog in one
preparation circuit dominate this implementation. Lower exact-row fractions
have not translated into lower total cost on these small workloads. We do
not extrapolate this comparison to 1,024 records or claim production memory
performance. Native and proof measurements answer separate questions.

## Validation and reproduction

- All 19 targeted Python tests passed.
- `go test ./... -p 1` passed across all Go packages.
- Both bundles pass fresh public-only replay: 20 proofs in total.
- Mutated cost changes and approved center inputs fail cryptographic replay.
- Circuit tests reject false moments, changed membership, duplicated original
  rows, omitted/duplicated corrections, changed correction records and false
  aggregate outputs. These tests are not a complete security audit.

From `code/`, reproduce the numerical experiment with a fresh output directory:

    python -m experiments.pilot_group_moments --extended --out <fresh-directory>

From `code/zk/`, build the proof command:

    go build -o bin/commitjl-group.exe ./cmd/commitjl-group

Owner benchmark inputs are retained under
`code/validation/outputs/local_group_proofs_20261023/`; proof runs use a fresh
output folder. From the project root, replay without private inputs:

    code/zk/bin/commitjl-group.exe --verify --out code/validation/outputs/local_group_proofs_20261023/fashion_n4
    code/zk/bin/commitjl-group.exe --verify --out code/validation/outputs/local_group_proofs_20261023/smoke

Regenerate the checked numerical/cost summary and hashes:

    python code/analysis/summarize_group_moments_local.py

## Decision

The sensitivity gap now has a mathematically exact candidate with a working
small authenticated implementation. The current implementation has not solved
the efficiency gap. A further optimization experiment should target moment
commitment cost and complete-preparation aggregation, then measure a controlled
larger catalog against Full before making a paper speed claim. No new Mobin
jobs were launched, no public publication was authorized, and the manuscript's
scientific claims were not silently replaced with these prototype results.
