# Completed exact-old group certificate control

All 284 proofs passed fresh public-only verification, and all 294 exported manifest hashes still match. The proof count is 128 exact-old row bindings, one group-moment link, 19 update aggregates and 136 switching batches. All 19 approved updates have correct conclusive improvement certificates. Every interval contains independently computed quantized Full truth, with no fallback. This timing workload contains zero natural regressions; it does not authenticate the separate 36-regression numerical suite.

| Workflow | Active minutes | Full / workflow |
|---|---:|---:|
| Exact reusable public-center Full | 196.70 | 1.00 |
| Previous rank63 relative certificate | 43.39 | 4.534 |
| Previous group v1 certificate | 68.88 | 2.856 |
| Exact-old group v2 certificate | 68.97 | 2.852 |

The 5-fold target was not reached. V2's observed total differs from v1 by 5.98 seconds, so this run does not demonstrate an overall speed improvement. It is 59.0% slower than the previous relative workflow, which offered weaker cost-decision guarantees. These measurements are one fit and one repetition, taken serially under changing laptop load; they do not establish an equal-guarantee algorithm ranking.

Only 2/19 updates take the aggregate-only fast path. The other 17 require switching proofs. Relative to v1, this saves 16 proof files, while the total protocol still pays for four cold setups, full-catalog authenticated head/energy and exact-old labels, the group link and query proof work. The aggregate fast path certifies direction with a deliberately broad conservative interval; it does not promise a narrow cost estimate or certify every new individual assignment.

The total includes all active attempts, native preparation, fitting, cold setup, proof generation/checks and fresh public replay. Two RAM-reserve stops contributed 384.92 charged seconds, including any work subsequently repeated. Waiting for free RAM and between-phase orchestration/export gaps are excluded. This is an interrupted active-phase total, not an uninterrupted cold run. Do not subtract failed work and label the remainder a measured uninterrupted result.

The underlying rank63 predictions are unchanged: the separate completed assignment check on this same 128-record catalog had 99.794% mean and 99.219% worst agreement. The broader stored-fit numerical recheck retained head decision coverage151/160 and natural-regression sensitivity30/36, with no incorrect conclusive decisions. These numerical cases differ from this timing workload and are not additional real-proof independent replications.

No Full fallback was required here, so nonzero fallback-generation cost is still unmeasured. Authenticated natural-regression replication, selective exact correction, independent-fit timing, and original Mobin export recovery remain unfinished. Passing public proof replay and hash checks is not a complete security/privacy audit. All artifacts remain private for supervisor review.

Source: `C:\PyProjects\zero-knowledge\code\validation\outputs\group_exact_control_n128_20261024`. The companion summary pins the evidence hashes. No manuscript speed claim has been upgraded to 5-fold.

## Matched certificate interval comparison

An additional per-update comparison against the completed v1 transcript finds exactly identical lower/upper cost-change bounds on all17 switching-path updates. On both aggregate-only updates, the v2 interval strictly contains the v1 interval: it is wider, although its negative upper bound still certifies improvement. V2 therefore does not uniformly strengthen the quantitative certificate.

| Same update subset | Updates | V1 mean interval width / old Full cost | V2 mean interval width / old Full cost |
|---|---:|---:|---:|
| Switching path | 17 | 1.6033% | 1.6033% |
| Aggregate-only shortcut | 2 | 0.6071% | 7050.9276% |

The denominator here is the same exact old catalog cost,95512379564 integer units, for both versions and every update. These widths describe uncertainty about the change, not assignment error or empirical approximation error. The shortcut's global range-based lower bound is deliberately very loose; a negative upper bound can prove direction despite that loose lower bound. This supports a direction-only use case and does not resolve a demand for tight cost estimates.

Exact integer endpoints, truth changes, modes and width normalizations are saved in `certificate_intervals_comparison.csv`. Relative to absolute true change, the17 switching-path intervals have identical mean width39.1992%; the two shortcut intervals go from24.2138% in v1 to298764.8509% in v2. Those are a different normalization from the old-cost percentages above and must not be mixed with absolute-cost interval-width figures.
