# Group-based cost-change certificates ? 24 October 2026

The new numerical bounds resolve natural regressions that the ordinary paired certificate cannot resolve. Efficient authenticated end-to-end performance is not yet established. Do not combine these numerical results with the separate4.53? speed result into a single demonstrated claim.

## Numerical evidence

All catalogs contain1024 committed quantized records at rank63 and scale16384. Historical development includes all64 requests from the eight previously selected regression-bearing fits:40 improvements and24 regressions. The previously fixed seeds67/79 add32 improvements and no regressions. Fresh confirmation at seeds167/193 includes64 requests,52 improvements and12 regressions, across both image datasets, k3/5, both ages and representations. These fresh seeds were fixed before evaluating the new bounds. Each dataset/seed split is shared across k/age/method configurations; updates are dependent observations.

| Cohort | Representation | Bound | Regressions resolved without exact rows |
|---|---|---|---:|
| Historical regression-bearing fits | learned | ordinary_paired | 0/24 |
| Historical regression-bearing fits | learned | group_separate | 18/24 |
| Historical regression-bearing fits | learned | group_paired | 18/24 |
| Historical regression-bearing fits | pca | ordinary_paired | 0/24 |
| Historical regression-bearing fits | pca | group_separate | 10/24 |
| Historical regression-bearing fits | pca | group_paired | 11/24 |
| Fresh fixed seeds167/193 | learned | ordinary_paired | 0/12 |
| Fresh fixed seeds167/193 | learned | group_separate | 12/12 |
| Fresh fixed seeds167/193 | learned | group_paired | 12/12 |
| Fresh fixed seeds167/193 | pca | ordinary_paired | 0/12 |
| Fresh fixed seeds167/193 | pca | group_separate | 7/12 |
| Fresh fixed seeds167/193 | pca | group_paired | 11/12 |

The head's inexpensive separate bound resolves18/24 historical regressions with no exact work,23/24 with at most6 selected rows, and24/24 with at most52 rows. Six of1024 rows is0.586%;52 is5.078%. The protocol names these ceiling-rounded0.5% and5% budgets. On all12 fresh regressions the head resolves the sign without exact rows; all64 fresh requests are conclusive and correct. In total the head resolves151/160 unique requests directly, including30/36 naturally worsening updates. A52-row maximum budget resolves all160 head requests numerically.

Across both representations and all declared variants/budgets there are zero incorrect conclusive decisions, and every interval contains the independently computed Full cost change. All failed/inconclusive cases, PCA results, and budgets are retained in all_outcomes.csv and the source result tables. Historical head bases are the previously persisted local refits; PCA historical bases match the older captures. The speed workload uses another ordered head basis. No uniform learned advantage or combined accuracy/speed replication is claimed.

## Why the certificate is stronger

For a fixed group g(i), let A be the exact sum of assigned-group cost changes, obtained from private group counts and coordinate sums. Define b_v(i)=min_j ||x_i-c_v,j||?-||x_i-c_v,g(i)||?. Then the true nearest-center change is exactly A+sum_i(b_new(i)-b_old(i)). The integer rounded-basis identity bounds each within-set distance difference. When the group is certified nearest, the switching advantage is exactly zero, so these records add no uncertainty. The paired variant subtracts the group's own motion before bounding the remaining residual direction. Group labels need not themselves be true old nearest-center labels for this identity and the current separate/paired bounds.

Width-priority refinement selects records using only correction interval widths, before consulting their exact values. A proof implementation must authenticate every selected original/head row, prevent duplicate corrections and charge unsuccessful earlier attempts. Lower exact-row counts alone are not proof speed evidence.

## Authentication prototype

The old whole-catalog group circuit was slower than Full on its bounded real pilot. The new experimental bridge binds rows and group sums using a polynomial fingerprint over the proof field, linked by commitments to private random masks. Memberships, sums and masks remain private. COMPLETE ordered row coverage and a consistent mask chain are mandatory.

Security requires an unpredictable nonzero auditor challenge issued AFTER the input, membership and moment commitments are fixed and pinned. A prover-selected/preannounced challenge is unsafe. For fixed false moments, a random challenge misses a nonzero degree-D polynomial with probability at most D/(p-1) per group, in addition to commitment/proof assumptions. The test suite deliberately constructs a known-challenge forgery to preserve this requirement. There is currently no production commit/challenge service or complete standalone group-audit verifier; these are prototype circuits/public-witness builders. The small proof fixtures use a fixed challenge to check implementation and do not establish interactive-protocol soundness by themselves.

Seven fresh synthetic proofs pass reconstruction from public inputs:four row bindings,one moment link,one assigned-cost aggregate andone switching-bound batch. Rounded nonorthogonal basis tests, false bounds, false moments with recomputed commitments, changed memberships, omitted/duplicated rows, broken mask chains and changed public contributions are covered. These are targeted correctness checks, not a complete security audit.

At N128,d784,k5,r63, compile-only counts are192411 per binding row,844879 for the link,919278 for the assigned aggregate, and425723 for a16-record switching batch. At N1024 the corresponding counts are195786,856697,935024 and461768. They fit the existing1.6-million-constraint guard. These counts are not runtime measurements; the extra link/setup/aggregate proofs must all be charged.

## Separate measured speed evidence

The completed original relative-certificate workload has128 records and20 center sets:Full196.80minutes,rank6343.46minutes(4.528?),rank3934.14minutes(5.765?). Rank63's worst agreement is99.21875%;rank39 fails the99% floor. All896 public proofs and numerical outputs passed checking.

The resumed cached rank63 workload completes with288 public proofs and matching numerical outputs. It consumes50.01 active phase minutes, excluding the user's offline pause but including restart/replay overhead. Its corresponding active phase ratio is3.933?. It is not an uninterrupted cold run and does not demonstrate a speed improvement from caching. The datasets/circuits were not interleaved across laptop clock/load conditions.

## Remaining work

1. Finish the complete post-commit auditor challenge, public verification and resumable benchmark harness for the new group relations.
2. Implement authenticated selective corrections and measure actual on-demand fallback costs, including unsuccessful attempts and Full baseline reuse.
3. Run matched group/Full/PCA timings across independent fitted models and scales. Keep early, mature, improvement, worsening and inconclusive outcomes separate; no positive-only selection.
4. The authenticated known-family affine baseline is implemented and is now proving the matched128-record/20-set workload with an85-center bank. Its eight-proof public-only smoke matches Full and rejects altered costs and out-of-bank centers. Final timings are pending. It supports preapproved centers; future unknown centers need new preparation.
5. Recover and independently replay the original Mobin exports when network access returns. Screenshots remain provisional. Keep artifacts private for supervisor review as the user instructed; author details remain the supervisor's task.
6. Update the manuscript only with validated results and an explicit final claim/contract. No claim that every reviewer objection is resolved, that the new full certificate is5? faster, or that acceptance is assured is supported yet.
