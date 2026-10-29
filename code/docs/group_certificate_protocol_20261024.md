# Experimental group cost-change certificate

This is an experimental protocol specification. The numerical reports are in `paper_results/review_controls/group_change_certificate_20261024/REPORT.md`. The v1 executable completed its matched 128-record benchmark: 300 public proofs, 19/19 correct conclusive improvement decisions, 68.88 active minutes, and a 2.856-fold Full ratio. It is slower than the previous relative pipeline. Selective-correction proofs, measured nonzero fallback costs and authenticated natural-regression replication remain unfinished. A separately keyed exact-old profile is described below; do not attribute v1 measurements to it.

## Statement and identity

The input is the full fixed committed integer catalog, the deployed basis/quantizer and ordered old centers. A fixed private group label is chosen for each record before future updates. New centers have the same count and index correspondence. Arbitrary subset audits require corresponding authenticated group totals; the existing totals cannot silently be reused for a different subset.

For a record's fixed group g, define b_v = min_j ||x-c_v,j||² - ||x-c_v,g||². If A is the sum of assigned-group cost changes, the true nearest-center cost change is exactly A + sum(b_new-b_old). Counts and coordinate sums compute A exactly. The rounded-basis within-set distance identity bounds each b_v, retaining its exact Gram correction. Old/new minimum-advantage bounds produce a sound correction interval; the optional paired numerical variant cancels the group's own motion first.

**In v1, groups are not proved to be old nearest-center labels. A alone is therefore not a valid sign certificate.** For x=0, old centers[0,10], new centers[9,1] and fixed group1, A=-99 but the true cost change is+1. The old switching advantage supplies the missing+100. Python and Go regression tests cover this counterexample. The separately keyed exact-old optimization below additionally proves every old argmin; choosing such labels only in native code would be insufficient.

The query outputs currently disclose the assigned-group aggregate A and per-batch switching intervals, in addition to the final interval/sign. Counts, coordinate sums, row values, group labels and masks stay private. This aggregate disclosure is part of the contract and must be stated explicitly.

## Scalable moment binding

1. Commit the original data, head/energy rows, fixed group-membership Merkle root and private group counts/coordinate sums. The moment commitment binds the model/input/statistic roots and the membership root.
2. An independent auditor issues and pins an unpredictable nonzero field challenge AFTER these commitments are fixed. A noninteractive replacement has not been implemented or justified here. A prover-selected or preannounced challenge is unsafe: the tests deliberately construct a false moment polynomial vanishing at a known challenge.
3. For every fixed catalog index, a row proof authenticates the original row, head/energy derivation and private group membership. It computes the polynomial1+sum_d x_d r^(d+1) and outputs one masked field contribution per group. Private random mask vectors are linked by indexed hash commitments before/after each row.
4. The verifier requires all indices0..N-1 exactly once and checks every adjacent mask commitment. It sums public masked contributions in the field. Missing rows, duplicate indices or broken chains are rejected.
5. One link proof opens the private moment commitment and the first/last private mask vectors. For each group it checks that the folded contribution equals count+sum_d sum_d r^(d+1), plus the endpoint-mask difference. Counts and coordinate sums are integer-range checked, and counts sum to N.

For fixed false moments, the discrepancy is a nonzero polynomial of degree at most d. An independent random challenge misses it with probability at most d/(p-1) per group, in addition to the underlying commitment and Groth16 assumptions. Integer ranges exclude distinct small integer coefficients becoming identical modulo the field. A full security argument, protocol challenge-pinning service and privacy review are still required; passing circuit tests is not a complete security audit.

## Query verification obligations

`BoundGroupAssignedChange` authenticates the same moment root, counts and sums and proves A for the auditor-approved original-space new centers. `BoundGroupSwitchBatch` authenticates each head/energy row and fixed group label and proves its correction bounds. Native parameters are derived from the full approved centers, not accepted as free prover claims. Public witness builders reproduce them independently.

The final service must require complete ordered query coverage of the same full catalog, the same model/original/statistic/membership roots across all relations, the separately pinned challenge and trusted verifier-key hashes. Signed integers must remain canonical and satisfy the checked field bounds. The auditor sums A*T^6 and all correction bounds using arbitrary-precision integers. A positive lower bound proves worsening; a negative upper bound proves improvement; otherwise the outcome is inconclusive.

`cmd/commitjl-bound/group_fast.go` supplies the experimental standalone relations and public-only verifier. `validation/check_group_fast_proofs.py` issues independent local auditor randomness after the commit receipt, exports an explicit public allowlist and checks missing/duplicate/changed rows, omitted batches, altered assigned aggregates/decisions, changed challenge/centers and duplicate approved batches. The auditor retains its owner pin outside the prover's bundle. Production transport authentication and a complete security/privacy audit remain outside this local simulation.

## Refinement and cost gate

The numerical refinement frontier selects the largest correction-interval widths before reading exact corrections. It uses ceiling-rounded row budgets0,6,11,21,52 and a complete-refinement endpoint for N1024. The circuits for selective exact corrections are not implemented yet.

An authenticated refinement must open both original and compressed rows plus membership, recompute the same base correction bounds, and replace selected intervals by exact corrections. It must forbid duplicate active indices across batches and prevent double-counting when a larger refinement replaces an earlier attempt. Public selected indices/active slots change the leakage contract. Failed earlier attempts, added key/setup costs and all verification must be charged.

The next actual benchmark should include preparation, the link proof, assigned-group proofs, correction batches, verifier parameter preparation, fresh public replay and any exact refinements/Full fallback generated prospectively. Use equally reusable Full and PCA. Compile counts only establish that each component fits the resource guard; they are not a speedup measurement.

The currently observed group bounds address cost-update decisions, not a universal guarantee that every approximate assignment is exact. Keep assignment agreement, certified assignment coverage and certified cost decisions as separate quantities.

## Exact-old profile v2: cheaper direction certificates

`group-exact-old-v2` uses a different commit receipt version and independently pinned `group_row_exact` and `group_switch_exact_Q` keys. Every original-row fingerprint proof additionally proves that its private group minimizes the exact old linear score `||old_j||^2 - 2*x dot old_j`. The common `||x||^2` cancels. Signed range bounds and comparisons are checked; ties are accepted. The old centers are fixed in the model before these proofs. Changing the deployed comparison base requires corresponding new preprocessing/proofs.

Consequently `b_old=0` and `Delta=A+sum(b_new)<=A`. An authenticated negative A certifies improvement without any per-row query proofs. Public identity centers give exact equality. The verifier permits those modes only after ALL exact-old row proofs and the moment link pass and the same assigned-change proof passes. A prover cannot invoke them under v1, for a positive A, with altered intervals, or with omitted preprocessing rows. Other updates require every new-switch batch and can remain inconclusive; no natural worsening is certified through the negative-A shortcut.

The aggregate-only path reports the conservative interval `[A-4*N*D*X^2,A]` in original integer cost units, where `X=2^input_bits`. It proves a useful direction, **not a narrow cost interval or exact individual new assignments**. This change in guarantee is deliberate and must remain visible. The fallback switching path uses the stronger exact-old correction bounds and the same rounded-basis/Gram treatment. Full fallback is generated after verified inconclusive decisions.

The first timing control charges four cold key setups, including the switching key even when all updates take the aggregate path. It retains the original 19-update workload and all attempts, setup, fitting, row/link/aggregate proofs and fresh public replay. Its binary is `code/zk/bin/commitjl-bound-group-exact.exe`; never overwrite the frozen v1 binary. The exact-old exported smoke passes ten proofs and twelve tampering cases, including a real switching-path worsening example. Python and Go tests reject a fully authenticated but nonnearest old label and accept ties. This is not yet real-data worsening replication or a complete security audit.

The native recheck uses all 80 stored archives and all 160 updates per representation without refitting. The proof timing queue starts only after both numerical cohorts finish with sound intervals and no loss against the predefined head coverage/sensitivity gate. This is a development gate on already stored fits, not newly independent validation. Paths: `code/validation/outputs/group_exact_confirmation_20261024/`, `group_exact_smoke_20261024/`, and `group_exact_control_n128_20261024_queue.json`.
