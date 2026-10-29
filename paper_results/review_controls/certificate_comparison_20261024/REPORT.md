# Certificate comparison

Assignment agreement and certificate coverage measure different things. A sound certificate may return inconclusive; this is not a wrong answer.

## All stored native fits: 160 updates per representation, 36 natural regressions

| Method | Numerical variant | Conclusive decisions | Natural regressions detected | Incorrect conclusive |
|---|---|---:|---:|---:|
| learned | ordinary_paired | 79/160 (49.38%) | 0/36 (0.00%) | 0 |
| learned | group_separate | 151/160 (94.38%) | 30/36 (83.33%) | 0 |
| learned | group_paired | 151/160 (94.38%) | 30/36 (83.33%) | 0 |
| pca | ordinary_paired | 75/160 (46.88%) | 0/36 (0.00%) | 0 |
| pca | group_separate | 124/160 (77.50%) | 17/36 (47.22%) | 0 |
| pca | group_paired | 135/160 (84.38%) | 22/36 (61.11%) | 0 |
| learned | group_exact_old | 151/160 (94.38%) | 30/36 (83.33%) | 0 |
| pca | group_exact_old | 135/160 (84.38%) | 17/36 (47.22%) | 0 |

Full computes the exact reference and resolves every nonzero change. These native rows do not represent 160 authenticated proof executions. Every persisted fit/update is retained; this recheck is not a new independent fit cohort.

## Separate completed 128-record proof workload: 19 proposed updates

| Method | Mean assignment agreement | Worst agreement | Mean certified assignment coverage |
|---|---:|---:|---:|
| full | 100.000% | 100.000% | 100.00% |
| head_r39 | 99.548% | 98.438% | 89.47% |
| head_r63 | 99.794% | 99.219% | 94.90% |

The group certificate changes the cost-change guarantee, not the rank63 predictions. Those unchanged predictions retain 99.219% minimum agreement on this workload; the group relation alone does not certify every assignment.

Completed real-proof group v1: 19/19 correct conclusive decisions, all improvements, 300 fresh publicly replayed proofs, 68.88 active minutes and a 2.856-fold Full ratio. Previous relative rank63: 43.39 active minutes and a 4.534-fold ratio, with weaker cost-decision guarantees. Neither is evidence of a fivefold fully certified improvement.

Exact-old v2 is complete: 284 proofs passed fresh public replay; 19/19 updates have correct conclusive certificates, all improvements, and no fallback. Active time including RAM retry attempts is 68.97 minutes, Full ratio2.852. The 5-fold target is not met. Only2/19 updates skip switching proofs. The native regression suite remains separate from this all-improvement timing workload.

All results stay private for supervisor review.
