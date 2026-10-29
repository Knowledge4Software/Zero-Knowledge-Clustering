# Local accuracy and certificate search ? 24 October 2026

The latest method improves empirical fidelity and the assignment certificate. A matched end-to-end proof timing campaign is running. The approximately 5? goal has not been demonstrated for these settings.

## Current fixed update sequence

MNIST, k=5, fit seed11, ordered training-only head, T=16384. All 20 distinct training Lloyd iterates are retained; exact duplicate iterates21?24 were cached before auditing. This is one fitted model, not 20 independent fits. The 128-record proof catalog uses the first128 frozen audit records with no row selection. The earlier n32 accuracy failures remain in the original campaign preparation.

| Rank | Catalog | Mean agreement | Worst agreement | Mean relative certified coverage | Worst relative coverage | Maximum absolute cost error | Maximum absolute interval width |
|---|---|---:|---:|---:|---:|---:|---:|
| 39 | native1024 | 99.6631% | 99.4141% | 91.5576% | 89.9414% | 0.0819% | 4.7118% |
| 39 | proof_catalog | 99.5703% | 98.4375% | 90.0000% | 87.5000% | 0.0687% | 4.9652% |
| 63 | native1024 | 99.8340% | 99.6094% | 95.2100% | 93.6523% | 0.0323% | 2.6918% |
| 63 | proof_catalog | 99.8047% | 99.2188% | 95.1562% | 90.6250% | 0.0276% | 2.8058% |

Agreement means agreement with exact quantized Full, not ground-truth classification accuracy. The empirical error denominator and interval-width denominator are the exact quantized nearest-center cost. The relative certificate guarantees assignments for the reported certified count. The remaining records are uncertified; the full approximate assignment set is not guaranteed exact. The absolute cost interval remains conservative and does not imply that a cost-change decision is conclusive.

Rank39 fails the 99% floor on at least one set in the actual n128 proof catalog. Rank63 meets the floor on every set there. The original n32 proof catalog fails the floor for both ranks. No failing rows or updates were removed.

## Broader numerical screen

For the four actual-update screen on MNIST/Fashion, k3/5 and ages1/8, rank39 reached minimum99.3164% agreement on the development seed and99.1211% on additional seeds101/131. Rank39 was selected after looking at those seeds, so they are validation data for this choice, not an untouched final confirmation set. Tested low-rank JL residual corrections failed the per-update99% gate. All maps, weights and failures were retained.

On the development rank63 grid, the new relative assignment bound increased pooled mean certified coverage from96.4920% to97.9355%, with zero false certified labels. This is a different aggregation from the 20-center-set table above. It does not resolve every natural cost-regression audit.

## Proof checks and timing

The real-proof smoke test freshly verified8 public proofs, matched Python/Go relative counts and assignments, rejected altered public certified counts and changed approved centers, and confirmed resumability without replacing keys/proofs. Public exports exclude private row witnesses and resume files. Targeted relative-circuit and verifier-coverage tests passed. These are implementation checks, not a complete cryptographic security audit.

The n128 timing campaign compares head ranks39/63 with one equally reusable public-center exact Full circuit. Each method proves all records for all20 center sets. Head batch16 and Full batch8 are separately chosen under the same1.6-million-constraint guard. The resource logs measure preparation, compilation/setup, native answer and witness/proof generation, worker verification and fresh public-only replay; representation fitting is charged to head. Phase-wall sums exclude the public-bundle copy and Python work between phases. Invocation timestamps are retained so the final analysis can also report the enclosing case wall time. No final ratio is claimed until that accounting is checked. An interrupted invocation retains its measured work; restarted wall time is not substituted for a cold run. Native quality screening is separately recorded and is not a deployment component.

The completed group-moment exact-correction branch was slower than Full and is retained as a failed speed candidate. The new relative-certificate branch has no final measured speed ratio yet. No paper claim of5? speed, universal99% accuracy, or solved natural-regression sensitivity is supported at this stage.

Live campaign: `code/validation/outputs/local_fast_tradeoff_n128_20261024/status.json`.
Proof checks: `code/validation/outputs/relative_reuse_smoke_20261024/checks.json`.
