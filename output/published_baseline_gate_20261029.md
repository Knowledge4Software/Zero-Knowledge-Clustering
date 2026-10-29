# Local gate against paper-derived baselines — 29 October 2026

Study status: native complete; cold proof queue complete=True; RaBitQ follow-up complete=True; equal-routing warm follow-up complete=True.

The current four-stage methodology and all manuscript sources are unchanged. This gate uses new, independently refitted configurations of previously inspected public sources. It is exploratory evidence, not untouched confirmation.

Accuracy below means agreement with exact quantized nearest-center labels or the sign of exact quantized cost change. It is not ground-truth class accuracy. Coresets estimate aggregate costs and do not return labels for unsampled records.

## Assessment before renting a server

The current method has a useful verified-reuse operating region, but these results do not establish broad superiority over published systems. Its nearest-center label agreement exceeds the raw random sketches and one-bit RaBitQ on these fitted cohorts. Its deterministic decision coverage is strong for Fashion and HAR, while text is difficult. PCA is a near tie in the measured proof cohort. The measured ZKML primitive adapter is faster in both cold cohorts, with a different output/backend/commitment contract.

Small coresets can estimate aggregate decisions well without producing a deterministic, authenticated full-catalog sign certificate. Their speed and sign accuracy are useful controls, not interchangeable verified services. The text point predictor also needs caution: overall sign agreement can equal the majority-class baseline, as the balanced metric below shows.

A larger server campaign is justified to measure replication, scale and reuse economics if this narrower contribution is the intended claim. It is not justified by a demonstrated advantage of the current basis over PCA. The present proof cohort has only one fit and one timing execution per cell; none of the small differences should be read as statistically established.

## Added published algorithms

- **Random Projections for k-means Clustering** (Boutsidis, Zouzias, Drineas, NeurIPS 2010). [Source](https://papers.neurips.cc/paper/3901-random-projections-for-k-means-clustering.pdf). Algorithm 1 random sign embedding; supplied banks, no cluster refit; rank63 is an empirical budget, not a certified theorem parameter.
- **Database-friendly random projections: Johnson-Lindenstrauss with binary coins** (Achlioptas, JCSS 2003). [Source](https://doi.org/10.1016/S0022-0000(03)00025-4). Independent -sqrt(3/m),0,+sqrt(3/m) entries with probabilities1/6,2/3,1/6.
- **Scalable k-Means Clustering via Lightweight Coresets** (Bachem, Lucic, Krause, KDD 2018). [Source](https://las.inf.ethz.ch/files/bachem18scalable.pdf). Algorithm1, sampling with replacement, exact importance weights; fixed empirical sample budgets, no claimed epsilon/delta at these sizes.
- **RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search** (Gao and Long,SIGMOD2024). [Paper](https://arxiv.org/abs/2405.12497), [author implementation](https://github.com/gaoj0017/RaBitQ). Separate numerical follow-up, registered after the original gate's native outcomes. Independent float64 one-bit estimator with old groups as IVF anchors; exhaustive all-center scan, no SIMD or query quantization. It is not the full ANN engine.
- **ZKML: An Optimizing System for ML Inference in Zero-Knowledge Proofs** (EuroSys2024). [Paper](https://doi.org/10.1145/3627703.3650088), [author implementation](https://github.com/uiuc-kang-lab/zkml). Existing exact-cost primitive adapter is measured on the new matched cohort; this is not the complete published optimizing compiler.

Raw projections and coresets do not provide a deterministic full-catalog sign certificate at these empirical budgets. Their absence of a certificate is not a measured certificate failure. Random-projection certificate arms attach our existing authenticated relation and bounded refinement; they are adapters, not published ZK systems. Their residual-aware predictor is different from raw projected-distance prediction.

[Standalone scientific summary](./published_baseline_gate_20261029_summary.png) ([editable vector](./published_baseline_gate_20261029_summary.svg)). The chart separates label agreement, estimated decisions, deterministic coverage and cold proof-service time.

## Native quality and request cost

Each table pools eight candidates from each of three fits. Per-fit rows are supplied in the CSV. Median milliseconds summarize three timed passes within each fit, then the median across fits. Timing repetitions and correlated candidates are not independent learning replications. Cold basis fitting and reusable numerical preparation are retained in the raw JSON, outside these recurring query times.

### fashion,128 records

| Method | Label agreement % | Estimated-sign agreement % | Balanced sign % | Base coverage % | After16 corrections % | Native cost/prediction ms | Native bound/refine ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exact Full (native) | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 0.955 | N/A |
| Bachem lightweight coreset,16 draws | N/A | 79.17 | 89.13 | N/A | N/A | 0.130 | N/A |
| Bachem lightweight coreset,32 draws | N/A | 87.50 | 93.48 | N/A | N/A | 0.249 | N/A |
| Current centroid certificate | 99.80 | 100.00 | 100.00 | 95.83 | 100.00 | 31.070 | 219.292 |
| PCA certificate | 99.84 | 100.00 | 100.00 | 95.83 | 100.00 | 30.913 | 215.929 |
| RaBitQ one-bit estimator (2024) | 98.50 | 100.00 | 100.00 | N/A | N/A | 1.944 | N/A |
| Boutsidis sign RP (raw) | 92.84 | 100.00 | 100.00 | N/A | N/A | 0.213 | N/A |
| Achlioptas sparse RP (raw) | 92.61 | 91.67 | 47.83 | N/A | N/A | 0.214 | N/A |
| Uniform sample,16 draws | N/A | 91.67 | 95.65 | N/A | N/A | 0.132 | N/A |
| Uniform sample,32 draws | N/A | 100.00 | 100.00 | N/A | N/A | 0.245 | N/A |

True decision mix: 1 improvements, 23 worsenings, 0 equalities. Always predicting worsening scores 95.83%. Balanced sign agreement averages recall over the observed true classes; it is a post-hoc descriptive analysis of the frozen outcomes. False-improvement and missed-improvement counts are in JSON. Native certificate time is mathematical bound/refinement work, without cryptographic authentication or proving. It cannot be compared with ZK seconds.

### text,128 records

| Method | Label agreement % | Estimated-sign agreement % | Balanced sign % | Base coverage % | After16 corrections % | Native cost/prediction ms | Native bound/refine ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exact Full (native) | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 0.796 | N/A |
| Bachem lightweight coreset,16 draws | N/A | 75.00 | 57.14 | N/A | N/A | 0.082 | N/A |
| Bachem lightweight coreset,32 draws | N/A | 95.83 | 97.62 | N/A | N/A | 0.133 | N/A |
| Current centroid certificate | 93.49 | 87.50 | 50.00 | 12.50 | 12.50 | 19.130 | 133.208 |
| PCA certificate | 92.55 | 87.50 | 50.00 | 12.50 | 12.50 | 18.868 | 127.143 |
| RaBitQ one-bit estimator (2024) | 89.32 | 83.33 | 90.48 | N/A | N/A | 0.889 | N/A |
| Boutsidis sign RP (raw) | 68.68 | 91.67 | 66.67 | N/A | N/A | 0.177 | N/A |
| Achlioptas sparse RP (raw) | 70.67 | 83.33 | 90.48 | N/A | N/A | 0.175 | N/A |
| Uniform sample,16 draws | N/A | 79.17 | 59.52 | N/A | N/A | 0.083 | N/A |
| Uniform sample,32 draws | N/A | 100.00 | 100.00 | N/A | N/A | 0.130 | N/A |

True decision mix: 3 improvements, 21 worsenings, 0 equalities. Always predicting worsening scores 87.50%. Balanced sign agreement averages recall over the observed true classes; it is a post-hoc descriptive analysis of the frozen outcomes. False-improvement and missed-improvement counts are in JSON. Native certificate time is mathematical bound/refinement work, without cryptographic authentication or proving. It cannot be compared with ZK seconds.

### har,128 records

| Method | Label agreement % | Estimated-sign agreement % | Balanced sign % | Base coverage % | After16 corrections % | Native cost/prediction ms | Native bound/refine ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exact Full (native) | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 0.736 | N/A |
| Bachem lightweight coreset,16 draws | N/A | 79.17 | 72.11 | N/A | N/A | 0.107 | N/A |
| Bachem lightweight coreset,32 draws | N/A | 91.67 | 94.74 | N/A | N/A | 0.186 | N/A |
| Current centroid certificate | 99.45 | 91.67 | 80.00 | 87.50 | 100.00 | 23.970 | 176.479 |
| PCA certificate | 99.51 | 91.67 | 80.00 | 87.50 | 100.00 | 24.059 | 165.547 |
| RaBitQ one-bit estimator (2024) | 98.44 | 95.83 | 97.37 | N/A | N/A | 1.309 | N/A |
| Boutsidis sign RP (raw) | 93.33 | 83.33 | 67.37 | N/A | N/A | 0.191 | N/A |
| Achlioptas sparse RP (raw) | 92.58 | 91.67 | 87.37 | N/A | N/A | 0.188 | N/A |
| Uniform sample,16 draws | N/A | 75.00 | 69.47 | N/A | N/A | 0.115 | N/A |
| Uniform sample,32 draws | N/A | 75.00 | 62.11 | N/A | N/A | 0.196 | N/A |

True decision mix: 5 improvements, 19 worsenings, 0 equalities. Always predicting worsening scores 79.17%. Balanced sign agreement averages recall over the observed true classes; it is a post-hoc descriptive analysis of the frozen outcomes. False-improvement and missed-improvement counts are in JSON. Native certificate time is mathematical bound/refinement work, without cryptographic authentication or proving. It cannot be compared with ZK seconds.

### Certificate adapters,128 records

The following basis controls share our deterministic interval relation. These are not raw published predictors or independent published proof systems. In particular, a nonorthogonal JL matrix can have useful raw distance estimates while its residual norm is too large for this particular deterministic envelope. Poor adapter coverage does not refute the original randomized approximation theorem.

| Data | Basis in our wrapper | Base coverage % | After16 corrections % | Native bound/refine ms |
|---|---|---:|---:|---:|
| fashion | Current centroid certificate | 95.83 | 100.00 | 219.292 |
| fashion | PCA certificate | 95.83 | 100.00 | 215.929 |
| fashion | Sign RP + our certificate adapter | 4.17 | 4.17 | 219.178 |
| fashion | Sparse RP + our certificate adapter | 4.17 | 4.17 | 189.964 |
| har | Current centroid certificate | 87.50 | 100.00 | 176.479 |
| har | PCA certificate | 87.50 | 100.00 | 165.547 |
| har | Sign RP + our certificate adapter | 16.67 | 16.67 | 167.783 |
| har | Sparse RP + our certificate adapter | 16.67 | 16.67 | 149.023 |
| text | Current centroid certificate | 12.50 | 12.50 | 133.208 |
| text | PCA certificate | 12.50 | 12.50 | 127.143 |
| text | Sign RP + our certificate adapter | 12.50 | 12.50 | 132.587 |
| text | Sparse RP + our certificate adapter | 12.50 | 12.50 | 114.554 |

### RaBitQ probabilistic interval follow-up

RaBitQ uses its one-bit distance estimator and a spherical-coordinate quantile envelope, with a union-bound budget0.01 over all predeclared row/center/bank pairs in each fit/catalog case. Candidate banks were fixed independently of the random rotation. Exact old cost is computed during numerical old-group preprocessing. This is an ideal-arithmetic finite-query statistical statement, not an authenticated integer certificate; this float64 implementation has no outward-rounding proof. The query-time estimate includes interval construction.

| Data,128 rows | Label agreement % | Estimated-sign agreement % | Statistical decision coverage % | Wrong statistical decisions |
|---|---:|---:|---:|---:|
| fashion | 98.50 | 100.00 | 8.33 | 0 |
| har | 98.44 | 95.83 | 12.50 | 0 |
| text | 89.32 | 83.33 | 0.00 | 0 |

The observed interval-validity counts and individual request results are retained, including any failure. Statistical interval coverage is not included in the deterministic-certificate columns above.

## Real-proof adapter gate

One pinned fit per source,32 rows and the first four training-only candidates, with every candidate retained. All arms use eight workers serially under the same laptop guards. The table uses enclosing measured invocation wall time; certificate totals additionally charge basis fitting. It includes orchestration/validation, not just prover time. Active-phase sums are retained separately in JSON. ZKML diagnostic MockProver time is excluded explicitly. Full and adapter exact public costs reveal more than the certificate output.

| Data | Method | Status | Minutes | Coverage before Full | Full fallback |
|---|---|---|---:|---:|---:|
| fashion | Exact Full / Groth16 | complete | 7.870 | 4/4 | 0 |
| fashion | Current centroid certificate | complete | 6.690 | 4/4 | 0 |
| fashion | PCA certificate | complete | 6.699 | 4/4 | 0 |
| fashion | Sign RP + our certificate adapter | complete | 14.085 | 3/4 | 1 |
| fashion | Sparse RP + our certificate adapter | complete | 14.017 | 3/4 | 1 |
| fashion | ZKML primitive adapter / Halo2 | complete | 3.957 | 4/4 | 0 |
| text | Exact Full / Groth16 | complete | 3.926 | 4/4 | 0 |
| text | Current centroid certificate | complete | 11.046 | 1/4 | 3 |
| text | PCA certificate | complete | 11.014 | 1/4 | 3 |
| text | Sign RP + our certificate adapter | complete | 12.634 | 0/4 | 4 |
| text | Sparse RP + our certificate adapter | complete | 11.875 | 0/4 | 4 |
| text | ZKML primitive adapter / Halo2 | complete | 2.290 | 4/4 | 0 |

## Equal cost-aware warm comparison

A separate engineering follow-up applies the existing routing policy equally to centroid, PCA and both projection adapters. Cost calibration is frozen from the earlier seed521 Q8 pilot, not tuned to this new cohort. Each known batch contains four fixed requests; the whole native preflight is charged before batch activation. Authenticated rows, applicable old-cost proofs and setup are reused; all changing-request proofs and public replay are fresh. This is a warm phase-wall comparison, not independently measured RPC latency or a cold crossover.

One newly measured warm Full arm is shared per dataset. It is not counted as four baseline repetitions. Copy/export and independent integer oracle checks are outside these phase sums. Missing cells establish no ratio.

| Data | Basis / policy | Compact requests | Direct Full | Minutes | Warm Full min. | Full / service |
|---|---|---:|---:|---:|---:|---:|
| fashion | Current centroid certificate | 4 | 0 | 2.124 | 4.730 | 2.227 |
| fashion | PCA certificate | 4 | 0 | 2.068 | 4.730 | 2.287 |
| fashion | Sign RP + our certificate adapter | 3 | 1 | 3.597 | 4.730 | 1.315 |
| fashion | Sparse RP + our certificate adapter | 3 | 1 | 3.661 | 4.730 | 1.292 |
| text | Sparse RP + our certificate adapter | 0 | 4 | 2.283 | 2.276 | 0.997 |
| text | Sign RP + our certificate adapter | 0 | 4 | 2.241 | 2.276 | 1.016 |
| text | PCA certificate | 0 | 4 | 2.282 | 2.276 | 0.998 |
| text | Current centroid certificate | 0 | 4 | 2.276 | 2.276 | 1.000 |

Equal-routing warm follow-up complete: **True**. The initial unexecuted warm plan is retained; the active v2 plan uses portable dependency paths. The case list, method policy and calibration are unchanged.

Proof queue status: **complete**. Missing or stopped cells establish no speed ratio.

Raw/projection algorithms use optimized floating-point sketches and coresets use exact integer sampled distances plus floating weights; our native certificates use conservative arbitrary-precision integer arithmetic. Native speed comparisons therefore concern these explicit implementations, not a universal mathematical lower bound.

## Applicability of other papers

- ezDPS proves a DWT–PCA–SVM inference pipeline with model privacy. Its original classification accuracy and runtime are not measurements of our private-catalog, original-space cost-change task. No imported speed ratio is claimed.
- Zhang et al.2021 uses secure multiparty computation for outsourced clustering. It has a different statement and trust/interaction contract; no same-task implementation or runtime is claimed here. The frozen initial source note's word "encryption" was overbroad; this description follows the publisher's abstract.
- ZKML is retained as a separately measured primitive adapter. This does not benchmark its complete optimizing compiler.

## Validation and environment

All reported certificate-adapter proof cells passed their independent exact-integer oracle and public verifier checks, including paid fallback. No wrong deterministic certified decision was observed in the native gate. These checks establish correctness on the recorded cases, not a population guarantee.

Targeted Linux test receipt: complete=True, exit=0. Command: `-m pytest code/tests/test_published_audit_baselines.py code/tests/test_rabitq_audit_reference.py code/tests/test_published_baseline_gate_reporting.py code/tests/test_cost_aware.py code/tests/test_referee_learning_workflows.py -q`. Retained log summary: `28 passed in 7.70s`. It covers baseline formulas, overflow handling, valid intervals, statistical/deterministic report separation, routing and learning workflows.

Measurement host: AMD Ryzen 7 5800H with Radeon Graphics; Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39; Python 3.12.3. Proof receipts record eight workers, affinity0–7 and process-tree peak RSS. Numerical BLAS work uses one thread. Pytest was installed into the isolated validation environment after cold measurements; numerical package versions were unchanged. No server or GPU was used.

## Reproduction

Use a fresh root inside this checkout for new timing measurements. The existing gate retains its immutable plan, row IDs, model inputs, source pins, all attempts, public proof bundles and exact integer checks.

```sh
PYTHONPATH=code OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.published_baseline_gate prepare --root code/validation/outputs/fresh_gate
PYTHONPATH=code OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.published_baseline_gate native --root code/validation/outputs/fresh_gate
# Prepare the follow-ups after the first native gate, before their own outcomes:
PYTHONPATH=code python -m experiments.rabitq_baseline_gate prepare --root code/validation/outputs/fresh_gate
PYTHONPATH=code python -m campaigns.published_baseline_warm_gate prepare --root code/validation/outputs/fresh_gate
# Real proofs require the pinned Linux binaries and the guarded laptop wrapper:
PYTHONPATH=code python -m experiments.published_baseline_gate proofs --root code/validation/outputs/fresh_gate
PYTHONPATH=code python -m experiments.rabitq_baseline_gate run --root code/validation/outputs/fresh_gate
ln -s ../STOP code/validation/outputs/fresh_gate/warm_followup_v2/STOP
PYTHONPATH=code python -m campaigns.published_baseline_warm_gate run --root code/validation/outputs/fresh_gate
PYTHONPATH=code python -m analysis.published_baseline_gate_report --root code/validation/outputs/fresh_gate --stem fresh_gate_report
```
