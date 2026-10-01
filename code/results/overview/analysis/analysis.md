# Paper experiment analysis

All numerical entries below come from saved rows. Confidence intervals are 95% Student-t intervals over the ten recorded seeds within each configuration; they are descriptive and may extend outside bounded metric ranges. No CI pools datasets or treats repeated rows as independent datasets.

## Run inventory

| experiment | rows | ok | failed |
| --- | --- | --- | --- |
| RQ1 fidelity | 7180 | 7180 | 0 |
| RQ3 overhead | 90 | 90 | 0 |
| RQ3 grinding | 540 | 540 | 0 |
| RQ4 certificates | 265 | 265 | 0 |
| RQ5 scaling | 200 | 200 | 0 |
| Precision 20-bit reference | 120 | 120 | 0 |
| RQ2 proof | 2 | 0 | 2 |
| RQ3 paired supplement | 30 | 30 | 0 |
| RQ4 real supplement | 100 | 100 | 0 |
| RQ5 separation supplement | 80 | 80 | 0 |
| Precision 24-bit reference | 150 | 150 | 0 |

## RQ1: fidelity

Measured 7180 rows on 7 dataset families. Each configuration has ten seed rows. ARI compares projected assignments with the full-space reference; it is not ground-truth label accuracy.

[All 716 configuration-level summaries](../../rq1/analysis/rq1_all_configurations.csv) and [smallest CI-eligible dimensions](../../rq1/analysis/rq1_smallest_ci_eligible_m.csv) are available. Eligibility here requires lower ARI CI ≥ 0.8 and the entire cost-ratio CI within 1 ± 0.1.

[ARI versus dimension](../../rq1/analysis/rq1_ari_vs_m.png); [configuration statistics](../../rq1/analysis/rq1_sparse_q2_k5.csv).

| dataset_name | n | d | m | repetitions | ari_mean | ari_ci95_low | ari_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian_mixture | 2000 | 256 | 16 | 10 | 1 | 1 | 1 |
| gaussian_mixture | 2000 | 256 | 32 | 10 | 1 | 1 | 1 |
| gaussian_mixture | 2000 | 256 | 64 | 10 | 1 | 1 | 1 |
| gaussian_mixture | 2000 | 256 | 128 | 10 | 1 | 1 | 1 |
| gaussian_mixture | 2000 | 512 | 16 | 10 | 0.8487 | 0.7853 | 0.912 |
| gaussian_mixture | 2000 | 512 | 32 | 10 | 0.9706 | 0.9471 | 0.994 |
| gaussian_mixture | 2000 | 512 | 64 | 10 | 0.9999 | 0.9995 | 1 |
| gaussian_mixture | 2000 | 512 | 128 | 10 | 1 | 1 | 1 |
| gaussian_mixture | 2000 | 512 | 256 | 10 | 1 | 1 | 1 |
| dbscan_margin | 1000 | 128 | 16 | 10 | 0.1801 | 0.111 | 0.2492 |
| dbscan_margin | 1000 | 128 | 32 | 10 | 0.4109 | 0.3147 | 0.507 |
| dbscan_margin | 1000 | 128 | 64 | 10 | 0.6953 | 0.579 | 0.8116 |
| mnist | 5000 | 784 | 16 | 10 | 0.3802 | 0.3524 | 0.4079 |
| mnist | 5000 | 784 | 32 | 10 | 0.522 | 0.4964 | 0.5477 |
| mnist | 5000 | 784 | 64 | 10 | 0.648 | 0.626 | 0.6699 |
| mnist | 5000 | 784 | 128 | 10 | 0.7633 | 0.7492 | 0.7773 |
| mnist | 5000 | 784 | 256 | 10 | 0.8364 | 0.827 | 0.8457 |
| fashion_mnist | 5000 | 784 | 16 | 10 | 0.669 | 0.6468 | 0.6912 |
| fashion_mnist | 5000 | 784 | 32 | 10 | 0.7626 | 0.7444 | 0.7807 |
| fashion_mnist | 5000 | 784 | 64 | 10 | 0.8292 | 0.8149 | 0.8434 |
| fashion_mnist | 5000 | 784 | 128 | 10 | 0.8822 | 0.8723 | 0.892 |
| fashion_mnist | 5000 | 784 | 256 | 10 | 0.9158 | 0.9079 | 0.9238 |
| gisette | 3000 | 5000 | 16 | 10 | 0.1624 | 0.1441 | 0.1807 |
| gisette | 3000 | 5000 | 32 | 10 | 0.2928 | 0.2791 | 0.3065 |
| gisette | 3000 | 5000 | 64 | 10 | 0.4368 | 0.4213 | 0.4524 |
| gisette | 3000 | 5000 | 128 | 10 | 0.5681 | 0.5506 | 0.5856 |
| gisette | 3000 | 5000 | 256 | 10 | 0.693 | 0.6791 | 0.7069 |
| arcene | 100 | 10000 | 16 | 10 | 0.9437 | 0.9072 | 0.9802 |
| arcene | 100 | 10000 | 32 | 10 | 0.9743 | 0.9493 | 0.9992 |
| arcene | 100 | 10000 | 64 | 10 | 0.9823 | 0.9615 | 1.003 |

## RQ3: protocol overhead and grinding

The existing overhead run used six concurrent processes. Its per-job timings include CPU contention. The beacon is a **simulated local trace**, not an external beacon measurement.

| shape | seed_source | time_commit_X | time_record_theta | time_issue_nonce | time_expand_sparse_and_CA | time_project_and_cluster |
| --- | --- | --- | --- | --- | --- | --- |
| gaussian_mixture 1000x1024 | beacon | 576.6 | 0.6849 | 3.62e-05 | 0.5893 | 0.01496 |
| gaussian_mixture 1000x1024 | commit_reveal | 577 | 0.6844 | 6.12e-05 | 0.5887 | 0.01498 |
| gaussian_mixture 1000x1024 | nonce | 576.6 | 0.6869 | 1.535e-05 | 0.5868 | 0.01528 |
| gaussian_mixture 500x256 | beacon | 70.46 | 0.1741 | 3.545e-05 | 0.1428 | 0.002574 |
| gaussian_mixture 500x256 | commit_reveal | 70.49 | 0.1735 | 5.75e-05 | 0.1431 | 0.002468 |
| gaussian_mixture 500x256 | nonce | 70.63 | 0.1711 | 1.305e-05 | 0.1431 | 0.002651 |
| mnist 500x784 | beacon | 275.5 | 0.5249 | 3.04e-05 | 0.4441 | 0.005891 |
| mnist 500x784 | commit_reveal | 275 | 0.5272 | 5.255e-05 | 0.4456 | 0.005898 |
| mnist 500x784 | nonce | 275.2 | 0.5186 | 1.29e-05 | 0.4459 | 0.00592 |

[Grinding statistics](../../rq3/analysis/rq3_grinding_summary.csv) and [chart](../../rq3/analysis/rq3_grinding.png). The plotted union bound is an empirical plug-in quantity, not a proved population guarantee.

Paired pre-published baseline: map generation/authentication occurs before the timed transaction and is recorded separately. This is native timing only; it does not establish ZK proof-cost savings.

| dataset_name | n | d | repetitions | incremental_postcommit_s_mean | incremental_postcommit_s_ci95_low | incremental_postcommit_s_ci95_high | incremental_after_commit_s_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian_mixture | 500 | 256 | 10 | 0.3296 | -0.3134 | 0.9725 | 0.1599 |
| gaussian_mixture | 1000 | 1024 | 10 | -0.5016 | -32.14 | 31.14 | 0.6514 |
| mnist | 500 | 784 | 10 | 0.6475 | -17.88 | 19.18 | 0.5407 |

## RQ4: certificate behavior

| cert_type | m | repetitions | coverage_mean | q | violation_rate_mean | uncertified_stability_mean |
| --- | --- | --- | --- | --- | --- | --- |
| centroid_margin | 16 | 10 | 0.7326 | 1 | 0.1233 | 0.3678 |
| dbscan_band | 16 | 10 | 0.9995 | 1 | 0 | 0.5652 |
| centroid_margin | 16 | 10 | 0.7326 | 2 | 0.1022 | 0.2705 |
| dbscan_band | 16 | 10 | 0.9995 | 2 | 0 | 0.5956 |
| centroid_margin | 16 | 10 | 0.7326 | 4 | 0.1098 | 0.3544 |
| dbscan_band | 16 | 10 | 0.9995 | 4 | 0 | 0.5776 |
| centroid_margin | 32 | 10 | 0.7326 | 1 | 0.03446 | 0.381 |
| dbscan_band | 32 | 10 | 0.9995 | 1 | 0 | 0.6196 |
| centroid_margin | 32 | 10 | 0.7326 | 2 | 0.03323 | 0.4197 |
| dbscan_band | 32 | 10 | 0.9995 | 2 | 0 | 0.6244 |
| centroid_margin | 32 | 10 | 0.7326 | 4 | 0.03922 | 0.4115 |
| dbscan_band | 32 | 10 | 0.9995 | 4 | 0 | 0.6312 |
| centroid_margin | 64 | 10 | 0.7326 | 1 | 0.005939 | 0.4311 |
| dbscan_band | 64 | 10 | 0.9995 | 1 | 0 | 0.6644 |
| centroid_margin | 64 | 10 | 0.7326 | 2 | 0.006104 | 0.4699 |
| dbscan_band | 64 | 10 | 0.9995 | 2 | 0 | 0.6736 |
| centroid_margin | 64 | 10 | 0.7326 | 4 | 0.01119 | 0.4548 |
| dbscan_band | 64 | 10 | 0.9995 | 4 | 0 | 0.6576 |
| centroid_margin | 128 | 10 | 0.7326 | 1 | 0.0004525 | 0.5107 |
| dbscan_band | 128 | 10 | 0.9995 | 1 | 0 | 0.7296 |
| centroid_margin | 128 | 10 | 0.7326 | 2 | 0 | 0.5121 |
| dbscan_band | 128 | 10 | 0.9995 | 2 | 0 | 0.7212 |
| centroid_margin | 128 | 10 | 0.7326 | 4 | 0.0006718 | 0.4692 |
| dbscan_band | 128 | 10 | 0.9995 | 4 | 0 | 0.7136 |

The [phase diagram](../../rq4/analysis/rq4_phase_diagram.png) contains 25 single-seed cells; it has no seed-level uncertainty interval.

The DBSCAN generator concentrates difficulty in paired near-radius points. Overall pair coverage is dominated by far-apart pairs; use the uncertified near-band denominator when judging this stress test.

| m | q | repetitions | uncertified_pairs_mean | uncertified_stability_mean |
| --- | --- | --- | --- | --- |
| 16 | 1 | 10 | 250 | 0.5652 |
| 16 | 2 | 10 | 250 | 0.5956 |
| 16 | 4 | 10 | 250 | 0.5776 |
| 32 | 1 | 10 | 250 | 0.6196 |
| 32 | 2 | 10 | 250 | 0.6244 |
| 32 | 4 | 10 | 250 | 0.6312 |
| 64 | 1 | 10 | 250 | 0.6644 |
| 64 | 2 | 10 | 250 | 0.6736 |
| 64 | 4 | 10 | 250 | 0.6576 |
| 128 | 1 | 10 | 250 | 0.7296 |
| 128 | 2 | 10 | 250 | 0.7212 |
| 128 | 4 | 10 | 250 | 0.7136 |

Real-data centroid certificate supplement:

| dataset_name | m | repetitions | coverage_mean | violation_rate_mean | embedding_event_rate | n_certified_with_pointwise_event | n_certified_pointwise_event_violations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mnist | 32 | 10 | 0.427 | 0.03489 | 0 | 228 | 0 |
| mnist | 64 | 10 | 0.427 | 0.006473 | 0 | 752 | 0 |
| fashion_mnist | 32 | 10 | 0.7691 | 0.02539 | 0 | 227 | 0 |
| fashion_mnist | 64 | 10 | 0.7691 | 0.007021 | 0 | 790 | 0 |
| gisette | 32 | 10 | 0.0072 | 0.04242 | 0 | 16 | 0 |
| gisette | 64 | 10 | 0.0072 | 0.03333 | 0 | 22 | 0 |
| arcene | 32 | 10 | 0.963 | 0.005273 | 0 | 43 | 0 |
| arcene | 64 | 10 | 0.963 | 0.002128 | 0 | 164 | 0 |
| 20ng_tfidf | 32 | 10 | 0.1213 | 0.002608 | 0 | 55 | 0 |
| 20ng_tfidf | 64 | 10 | 0.1213 | 0.0003009 | 0 | 160 | 0 |

A certified assignment is guaranteed stable only when the point-to-centroid embedding event holds. The final two columns count certified points satisfying their pointwise event and violations among those points. Unconditional violations on failed-event rows do not contradict the conditional theorem.

## RQ5: scaling and sensitivity

The reported `t_total_s` is a sum of cached component timings and assignment timing, not an end-to-end wall-clock stopwatch or ZK proving time.

| sweep_factor | sweep_value | repetitions | ari_mean | r_phi_mean | t_total_s_mean |
| --- | --- | --- | --- | --- | --- |
| n | 250 | 10 | 1 | 1 | 0.01378 |
| n | 500 | 10 | 1 | 1 | 0.008607 |
| n | 1000 | 10 | 1 | 1 | 0.01113 |
| n | 2000 | 10 | 1 | 1 | 0.0149 |
| d | 64 | 10 | 1 | 1 | 0.004363 |
| d | 128 | 10 | 1 | 1 | 0.006407 |
| d | 256 | 10 | 1 | 1 | 0.01124 |
| d | 512 | 10 | 1 | 1 | 0.01972 |
| k | 2 | 10 | 1 | 1 | 0.009756 |
| k | 5 | 10 | 1 | 1 | 0.01119 |
| k | 10 | 10 | 0.9984 | 1 | 0.01245 |
| k | 20 | 10 | 0.9988 | 1 | 0.01575 |
| m | 16 | 10 | 1 | 1 | 0.01066 |
| m | 32 | 10 | 1 | 1 | 0.01129 |
| m | 64 | 10 | 1 | 1 | 0.01178 |
| m | 128 | 10 | 1 | 1 | 0.01347 |
| q | 1 | 10 | 1 | 1 | 0.01077 |
| q | 2 | 10 | 1 | 1 | 0.01128 |
| q | 4 | 10 | 1 | 1 | 0.01195 |
| q | 8 | 10 | 1 | 1 | 0.01341 |

[Scaling ARI chart](../../rq5/analysis/rq5_scaling_ari.png) and [reported component-time chart](../../rq5/analysis/rq5_scaling_time.png).

Controlled separation supplement:

| sweep_value | repetitions | ari_mean | ari_ci95_low | ari_ci95_high | certificate_coverage_mean | certificate_violation_rate_mean | embedding_event_rate | n_certified_with_pointwise_event | n_certified_pointwise_event_violations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 10 | 0.07788 | 0.06752 | 0.08824 | 0 | 0 | 0 | 0 | 0 |
| 0.1 | 10 | 0.07352 | 0.06388 | 0.08315 | 0 | 0 | 0 | 0 | 0 |
| 0.25 | 10 | 0.2603 | 0.2313 | 0.2893 | 0 | 0 | 0 | 0 | 0 |
| 0.5 | 10 | 0.7394 | 0.6923 | 0.7865 | 0.7178 | 0.08844 | 0 | 466 | 0 |
| 1 | 10 | 0.993 | 0.9834 | 1.003 | 0.9801 | 0.001624 | 0 | 281 | 0 |
| 2 | 10 | 0.999 | 0.9968 | 1.001 | 0.999 | 0.0007071 | 0 | 204 | 0 |
| 3 | 10 | 1 | 1 | 1 | 1 | 0 | 0 | 126 | 0 |
| 4 | 10 | 1 | 1 | 1 | 1 | 0 | 0 | 123 | 0 |

Precision sweep shown with the 24-bit common reference. Do not combine rows from different normalization references as one precision curve.

| dataset_name | precision_bits | repetitions | ari_mean | ari_ci95_low | ari_ci95_high | mean_abs_quantization_error_mean |
| --- | --- | --- | --- | --- | --- | --- |
| gaussian_mixture | 8 | 10 | 0 | 0 | 0 | 4.193e-05 |
| gaussian_mixture | 12 | 10 | 0.9597 | 0.8934 | 1.026 | 4.11e-05 |
| gaussian_mixture | 16 | 10 | 1 | 1 | 1 | 3.807e-06 |
| gaussian_mixture | 20 | 10 | 1 | 1 | 1 | 2.384e-07 |
| gaussian_mixture | 24 | 10 | 1 | 1 | 1 | 1.489e-08 |
| dbscan_margin | 8 | 10 | 0 | 0 | 0 | 2.474e-05 |
| dbscan_margin | 12 | 10 | 0.05428 | 0.02045 | 0.08812 | 2.443e-05 |
| dbscan_margin | 16 | 10 | 0.1309 | 0.03386 | 0.228 | 3.815e-06 |
| dbscan_margin | 20 | 10 | 0.383 | 0.2829 | 0.4831 | 2.387e-07 |
| dbscan_margin | 24 | 10 | 0.4059 | 0.3131 | 0.4987 | 1.492e-08 |
| mnist | 8 | 10 | 0 | 0 | 0 | 2.869e-05 |
| mnist | 12 | 10 | 0.4288 | 0.3624 | 0.4952 | 9.034e-06 |
| mnist | 16 | 10 | 0.4748 | 0.4079 | 0.5417 | 7.733e-07 |
| mnist | 20 | 10 | 0.5135 | 0.4481 | 0.5789 | 5.596e-08 |
| mnist | 24 | 10 | 0.5431 | 0.5149 | 0.5713 | 2.941e-09 |

At the 24-bit common physical reference, all 30 observed 8-bit rows quantized every coordinate to zero. These are measured precision-collapse cases, not valid evidence of useful 8-bit clustering.

## Proof and remaining evidence

| dataset_name | n | d | m | _status | failure_kind | peak_rss_bytes_psutil_max | _error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian_mixture | 2000 | 256 | 16 | proof_failed | memory_limit | 6796668928 | proof process exceeded RSS budget of 6721929216 bytes |
| arcene | 100 | 10000 | 16 | proof_failed | memory_limit | 6855467008 | proof process exceeded RSS budget of 6841827328 bytes |

These failed proof rows provide no measured constraint count or proving time. The memory watchdog threshold is not the RAM needed for a successful proof.

Backend/hardware proof sensitivity and proof-based ablation remain unmeasured. Symbolic constraint units are never presented here as measured gnark counts.
