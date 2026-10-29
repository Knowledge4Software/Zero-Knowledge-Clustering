# Preparation versus request work — 28 October 2026

Identified reusable stages versus residual response-phase work. Key serialization/loading, file I/O and public replay not independently isolated remain in the response residual. This is component accounting, not independently timed single-request latency.

Each request evaluates new public centers against one fixed catalog and old reference. Changing the catalog/reference requires charging new authentication in the current implementation.

All times below are seconds. Identified reusable work includes native commitment preparation, base keys, authentication of old rows, lazy correction-key setup and any measured exact-fallback key/old-reference proof. The response column retains all residual query, correction, fallback and verification work.

| Execution | Data | Requests | Method | Native data/summary preparation | Summary-authentication proofs | Base keys | Other reusable stages | Identified reusable total | Mean residual response | Enclosing accounting gap | Measured enclosing total |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| previous8 | fashion_n32 | 8 | Full-Groth16 | 1.10 | 0.00 | 184.06 | 67.78 | 252.94 | 65.33 | 9.00 | 784.57 |
| previous8 | fashion_n32 | 8 | ZKML-adapter | 0.60 | 0.00 | 32.44 | 46.50 | 79.54 | 45.65 | 0.00 | 444.78 |
| previous8 | fashion_n32 | 8 | Centroid | 2.09 | 78.65 | 201.61 | 46.46 | 328.81 | 30.87 | 15.10 | 590.85 |
| previous8 | fashion_n32 | 8 | PCA-certificate | 2.08 | 79.80 | 202.58 | 46.55 | 331.01 | 31.51 | 15.32 | 598.41 |
| previous8 | text_n32 | 8 | Full-Groth16 | 1.13 | 0.00 | 84.47 | 27.87 | 113.47 | 29.45 | 8.45 | 357.50 |
| previous8 | text_n32 | 8 | ZKML-adapter | 0.29 | 0.00 | 14.80 | 20.57 | 35.65 | 23.63 | 0.00 | 224.73 |
| previous8 | text_n32 | 8 | Centroid | 1.05 | 45.36 | 129.92 | 143.11 | 319.45 | 76.87 | 30.67 | 965.05 |
| previous8 | text_n32 | 8 | PCA-certificate | 1.06 | 44.35 | 130.04 | 144.47 | 319.92 | 77.56 | 30.41 | 970.80 |
| new16 | text_n32 | 16 | Full-Groth16 | 1.12 | 0.00 | 89.00 | 30.53 | 120.65 | 30.59 | 14.32 | 624.42 |
| new16 | text_n32 | 16 | ZKML-adapter | 0.32 | 0.00 | 17.03 | 26.16 | 43.52 | 25.21 | 0.00 | 446.94 |
| new16 | text_n32 | 16 | Centroid | 1.06 | 46.36 | 134.32 | 147.88 | 329.63 | 79.52 | 49.56 | 1651.45 |
| new16 | text_n32 | 16 | PCA-certificate | 1.06 | 46.41 | 133.41 | 147.57 | 328.45 | 79.41 | 49.69 | 1648.70 |

## Conditional crossover model

T(Q)=P+Q*c using the observed mixture and batching policy. Crossover counts are conditional estimates, not measured guarantees. Fallback and correction costs are retained.

| Data / calibration | Baseline | Estimated first better count | Interpretation |
|---|---|---:|---|
| text_n32 / Q16 | Full-Groth16 | No eventual crossing in model | No eventual crossover in this constant-mean model. More requests alone do not change the slope; future workload mixes can differ. |
| text_n32 / Q16 | ZKML-adapter | No eventual crossing in model | No eventual crossover in this constant-mean model. More requests alone do not change the slope; future workload mixes can differ. |
| text_n32 / Q16 | PCA-certificate | Not reliable: below5% single-run margin | No eventual crossover in this constant-mean model. More requests alone do not change the slope; future workload mixes can differ. |
| fashion_n32 / Q8 | Full-Groth16 | Already favorable at measured Q=8; lower-count crossing unmeasured | Estimated eventual crossover under unchanged average workload; not a measured threshold. |
| fashion_n32 / Q8 | ZKML-adapter | 17 | Estimated eventual crossover under unchanged average workload; not a measured threshold. |
| fashion_n32 / Q8 | PCA-certificate | Not reliable: below5% single-run margin | Estimated eventual crossover under unchanged average workload; not a measured threshold. |
| text_n32 / Q8 | Full-Groth16 | No eventual crossing in model | No eventual crossover in this constant-mean model. More requests alone do not change the slope; future workload mixes can differ. |
| text_n32 / Q8 | ZKML-adapter | No eventual crossing in model | No eventual crossover in this constant-mean model. More requests alone do not change the slope; future workload mixes can differ. |
| text_n32 / Q8 | PCA-certificate | Not reliable: below5% single-run margin | Estimated eventual crossover under unchanged average workload; not a measured threshold. |

These count estimates must not be presented as measured crossover thresholds. The first8 and first16 prefixes can have different proposal mixtures. Full and ZKML also reuse their setup in the model.

Differences below5% in a single execution, particularly the PCA comparison, do not establish a reliable speed advantage or crossover count.

The independently measured Fashion64 warm comparison remains 23.00min for the certificate versus 43.88min for the ZKML adapter. There is no measured Full-Groth16 Q64 run.

Adding the separately measured reusable setup reconstructs about 28.48min versus 45.21min at Fashion Q64. This is component accounting, not a fresh cold64 execution. It also shows why the Q8 constant-mean model must not be treated as an exact threshold prediction.

## Actual larger text run

{"complete": true, "status": "complete", "finished_unix": 1793222191.3638096}

The fresh16-request study supplies the previously fitted representations to all systems. Summary construction/authentication and proof keys are newly measured; basis-training time is not silently labeled compression.

The paper is unchanged. No cloud resources were used.
