# Four-hour local RQ study — 28 October 2026

Exploratory evidence about specified implementations and workloads. Different outputs/backend choices, setup reuse, synthetic cases and numerical-only studies remain explicit. No proof of universal or statistical superiority.

| Experiment | Case | Method | Minutes | Proof KiB | Peak RAM GiB | Assignment agreement % | Before fallback |
|---|---|---|---:|---:|---:|---:|---:|
| basic_cold | fashion_n32 | Exact Full / Groth16 | 13.093 | 6.891 | 1.775 | 100.000 | 8/8 |
| basic_cold | fashion_n32 | ZKML primitive adapter / Halo2-KZG | 7.430 | 52.875 | 0.749 | 100.000 | 8/8 |
| basic_cold | fashion_n32 | Centroid certificate | 9.867 | 11.484 | 1.252 | 100.000 | 8/8 |
| basic_cold | fashion_n32 | PCA certificate | 9.992 | 11.676 | 1.253 | 100.000 | 8/8 |
| basic_cold | text_n32 | Exact Full / Groth16 | 5.964 | 6.891 | 1.191 | 100.000 | 8/8 |
| basic_cold | text_n32 | ZKML primitive adapter / Halo2-KZG | 3.751 | 52.875 | 0.370 | 100.000 | 8/8 |
| basic_cold | text_n32 | Centroid certificate | 16.090 | 34.453 | 1.236 | 95.703 | 2/8 |
| basic_cold | text_n32 | PCA certificate | 16.186 | 34.453 | 1.244 | 95.703 | 2/8 |
| scaled_cold | fashion_n128 | Exact Full / Groth16 | 22.838 | 15.312 | 1.756 | 100.000 | 4/4 |
| scaled_cold | fashion_n128 | ZKML primitive adapter / Halo2-KZG | 15.525 | 29.375 | 2.393 | 100.000 | 4/4 |
| scaled_cold | fashion_n128 | Centroid certificate | 12.467 | 31.391 | 1.269 | 99.805 | 4/4 |

## Warm64

Actual warm certificate: 23.005min; exact Halo2: 43.885min; exact/certificate: 1.908x.
All64 request proofs were freshly generated for both systems; data authentication and setup reused for both. Certificate fallback updates: 0.

The host Windows OS issued a low-memory notification during the N128 PCA run. The guard stopped the run; PCA N128 is incomplete and is not counted as a win or a complete timing. The completed N128 centroid, Full and Halo2 runs remain independently checked.

## Per-baseline findings

### Exact Full / Groth16

Observed advantages:
- basic_cold:fashion_n32: observed total time (1.327x).
- basic_cold:fashion_n32: observed process-tree peak RAM (1.418x).
- scaled_cold:fashion_n128: observed total time (1.832x).
- scaled_cold:fashion_n128: observed process-tree peak RAM (1.384x).

Observed losses:
- basic_cold:fashion_n32: exported proof bytes (0.600x).
- basic_cold:text_n32: observed total time (0.371x).
- basic_cold:text_n32: exported proof bytes (0.200x).
- basic_cold:text_n32: assignment agreement pp (-4.297pp).
- scaled_cold:fashion_n128: exported proof bytes (0.488x).
- scaled_cold:fashion_n128: assignment agreement pp (-0.195pp).

### PCA certificate

Observed advantages:
- basic_cold:fashion_n32: exported proof bytes (1.017x).
- native_ablation:fashion_n32_r4_cap4: native-only conclusive coverage pp (+50.000pp).
- native_ablation:fashion_n128_r4_cap4: native-only conclusive coverage pp (+25.000pp).
- native_ablation:fashion_n128_r4_cap16: native-only conclusive coverage pp (+62.500pp).

Observed losses:
- None above the reporting thresholds.

### ZKML primitive adapter / Halo2-KZG

Observed advantages:
- basic_cold:fashion_n32: exported proof bytes (4.604x).
- basic_cold:text_n32: exported proof bytes (1.535x).
- scaled_cold:fashion_n128: observed total time (1.245x).
- scaled_cold:fashion_n128: observed process-tree peak RAM (1.886x).
- warm64:fashion_n32: measured warm time (1.908x).

Observed losses:
- basic_cold:fashion_n32: observed total time (0.753x).
- basic_cold:text_n32: observed total time (0.233x).
- basic_cold:text_n32: assignment agreement pp (-4.297pp).
- scaled_cold:fashion_n128: exported proof bytes (0.936x).
- scaled_cold:fashion_n128: assignment agreement pp (-0.195pp).

Proof bytes exclude keys, parameters and public task/witness files. A smaller Groth16 proof footprint versus Halo2 is a backend-level system tradeoff, not isolated evidence for the mathematical method. Timings are single executions.

## Real-data native screen (not proof timings)

| Dataset | Basis | N | Requests | Conclusive after bounded correction | Assignment agreement % |
|---|---|---:|---:|---:|---:|
| fashion | ordered_head | 32 | 8 | 8/8 | 100.000 |
| fashion | ordered_head | 32 | 64 | 64/64 | 99.805 |
| fashion | ordered_head | 128 | 8 | 8/8 | 99.902 |
| fashion | pca | 32 | 8 | 8/8 | 100.000 |
| fashion | pca | 32 | 64 | 64/64 | 99.707 |
| fashion | pca | 128 | 8 | 8/8 | 99.707 |
| text | ordered_head | 32 | 8 | 2/8 | 95.703 |
| text | ordered_head | 32 | 64 | 20/64 | 94.092 |
| text | ordered_head | 128 | 8 | 0/8 | 93.848 |
| text | pca | 32 | 8 | 2/8 | 95.703 |
| text | pca | 32 | 64 | 19/64 | 94.482 |
| text | pca | 128 | 8 | 0/8 | 95.020 |

## Certificate-rank ablation (native only)

| Dataset | Basis | N | Certificate rank | Correction cap | Conclusive | Exact rows |
|---|---|---:|---:|---:|---:|---:|
| fashion | ordered_head | 32 | 4 | 0 | 1/8 | 0 |
| fashion | ordered_head | 32 | 4 | 4 | 5/8 | 25 |
| fashion | ordered_head | 32 | 4 | 16 | 8/8 | 28 |
| fashion | ordered_head | 32 | 39 | 0 | 4/8 | 0 |
| fashion | ordered_head | 32 | 39 | 4 | 8/8 | 7 |
| fashion | ordered_head | 32 | 39 | 16 | 8/8 | 7 |
| fashion | ordered_head | 32 | 63 | 0 | 6/8 | 0 |
| fashion | ordered_head | 32 | 63 | 4 | 8/8 | 5 |
| fashion | ordered_head | 32 | 63 | 16 | 8/8 | 5 |
| fashion | ordered_head | 32 | 79 | 0 | 6/8 | 0 |
| fashion | ordered_head | 32 | 79 | 4 | 8/8 | 4 |
| fashion | ordered_head | 32 | 79 | 16 | 8/8 | 4 |
| fashion | ordered_head | 128 | 4 | 0 | 1/8 | 0 |
| fashion | ordered_head | 128 | 4 | 4 | 3/8 | 25 |
| fashion | ordered_head | 128 | 4 | 16 | 6/8 | 72 |
| fashion | ordered_head | 128 | 39 | 0 | 7/8 | 0 |
| fashion | ordered_head | 128 | 39 | 4 | 8/8 | 1 |
| fashion | ordered_head | 128 | 39 | 16 | 8/8 | 1 |
| fashion | ordered_head | 128 | 63 | 0 | 8/8 | 0 |
| fashion | ordered_head | 128 | 63 | 4 | 8/8 | 0 |
| fashion | ordered_head | 128 | 63 | 16 | 8/8 | 0 |
| fashion | ordered_head | 128 | 79 | 0 | 8/8 | 0 |
| fashion | ordered_head | 128 | 79 | 4 | 8/8 | 0 |
| fashion | ordered_head | 128 | 79 | 16 | 8/8 | 0 |
| fashion | pca | 32 | 4 | 0 | 1/8 | 0 |
| fashion | pca | 32 | 4 | 4 | 1/8 | 28 |
| fashion | pca | 32 | 4 | 16 | 8/8 | 80 |
| fashion | pca | 32 | 39 | 0 | 4/8 | 0 |
| fashion | pca | 32 | 39 | 4 | 8/8 | 9 |
| fashion | pca | 32 | 39 | 16 | 8/8 | 9 |
| fashion | pca | 32 | 63 | 0 | 6/8 | 0 |
| fashion | pca | 32 | 63 | 4 | 8/8 | 5 |
| fashion | pca | 32 | 63 | 16 | 8/8 | 5 |
| fashion | pca | 32 | 79 | 0 | 6/8 | 0 |
| fashion | pca | 32 | 79 | 4 | 8/8 | 5 |
| fashion | pca | 32 | 79 | 16 | 8/8 | 5 |
| fashion | pca | 128 | 4 | 0 | 1/8 | 0 |
| fashion | pca | 128 | 4 | 4 | 1/8 | 28 |
| fashion | pca | 128 | 4 | 16 | 1/8 | 112 |
| fashion | pca | 128 | 39 | 0 | 7/8 | 0 |
| fashion | pca | 128 | 39 | 4 | 8/8 | 1 |
| fashion | pca | 128 | 39 | 16 | 8/8 | 1 |
| fashion | pca | 128 | 63 | 0 | 8/8 | 0 |
| fashion | pca | 128 | 63 | 4 | 8/8 | 0 |
| fashion | pca | 128 | 63 | 16 | 8/8 | 0 |
| fashion | pca | 128 | 79 | 0 | 8/8 | 0 |
| fashion | pca | 128 | 79 | 4 | 8/8 | 0 |
| fashion | pca | 128 | 79 | 16 | 8/8 | 0 |
| text | ordered_head | 32 | 4 | 0 | 2/8 | 0 |
| text | ordered_head | 32 | 4 | 4 | 2/8 | 24 |
| text | ordered_head | 32 | 4 | 16 | 2/8 | 96 |
| text | ordered_head | 32 | 39 | 0 | 2/8 | 0 |
| text | ordered_head | 32 | 39 | 4 | 2/8 | 24 |
| text | ordered_head | 32 | 39 | 16 | 2/8 | 96 |
| text | ordered_head | 32 | 63 | 0 | 2/8 | 0 |
| text | ordered_head | 32 | 63 | 4 | 2/8 | 24 |
| text | ordered_head | 32 | 63 | 16 | 2/8 | 96 |
| text | ordered_head | 32 | 79 | 0 | 2/8 | 0 |
| text | ordered_head | 32 | 79 | 4 | 2/8 | 24 |
| text | ordered_head | 32 | 79 | 16 | 2/8 | 96 |
| text | ordered_head | 128 | 4 | 0 | 0/8 | 0 |
| text | ordered_head | 128 | 4 | 4 | 0/8 | 32 |
| text | ordered_head | 128 | 4 | 16 | 0/8 | 128 |
| text | ordered_head | 128 | 39 | 0 | 0/8 | 0 |
| text | ordered_head | 128 | 39 | 4 | 0/8 | 32 |
| text | ordered_head | 128 | 39 | 16 | 0/8 | 128 |
| text | ordered_head | 128 | 63 | 0 | 0/8 | 0 |
| text | ordered_head | 128 | 63 | 4 | 0/8 | 32 |
| text | ordered_head | 128 | 63 | 16 | 0/8 | 128 |
| text | ordered_head | 128 | 79 | 0 | 0/8 | 0 |
| text | ordered_head | 128 | 79 | 4 | 0/8 | 32 |
| text | ordered_head | 128 | 79 | 16 | 0/8 | 128 |
| text | pca | 32 | 4 | 0 | 2/8 | 0 |
| text | pca | 32 | 4 | 4 | 2/8 | 24 |
| text | pca | 32 | 4 | 16 | 2/8 | 96 |
| text | pca | 32 | 39 | 0 | 2/8 | 0 |
| text | pca | 32 | 39 | 4 | 2/8 | 24 |
| text | pca | 32 | 39 | 16 | 2/8 | 96 |
| text | pca | 32 | 63 | 0 | 2/8 | 0 |
| text | pca | 32 | 63 | 4 | 2/8 | 24 |
| text | pca | 32 | 63 | 16 | 2/8 | 96 |
| text | pca | 32 | 79 | 0 | 2/8 | 0 |
| text | pca | 32 | 79 | 4 | 2/8 | 24 |
| text | pca | 32 | 79 | 16 | 2/8 | 96 |
| text | pca | 128 | 4 | 0 | 0/8 | 0 |
| text | pca | 128 | 4 | 4 | 0/8 | 32 |
| text | pca | 128 | 4 | 16 | 0/8 | 128 |
| text | pca | 128 | 39 | 0 | 0/8 | 0 |
| text | pca | 128 | 39 | 4 | 0/8 | 32 |
| text | pca | 128 | 39 | 16 | 0/8 | 128 |
| text | pca | 128 | 63 | 0 | 0/8 | 0 |
| text | pca | 128 | 63 | 4 | 0/8 | 32 |
| text | pca | 128 | 63 | 16 | 0/8 | 128 |
| text | pca | 128 | 79 | 0 | 0/8 | 0 |
| text | pca | 128 | 79 | 4 | 0/8 | 32 |
| text | pca | 128 | 79 | 16 | 0/8 | 128 |

The prediction model remains rank63. These are checked numerical bounds, not proof timing results.

At certificate rank4 and cap16, Fashion N128 has6/8 conclusive decisions for the centroid basis versus1/8 for PCA. This is a numerical representation/coverage result, not a measured low-rank ZK runtime. The default high-rank profile remains close to PCA.

## Boundaries

- Native/synthetic controls are separated from real proof measurements.
- The Fashion N128 choice followed an exploratory native screen; it is not independent confirmation.
- ezDPS and Zhang2021 were not implemented as matched systems and have no claimed runtime.
- The paper is unchanged. No cloud server was purchased or accessed.
- All unfinished or resource-stopped cells remain in the queue/receipts and are not counted as wins.
