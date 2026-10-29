# Basic laptop gate before server rental — 28 October 2026

New fit seed521; same32 original integer rows and first8 proposed updates per dataset; actual proofs, fresh setup, standalone replay, bounded refinement and new Full fallback. Frozen published adapter binaries reused, historical workload times not reused. One exploratory timing execution.

| Dataset | Method | Cold minutes incl. fitting | Assignment agreement % | Conclusive before Full | Fallback | Proof KiB | Peak RSS GiB |
|---|---|---:|---:|---:|---:|---:|---:|
| fashion_n32 | Exact Full / Groth16 | 13.093 | 100.000 | 8/8 | 0 | 6.891 | 1.775 |
| fashion_n32 | ZKML primitive adapter / Halo2-KZG | 7.430 | 100.000 | 8/8 | 0 | 52.875 | 0.749 |
| fashion_n32 | Centroid certificate | 9.867 | 100.000 | 8/8 | 0 | 11.484 | 1.252 |
| fashion_n32 | PCA certificate | 9.992 | 100.000 | 8/8 | 0 | 11.676 | 1.253 |
| text_n32 | Exact Full / Groth16 | 5.964 | 100.000 | 8/8 | 0 | 6.891 | 1.191 |
| text_n32 | ZKML primitive adapter / Halo2-KZG | 3.751 | 100.000 | 8/8 | 0 | 52.875 | 0.370 |
| text_n32 | Centroid certificate | 16.090 | 95.703 | 2/8 | 6 | 34.453 | 1.236 |
| text_n32 | PCA certificate | 16.186 | 95.703 | 2/8 | 6 | 34.453 | 1.244 |

## Reuse diagnostic

| Dataset | Certificate basis | Observed online component seconds | Exact online component seconds | Exact / certificate |
|---|---|---:|---:|---:|
| fashion_n32 | ordered_head | 293.288 | 365.238 | 1.245 |
| fashion_n32 | pca | 298.520 | 365.238 | 1.223 |
| text_n32 | ordered_head | 757.983 | 189.073 | 0.249 |
| text_n32 | pca | 764.887 | 189.073 | 0.247 |

These are phase-component diagnostics, not independent warm measurements or predictions. Certificate online costs include all base verification, lazy correction setup and paid fallback. Exact online costs include shared loading and new-bank public replay.

## Native screen (not proof timings)

| Dataset | Basis | N | Requests | Conclusive after <=16 corrections | Correction rows | Agreement % |
|---|---|---:|---:|---:|---:|---:|
| fashion | ordered_head | 32 | 8 | 8 | 4 | 100.000 |
| fashion | ordered_head | 32 | 64 | 64 | 18 | 99.805 |
| fashion | ordered_head | 128 | 8 | 8 | 0 | 99.902 |
| fashion | pca | 32 | 8 | 8 | 5 | 100.000 |
| fashion | pca | 32 | 64 | 64 | 22 | 99.707 |
| fashion | pca | 128 | 8 | 8 | 0 | 99.707 |
| text | ordered_head | 32 | 8 | 2 | 96 | 95.703 |
| text | ordered_head | 32 | 64 | 20 | 789 | 94.092 |
| text | ordered_head | 128 | 8 | 0 | 128 | 93.848 |
| text | pca | 32 | 8 | 2 | 96 | 95.703 |
| text | pca | 32 | 64 | 19 | 808 | 94.482 |
| text | pca | 128 | 8 | 0 | 128 | 95.020 |

## Boundaries

- New seed521, existing public data sources; exploratory, one timing execution, N32, eight chronological requests per dataset.
- All first8 requests retained; only identical public centers cached equally. The64-request native screen did not select proof candidates.
- Agreement measures exact quantized assignments, not classification accuracy. All interval endpoints independently checked.
- ZKML primitive adapter, not original compiler; different commitment and proof backend. Exact-cost output is stronger than sign-only output.
- No server purchase or remote access, and no manuscript edits.
