# Clean Benchmark Comparison

| Measure | Previous broken baseline | New clean measurement |
|---|---:|---:|
| Audit PASS / FAIL / PARTIAL / UNVERIFIED | 10 / 1 / 1 / 1 | 17 / 0 / 0 / 0 |
| RQ1 successful rows | 769 (included stale hash) | 768 |
| RQ1 median ARI | 0.8968 | 0.8968 |
| RQ1 median distortion q95 | 0.5603 | 0.5603 |
| RQ2 real proof rows | 1 (unreduced d=8,m=8) | 24 (16 reduced) |
| RQ2 median proving time | 2.2345 s | 3.1861 s |
| RQ2 median constraints | 174325 | 628450 |
| RQ2 median peak RSS | 0.353 GiB | 1.788 GiB |
| Ablation real / symbolic / native-only | 2 / 12 / 0 | 12 / 0 / 2 |
| RQ4 successful / failed rows | 387 / 25 | 27 / 0 |
| RQ5 measured rows with ARI/distortion | 0 | 768 |

RQ2 clean shapes: 12x32->16, 12x32->32, 12x32->8, 8x32->16, 8x32->32, 8x32->8.

The clean result files contain no symbolic-only rows. `incircuit-gen` remains explicitly labeled `native_only` because that circuit is not implemented.
