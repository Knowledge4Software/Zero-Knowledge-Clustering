# Completed Mobin results - 20 October 2026

All 14 cases finished successfully; the final exact N=1024 case finished at 20:16:39 Tehran time. All 6,848 proofs passed worker public-only replay. Independent local numerical analysis checks 208 captured metadata hashes and 33,280 record/set certificate evaluations.

| Workload | Method | Plain (min) | Certified (min) | Exact Full / certified | Agreement (%) | Certified coverage (%) | Cost width (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CIFAR/ResNet N=64, rank 15 | Learned + energy | 10.66 | 11.05 | 8.07 | 99.48 | 60.76 | 24.00 |
| CIFAR/ResNet N=64, rank 15 | PCA + energy | 10.50 | 10.78 | 8.28 | 98.44 | 39.76 | 33.58 |
| MNIST N=1024, rank 15 | Learned + energy | 79.64 | 81.02 | 5.08 | 98.25 | 61.63 | 20.65 |
| MNIST N=1024, rank 15 | PCA + energy | 80.40 | 81.74 | 5.03 | 97.89 | 37.65 | 37.97 |
| MNIST N=256, rank 15 | Learned + energy | 21.24 | 21.82 | 4.76 | 98.31 | 59.20 | 20.51 |
| MNIST N=256, rank 63 | Learned + energy | 28.85 | 30.58 | 3.40 | 99.74 | 85.20 | 7.26 |
| MNIST N=256, rank 15 | PCA + energy | 21.19 | 21.83 | 4.76 | 97.44 | 35.98 | 38.10 |
| MNIST N=256, rank 63 | PCA + energy | 29.23 | 30.84 | 3.37 | 99.70 | 80.51 | 8.88 |
| MNIST N=64, rank 15 | Learned + energy | 6.03 | 6.89 | 4.15 | 98.26 | 52.26 | 20.80 |
| MNIST N=64, rank 15 | PCA + energy | 6.12 | 7.00 | 4.09 | 96.53 | 30.03 | 38.19 |

All requested catalog rows and ten center sets are actually proved. Costs charge preparation and setup once per chosen alternative; they exclude verification, warmups, fitting, I/O and public-center preparation. Exact Full and approximate reuse provide different fidelity guarantees. These are one-host measured component-cost ratios, not equal-fidelity acceleration of exact computation.

Quality percentages are means across nine updated center sets. No certified assignment is incorrect and every checked interval contains the exact quantized optimum. Zero of 90 updated method/set cases passes the illustrative 99% certified coverage / 1% cost-width policy. The rank-63 pilots also change basis precision to T=16384 and query batch to 16; their differences are combined configuration effects.

The [current manuscript](../manuscript.tex) and [paper delivery](README.md) incorporate these completed results. The [numerical manifest](../paper_results/review_controls/generated_completed_20261020/manifest.json) and [reproduction guide](../code/docs/review_controls_20261020.md) retain the precise evidence and timing boundaries. Author review and declarations remain pending.
