# Fresh-seed rank79 numerical replication

Four dataset/seed/k split configurations; eight deployed models and sixteen head/PCA representation fits. All predeclared natural updates remain, including unchanged/duplicate proposals. Ages, representations, ranks, updates and N128/N1024 catalogs share splits. These are fresh fits on the same two public source datasets, not new datasets or independent draws per update. Numerical results below are separate from the fresh proof timings still recorded by the laptop queue.

The primary table counts distinct nonidentity candidate center sets once per deployed fit, rather than repeated Lloyd requests. In particular, the three N1024 worsening submissions at Fashion-MNIST/k3/seed101/age8 iterations12/16/24 are one identical candidate. That is ONE distinct regression, not three independent regression cases. Raw proposal-level exports remain unchanged.

| N | Representation | Certificate rank | Conclusive distinct updates | Distinct regressions detected | Mean prediction agreement (%) | Worst agreement (%) | Mean assignment-certified fraction (%) | Mean change width / old cost (%) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1024 | ordered_head | 63 | 25/25 | 1/1 | 99.898438 | 99.511719 | 97.5781 | 0.3507 |
| 1024 | ordered_head | 79 | 25/25 | 1/1 | 99.898438 | 99.511719 | 97.5781 | 0.2560 |
| 1024 | pca | 63 | 24/25 | 0/1 | 99.906250 | 99.707031 | 97.3594 | 0.3243 |
| 1024 | pca | 79 | 24/25 | 0/1 | 99.906250 | 99.707031 | 97.3594 | 0.2314 |
| 128 | ordered_head | 63 | 23/25 | 6/7 | 99.937500 | 99.218750 | 97.4375 | 0.3605 |
| 128 | ordered_head | 79 | 23/25 | 6/7 | 99.937500 | 99.218750 | 97.4375 | 0.2645 |
| 128 | pca | 63 | 22/25 | 5/7 | 99.906250 | 99.218750 | 97.0625 | 0.3322 |
| 128 | pca | 79 | 23/25 | 6/7 | 99.906250 | 99.218750 | 97.0625 | 0.2402 |

Prediction rank63 stays frozen in both certificate profiles. Thus prediction agreement and assignment-certified fraction are identical between63 and79 by construction; stronger cost-change bounds do not manufacture better assignment predictions or certify every label. Agreement below the user-selected99% target is retained and explicitly counted in summary.json. No universal acceptance threshold is assumed.

The following older paired count includes all32 proposal steps per representation, including identities and repeats; it is a containment check, not32 distinct updates.

| N | Representation | Rank79 intervals inside63 | Decisions gained | Decisions lost |
|---:|---|---:|---:|---:|
| 1024 | ordered_head | 32/32 | 0 | 0 |
| 1024 | pca | 32/32 | 0 | 0 |
| 128 | ordered_head | 32/32 | 0 | 0 |
| 128 | pca | 32/32 | 1 | 0 |

All256 recorded intervals contain independently recomputed quantized Full truth; all16 fit-archive hashes match. No false conclusive outcome or false certified assignment was found. This is not a complete security audit. Cases that remain inconclusive still need verified fallback. Native fitting durations can reflect overlapping one-thread development screens and varying laptop load.

Two N128 proof cases were fixed before seeing these results: Fashion-MNIST/k3/seed101/age8 and MNIST/k5/seed113/age1. Fresh Full/head79/head63/PCA79 proof generation and public replay, including every required newly generated fallback, are managed by code/validation/outputs/rank79_independent_replication_20261025/queue.json. No new speed ratio is supported until those stages complete. Artifacts stay private for supervisor review.
