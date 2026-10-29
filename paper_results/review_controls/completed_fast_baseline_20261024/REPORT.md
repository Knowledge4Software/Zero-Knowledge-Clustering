# Completed local proof comparison ? 24 October 2026

MNIST, one fitted model (k=5, seed11),128 records and all20 distinct training-only Lloyd center sets. Head batch16 and exact Full batch8, each under the same1.6-million-constraint guard. One repetition with two Go scheduler threads. Full accepts public centers and reuses its setup across all sets.

| Method | Complete case minutes | Full/method | Mean agreement | Worst agreement | Mean certified assignment coverage | Max absolute cost error |
|---|---:|---:|---:|---:|---:|---:|
| full | 196.80 | 1.000? | 100.0000% | 100.0000% | 100.0000% | 0.0000% |
| head_r39 | 34.14 | 5.765? | 99.5703% | 98.4375% | 90.0000% | 0.0687% |
| head_r63 | 43.46 | 4.528? | 99.8047% | 99.2188% | 95.1562% | 0.0276% |

All896 included proofs passed fresh public verification:288 for each head rank and320 for Full. The separate Python numerical audit matched every claimed assignment count, aggregate approximate cost, absolute interval and exact Full cost against the frozen inputs. All reported absolute intervals contain the exact quantized cost.

The clocks include native preparation/answer construction, compilation/setup, proof generation and worker checks, public export and fresh public replay; head fitting is charged. The enclosing case spans add the few seconds of export/orchestration omitted by the original phase sums. Full kept running while its failed monitor was replaced; its proof process was never restarted, and its entire elapsed time is retained.

Rank63 meets the user-selected99% agreement floor on all20 sets with a4.528? complete-workflow ratio in this pilot. Rank39 reaches5.765? but fails that floor on some sets. Neither a5? speedup with the required accuracy nor consistent independent-fit performance is established. Timing was not randomized or repeated, and laptop workload/clock variation remains a limitation.

The assignment certificate remains partial and does not establish conclusive exact cost-update decisions. This comparison does not measure the known-family authenticated affine baseline or show superiority over an equally reusable PCA configuration.

The cached-parameter rank63 run is still running separately. Its final timing and output audit must be completed before updating this comparison.
