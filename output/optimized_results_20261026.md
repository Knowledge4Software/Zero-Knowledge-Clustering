# Optimized frozen-profile evidence, 26 October 2026

The selected scientific profile is unchanged: predictor63, certificate79, precision14, at most16 width-ordered exact rows.
Native quality is separated from actual proof performance. No superiority over unmeasured baselines is inferred.

| Dataset | Basis | Updates certified | Regressions certified | Mean agreement | Worst agreement | Label coverage | Below99 |
|---|---|---:|---:|---:|---:|---:|---:|
| fashion_mnist | ordered_head | 28/28 | 5/5 | 99.948% | 99.805% | 98.982% | 0 |
| fashion_mnist | pca | 28/28 | 5/5 | 99.916% | 99.707% | 98.525% | 0 |
| mnist | ordered_head | 32/32 | 0/0 | 99.942% | 99.707% | 96.613% | 0 |
| mnist | pca | 32/32 | 0/0 | 99.887% | 99.609% | 95.990% | 0 |

New completed method--Full timing pairs: 21. A complete repetition requires all three certificate/moments methods.
Optimizations preserve mathematical outputs. Quality differences between old and new fitted workloads are not attributed to code optimization.
Stress limits: Fashion k10 resolves6/8 updates and2/3 natural regressions with both bases; the other2 need Full. MNIST k20 includes1 centroid request below99% agreement.
At16384 real records (primary MNIST age1), centroid resolves4/4 updates; mean agreement99.786%, worst99.762%, label coverage94.958%. These are native, not proof timing, results.
All source records, inconclusive outcomes, below-target cases and resource exclusions remain retained.
Timing/control progress is in code/validation/outputs/referee_optimized_20261026/queue.json.

On27October the user limited further proof generation to at most1024 records. The4096/16384 proof benchmarks are deferred for laptop practicality, after partial results; their completed native quality results and the original study plan are retained. This is an execution-scope change, not a completed scale proof or an outcome-based deletion.
