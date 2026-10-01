# Paper-readiness review (2026-10-01)

The completed native experiments are usable as measured, explicitly labeled
native results. The current artifact is **not ready for a paper whose main claim
is lower ZK proving cost at useful fidelity**: both selected full-shape RQ2
proofs stopped at the memory watchdog, so no measured constraints, proving
time, verification rate, or proof-based ablation is available.

| Evidence | Measured result | Supported interpretation |
| --- | --- | --- |
| Completion | All 360 supplemental rows succeeded: RQ3 30, RQ4 100, separation 80, precision 150. The earlier native RQ1/RQ3/RQ4/RQ5 campaigns also have zero failed rows. | The configured native campaigns completed. This is not a correctness proof of every research claim. |
| Dataset scope | RQ1 includes all seven configured dataset families but uses subsets: 5,000 MNIST/Fashion-MNIST rows, 3,000 Gisette and 20 Newsgroups rows, and 100 Arcene rows. RQ3-RQ5 use smaller targeted samples. | Report the actual analyzed `n` and preprocessing for every table; source-dataset sizes in the manuscript are not the experimental sample sizes. |
| RQ1 fidelity | Sparse JL meets the predeclared ARI/cost-ratio **CI** thresholds in 9 of 16 dataset-shape/cluster-count groups tested at reduced dimension. At `m=256`, `q=2`, `k=5`, mean ARI is 0.552 on 20 Newsgroups and 0.693 on Gisette; neither meets the 0.80 ARI threshold. | Fidelity is data-dependent. Do not claim a universally fidelity-eligible reduction. |
| Same-dimension native baselines | At `m=64`, `k=5`, mean MNIST ARI is about 0.651 for sparse JL versus 0.999 for PCA; Gisette is about 0.436 versus 0.985. | PCA is stronger for native assignment fidelity in these cases. PCA computation/proof scope differs, so this is not a ZK-cost comparison. |
| RQ3 pre-published comparison | Mean incremental **after-commit** host latency is 0.160 s (500x256 Gaussian), 0.651 s (1000x1024 Gaussian), and 0.541 s (500x784 MNIST), each with a positive seed-level 95% CI. These are approximately 0.22%, 0.11%, and 0.16% of the corresponding mean post-commit transaction times. The CI for the full transaction-time difference crosses zero in all three shapes. | The seed/expansion stage has a measurable local cost. The complete native transaction has no statistically resolved timing difference in these ten-seed samples. Pre-published setup is recorded separately and excluded from its transaction. No external beacon or proof overhead was measured. |
| RQ4 centroid certificate | All 100 real-data rows failed the **global** point-to-centroid embedding event at epsilon 0.15. Across the 100 rows, 2,457 certified points satisfied their own pointwise event, and none changed assignment. There were 466 unconditional certified assignment changes on rows where the global event failed. | The conditional theorem is consistent with the checked pointwise subset; the global guarantee was never activated in this real-data grid. Do not report unconditional zero violations. |
| RQ4 DBSCAN certificate | Each `n=1000` synthetic row has 499,500 pairs, of which only 250 are in the uncertified near-radius band. Thus the approximately 99.95% overall certified-pair coverage is dominated by far-apart pairs. Near-band stability is about 0.57-0.73 over the tested dimensions and sparsities. | Report the near-band denominator and behavior. Overall coverage alone makes this stress test look easier than it is. |
| RQ5 controlled separation | Mean assignment ARI rises from 0.078 at gamma 0, to 0.739 at gamma 0.5, to 0.993 at gamma 1. The original scaling grid at gamma 3 has ARI near 1 for almost every factor level. The global epsilon-0.15 embedding event holds in none of the 80 separation rows. | Separation matters strongly. The easy original scaling grid cannot by itself establish robust fidelity across difficult data; the gamma results are empirical fidelity measurements, not instances of the global theorem premise. |
| RQ5 precision | Under one common 24-bit physical reference, every 8-bit row collapses to an all-zero quantized matrix. Mean ARI at 16 versus 24 bits is 0.131 versus 0.406 on DBSCAN margin and 0.475 versus 0.543 on MNIST. Paired 24-minus-16 mean ARI differences are 0.275 [0.145, 0.405] and 0.068 [0.004, 0.132], respectively. | Sixteen bits is not established as optimal. The old 20-bit-reference and new 24-bit-reference sweeps must not be merged into one curve. Symbolic constraint units do not establish a precision/gnark-cost tradeoff. |
| RQ5 timing | `t_total_s` is a sum of cached KMeans/projection component timings and assignment timing. It excludes data commitment and ZK proving. | Use it only as a native component-time diagnostic, not a measured end-to-end scaling or prover-cost curve. |

Statistical and reporting work remains before submission. The current analysis
uses ten seed rows per configuration and Student-t intervals; bounded metrics
can receive intervals above 1. The manuscript describes 30 independent
projection seeds, ten paired clustering starts, hierarchical bootstrap, and
Holm-adjusted comparisons. Those are not the design or inference procedure of
these files. Either collect the additional observations and implement that
analysis or revise the paper's methods to match the completed experiment.

The manuscript also includes backend/hardware sensitivity and a proof-based
ablation, neither of which is measured. No estimate should be inserted into
their measured proof columns. The primary next engineering decision is whether
to redesign the full-data commitment circuit or obtain a measured memory
scaling curve before selecting hardware for full-shape RQ2.

See [full analysis](analysis.md), [run inventory](run_inventory.csv),
[RQ1 CI-eligible dimensions](../../rq1/analysis/rq1_smallest_ci_eligible_m.csv),
[RQ3 paired statistics](../../rq3/analysis/rq3_paired_summary.csv),
[RQ4 real-data statistics](../../rq4/analysis/rq4_real_summary.csv), and
[RQ5 separation](../../rq5/analysis/rq5_gamma_summary.csv) and
[precision](../../rq5/analysis/precision_summary_analysis.csv) summaries.
