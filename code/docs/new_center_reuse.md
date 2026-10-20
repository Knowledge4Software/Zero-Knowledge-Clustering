# New-center reuse study

**19 October proof follow-up:** This guide describes the original native study. A separate [reuse workload](../experiments/reuse_workload.py) now implements actual same-k new-center proofs; both N=64 cohorts and the N=256 MNIST scale check are complete; the separate five-start native fidelity run is also complete. The [current manuscript revision guide](../../paper_results/reuse_revision/README.md) covers the new circuit, cost accounting and verification limits. Historical statements below about absent new-center proofs refer to the original native experiment.

The completed native study is `experiments/new_center_reuse.py`; its server output is `/root/commitjl/results/new_center_reuse_20261018` on `cloudfpsmobin`. The initial [18 October snapshot](../validation/outputs/new_center_reuse_status_20261018.json) records observed running state. The [19 October completion check](../validation/outputs/new_center_reuse_status_20261019.json) confirms all 60 jobs. Downloaded results, corrected comparisons and validation are in the [completed result bundle](../results/new_center_reuse_20261018/README.md).

## What is reused

Fit deployed centers C0 on the 60% training split. Let U span their differences relative to c0, and decompose each held-out point into head h(x), residual r(x), and energy e(x). For any later center c', decompose it using the same anchor and basis:

```
||x-c'||^2 = ||h(x)-h(c')||^2 + e(x) + ||r(c')||^2 - 2 <r(x), r(c')>.
```

Head+energy recovers this exactly when the new center lies in the old affine span. For a center outside that span, the stored random tail estimates the missing inner product. JL is the estimator used here; other representations or access to original records are also possible. Neither a general JL guarantee at these widths nor superiority over PCA follows from this identity.

The stored sketch is constructed once per fitted start/budget/map and reused across all scenarios. Reference distances use original-space held-out points. Public class labels do not define the reference; k-means need not learn one center per MNIST digit.

## Design

Ten corpora, k=5/10, three starts: 60 planned job CSVs, subject to unavailable embedding caches. The grid uses total stored-scalar budgets 16/32/64/128/256 where smaller than input dimension, q=1/4, and ten maps per q. Bases, centers and PCA use only the training split; zero-norm rows are removed before splitting.

| Scenario | Count | Construction |
| --- | ---: | --- |
| Same centers | 1 | Deployed centers, sanity control |
| Refit | 2 | Different k-means seeds |
| Subsample | 2 | Centers from random half-training subsets |
| Different k | 2 | k/2 and 2k |
| Lloyd snapshots | 4 | 0/1/3/10 full-space steps from another initialization |
| Perturbation | 2 | Random shifts at 0.1/0.3 median center spacing |

There are **12 new scenarios plus the same-center control**. The Lloyd snapshots are generated in original training space; this does not test an adaptive optimization loop driven by sketch outputs.

The methods are energy-plus-JL cross-term estimation (`ours_cross`), head-plus-tail norm (`ours_tailnorm`), head+energy with the cross term dropped (`ours_nocross`), plain sparse JL, PCA, and PCA plus residual energy. `ours_tailnorm` stores B-1 scalars, while `ours_nocross` stores only h+1. Equal nominal budgets do not imply equal bytes or model-storage costs.

## Analysis

From `code/`, after job CSVs are available:

```powershell
python -m analysis.new_center_reuse_report results/NEW_CENTER_RUN
```

The reporter averages maps within each fitted start/scenario, excludes the same-center control from new-center summaries, and exports:

- `coverage_by_method.csv`: available cells by method/budget.
- `matched_comparisons.csv`: mean metrics and paired differences on common corpus/k/start/scenario/budget cells.
- `paired_wins.csv`: common-cell agreement wins and ties.
- Descriptive overall/family/dataset summaries and scenario shift/residual descriptors.

PCA is excluded when its requested width is not below the training sample count. Other methods can still have those cells. Use matched comparisons for ranking; the descriptive grand means can use different corpus mixes. More scenario rows do not create more independent datasets. No confidence intervals or broad superiority test are currently supplied by this report.

`r_phi` evaluates selected labels in original space against the exact optimum. `psi_rel_error` compares the estimated selected cost with that optimum; it combines assignment and cost-estimation effects. The cross-term estimate can produce negative estimated squared distances at small widths; no new-center circuit constrains those estimates in this study.

## Server workflow and limits

`campaigns/server/run_new_center_reuse.sh` queues CIFAR-10/ResNet50 and 20 Newsgroups/MiniLM embedding builds, then starts four single-thread workers. At the first check on 18 October, embeddings were still building and no job CSVs existed. The experiment and embedding-builder source hashes matched the local files. The reporter was absent on the server at that check.

This is float64 native evaluation. It generates no commitments, Stage-1 proofs, new-center Stage-2 proofs, or matched per-method proof timings. The current paper's fixed-center proof circuit embeds constant centers and does not implement the new-center cross-term relation. A new proof study must separately bind the approved new centers and old basis/tail context and measure its actual costs.

The current launcher has two operational limitations: it kills an existing tmux session named `ncr` if invoked again, and the generated run script can continue after embedding failure and does not collect each worker's exit status separately. Do not relaunch it over the active session. Check all expected job files and individual logs; `RUN_DONE` alone is not a completeness check. The active workflow was inspected without restarting it.

## Completed findings - 19 October

All 60 jobs and both embedding corpora completed. The 227,604 raw rows cover ten datasets and 12 new-center scenarios per fitted start. The same-center identity passes its numerical sanity check. On matched new-center cells, PCA+energy has better mean agreement and aggregate-cost accuracy than head+energy+JL at every tested budget. The tail helps relative to plain JL and, at larger widths, dropping the cross term. See the [result interpretation](../results/new_center_reuse_20261018/README.md) for exact denominators and measurements. The corrected reporter was uploaded without changing or restarting the numerical run.
