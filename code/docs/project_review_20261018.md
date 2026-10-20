# Project and paper review - 18 October 2026

Claude's central mathematical distinction is sound: the measured head+energy construction already answers the fixed-center task, and the random tail becomes useful when later queries need residual inner products. The current paper recognizes this distinction. The completed new-center experiment now supplies native results: its tail helps against plain JL, but PCA+energy gives better mean agreement and cost accuracy on matched new-center cells. It supplies no new-center proof-speed measurements.

## Which paper versions were inspected

| Version | Source | Interpretation |
| --- | --- | --- |
| Original planning lineage | `git show f1a4af6:manuscript.tex` | Commit-then-sample sparse JL, fixed public centroids, conditional geometry, and planned ZK comparisons; contains result placeholders |
| Pre-results draft described by the user | [preserved source](../../paper_history/supervisor_draft.tex) | Recovered unchanged from the 17 October pre-results root backup; retains the original planning abstract |
| Latest results revision | [canonical manuscript](../../manuscript.tex), [protected method](../../paper_results/protected_method.tex), [experiments](../../paper_results/experiments.tex) | Fixed-center sufficient statistics, original-to-statistic binding, subset queries, exact affine/cached controls, separately scoped optional geometry |

The latest compiled PDF has 29 pages. Its title/abstract, fixed-center method, proof-cost tables and discussion were inspected in source; representative rendered pages were checked. The current manuscript and its numerical sources were preserved during cleanup. This review did not recheck every bibliography entry or constitute a formal cryptographic audit.

## What changed scientifically

The original draft proposes proving a challenged sparse projection and clustering calculations, then using conditional geometric bounds to interpret projected answers. It already fixes public centers in its primary statement; it does not primarily prove k-means training.

The latest paper's measured construction instead stores a head spanning center differences, residual energy, and an optional random tail. The real identity is exact for the fixed-center task. The integer implementation proves a rounded relation and therefore has small original-space approximation error. A no-JL control has identical fixed-center query outputs. Stage 1 authenticates statistic derivation, while Stage 2 authenticates selected rows and checks the fixed task.

This is a narrower, empirically supported result than a general claim that sparse JL makes clustering better or proofs uniformly faster. The paper makes that scope visible in its abstract, main method and discussion. Native fidelity on full cached data is separate from small-catalog actual proofs. Existing records include the 64-row MNIST query at about 5.09 seconds for head+energy+JL, 6.30 for PCA-16 and 3.44 for cached exact answers. The four-distinct-subset reconstructed workload reports a 3.85% advantage over PCA after included preprocessing components. These are saved measurements with stated exclusions, not fresh timings from this review.

## Corrections to Claude's explanation

1. New centers **inside the old affine center span** also work exactly with the old head+energy. A new direction outside that span introduces the missing residual cross term.
2. JL is the mechanism tested here, not the only possible solution. Other sketches, fuller stored statistics, or access to original records are alternatives. New centers do not logically force approximate answers if originals remain accessible.
3. A k=10 MNIST model need not contain one center for each digit. Its clusters are unsupervised, and the experiment compares assignments against original-space nearest centers rather than digit labels.
4. The code has **12 new scenarios**, not 11, plus one unchanged-center sanity control.
5. The running study is **native float64**, with no cryptographic data commitment or new-center proof relation. It cannot yet establish authenticated model updates, ZK Lloyd training, or proof-time gains.
6. About three seconds per row describes one measured preprocessing proof shape. It is not a universal cost of rebuilding a native sketch or every Stage-1 configuration.

The exact reused-sketch distance formula and scenario grid are in the [new-center study guide](new_center_reuse.md). Existing sketch error can matter even when labels remain stable, and adaptive centers chosen using the revealed map require a separate analysis. The tested Lloyd snapshots come from original-space training, not an adaptive sketch-driven loop.

## Server observation

The run is under `/root/commitjl/results/new_center_reuse_20261018` on `cloudfpsmobin`, reachable through WSL SSH. The [timestamped status snapshot](../validation/outputs/new_center_reuse_status_20261018.json) records process state, completed jobs, progress and source hashes. At the initial check, the `ncr` tmux workflow was building CIFAR embeddings; four numerical workers were queued afterward, with zero completed job CSVs. The experiment and embedding source hashes matched local source. Slow dataset download accounted for about an hour before embedding extraction began. No active workflow was restarted or killed.

Operational completion must be checked against expected job files and worker logs. The launcher does not reliably propagate all worker failures, and missing embedding caches can skip corpora. The existing run was left intact.

## Reporting fix and verification

The old new-center summary could compare grand means on different corpus mixes because PCA has a training-rank restriction. The reporter now adds `coverage_by_method.csv` and `matched_comparisons.csv`, with paired method means/differences restricted to identical cells at each budget. Existing descriptive summaries remain labeled as such. A regression test checks that a corpus missing PCA cannot alter its paired comparison.

Before cleanup, all **83 recorded input/output hashes checked** in the current numerical manifests matched their files. The stored audit summary reports 22,645 checks and 1,109 saved-proof verifications; those are preserved historical results, not proof verifications repeated in this housekeeping pass. Focused head/projection/reuse tests passed; the final cleanup verification record gives the exact checks run after changes.

## Cleanup

The [receipt](../validation/outputs/cleanup_20261018.json) records **1,028 archived files, 220,937,114 bytes**, including 14 superseded report scripts, old PDF/ZIP deliveries, temporary deployment scripts, intermediate manuscript backups, an unused alternate section tree and Springer template examples. They were moved to `C:/PyProjects/zero-knowledge-archive/cleanup-20261018/` and SHA-256 verified after the move. The pre-results draft remains directly accessible in `paper_history/`.

No raw experiment directory, dataset cache, circuit, test, or required PDF figure was removed. Approximately 32.8 GB of result/recovery evidence was retained. Old per-RQ result directories were already absent and tracked as deleted before this task; those pre-existing changes were not reverted. The external archive also records the original Git status and diff.

The current build produces one canonical manuscript PDF and one source ZIP. Generated temporary builds and delivery artifacts are ignored by Git. Root and results documentation now identify the actual current evidence instead of directing readers to the retired per-RQ layout.

## Completed follow-up - 19 October 2026

All 60 expected jobs completed, with no skipped corpora or recorded worker exceptions. Both embeddings finished successfully. Downloaded results contain 227,604 rows; all 75 transferred files match server hashes. All job names and scenario counts, unique result keys, finite numerics and same-center sanity checks passed. The [completed bundle](../results/new_center_reuse_20261018/README.md) contains exact comparisons and reproducible source snapshots.

At budget 16 on 720 matched cells, q=1 head+energy+JL gives 73.92% agreement and 5.852% mean absolute cost error versus PCA+energy at 93.27% and 0.673%. At budget 256 on 504 matched cells, the figures are 91.16%/0.326% versus 98.13%/0.037%. At higher widths the tail helps versus dropping the cross term, but these results do not establish superiority over PCA. Correlated scenarios/maps and only three starts limit statistical interpretation. The existing paper remains textually unchanged; a future revision must present these new-center results separately and honestly.
