# RQ1 — clustering quality after reduction

`runs/native/` holds the completed ten-seed RQ1 grid (7,180 native rows), including `rq1_fidelity.parquet` and its summaries. `runs/pilot/` holds the separate five-seed preliminary pilot. Do not combine the pilot with the full run as additional repetitions.

`analysis/` holds the per-configuration CSVs and ARI-versus-dimension plot. The [cross-method quality chart](analysis/method_quality.png) fixes `m=64`, `k=5`, and SparseJL `q=2`. The [objective-ratio chart](analysis/objective_ratio.png) is a clustering-cost comparison, not a runtime or proof-cost comparison.

The native run is launched by `code/campaigns/native.ps1` (or the `FullNative` stage). The [experiment runbook](../../docs/experiments.md) gives the exact command. ARI compares assignments with full-space K-means, not ground-truth labels.
