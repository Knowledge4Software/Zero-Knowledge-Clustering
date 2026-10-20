# Experiment runbook (Windows PowerShell)

For the new 10-seed Ã— 5-start Ubuntu/56 GB campaign, use the
[RQ1 runner guide](rq1_complete.md) and [RQ2â€“RQ5 server guide](rq2345_complete.md).
The observations listed below describe the historical campaign, rather than
results from these new runners.

Run the commands from the repository root. The completed paper observations are already organized under `code/results/rq1/` through `rq5/`, with each RQ's raw files in `runs/` and summaries/figures in `analysis/`. [The results index](../results/README.md) explains the folders. `code/results/archive/` preserves the original mixed campaign bundles and is not an extra set of independent seeds.

## Current status

| Study | Canonical run folder | Outcome |
| --- | --- | --- |
| RQ1 | `results/rq1/runs/native/` | 7,180 successful native rows. |
| RQ2 | `results/rq2/runs/representative/` | Two attempted full-size proofs; both stopped at the memory watchdog. |
| RQ3 | `results/rq3/runs/native/` and `runs/paired/` | 90 overhead, 540 grinding, and 30 paired native rows. |
| RQ4 | `results/rq4/runs/native/` and `runs/real/` | 265 synthetic and 100 real-data native rows. |
| RQ5 | `results/rq5/runs/native/`, `runs/gamma/`, and the two precision folders | 200 scaling, 80 separation, 120 older precision, and 150 shared-24-bit-reference precision rows. |

The completed main campaign used six parallel workers for RQ3-RQ5. Its RQ1 rows originated in the serial native campaign. Per-job RQ3 timing includes CPU contention. The RQ3 beacon label means **simulated local trace**. RQ2's input commitment scales with original dimension `d`, even when projected distance work scales with reduced dimension `m`.

## Independent commands

```powershell
# 1. Preliminary five-seed RQ1 pilot
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\code\campaigns\stages.ps1 -Stage AccuracyPilot

# 2. Original precision sweep, 20-bit physical reference
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\code\campaigns\stages.ps1 -Stage PrecisionSweep

# 3a. Full native serial campaign; RQ1 writes to runs/native, RQ3-RQ5 to runs/serial
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\code\campaigns\stages.ps1 -Stage FullNative -GrindingWorkers 8

# 3b. Parallel RQ3-RQ5 campaign; requires completed RQ1, six workers for the existing run
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\code\campaigns\native_parallel.ps1 -Stage All -Workers 6

# 4. Select RQ1 shapes and attempt representative RQ2 proofs
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\code\campaigns\stages.ps1 -Stage RepresentativeProof -MaxProofShapes 2 -NativeMode Parallel

# 5. Proof-based ablation on selected shapes (resource-intensive)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\code\campaigns\stages.ps1 -Stage ProofAblation -MaxProofShapes 2 -NativeMode Parallel

# 6. Strict final tables and figures; requires successful proof inputs
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\code\campaigns\stages.ps1 -Stage FinalReport -NativeMode Parallel
```

`-NativeMode Parallel` selects the per-RQ parallel native files for final table inputs and is the default. `-NativeMode Serial` selects `runs/serial/` for RQ3-RQ5. The `-NativeCampaignName paper_native_parallel_v1` or `paper_native_rqs_v2` arguments remain supported on the new launcher path. The old loose launcher paths have been retired; the separate pilot launcher was redundant with `stages.ps1 -Stage AccuracyPilot`. The proof and strict final-report stages remain incomplete until selected RQ2 proofs succeed. The ablation labels `incircuit-gen` **evaluated as native-only baseline** and must not invent proving time or circuit counts.

## Supplements and analysis

```powershell
# Completed design: RQ3 paired, RQ4 real data, RQ5 separation, RQ5 24-bit-reference precision
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\code\campaigns\supplements.ps1 -Stage All -Workers 6

# Refresh summaries and four author charts without rerunning experiments
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\code\campaigns\supplements.ps1 -Stage Analysis -Workers 6
```

Individual supplement stages are `RQ3`, `RQ4`, `RQ5Gamma`, and `RQ5Precision`. They write to the corresponding RQ's `runs/` folder. The wrapper then regenerates per-RQ `analysis/` outputs and `results/overview/analysis/`. The newer precision run tests `{8, 12, 16, 20, 24}` under one 24-bit physical reference; do not pool it with the older 20-bit-reference run.

## Resume and interpretation

Runners checkpoint completed rows in Parquet and validate row, source, and context hashes before skipping them. Resume with the same configuration and worker count. The copied supplemental Parquet rows were checked against their original source and context hashes after the folder move. A hard interruption may require rows since the last checkpoint to run again. Keep logs and metadata with the raw rows.

Source fingerprints include exact file bytes. `.gitattributes` preserves the
fingerprinted implementation/configuration files and supplemental engine without
line-ending conversion so a new checkout retains the recorded provenance. Avoid
bulk line-ending normalization of those files when resuming existing runs.

RQ1 ARI measures agreement with full-space K-means assignments, not class-label accuracy. RQ1 `r_phi` is a clustering-objective ratio, not runtime or prover cost. RQ5 `t_total_s` sums cached native component timings and is not an end-to-end stopwatch. RQ4 certificate claims require their realized embedding premise. For paper use, read the [consolidated server study](../results/reports/full_persian_20261007/report.html) and [table coverage](../results/reports/paper_tables_english_20261007/report.html). Earlier laptop result folders were removed; these legacy commands generate new outputs.
