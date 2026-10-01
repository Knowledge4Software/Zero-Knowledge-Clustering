# Research results

The current paper evidence is grouped by research question. Each RQ folder has `runs/` for raw Parquet/checkpoint outputs and `analysis/` for CSV summaries and figures. Use the per-RQ README to find the exact command and scope. `overview/analysis/` contains the cross-RQ [author overview](overview/analysis/project_overview_for_authors.md), [full analysis](overview/analysis/analysis.md), [readiness review](overview/analysis/paper_readiness_review.md), and run inventory.

| Folder | Research question | Current run status |
| --- | --- | --- |
| [`rq1/`](rq1/) | Clustering fidelity and reduced dimension | 7,180 successful native rows; preliminary pilot retained separately. |
| [`rq2/`](rq2/) | ZK proving and ablation | Two selected full-size proof attempts stopped at the memory watchdog. |
| [`rq3/`](rq3/) | Randomness, grinding, and post-commit overhead | 90 overhead, 540 grinding, and 30 paired native rows. |
| [`rq4/`](rq4/) | Stability certificates | 265 synthetic and 100 real-data native rows. |
| [`rq5/`](rq5/) | Scaling, separation, and precision | 200 scaling, 80 separation, 120 older precision, and 150 shared-24-bit-reference precision rows. |
| [`overview/`](overview/) | Cross-RQ reports and campaign metadata | English summaries and comparison charts. |
| [`archive/`](archive/) | Original mixed campaign bundles | Preserved for historical provenance; use the RQ folders for current analysis. |

The archive is a copy of the original campaign directory structure and is not a second independent experiment. Likewise, `rq1/runs/native/` contains the same completed RQ1 rows that appeared in both the serial and parallel campaign bundles. Never pool those copies as additional seeds. The older 20-bit-reference and newer 24-bit-reference precision runs use different physical normalization and must remain separate.

`archive/legacy_inputs/` contains the older loose Parquet files, tables, and figures. Consult their row provenance before use; they are not substitutes for the named paper runs. The [experiment runbook](../docs/experiments.md) contains commands and resume rules. Temporary `.parquet.lock` files are process locks, not observations.
