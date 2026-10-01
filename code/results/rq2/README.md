# RQ2 — ZK proving cost and memory

`runs/representative/` contains the selected shapes, proof attempt rows, and logs. The two full-size shapes were Gaussian `2,000 x 256 -> 16` and Arcene `100 x 10,000 -> 16`; both stopped at the memory watchdog. `analysis/rq2_proof_status.csv` lists their status and observed resource metadata. No full-size proving time or constraint count was measured successfully.

The `RepresentativeProof` stage in `code/campaigns/stages.ps1` selects shapes from the completed RQ1 rows. `runs/ablation/` is reserved for measured proof-based ablations; it must not be filled with symbolic circuit numbers. The [readiness review](../overview/analysis/paper_readiness_review.md) explains the outstanding prover-memory problem.
