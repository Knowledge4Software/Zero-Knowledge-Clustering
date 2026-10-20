# Targeted server follow-up (6 October 2026)

This is a separate campaign, not a replacement for `paper_server_20261006`.
Launcher: `campaigns/followup_server.py`. Server service:
`commitjl-followup-20261006.service`. Outputs: `results/followup_server_20261006`.

| Phase | Actual scope |
|---|---|
| sizing | Synthetic 96x128: full, sparse, dense; optional 128x128 arms only if a buffered RAM forecast is below 38 GiB. Five measured proofs per arm, one warmup. |
| rq1_targeted | Original 20 Newsgroups (3,000x2,000), DBSCAN-margin (1,000x128), Gaussian easy (2,000x256); sparse q=4/8, m=64/128/256/512/1024/1536, k=5/10. Dimensions above d are explicitly excluded. 10 seeds x5 starts. |
| precision | All eight original native corpora, 16/24 bits, fixed physical normalization at 24 bits, RQ1-selected sparse m/q/k. 20 Newsgroups uses eligible k=10; DBSCAN-margin uses explicitly exploratory m=128,q=4,k=5. 10x5 measurements per cell. |
| real | Original frozen corpora, fixed randomly selected rows (seed 20261005), all features: Gaussian easy 32x256, Gaussian hard 16x512, MNIST 8x784, Fashion-MNIST 8x784. k=5. Sparse uses the original selected m/q, but must pass a new quantized subset ensemble CI and fresh-witness check. Full baseline is measured independently. These are tiny feasibility subsets, not full-dataset proof evidence. |

Arcene/Gisette are not feature-truncated to fit proofs. Original full-size proof
results remain exclusions. Qualification failures are preserved rather than
rerolling until a successful challenge appears. Native precision is physical
fidelity, while RQ2 subset qualification is relative to the integer witness.

Proof runs are serial with 16 Go threads, 42 GiB prover RSS cap, 46 GiB task-tree
cap. Precision uses 4 workers x4 BLAS threads and 10 GiB per task-tree cap.
RQ1 uses 3 dataset workers x5 BLAS threads with 12 GiB caps. Systemd caps the
aggregate to 50 GiB, disables swap, and owns the complete process tree.
The queue has process-tree monitoring, finite phase deadlines, an inactivity
watchdog and disk checks. A failed phase is recorded; independent phases can
continue. There are no automatic retries or unbounded loops.

Preliminary expected duration: **4-8 hours**, based on previous pilot timings;
wide real subset proofs are not yet measured, so this is an estimate.
The sum of phase hard deadlines is **17 hours**, with an 18-hour service cap.
Timings will remain isolated from other experiment work; downloading existing
results can continue but heavy disk/CPU work should be avoided during timing.
Median prove time excludes setup/compile; report those separately.

Read-only progress: inspect `queue_state.json`, each `batch_state.json`,
`inventory.csv`, and controller/task logs. Original results and arrays are
preserved. The service persists when the laptop or SSH connection closes.
