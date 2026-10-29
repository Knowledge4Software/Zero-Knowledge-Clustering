# Auxiliary certificate rank: complete stored-fit numerical screen

All160 frozen head-update requests were evaluated, including36 natural
regressions. Precision stays14 bits. The original63-dimensional prediction
basis is unchanged; added certificate directions use training data only.

| Certificate rank | Conclusive updates | Natural regressions detected |
|---|---:|---:|
|63|151/160 (94.375%)|30/36 (83.333%)|
|79|156/160 (97.500%)|34/36 (94.444%)|
|95|157/160 (98.125%)|35/36 (97.222%)|

The table above retains the original proposal-level counts, including repeated
center sets. A subsequent hash-checked audit of every source archive groups
requests by cohort, deployed fit and the complete quantized proposed center
array. It removes12 repeated requests, leaving148 distinct nonidentity updates
and27 distinct natural regressions. Rank63 resolves140/148 updates and22/27
regressions; rank79 resolves144/148 and25/27; rank95 resolves145/148 and26/27.
Thus the rank79 extension gains three distinct natural regressions over63.
Different proposals, ranks and ages sharing a fit are paired observations,
not independent replications. `distinct_summary.json` records every center
equivalence class and source hash; the original `summary.json` and `results.csv`
are preserved. Reproduce these counts with
`python -m analysis.summarize_distinct_rank_confirmation --root <this directory>`
from an environment with `code` on `PYTHONPATH`.

No incorrect conclusive outcomes were observed. Every interval contains the
independently recomputed quantized Full cost change. The standalone higher-rank
counts equal the counts obtained by intersecting their intervals with the old
rank63 intervals. The numerical run completed in548.010seconds (9m8s).

These percentages describe certificate decisions and regression sensitivity,
not assignment accuracy. These are previously stored configurations, including
the cohorts used for development and confirmation; this is not a new independent
fit cohort. The experiment is numerical only, without SNARK generation or a
measured deployment-speed result. In particular, do not combine34/36 or35/36
with the3.51174-fold speed measured for the separate rank63 proof control as
if they were one jointly measured result.

Rank79 is the next cost-control candidate: it is the smaller tested extension
and resolves four additional regression proposals (three distinct regressions).
A fresh real-proof control
has started at
`code/validation/outputs/group_compact_auxiliary_r79_n128_20261025/control`.
It retains the original N128 prediction model/catalog, all19 updates, q16,
precision14 and the same frozen Go binary. All setup/proof/replay work and the
extra training-only fitting cost are charged. It does not share rank63 proof
work or claim an implemented adaptive-rank protocol. Its runtime remains
unmeasured until completion; a provisional60-90minute budget is based on the
56minute rank63 control, allowing additional cost and laptop-load variation.
Fallback, RAM stops or a larger proving domain can extend that budget.

The companion `summary.json`, `results.csv`, fitted `.npz` bases and source
hashes preserve the complete evidence. Artifacts remain private for supervisor
review.
