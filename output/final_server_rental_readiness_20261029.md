# One final paid allocation: purchase readiness

The user can fund only one server rental. The rental remains 72 hours,
with an initial campaign targeting 32 hours and capped at 36 hours. At least
36 hours remain uncommitted for recovery and final verification/transfer.
There is no extension and no later paid campaign. One rental contains multiple
experiments, fits and repeated timings; it is not a single benchmark execution.

**Current decision: do not start billing yet.** The 128 GB RAM / 16 dedicated
physical-core specification is suitable for the proposed bounded study. The
complete server queue has not yet been frozen and validated end to end.

The completed [published-baseline gate](published_baseline_gate_20261029.md)
supports a regime-specific verification/reuse contribution. It does not show a
reliable advantage over PCA or universal superiority over published systems.
More compute may confirm those limitations rather than remove them.

Before payment, the execution package must contain:

1. A fixed primary comparison across independent fits, with the same scheduling
   optimization for current and PCA, plus matched Full and ZKML-adapter controls.
2. Pinned candidate streams and a complete-response timing implementation that
   charges preparation, planning, key loads, corrections, fallback and replay.
3. A locally checked checkpoint/resume path and verified evidence export.
4. A bounded queue with the existing optional-cell drop order and a reserve for
   failures, independent replay and transfer. In the initial campaign no new
   proving job starts after hour 26.5; proof jobs drain by hour 28 and a verified
   backup targets hour 32. The initial campaign cannot exceed hour 36.

The first server canaries must replace laptop runtime projections. If the
forecast exceeds the budget, reduce optional scope using the predeclared rule;
retain paired primary controls and audit/transfer time. Canaries themselves
consume the same single rental. The forecast cannot be certified in advance
from CPU/RAM specifications. Recovery starts when the initial campaign finishes,
targets failed/invalidated paired cells and necessary missing checks, and cannot
be spent on new optional scope. Recovery proof jobs drain by hour 64; hours64–72
are reserved for final verification and transfer. A full campaign rerun is not
guaranteed within that reserve.

We can commit to this being the **last paid allocation**. We cannot honestly
promise it will be the last experiment a reviewer might request, that every
optional cell will finish, or that the results will make the paper acceptable.
If purchasing depends on that guarantee, the purchase is not justified.

This note changes the budget and readiness assessment only. No server was
purchased or accessed, and the methodology and manuscript remain unchanged.
