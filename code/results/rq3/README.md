# RQ3 — randomness and post-commit overhead

`runs/native/` contains the completed 90 overhead and 540 grinding rows from the six-worker parallel campaign, with separate logs. `runs/paired/` contains 30 native paired comparisons of post-commit generation against a prepublished map. `analysis/` contains their CSV summaries and timing/grinding charts.

The beacon source in the saved overhead experiment is a **simulated local trace**, not an external beacon measurement. Per-job times include CPU contention. The small measured after-commit stage does not establish a statistically resolved difference in total native transaction time, and no ZK proof time was measured here. See the [experiment runbook](../../docs/experiments.md) for launch commands.
