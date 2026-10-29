# Local experiment orchestration

The active review campaign uses `referee_two_reviews_local.py`, followed
serially by `referee_review_followups.py`. State is under
`code/validation/outputs/referee_two_reviews_20261027/`. A separate completed-
evidence finalizer builds an unreviewed PDF candidate after the queue ends.

Inspect current execution using `python -m analysis.two_review_campaign_status`
from the repository root with `PYTHONPATH=code`. Do not start a duplicate
coordinator while the recorded live supervisor owns the queue.

The shared campaign budget is 24 hours. Proof catalogs are at most 1024
records, with memory, RSS, disk and explicit-stop guards. All attempts,
resource exclusions, fallback costs and negative outcomes remain recorded.
The current policy is `validation/outputs/referee_round2_20261026/execution_policy.json`.

Older orchestration modules remain source for existing proof receipts and
controlled reruns. Their old output directories and interim launch reports
are not prerequisites for the current queue. Mobin is permanently retired;
there is no remote target and no server work should be launched.

Cost-aware laptop control: `run_cost_aware_control.py` selects compact or Full
before query proving and checks whether a batch amortizes key loading. See
`code/docs/cost_aware_control_20261029.md` for commands and timing scope.
`cost_aware_pilot.py` is the paired warm control; completed receipts are kept in
`validation/outputs/cost_aware_20261029` and its final text follow-up. These
engineering tests do not modify the paper or establish wins over every baseline.

The 29 October published-baseline gate adds random-sign projections, sparse
random projections and lightweight coresets, with raw numerical comparisons
and separately identified proof adapters. Its immutable inputs and guarded
attempts are under `validation/outputs/published_baseline_gate_20261029/`.
See [scope and commands](../docs/published_baseline_gate_20261029.md).
