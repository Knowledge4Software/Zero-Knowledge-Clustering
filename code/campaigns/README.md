# Campaign launchers

The current review-control campaign finished all 14 cases on 20 October 2026. The [completed guide](../docs/review_controls_20261020.md) records both remote roots, checked evidence and reproduction commands. Launchers are retained for new, explicitly named campaigns; do not relaunch a completed campaign to inspect its status.

Earlier new-center and per-RQ workflows below target separate historical raw runs. Their local outputs are excluded from Git, and their measurements must not be pooled with the current matched proof cohort.

Run the PowerShell launchers from any working directory. They resolve the code
and results folders relative to their own location. Python module commands must
be run from `code/`.

| Entry point | Purpose |
| --- | --- |
| `bash campaigns/paper_server.sh rq1\|pilot\|native\|selected\|proof` | Current balanced Ubuntu campaign; see the [server guide](../docs/rq2345_complete.md). |
| `python -m campaigns.tiny_all_rq_check` | Tiny real check of every stage and proof ablations; engineering evidence only. |
| `stages.ps1` | Pilot, precision, full native, selected proofs, ablation, final report. |
| `native.ps1` | Sequential native campaign with preflight checks. |
| `native_parallel.ps1` | Parallel RQ3 orchestration and remaining native studies. |
| `supplements.ps1` | Supplemental studies and report generation. |
| `python -m campaigns.rq3_parallel` | Process-based RQ3 overhead measurements. |
| `python -m campaigns.supplements` | Supplemental measurements with per-RQ outputs. |

The [experiment runbook](../docs/experiments.md) gives exact stage commands.
The PowerShell entries below the current Ubuntu commands retain the earlier
campaign designs; do not mix their rows into the balanced campaign.
The old loose `run_*.ps1` launchers have been moved here. Use `stages.ps1 -Stage
AccuracyPilot` instead of the redundant standalone pilot launcher.

`supplements/engine.py` retains the original measured implementation byte for
byte: its SHA-256 is part of existing checkpoint contexts. `supplements/__main__.py`
provides the current folder routing without changing measured algorithms or
invalidating those checkpoints. Resume with the original worker count.

## 20 October reviewer controls

`launch_review_full.py` and `launch_review_certified.py` create fresh timestamped Mobin releases with frozen sources, inputs and binaries. The certified queue waits for the exact queue so numerical benchmarks do not contend. Both queues completed successfully; inspect their preserved manifests and final statuses instead of relaunching them. The [review-control guide](../docs/review_controls_20261020.md) records the exact roots and completed measurements.
