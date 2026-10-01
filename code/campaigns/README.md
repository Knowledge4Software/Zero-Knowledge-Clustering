# Campaign launchers

Run the PowerShell launchers from any working directory. They resolve the code
and results folders relative to their own location. Python module commands must
be run from `code/`.

| Entry point | Purpose |
| --- | --- |
| `stages.ps1` | Pilot, precision, full native, selected proofs, ablation, final report. |
| `native.ps1` | Sequential native campaign with preflight checks. |
| `native_parallel.ps1` | Parallel RQ3 orchestration and remaining native studies. |
| `supplements.ps1` | Supplemental studies and report generation. |
| `python -m campaigns.rq3_parallel` | Process-based RQ3 overhead measurements. |
| `python -m campaigns.supplements` | Supplemental measurements with per-RQ outputs. |

The [experiment runbook](../docs/experiments.md) gives exact stage commands.
The old loose `run_*.ps1` launchers have been moved here. Use `stages.ps1 -Stage
AccuracyPilot` instead of the redundant standalone pilot launcher.

`supplements/engine.py` retains the original measured implementation byte for
byte: its SHA-256 is part of existing checkpoint contexts. `supplements/__main__.py`
provides the current folder routing without changing measured algorithms or
invalidating those checkpoints. Resume with the original worker count.
