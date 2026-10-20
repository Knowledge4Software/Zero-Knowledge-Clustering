# CommitJL implementation

The Python implementation, Go/gnark circuits, experiment runners and preserved results live here. Start with the [repository overview](../README.md) and [completed review-control guide](docs/review_controls_20261020.md).

## Install and verify

Use Python 3.10 or newer. From this directory:

```powershell
python -m pip install -r requirements.txt -e .
python -m pytest tests --ignore=tests/test_zk.py -q
```

For focused checks of the active reuse study:

```powershell
python -m pytest tests/test_new_center_reuse.py tests/test_certified_head.py tests/test_protected_jl.py -q
```

The Go module is in `zk/`. Build the main CLI from there with `go build -o bin/commitjl-prove.exe ./cmd/commitjl-prove` on Windows, or `go build -o bin/commitjl-prove ./cmd/commitjl-prove` on Linux. Bound and query studies use their corresponding `cmd/commitjl-bound` and `cmd/commitjl-query` tools. Go circuit tests and proof integration tests can be expensive; choose checks appropriate to the changed relation.

## Current entry points

| Task | Entry point |
| --- | --- |
| Completed review controls | [20 October guide](docs/review_controls_20261020.md), `python -m experiments.review_controls --help` |
| Certificate acceptance policy | `python validation/verify_certificate_policy.py --help` (public-only replay followed by integer accuracy thresholds) |
| New-center numerical reuse | [Study guide](docs/new_center_reuse.md), `python -m experiments.new_center_reuse --help` |
| New-center analysis | `python -m analysis.new_center_reuse_report results/NEW_CENTER_RUN` |
| Original-data/statistics binding | [Bound-audit protocol](docs/bound_audit.md), `python -m experiments.bound_audit --help` |
| Bound-audit analysis | `python analysis/bound_audit_report.py results/BOUND_AUDIT_RUN` |
| Current paper regeneration | Repository-root commands in [current analysis guide](analysis/README.md) |

Launchers are in [campaigns/](campaigns/README.md), reporting tools in [analysis/](analysis/README.md), and earlier run instructions in [docs/experiments.md](docs/experiments.md). Check campaign dates and circuit versions before using older instructions.

## Evidence and earlier studies

The [results index](results/README.md) distinguishes the tracked current evidence from local raw campaigns and recovery bundles. Superseded historical HTML/PDF reports were archived outside the workspace; the cleanup manifest records their hashes. They do not rebuild the current manuscript.

Other retained research includes [circuit optimization](docs/circuit_optimization.md), [protected-coordinate JL](docs/protected_jl_research.md), [native compression](docs/paper_compression.md), [its correctness audit](docs/paper_compression_audit.md), [laptop comparisons](docs/laptop_comparison.md), [reusable native representations](docs/reusable_native.md), and [residual maps](docs/residual_map_comparison.md). These experiments and tests retain their original source paths for reproducibility.

Dataset caches and local housekeeping material are under ignored `.cache/`; local binaries are under ignored `zk/bin/`. The loose files at this level are `README.md`, `Makefile`, `pyproject.toml`, and `requirements.txt`.
