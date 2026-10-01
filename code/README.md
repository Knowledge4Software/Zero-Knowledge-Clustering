# CommitJL implementation

This directory contains the Python implementation, the Go/gnark proof circuit, benchmark runners, tests, and saved results for the research project. Start with the [repository overview](../README.md) for the scientific status and layout.

## Install and verify

Use Python 3.10 or newer. From this directory:

```powershell
python -m pip install -r requirements.txt -e .
python -m pytest tests --ignore=tests/test_zk.py -q
```

The Go circuit is in `zk/`. From that directory, run `go build ./...` and `go test ./...`. On Windows, build the CLI with `go build -o bin/commitjl-prove.exe ./cmd/commitjl-prove`. On Unix-like systems, use `go build -o bin/commitjl-prove ./cmd/commitjl-prove`. Then return here to run `python -m pytest tests -q` for the complete Python suite, including proof integration tests. The CLI executable is a local build product and is ignored by Git.

## Experiments and saved evidence

The [experiment runbook](docs/experiments.md) lists independent PowerShell commands, resume behavior, and output locations. The [author overview](results/overview/analysis/project_overview_for_authors.md) summarizes the completed native results and charts. The [readiness review](results/overview/analysis/paper_readiness_review.md) explains why the selected full-size RQ2 proofs are not yet paper-ready.

The main code paths are `commitjl/` for protocol and mathematics, `data/` for loaders, `experiments/` for benchmark logic and configuration, `zk/` for the proof circuit, `tests/` for verification, and `validation/` for the independent audit. Campaign launchers are in [`campaigns/`](campaigns/README.md), reporting scripts in [`analysis/`](analysis/README.md), and notebooks in `notebooks/`. The `env_capture/` module records environment metadata.

Only `README.md`, `Makefile`, `pyproject.toml`, and `requirements.txt` belong as
loose files at this level. `commitjl.egg-info/` is ignored metadata for the local
editable installation. Downloaded data and reversible housekeeping backups are
under the ignored `.cache/` directory; they are not source code.

`results/rq1/` through `results/rq5/` each contain `runs/` and `analysis/`; `results/overview/` holds cross-RQ reports; `results/archive/` retains original mixed campaign bundles. Do not combine rows from different campaigns without checking their source fingerprints, normalization policy, and sample sizes.

To refresh the completed campaign analysis without rerunning benchmarks, run `python -m analysis.paper_results` and `python -m analysis.author_overview` from this directory. These scripts write each RQ's summaries and charts under `results/rq*/analysis/` and the cross-RQ reports under `results/overview/analysis/`.
