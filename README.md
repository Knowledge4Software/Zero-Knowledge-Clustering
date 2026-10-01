# CommitJL: sparse JL clustering with zero-knowledge proofs

This research repository studies whether a sparse, challenge-derived random projection can preserve clustering results and be authenticated in a zero-knowledge proof. The Python package implements the protocol, projection, metrics, certificates, and experiments; `code/zk/` contains the Go/gnark circuit.

The completed laptop campaign provides native clustering and sensitivity results. Its two selected full-size RQ2 proof attempts stopped at the memory watchdog, so the current results do not demonstrate full-size ZK proving cost or a proof-cost advantage. See the [author overview](code/results/overview/analysis/project_overview_for_authors.md) for the dataset inventory, four charts, and a plain-language assessment. The [paper-readiness review](code/results/overview/analysis/paper_readiness_review.md) records limitations that matter for publication.

## Repository map

| Path | Purpose |
| --- | --- |
| [`manuscript.tex`](manuscript.tex), [`figures/`](figures/) | Paper source and paper figures. |
| [`code/commitjl/`](code/commitjl/) | Python protocol and mathematical implementation. |
| [`code/zk/`](code/zk/) | Go/gnark circuit and proof CLI. |
| [`code/data/`](code/data/) | Dataset loaders and local SHA-256 manifest generation. |
| [`code/experiments/`](code/experiments/) | Benchmark runners, configurations, tables, and figure generators. |
| [`code/campaigns/`](code/campaigns/) | PowerShell campaign commands and parallel orchestration. |
| [`code/analysis/`](code/analysis/) | Statistics, comparison charts, and author reports. |
| [`code/notebooks/`](code/notebooks/) | Interactive and Kaggle notebooks. |
| [`code/tests/`](code/tests/) | Python verification tests. |
| [`code/results/`](code/results/) | Per-RQ `runs/` and `analysis/` folders, an overview, and historical campaign bundles in `archive/`. |
| [`code/docs/experiments.md`](code/docs/experiments.md) | Windows campaign commands, checkpoints, and result locations. |
| [`code/validation/`](code/validation/) | Independent audit, baseline comparison, and verification records. |

The local Overleaf copy in `Sparse_JL_for_ZK_Clustering/` and downloaded datasets in `code/.cache/` are excluded from Git. They are not required to browse the saved results.

## Quick start

From `code/`, install Python dependencies with `python -m pip install -r requirements.txt -e .`. Run Python tests with `python -m pytest tests --ignore=tests/test_zk.py -q`. The full proof tests additionally require Go and a built CLI; on Windows, run `go build -o bin/commitjl-prove.exe ./cmd/commitjl-prove` from `code/zk/`, then run the relevant ZK tests. The [implementation guide](code/README.md) gives more detail.

The experiment launchers are grouped in `code/campaigns/`. Use the updated commands in the [experiment runbook](code/docs/experiments.md); saved checkpoints remain compatible. The native campaign and the full-size proof campaign have very different memory requirements.

The saved benchmark rows are observations on the sample sizes recorded in each result file. They should not be described as whole-dataset results or as completed RQ2 proofs.
