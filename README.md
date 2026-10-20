# CommitJL: verifiable clustering audits

CommitJL implements reusable zero-knowledge audits of nearest-center assignments and clustering costs over committed records. Python supplies representations, integer references and experiments; Go/gnark supplies the proof relations and public verifier.

The current [Neurocomputing manuscript](manuscript.tex) incorporates the completed 20 October Mobin campaign. All **14 cases** finished: exact public-center Full and learned/PCA plain/certified reuse on MNIST catalogs of 64, 256 and 1,024 records and CIFAR/ResNet with 64 records, plus two higher-rank proof pilots. All **6,848 proofs** passed worker public-only replay. The local analysis checks 208 captured hashes, exact Full outputs and 33,280 record/set certificate evaluations.

At 1,024 MNIST records and ten public center sets, component costs are 24,681.3 s for exact Full, 4,778.1 s for plain learned reuse and 4,861.0 s for certified reuse. The 5.08-fold Full/certified ratio compares different accuracy guarantees. Learned empirical agreement averages 98.25%, certified assignment coverage 61.63%, and cost-interval width 20.65%. No updated case meets the illustrative 99% coverage / 1% width policy. PCA/energy is equally reusable; no universal learned-over-PCA improvement is claimed.

## Start here

- [Completed results](output/mobin_completed_results_20261020.md): every matched method and representation configuration.
- [Current PDF and source package](output/README.md): latest paper plus the three preserved earlier versions.
- [Reproduction guide](code/docs/review_controls_20261020.md): captures, verification, timing policy and commands.
- [Numerical manifest](paper_results/review_controls/generated_completed_20261020/manifest.json): inputs, checks and exact measurements.

## Layout

| Path | Purpose |
| --- | --- |
| `manuscript.tex`, `references.bib`, `figures/` | Canonical paper and required assets |
| `paper_results/` | Authored sections, generated tables, plots and provenance |
| `paper_history/supervisor_draft.tex` | Supervisor original, preserved byte for byte |
| `paper_versions/20261020/` | Frozen original, red/green comparison and previous clean revision |
| `output/` | Latest PDF/source package, delivery checks and completed-result report |
| `code/commitjl/`, `code/data/` | Mathematical implementation and dataset loaders |
| `code/zk/` | Circuits, proof CLIs and public verification |
| `code/experiments/`, `code/campaigns/` | Experiment runners and launchers |
| `code/analysis/`, `code/tests/` | Numerical export and implementation tests |
| `code/validation/outputs/` | Selected hashed evidence, benchmark owner inputs and audit receipts |
| `code/results/`, `code/.cache/`, `tmp/` | Ignored local raw runs, dataset caches and build scratch space |

## Reproduce and check

From the repository root:

```powershell
python -m pip install -r code/requirements.txt -e code
python code/analysis/export_review_matched.py --full-capture code/validation/outputs/review_full_capture_followup_20261020 --reuse-capture code/validation/outputs/review_final_capture_20261020_1907Z --out tmp/reproduced_review_controls
python code/validation/check_layout.py
python code/analysis/build_reuse_manuscript.py --tag LOCAL_REVIEW_BUILD
```

Use a fresh export directory and build tag. The PDF builder requires TeX Live. Run Python tests from `code/` with `python -m pytest tests --ignore=tests/test_zk.py -q`, and Go checks from `code/zk/` with `go test ./... -p 1`. Proof integration tests require locally built CLI binaries.

## Evidence and cleanup

The repository includes source code, compact historical paper exports, completed reviewer-control metadata, public-benchmark owner inputs and selected public replay artifacts. The benchmark inputs are not sensitive participant records. Captured diagnostic assignments are distinct from the public proof exports, which omit individual assignments and certificate flags. Bulk raw arrays, dataset downloads, proving artifacts, local binaries and recovery archives remain local; they are not required to compile the paper.

Superseded reports, intermediate paper drafts and duplicate evidence views were moved to `C:/PyProjects/zero-knowledge-archive/cleanup-publication-20261020/`. The [cleanup receipt](code/validation/outputs/cleanup_publication_20261020.json) records 564 files with original paths and SHA-256 hashes. Earlier cleanup receipts and Git history preserve previous layouts. No raw measurement campaign or supervisor original was deleted.

The target is Neurocomputing, using the subscription publication route given the authors' budget. Author details, declarations and final scientific approval remain to be completed before submission. This repository makes no prediction of journal acceptance.
