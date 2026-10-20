# Numerical analysis and paper builds

## Current completed-results workflow

Run from the repository root:

```powershell
python code/analysis/export_review_matched.py --full-capture code/validation/outputs/review_full_capture_followup_20261020 --reuse-capture code/validation/outputs/review_final_capture_20261020_1907Z --out tmp/reproduced_review_controls
python code/validation/check_layout.py
python code/analysis/build_reuse_manuscript.py --tag LOCAL_REVIEW_BUILD
```

Use fresh output paths. `export_review_matched.py` checks captured hashes, complete public task coverage, identical catalog/center data, exact integer Full outputs, and every integer certificate against Python. It writes matched cost and quality CSVs, separate LaTeX tables and a 1,024-record plot. An exact deployment and each plain/certified deployment pay their own preparation once. Native answers are included; verification, warmups, fitting, I/O and public center preparation are separate.

`capture_review_controls.py` performs read-only, compressed SSH transfers of completed Mobin case metadata into a fresh hashed snapshot. `review_certificate_preflight.py` and `review_certificate_budget.py` compute native controls; those calculations are not proof evidence. `export_review_controls.py` preserves their earlier native tables and exact-control normalization.

The [review-control guide](../docs/review_controls_20261020.md) records paths, certificate semantics and reproduction boundaries. The [paper output index](../../output/README.md) identifies the current PDF and preserved historical versions.

For PDF/source release checks, install the optional dependencies with `python -m pip install -r code/requirements-paper.txt` and run `python code/validation/verify_paper_release.py --tag LOCAL_REVIEW_BUILD` from the repository root. A completed visual-review flag must only be supplied after inspecting every final rendered page.

## Earlier research workflows

Analysis modules for compression, residual maps, laptop, circuit and native new-center studies retain their original paths. They require the corresponding local raw result folders, which are intentionally excluded from Git. `paper_results.py` and `author_overview.py` target a retired per-RQ layout and must not overwrite the current paper's exports.

Educational guide builders remain source-only utilities; they do not run or validate the paper campaign. Superseded intermediate reports and paper builds are listed in the [cleanup receipt](../validation/outputs/cleanup_publication_20261020.json).
