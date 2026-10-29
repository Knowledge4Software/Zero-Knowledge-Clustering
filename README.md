# Reusable zero-knowledge prototype-update audits

The current method combines authenticated deployed groups and moments,
rounded-basis switching bounds, and up to 16 exact-row corrections before
paid exact Full fallback. The evaluated profile uses prediction rank 63,
certificate rank 79 and precision 14. It uses no JL projection tail.

## Current paper and supervisor handoff

- [Full paper sources](manuscript.tex) and [bibliography](references.bib).
- [Current paper with the latest baseline results, 20 pages](output/pdf/prototype_audit_latest_baselines_20261029.pdf).
- [Editable paper package](output/prototype_audit_latest_baselines_20261029_sources.zip).
- [Current-method consistency audit](output/current_method_audit_20261029.md).
- [32-hour initial campaign and recovery reserve](output/server_initial_32h_20261029_fa.md).
- [Current paper scope and SVG figures](paper_results/ccd_pilot_20261029/README.md).
- [New methodology excerpt](output/pdf/prototype_update_methodology_20261027.pdf).
- [Editable methodology and integration notes](paper_results/supervisor_methodology_20261027/README.md).
- [Two-referee response and measured results](output/two_review_response_20261027.md).
- [Deliverables](output/README.md).

The completed four-stage methodology is now integrated without changes to its
prose or equations. The existing main abstract and introduction are preserved;
the supplied source ZIPs remain archived unchanged. New evaluation sections
report completed local pilots and keep the proposed server campaign prospective.

## Code and evidence

The [local comparison against paper-derived baselines](output/published_baseline_gate_20261029.md)
adds random projections, lightweight coresets and RaBitQ, with separate native
quality and real-proof adapter measurements. It preserves the paper methodology
and reports PCA ties and negative text results alongside useful operating regions.

Python source is in `code/`; Go/gnark circuits and commands are in `code/zk/`.
Current numerical records, public proofs, verification keys, source receipts
and guarded execution state are in `code/validation/outputs/`. The current
paper's new generated tables are in `paper_results/ccd_pilot_20261029/`, alongside
retained earlier tables in `paper_results/referee_final_20261025/`.
See [evidence documentation](code/validation/README.md) and
[current execution handoff](code/docs/two_review_revision_20261027.md).

The serial laptop campaign remains authorized and resource guarded. Future
proof catalogs are capped at 1024 records; larger completed native checks
are not described as proof timings. Mobin is permanently retired. Author
declarations and journal submission remain with the supervisor.

From the repository root in PowerShell:

```powershell
$env:PYTHONPATH = Join-Path $PWD 'code'
python -m analysis.two_review_campaign_status
python -m pytest code/tests/test_two_review_evidence.py code/tests/test_referee_learning_workflows.py code/tests/test_optimized_evidence_export.py code/tests/test_online_affine_control.py -q
```

The cleanup requested on 27 October actually deletes retired outputs,
old deliveries and inactive proving/build caches. It does not create another
archive. [Deletion receipt](output/project_cleanup_20261027.json).

## Repository contents

Git retains the current manuscript sources, required vector figures, experiment
and verifier code, tests, selected result exports and provenance receipts.
Root reference ZIPs, generated PDF/source deliveries, temporary builds, local
environments, dataset caches and proving caches stay local under `.gitignore`.
The PDF and ZIP links above refer to local generated deliveries. Build the
paper from `manuscript.tex`; the received ZIPs are not needed for compilation.
Large proof/data stores in `code/validation/outputs/` remain local unless an
individual evidence record was deliberately tracked. A source checkout is not
the complete private cryptographic reproduction archive.
