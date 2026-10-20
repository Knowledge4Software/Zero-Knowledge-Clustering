# Evidence and verification

The active [layout/evidence check](check_layout.py) validates current manuscript dependencies, citations, primary documentation links, preserved supervisor bytes, and the hashes and completeness of the finished review-control captures. It writes `outputs/layout_verification.json`.

From the repository root:

```powershell
python code/validation/check_layout.py
```

For independent integer recomputation of the full matched campaign, run the exporter in the [analysis guide](../analysis/README.md). Its canonical output is `paper_results/review_controls/generated_completed_20261020/manifest.json`.

## Included evidence

- `outputs/review_full_capture_followup_20261020/`: complete three-case small-catalog exact metadata.
- `outputs/review_final_capture_20261020_1907Z/`: all 11 later completed cases, including the exact N=1024 baseline.
- `outputs/review_owner_inputs_20261020/`, `review_certificate_preflight_20261020/`, `review_certificate_budget_20261020/`: public-benchmark owner inputs, native grids and predeclared pilot records.
- `outputs/completed_reuse_20261020/`: earlier historical D/F evidence, separate from the new controls.
- Selected public replay, smoke, negative-policy and verifier-release receipts retain targeted engineering evidence.

Captured phase diagnostics may contain individual benchmark assignments. They are analysis inputs, not the privacy-preserving public proof export. `verify_certificate_policy.py` uses only public proof metadata, independently approved tasks and setup pins; it applies explicit integer accuracy thresholds after cryptographic replay.

All 6,848 campaign proofs passed the workers' separate public-only verification phase. Local numerical re-analysis is a further check and is not described as a fresh local cryptographic replay of every saved proof.

## Historical tools

`audit.py` and the earlier written formula/implementation reviews concern retired per-RQ campaigns and their historical assumptions. Their required raw inputs are not part of a fresh clone. Their old outcomes must not be presented as current campaign verification. The modern checker retains `provenance_flags` for the historical byte/provenance regression tests.

Superseded status reports and duplicate evidence views are recorded in the [cleanup manifest](outputs/cleanup_publication_20261020.json). Bulk raw runs and local binaries are excluded from Git; current compact evidence and provenance are included.
