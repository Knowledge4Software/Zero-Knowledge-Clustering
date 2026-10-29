# Current analysis entry points

Set `PYTHONPATH` to the repository's `code` directory before using module
commands from the project root.

- `analysis.two_review_campaign_status`: inspect actual local processes and
  write a timestamped queue snapshot.
- `analysis.update_optimized_evidence`: validate/export the retained image
  confirmation study and completed matched proof comparisons.
- `analysis.export_two_review_evidence`: completed-only learning-workload,
  exact-backend, verifier-resource and timing-variation exports.
- `analysis.release_referee_final_paper`: build the full private paper with
  a fresh tag; automatic candidates still need visual review.
- `analysis.check_optimized_release`: bind a reviewed render to its PDF and
  check source/evidence pins.
- `analysis.prepare_supervisor_methodology`: preserve the supplied input
  ZIPs and prepare/package the independent methodology handoff.
- `analysis.check_supervisor_methodology`: check rewritten algebra, source
  preservation and the methodology build.
- `analysis.published_baseline_gate_report`: export the completed numerical
  and real-proof cells for the 29 October paper-derived baseline gate,
  retaining missing comparators and checking manuscript preservation.

Other modules support retained controls or historical experimental workflows.
Obsolete outputs and old paper-version folders are deliberately absent after
cleanup. Current source receipts and public proof bundles remain in
`code/validation/outputs/`; proving caches outside active/queued runs can be
regenerated if a future experiment requires them.
