# Two-review revision, 27 October 2026

User explicitly authorized fixes, additional experiments and manuscript revision
within laptop limits. No Mobin, publication, external messages or author-field
completion is authorized. Keep proof catalogs <=1024; no4096/16384 proof work.
Source reviews are the two attachment files ending c6d2afcf and30ccc683.

## Completed in this revision

- Read both reviews. Main scientific priorities: learning application breadth,
  an actual alternative exact backend, fixed-reference motivation, privacy
  boundaries, baseline justification, statistics and a consolidated evaluation.
- Added data/referee_revision_data.py. Public UCI HAR official subject-separated
  training/test features, and public20 Newsgroups train/test features encoded
  with fixed MiniLM ONNX model. Archive/model/source hashes are in the study.
  MiniLM revision1110a243fdf4706b3f48f1d95db1a4f5529b4d41, model.onnx,
  masked-mean pooling plus L2 normalization, max128 tokens, batch16, CPU2threads.
- Added experiments/referee_learning_workflows.py. Native study complete:
  8representation fits, two row-disjoint configurations per corpus, past500
  training records, incoming500, audit1024; first incoming/second incoming250
  each give three fixed update rates. Bases/scaling fit past data only.
  All six requests retained; Lloyd continuations are separate diagnostics.
- At N1024: HAR centroid11/12 decisions,3/4 regressions, mean agreement99.691%,
  worst99.023%; PCA11/12,3/4, mean99.642%, worst98.926%.
  Neural text centroid8/12,0/4 regressions, mean96.370%, worst92.188%; PCA8/12,
  0/4, mean95.793%, worst91.992%. All intervals contain truth, no false sign.
  These failures remain in the manuscript. This does not solve a universal99%
  agreement target or neural-text regression sensitivity.
- Native rank39/63/79 and cap0/4/8/16/32 ablation: on first128-row HAR fit,
  correction budget4 increases5/6 to6/6; neural text remains4/6 even at32.
  No resolution advantage of79 over39 on these cases; do not claim optimality.
- Modified only full-plonk-control main to separate SRS loading/generation and
  circuit setup, and support a private unsafe test-SRS cache/preparation mode.
  New frozen binary72ef6fa9731af71745f87b2b9a52a52a28e9dc36c1825f1ce35ba93ab88ee68b
  is code/zk/bin/full-plonk-control-referee-20261027.exe. Old binaries unchanged.
  Four tiny proofs and16 real D384 neural-feature pilot proofs passed public
  verification. Pilot runtime selected N64 (not accuracy/outcome selection).
- Source plan and all work: code/validation/outputs/referee_two_reviews_20261027/.
  Root plan pinned before fitted results; proof budget24hours, serial work.
- Native tests4pass; new exporter analysis/export_two_review_evidence.py emits
  completed-only quality, rank/cap, proof-resource, timing-SD/conditional-CI,
  backend tables and two_review_evidence_receipt.json. Initial export passed
  proof-file counts, all native interval/sign invariants, and prior manifests.
- Main text edited: simpler abstract, release-batch motivation, explicit
  aggregate disclosure and no adaptive record privacy; precise novelty delta.
  Stale A.6 online-affine text fixed; table caption generator no longer mentions
  absent PLONK rows. Old-two versus extended-eight split wording corrected.
  BN254 bound example added. V3DB v2 and ZK-Value v1 archived arXiv IDs/DOIs
  verified online. HAR and MiniLM model-card references added.
- experiments.tex now includes evaluation_two_reviews.tex: consolidated quality,
  timing/resource/backend evidence and main-text break-even figure. Old numerical
  generated tables are preserved; most overlapping development tables no longer
  appear in main text. Discussion now consolidates limitations.
- release_referee_final_paper supports focused text checks and adds new evidence
  sources/code to editable packages. No new PDF has been built for this revision.

## Current proof jobs and the sleep incident

campaigns.referee_two_reviews_local.py runs the actual new comparisons.
Initial scheduler9888 stopped on a missing backend parent output directory,
before proving. Fixed directory creation and explicit logged restart10028.
Completed pilot/inputs/SRS reused; failed attempt retained and charged.
Root stage backend_r1 uses Go PID37752 (verify live identities, not old PIDs).
Plan: three PLONK repetitions on identical N64 neural features/7banks, public
replay, matched Groth Full/centroid/PCA at64, then HAR and text at128.

An independently logged Windows sleep during backend_r1 adds approximately
5360seconds to one raw proving sample (normal samples~5.85sec). Windows
Power-Troubleshooter event123607 has SleepTime09:55:47Z/WakeTime11:25:07Z.
Raw values and valid proofs are retained. backend_timing_interruption_amendment
excludes repetition01 from clean timing statistics and predeclares replacement04.
Do not present raw sleeping time as an algorithm speed comparison.

To avoid its original2hour calendar cap killing a healthy worker, guardian
PID13292 runs campaigns.referee_sleep_guardian.py. It suspended only scheduler
10028, preserves prover37752 and monitors RAM/disk/STOP, then resumes scheduler
after the phase exits. No proof/key/input restarted. Guardian has12hour cap;
global study budget remains24hours. Inspect actual processes and JSON ledgers.

## Next required work

1. Verify current new proof jobs. Finish/export each completed comparison. Add
   clean replacement PLONK repetition04 serially after current pipeline; optional
   matching Groth batch1 control separates backend from batching effects.
2. Resume only the interrupted current1024 Fashion online affine bank rebuild
   under the user's renewed experiment authorization; bank000 complete, bank001
   partial. Preserve and charge all samples/resource attempts. Do not restart
   old proof series or4096/16384. Prioritize this over redundant1024 repetitions.
3. Consider a preregistered follow-up neural activity prototype head on HAR,
   since current MiniLM study is a negative regime. New plan amendment must
   disclose post-result motivation and retain every fit/outcome. Run serially.
4. Add tests for exporter resource/count, repeat-unit and interrupted-timing
   behavior. Inspect source consistency, bibliography and undefined references.
5. Run analysis.update_optimized_evidence and export_two_review_evidence. Rebuild
   private paper with new tag prototype_audit_two_reviews_20261027, render and
   visually inspect all pages before marking ready. PDF skill was read earlier
   in this conversation. Preserve Oct26/Oct27 reviewed PDFs and supervisor draft.
6. Produce response matrix with resolved/measured/pending/fundamentally limited
   items. Do not promise minor revision or claim all scientific issues fixed.
   Author metadata and public artifact remain intentionally pending by user.

## Completed delivery and continuation wiring

The revised paper is now `output/pdf/prototype_audit_two_reviews_20261027.pdf`,
15 pages, SHA256
`f82b03daa48b7123aaffd5cae7d368b85a432610ea1c64e07e7caf6ead1d88be`.
All pages were visually inspected; after the last wording adjustment only
page 4 changed and was reinspected at full size. The source ZIP and manifest
passed validation, all 14 historical numerical tables stayed unchanged, and
the new evidence pins passed. Its status is ready for private supervisor
review, not submission-ready. The preceding 17-page PDF was later deleted under the user's explicit project-cleanup request; the current 15-page release remains.

`output/two_review_response_20261027.md` gives a point-by-point response and
separates completed numerical evidence from pending proofs. The four targeted
test files passed 14 tests; added backend checks recompute every exact batch
and reject omitted or duplicated rows even when aggregate costs happen to
match. Public-resource export now verifies manifest hashes before counting.

Backend repetition 1 completed 448 proofs and fresh public replay. Its raw
sleep-interrupted timing is excluded. The guardian completed and resumed
the same scheduler; repetition 2 is now running. Verify live PIDs rather
than relying on earlier identities. No proof worker was restarted for sleep.

Serial follow-ups are already running as a waiting coordinator, using
`campaigns.referee_review_followups.py`. They perform the trained neural HAR
diagnostic and first-seed proofs, replacement PLONK repetition 4, three
matched batch-1 Groth16 controls, largest feasible PLONK batch from 8/4/2,
and only the interrupted 1024 Fashion online-affine continuation. They share
the original 24-hour budget. Every outcome and resource exclusion is retained.
Training preflight confirms all six classes in each past-500 subset.

`analysis.finalize_two_review_revision.py` is a separate live observer. After
the serial queues finish or stop, it exports completed evidence and builds
`prototype_audit_two_reviews_20261027_completed_candidate.pdf`; the reviewed
PDF is preserved. `reviewed_authoring_ready.json` pins the reviewed authoring
state. The finalizer refuses changed authored text and marks its PDF awaiting
visual review. It holds a temporary Windows idle-sleep request for its process
lifetime; no permanent power setting changed, and manual user sleep still
remains possible. Inspect `paper_finalizer.json` for its actual state.

Exporter support for completed matched Groth16 batch-1 and larger PLONK
controls is implemented. Future numerical results append only completed
rows to the generated tables; no automatically produced PDF is preapproved.

Everything remains private; no push/public release. No new subagents permitted.

## Explicit disk cleanup, 27 October

The user subsequently requested literal deletion of old files and archives.
`output/project_cleanup_20261027.json` records the removals. Retired
`code/results` campaigns, duplicate paper snapshots/deliveries, obsolete
reports/binaries and inactive proving caches are removed. Source code, current
paper dependencies, public proofs/verification keys, numerical records and
receipts remain. Every cache in this active campaign and the queued current
1024 Fashion online-affine directory is preserved. No running job is paused
or restarted for cleanup. Older narrative statements that a superseded PDF
was preserved describe the state before this explicit deletion request.
