# Current referee campaign, 25 October 2026

26 October latest update: all21 main phases are COMPLETE; saved old PIDs below
are historical. Fresh additional public-only verification passed 7,679
current-control proofs. No prover/campaign is active. Heavy laptop runs remain
held. Mobin is retired because the user no longer has it; do not connect or
launch work there. No replacement execution host is assigned. Do not
automatically resume an old queue. Completed local evidence is preserved.
Policy: `code/validation/outputs/referee_round2_20261026/execution_policy.json`.
Current manuscript/results: `output/referee_round3_report_20261026_fa.md` and
`output/prototype_audit_referee_revision_20261026_validation.json`. The older
failed extra replay receipt is preserved; its wrapper was fixed and a fresh
audit completed in `public_replay_round3_20261026`. This is the same frozen
verifier implementation, not an independent security audit.
The37server packages have been recovered with transfer hashes, but their
comprehensive fresh replay is still pending. Online-affine preparation code
is not a completed benchmark. Main-bank affine is in a separate appendix.

The user authorizes autonomous local experimentation and manuscript revision
for approximately ten hours on 25 October. Keep all material private. Supervisor
handles author and declaration fields. Do not restart superseded queues.

## Current method and comparisons

Selected method: compact exact-old group cost-change certificate, prediction
rank 63 and training-only certificate rank 79, precision 14. On an inconclusive
interval, authenticate up to16 exact-row switching corrections ordered by
canonical interval width (index tie breaks), stopping on a conclusive result.
Residual ambiguity gets newly generated Full. No JL tail.
Compare reusable public-center Full, equally reusable PCA79, authenticated
projection-free moments-only, and exact known-family affine. The affine
baseline requires its center bank before preprocessing; this is a different
information contract from unknown future centers.

Three diagnostic cohorts retain every distinct supplied update: mature MNIST
k3/seed101/N128 (natural switching regressions and paid fallback), early
Fashion-MNIST k3/seed101/N128, and early MNIST k5/seed113/N1024. These fits were
already observed in the native grid; the cases are challenge workloads, not
new independent fits or unbiased deployment-rate samples.

Every new method and Full fallback uses eight Go workers. Do not compare new
eight-worker times against old two-worker measurements. Charge fitting where
applicable, setup, original-row proofs, queries, fresh public replay, all failed
attempts and newly generated Full fallback. Queue/offline gaps are excluded.

## Live evidence and recovery

- Queue: `code/validation/outputs/referee_final_20261025/queue.json`.
- Immutable plan: same directory, `plan.json`.
- Frozen binary: `code/zk/bin/commitjl-bound-referee-final-20261025.exe`.
- SHA256: `67d40e423bbb98ec4d6dbcd9db196e447ddbedcad2f5f4c4b720026dedd70190`.
- Input/source/binary hashes, smoke receipts, resources and public manifests
  remain in the campaign directory. Do not overwrite any completed evidence.

Check actual processes, not just the saved `running` flag. A previous turn
interruption left stale flags after process termination. Resume with the same
frozen executable and `--resume`; complete proof caches and elapsed attempt
costs must be retained. Do not launch a second coordinator while one is live.
Use hidden `Start-Process` on Windows for a detached launcher.

The selected bounded-refinement MNIST mature pair is COMPLETE:4/4 natural
worsenings with2/7/12/13 switching rows. Only two new exact-row proofs are
needed (indices111,44).164base+2new proofs, no remaining fallback.
Method645.158s versus Full1258.234s, ratio1.95027. This is a shared measured
component sum plus new refinement, NOT an independently repeated end-to-end
trial. Base's old budget-zero fallback work remains archived and is excluded
from the selected algorithm. `refinement_execution_amendment.json` records
why obsolete fallback is not regenerated. It does not waive fresh fallback
for residual selected-method or moments-only ambiguity.

Development:both head/PCA reach25/25 at N128/N1024 with cap16. Two new frozen
seed/split configurations (MNIST137,Fashion149) confirm head16/16 at eachN;
PCA16/16 at128,15/16 at1024,missing the single large-catalog regression.
Native head worst agreement99.21875%/99.8046875%; label-certified fractions
are separate. Confirmation is native only, not a new proof timing.

The first complete matched cohort is now finished, including all baselines.
Head645.158s, PCA561.109s, Full1258.234s. Moments-only resolves0/4 and pays
1125.500s for80 newly generated Full proofs; total1457.281s. Exact known-family
affine takes282.610s (Full/affine4.452), under its preapproved bank contract.
PCA and affine are faster than Head here; retain that result.

A second cold Head proof execution is complete:555.221s versus the second
same-input Full execution1125.500s, ratio2.02712. First-pair ratio1.95027.
Each execution has166 new Head and80 new Full proofs with disjoint hashes.
The fit is fixed, and its recorded fitting component is charged again rather
than refitted. Receipt: `timing_repeat/checked_timing_repeat.json`.

At the 16:24UTC checkpoint, supervisor5392/coordinator20840 are live and
Fashion early Full is complete; Fashion Head is active. Verify actual process
command lines because PIDs and status files can become stale. Final public-only
replay watcher35808 waits for all three matched cohorts; its audit is not yet
complete. It uses the same frozen verifier implementations in fresh processes.

Final refinement binary: `commitjl-bound-refinement-v2-20261025.exe`, SHA256
`6d138f6f321877f110253928ed1c0c3270a1fb5ae36dbb6f77ab2ed22e7afa7d`.
Do not overwrite either frozen executable. Its smoke_v4 verified16base+3new
proofs and rejected8tamper checks, including missing base coverage and
duplicated refinement indices. Resume preserved proof bytes and statements.

Detached supervisor status: `autonomous_session/status.json` under the main
campaign. Check actual PID/command lines. It runs the serial coordinator,
retains caches and stops on scientific/programming failures. Its helper only
prevents automatic idle sleep while the supervisor is alive; no persistent
power plan changes. Laptop must remain powered for progress.

Scheduling amendment: read-only Windows low-memory notification plus1GiB
commit headroom and the unchanged3.5GiB RSS cap. After a memory stop, require
observed peak+768MiB before retry to prevent launch/kill loops. Initial Full
used the older650MiB physical reserve. All attempts remain charged; changing
available RAM makes these descriptive component timings, not a controlled
memory-scaling or production-economics experiment.

## Manuscript and release

Canonical `manuscript.tex` includes only `paper_results/referee_final_20261025/`
and generated checked tables. Earlier methods remain in development archives,
outside the paper's comparison set. The supervisor original is unchanged.

`code/analysis/release_referee_final_paper.py` checks evidence, builds the PDF,
renders every page and makes a source ZIP. Completed cohorts alone enter new
tables. The coordinator invokes the guarded automatic build after all three
cohorts complete. That produces a `_completed_candidate` PDF; it is not a
visually reviewed release. Inspect every rendered page before promoting it.
`authoring_pin.json` prevents silent overwriting of later supervisor edits.

Current manual release tag is `neurocomputing_referee_final_20261025`.
Read its validation JSON for the actual visual-review state. Never call a
pending PDF ready. The old 25 October current-results PDF remains preserved.

## Reviewer boundary

The original 19-update ablation finds A<0 on only 2/19; even the stronger
best-target-per-old-group moments upper bound resolves only those two, versus
19/19 switching decisions. This numerical necessity test is not a substitute
for timed authenticated moments-only and fallback comparisons.

Public assigned-change outputs allow coordinate-probe recovery of group
counts/sums. The paper explicitly excludes aggregate hiding from its privacy
objective. This is a disclosed output-leakage limitation, not a repaired privacy
mechanism. Conditional soundness and targeted tamper tests do not constitute a
complete protocol-security audit. A learned-head advantage and Neurocomputing
acceptance are not established by these experiments.
