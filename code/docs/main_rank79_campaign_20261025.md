# Main rank79 campaign and paper update

**Superseded:** the user has since authorized a new focused
[referee campaign](referee_final_campaign_20261025.md). This old queue and its
paper updater remain stopped; the pause description below is historical.

**Implementation update:** [short runtime optimization checks](runtime_optimization_20261025.md)
are separate from this paused campaign. Eight-worker proof microbenchmarks are
about three times faster than two-worker ones, including Full. Existing campaign
worker counts, binary pins, measurements and pause flags were not changed.

**Current state: USER-PAUSED before shutdown.** All three coordinators and
their research descendants are stopped; no experiment may resume without an
explicit user request. Completed first Fashion-MNIST N128 head79 and Full
are preserved:24.771min versus39.457min,1.593-fold,146group/48Full proofs,
2/2 distinct naturally worsening updates certified without fallback. Both
proved intervals have zero width because no membership switches are possible
on those two tasks. Frozen-predictor agreement and native assignment-certified
fraction are100% for that small cohort. OtherN128 profiles remain unfinished;
N256/N1024 main cases never started.

The user requested an immediate current-results paper update. Canonical
`manuscript.tex` now includes only complete evidence, with final8-page PDF at
`output/pdf/neurocomputing_current_results_20261025.pdf` and editable sources
beside it in `output/`. All pages were visually checked. Previous master is
preserved at `paper_versions/20261025_before_current_results/` and all earlier
PDFs/source sections are unchanged. Supervisor original hash is unchanged.
The automatic post-main updater remains paused and must be rebased before
any explicit future resume: preserve this new manuscript/current fresh-proof
section instead of promoting its stale pre-main candidate. The planned
canonical SHA guard intentionally prevents that accidental overwrite.

The user authorized the main runs and a paper update based on their results.
The approved method is compact exact-old group cost-change certification with
a frozen63-dimensional predictor and79-dimensional training-only auxiliary
certificate. It has no JL tail. All work remains private for supervisor review.

## Execution order

1. Finish the already running two fresh-fit N128 controls: Fashion-MNIST,
   k3/seed101/deployed age8, and MNIST, k5/seed113/age1. Each measures reusable
   exact public-center Full, head79, head63 and PCA79, with cold setup, fresh
   public replay and newly generated Full fallback for every inconclusive task.
2. N256 Fashion-MNIST mature control, same fit, all distinct original updates:
   Full/head79/head63/PCA79.
3. N1024 MNIST early control, same fit, all distinct original updates:
   Full/head79/PCA79.
4. Recheck complete receipts, public-manifest hashes, independent integer
   endpoints, prospective fallback costs and all eleven compressed comparison
   rows. Export paper tables directly from those checked results.
5. Promote the prepared focused manuscript only if its canonical predecessor
   has not been edited during the runs. Preserve the previous master, all old
   included sections, earlier PDFs and the supervisor original. Build/render a
   private PDF and editable source package. Inspect all final rendered pages
   before calling the PDF ready for final delivery.

All scenarios/profiles are pinned in the main plan before its N256 audit
outcomes are evaluated. The scale extension reuses the same two fits; it is
not independent model-fit replication. No workload is removed because it is
slow, inaccurate, regresses or needs fallback. Identities and repeated center
sets are cached before proof generation and exposed separately in statistics.
One timing repetition per case on a variably loaded laptop is exploratory.
Shorter two/four-update workloads amortize setup less than the earlier19-update
3.19-fold control; the old ratio is not promised for the main runs.

Expected expansion cost is about12--18 active hours based on saved phases,
plus the remaining N128 queue. RAM waits, pauses, fallback and retries can
extend completion. The long main phases have a12-hour invocation guard; the
default for earlier controls stays4hours. The longer limit propagates to
prospective Full fallback. RAM reserve650MiB, phase-start headroom and3.5GiB
process cap remain in force. No apps are closed and no startup task is installed.

## Status and pause/resume

- N128 queue:
  `code/validation/outputs/rank79_independent_replication_20261025/queue.json`.
- Main scale plan/state:
  `code/validation/outputs/main_rank79_20261025/{plan,queue}.json`.
- Paper update plan/state:
  `code/validation/outputs/main_paper_update_20261025/{plan,queue}.json`.
- Process registry: `code/validation/outputs/research_jobs_20261024.json`.
- Launch logs: `tmp/main_rank79_20261025_resume.*.log`,
  `tmp/rank79_replication_20261025_resume2.*.log`, and the paper-update logs.

The registry and live command lines are authoritative for process identities.
On a user pause, stop all three coordinators plus all proof/native/build
descendants. Preserve proof caches, mask/challenge pins, partial exports,
resource attempts and all queues. Resume only after the user's instruction,
using identical arguments plus `--resume`. The paper process must be resumed
too if it stopped because an upstream campaign paused. No visibility change,
upload, server action or publication is authorized.

Two scheduler-only restarts fixed result-field collisions, reference-copy
recovery and live-child adoption. The running Full generation process was
preserved throughout; its attempt clock and paid costs continued unchanged.
The maintenance/adoption receipts record this. Shared Full reference copies
are labelled as copies, and are never treated as new fallback generation.
Partial copies/exports are archived and recovered without deleting evidence.
Thirteen focused recovery/coordinator tests pass; these are not a security audit.

## Corrected statistical units

The initial fresh native report counted repeated Lloyd proposals. The primary
unit is now a distinct nonidentity quantized center set within each deployed
fit. Original raw results are unchanged; reproducible distinct exports preserve
the correction.

At N1024, head63/head79 resolve25/25 distinct updates and1/1 distinct natural
regression, versus PCA24/25 and0/1. The old3/3 worsening requests are one
identical candidate submitted at Lloyd12/16/24, not three regressions. Head
mean agreement on distinct updates is99.898438%, PCA99.906250%; assignment
certified fractions are97.578125% and97.359375%. N128 head79/PCA79 both
resolve23/25 and6/7 natural regressions; unresolved cases remain.

The old development screen has148 distinct updates and27 natural regressions:
63 resolves140/148 and22/27;79 resolves144/148 and25/27;95 resolves145/148
and26/27. Original160/36 counts remain proposal-level diagnostics, not the
primary paper units. The old single-fitN128/19-improvement real-proof frontier
is unchanged:56.013minutes/3.512 for compact63,61.637minutes/3.191 for79.
Those native sensitivity and proof timing figures remain separate studies.

## Manuscript boundary

The prepared draft focuses on exact assigned group change, bounded switching,
post-commit challenge/complete coverage, separate assignment metrics, equally
reusable PCA/Full and the faster exact known-family alternative under its
closed-bank contract. New proof tables are enabled only after all required
generation, public replay, fallback and independent checks succeed. Private
previews omit pending proof results entirely.

Final automatic PDF output is named with `_review.pdf` and marked pending
page-by-page visual review. Supervisor-owned author/declaration fields stay
pending. The update process stops rather than overwriting intervening edits
to the canonical master. All remaining security, dataset, replication and
server-export limitations are retained; no acceptance estimate or claim of
eliminating every possible referee objection is made.
