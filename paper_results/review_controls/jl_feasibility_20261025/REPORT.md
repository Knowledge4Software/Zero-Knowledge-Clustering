# JL feasibility, 25 October 2026

The current production method and frozen proof binaries are unchanged. These
are small numerical research studies, not a new proof-speed benchmark. No new
ZK proofs were generated. The existing measured rank79 result remains61.637
active minutes and3.1913 times the matched reusable Full control.

| Idea | Small-test result | Interpretation |
|---|---|---|
| Sparse-JL measurements orthogonalized into16/32 residual basis directions | All six variants detect0/6 hard regressions; the trained SVD16-direction control detects4/6 | Not a replacement for the present certificate basis on this stress cohort |
| Untrusted JL64 hints for exact-refinement ordering | All three seeds need33 exact rows in total, versus34 for simple interval-width ordering; both resolve6/6 numerically | About2.94% fewer rows in this tiny cohort; fewer rows do not establish faster proofs |
| JL on group summaries, with exact residual-sum energies | All19 matched improvements resolve, but so do the no-JL compressed control and the SVD control; all six hard regressions require precise-moment fallback | No incremental JL decision benefit; any compression advantage needs its own binding proof and full cost measurement |
| Shrunken sparse-JL residual correction for predictions | Development mean 99.794408% -> 99.821820%; worst remains99.218750% | Small post-hoc development improvement, with16 additional stored sketch values |
| Same fixed predictor on all160 stored requests, with three new sketch seeds | Original 99.919434%, JL 99.919027%;2 additional errors across491520 seed/record/update comparisons | The development improvement did not persist; do not promote it as a stronger predictor |

The confirmation has51 improved,49 worsened and380 equal seed/request pairs.
Its worst agreement falls from99.414062%
to99.316406%. All are comparisons with
exact quantized Full labels, not semantic/classification accuracy. All160
requests remain present. The three new random-map seeds test seed robustness;
the fitted prediction models are stored previous cohorts, not new independent
learning problems. The development grid did not find a rank39+JL candidate
passing the user-selected99% floor on every19-update/three-seed combination.

The quick stress cohort consists of the six regressions previously unresolved
at rank63; it is deliberately post-hoc. The matched workload has N128 and19
improvements. They are separate workloads. All725 recorded numerical envelope
checks were rechecked for containment and conclusive correctness. Three
oracle-based Python tests passed: rounded-basis and mixed-basis aggregate
containment, exact recovery with an identity sketch, and sound refinement even
with deliberately misleading hints. These are not a security audit.

The random-basis branch uses QR on sparse measurements after removing the
frozen head span, followed by integer rounding. Its deterministic interval uses
the existing exact Gram correction. It must not be advertised as a new sparse
JL theorem. The hint branch never accepts a certificate from a probabilistic
distance estimate: only verified exact corrections could tighten the bounds.
That selective correction proof is not implemented here. The aggregate branch
also needs authenticated counts, means and residual-sum norms before deployment;
the present experiment is only a numerical reference.

An initial attempt stopped after one complete quick-case calculation when RAM
headroom dropped. That case and its original log are preserved. The resumed
implementation streamed the identical5000 sampled image rows from existing
local ARFF caches, verified their canonical float64 SHA256, and used chunked
Full checks. It completed all six cases without closing user applications or
changing the underlying dataset. `resume_receipt.json` records the runner
revision and prior checkpoint; no native orchestration time is claimed as
deployment performance.

Recommendation: retain the trained rank79 certificate as the current measured
candidate. Keep these JL experiments as honest ablations. There is no present
evidence for claiming a faster production method, a stronger certificate, or
substantive novelty merely from appending JL. No large JL proof run is queued.
The summary pins all source results and code. Artifacts remain private.
