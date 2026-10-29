# Paper-derived baseline gate before renting a server

The user's 29 October request authorizes local implementation and comparison
of additional published baselines. This gate leaves the four-stage method,
its evaluated 63/79/14/16 profile, all manuscript files and earlier evidence
unchanged. It does not purchase or access a remote server.

The implemented additions are:

- Boutsidis, Zouzias and Drineas, *Random Projections for k-means Clustering*,
  NeurIPS 2010: the independent random-sign embedding from Algorithm 1.
- Achlioptas, *Database-friendly random projections: Johnson-Lindenstrauss
  with binary coins*, JCSS 2003: the three-point sparse projection.
- Bachem, Lucic and Krause, *Scalable k-Means Clustering via Lightweight
  Coresets*, KDD 2018: Algorithm 1 importance sampling and weights.
- Uniform sampling with identical draw budgets is an additional control.
- A separate numerical follow-up implements Gao and Long's RaBitQ one-bit
  estimator (SIGMOD2024) using the same frozen inputs. It is registered after
  the first numerical gate and before its own results; it is not presented
  as part of the original preregistered method list.

The algorithms are used to evaluate supplied reference/candidate banks, rather
than allowing each method to fit a different clustering and change the question.
Their numerical output is compared with an independent exact integer oracle.
Raw sketches and sampled costs do not provide deterministic exact-change sign
certificates at these empirical budgets. The papers' randomized approximation
theorems are not promoted into unverified per-request certificate claims.

The sign and sparse projection directions are also attached to our existing
authenticated moment/switch/refinement relations for a separate real-proof
adapter study. That study tests basis choice inside our protocol; it is not
a measurement of a published random-projection proof system. Raw projection
prediction uses projected distances. Adapter prediction uses the existing
residual-aware score and can differ substantially for nonorthogonal sketches.

## Frozen study

The immutable plan, row IDs, fit seeds, quantizer, candidate recipe and source
hashes are in `validation/outputs/published_baseline_gate_20261029/`.

- Native: Fashion, MiniLM384 text and HAR561; three new fits per source;
  catalogs32 and128; all first8 candidates; three request timing passes.
- Real proofs: Fashion and text; the first fixed fit; catalog32 and all first4
  candidates; one timing execution. Complete Full, PCA, centroid and ZKML
  controls are intended, together with both new projection adapters.
- No selection uses new native coverage or a desired speed ratio.
- These public sources have been inspected during development. New fit seeds
  add model variation, not an untouched confirmation dataset.

Resource settings retain serial execution, eight workers,3.5GiB process-tree
RSS,650MiB Linux RAM reserve,1GiB Windows commit headroom and Windows OS
low-memory stops. The proof queue has a six-hour bound and each invoked phase
has a30-minute guard. Failed and unfinished work remains in the evidence.

The first preparation/native wrapper receipts are under `tmp/`; every proof
attempt has a durable receipt/log under the gate's `proofs/resources/`.
The Windows guard is `tools/laptop_host_memory_guard.py` and the per-phase
Linux guard is `tools/guarded_linux_run.py`.

## Metrics and interpretation

Keep these outputs distinct:

1. nearest-center label agreement with the exact quantized objective;
2. estimated aggregate decision agreement, including wrong-sign counts;
3. base and refined deterministic decision coverage;
4. native mathematical runtime versus real cryptographic proving/replay time;
5. actual paid fallback and final publicly verified decision correctness.

Coresets do not predict assignments for unsampled rows. Their label agreement
and deterministic certificate coverage are N/A. Exact Full native timings are
not ZK timings. Approximate floating-point numerical work and exact integer
certificate work have different implementation costs. Per-fit results are
retained; timing repeats are not new independent learning fits.

The report generator exports completed cells only and marks missing comparators.
It checks that every pinned manuscript source remains unchanged. Its aggregate
report is `output/published_baseline_gate_20261029.md`, with JSON and per-fit/
per-request CSVs alongside it.

`campaigns.published_baseline_warm_gate` is a separate equal-routing engineering
follow-up, prepared after the first native outcomes and before its own proof
outcomes. The active `warm_followup_v2` plan fixes the existing10% margin and
16-row policy, using the earlier seed521 Q8 calibration equally for every
basis. It generates fresh warm request proofs and one fresh Full comparator
per dataset; it does not invent four Full repetitions. Whole known-batch
preflight is charged. The first unexecuted warm plan remains retained; v2
repairs Windows/Linux dependency-path portability without changing cases or
the routing rule. Its STOP link propagates the main Windows guard's stop.

## Applicability review

ezDPS's original statement is DWT/PCA/SVM inference, with model privacy and
client data supplied to the server. Zhang et al.2021 studies outsourced
k-means using secure multiparty computation. Their original statements and disclosures differ from this
private-catalog cost-change audit. They are documented as applicability reviews,
not assigned invented same-task timings. ZKML remains an explicitly identified
primitive adapter, not its complete optimizing compiler.

## Commands

Set `PYTHONPATH=code`, use the pinned Linux Python environment and place new
timings in a fresh output root. Run preparation, numerical checks and proofs
serially, under the guards above:

```sh
python -m experiments.published_baseline_gate prepare --root /path/to/new/gate
python -m experiments.published_baseline_gate native --root /path/to/new/gate
python -m experiments.published_baseline_gate proofs --root /path/to/new/gate
python -m analysis.published_baseline_gate_report --root /path/to/new/gate
```

The proof phase uses existing SHA256-pinned Linux Go and ZKML binaries. It
reuses previously checked relation-smoke receipts only after matching the
exact binary hash. Those smoke receipts are not new benchmark repetitions.

RaBitQ follow-up commands are `python -m experiments.rabitq_baseline_gate
prepare` and, after the serial proof queue is idle, `python -m
experiments.rabitq_baseline_gate run`. Its numerical intervals use a union
bound over the fixed finite query family and the spherical-coordinate
distribution in the paper's Lemma B.3/Theorem3.2. They are statistical
ideal-arithmetic envelopes; this float64 reference has no outward-rounding
proof and no authentication. It uses exact native old groups as IVF anchors
and exact old cost from their preprocessing. It does not import ANN engine
timings, SIMD kernels, query quantization or a privacy claim.

Fresh output roots must be inside the checkout, for example
`code/validation/outputs/fresh_gate`. Before running the warm follow-up in
Linux, link its stop marker with
`ln -s ../STOP code/validation/outputs/fresh_gate/warm_followup_v2/STOP`.

After the cold queue finishes, execute the RaBitQ numerical follow-up and warm
queue serially. For the latter, use `python -m
campaigns.published_baseline_warm_gate run`. The current-method and PCA arms
must have the same routing optimization. Report warm phase totals separately
from the cold enclosing totals; neither is an independently timed RPC.
