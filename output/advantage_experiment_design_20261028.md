# Advantage experiment design — 28 October 2026

The question is **where, if anywhere, does the current method win after all
costs are charged?** A positive result is not assumed. The paper is unchanged.

The machine-readable plan was frozen before the new numerical control:
[plan](../code/validation/outputs/advantage_design_20261028/plan.json),
[hash pin](../code/validation/outputs/advantage_design_20261028/pin.json),
[results](../code/validation/outputs/advantage_design_20261028/native_control.json).
The real-proof study specified below has **not been run**.

## 1. Repeated candidate audits: the main runtime question

An owner commits one catalog and an old reference. Audit distinct proposed
center banks against that reference. The centers can change between requests;
the old reference and its authenticated groups stay fixed within one epoch.
Changing the deployed reference starts a new epoch, with reauthentication paid.

Use Fashion-MNIST and MiniLM text, fit seeds 521, 523, 541, catalog sizes
32, 128, 512, 1024, and ordered request prefixes 1, 4, 16, 64. Pin source
versions and row IDs before fitting. New audit rows should be unused where
available; reused data must be described as exploratory. New seeds alone do
not turn previously inspected rows into independent confirmation.

Fit old centers at training-only Lloyd step8. Generate the64 proposals from
16 fixed disjoint incoming-training minibatches with rates .125, .25, .5, 1.
Audit rows never enter candidate generation. Retain every request. Cache exact
public center duplicates equally for every system and report the distinct count.

Compare the current centroid certificate, the same protocol with a PCA basis,
exact Groth16 Full, and the existing ZKML/Halo2 primitive adapter. Keep the
prediction rank63, certificate rank79, precision14 bits and maximum16 exact
corrections fixed. Refinement order is interval width with index ties.

Use exactly the same integer objective, quantization, tie rule, rows and banks.
Run serially in Linux/WSL with8 workers, equal CPU affinity, a3584MiB process-tree
RSS cap and650MiB free-RAM reserve. Compilation and diagnostic feasibility
checks are separate; deployment costs and unsuccessful runtime attempts count.

First test ascending sizes using4 fixed updates. Select the largest **common
resource-feasible size**, independent of certificate success or speed ratios.
At that size use3 paired timing blocks, rotating method order and retaining
every fit. Stop at the initial12-hour proof budget or a45-minute phase guard;
preserve incomplete cells. Resource failures do not establish a speed ratio.
The native grid can still cover all sizes without presenting native timings
as proof timings.

Record per-bank timings so cumulative prefixes are actual measured sums.
Within a block reuse genuine already generated evidence once; cold timing
blocks start with fresh bundles. Repeating an identical candidate is not64
new proof workloads.

Report cold total, warm query total, correction/fallback frequency, proof bytes,
peak RAM and individual paired timing ratios. Include basis fitting and owner
authentication in the appropriate cold profile. Allow every comparator to reuse
its parameters, keys and old-reference evidence under the same rules.

The accounting model is

    T_certificate(Q) = S_certificate + sum(query + replay + corrections + fresh fallback)
    T_exact(Q)       = S_exact       + sum(exact query + replay)

For a constant-cost planning approximation with per-request costs c and e,
amortization requires e > c; then the crossover is beyond
(S_certificate - S_exact)/(e-c). If corrections/fallback make c >= e, more
requests alone cannot create a sustained win.

The old Fashion N32 receipts give an **optimistic planning crossover around38
updates** when verification, corrections, process gaps and workload changes are
ignored. This is why64 distinct candidates are included. It is not an observed
speedup or a forecast with uncertainty bounds. Text also has a superficially
favorable base-query slope, but its observed corrections and fallback are
substantial; those cannot be omitted from the actual comparison.

Primary correctness is the verified sign of the exact original-space cost
change. An inconclusive certificate must pay for new exact fallback. Exact
individual labels are a separate output requirement; if required, their exact
correction must also be paid. Exact Full publishes a stronger exact-cost result;
a later strong decision-only comparison should include an exact sign-only
circuit too.

Groth16 Full provides the internal backend control. Halo2 provides a system
comparison with a different commitment construction; it does not isolate the
mathematical algorithm. A2x ratio is a practical target, not a promised outcome.
Require a consistent regime-specific effect across all3 fixed fits of at least
one real dataset and3 timing blocks before claiming a real-data runtime benefit.
Report paired runs and dispersion; repeated timings are not new dataset samples.

## 2. Low-variance decision directions: a mechanism control against PCA

Construct128-dimensional data with5 clusters. Their center differences live
in a4-dimensional subspace of the first5 coordinates. Increase independent
nuisance variation in the other123 coordinates while holding the decision
geometry fixed. These coordinates affect distances but cancel in pairwise
center comparisons. This deliberately tests the difference between retaining
variance and retaining center-difference directions.

Use3 fixed seeds,256 training rows,64 separate audit rows, nuisance standard
deviations8,32,128, signal amplitude30 and signal noise2. Old centers have
amplitude32; new candidates have amplitudes28,30,34,36. Fit each basis from
training and old centers only; keep it frozen for all4 new candidates.
Use the actual rounded-basis integer certificate implementation, prediction
rank63, certificate rank79, and the same16-row correction cap for both methods.

**This numerical control has now run**, with72 method/update cases. Each row
below pools12 updates (3 seeds times4 amplitudes). All intervals contained the
independently computed exact change; no certified assignment was incorrect.

| Nuisance standard deviation | Representation | Mean agreement with exact labels | Conclusive decisions after bounded correction |
|---:|---|---:|---:|
| 8 | Centroid method | 100.00% | 12/12 |
| 8 | PCA | 100.00% | 12/12 |
| 32 | Centroid method | 100.00% | 12/12 |
| 32 | PCA | 19.79% | 5/12 |
| 128 | Centroid method | 100.00% | 12/12 |
| 128 | PCA | 19.79% | 5/12 |

This establishes a numerical mechanism example against this PCA configuration.
It is deliberately synthetic and has no generated cryptographic proofs. It
does not establish a proof-speed advantage, real-data prevalence, novelty, or
an advantage over exact ZKML. Native exact-correction values are an oracle for
checking the numerical algorithm; their authenticated proof costs remain to
be measured in the real-proof study.

Reproduce the numerical control into a fresh directory (existing evidence is
never overwritten):

```sh
PYTHONPATH=code OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/koush/.local/share/commitjl-external/python/bin/python \
  -m experiments.advantage_design_20261028 --root /path/to/new/evidence
```

Implementation: [experiment script](../code/experiments/advantage_design_20261028.py).
If the mechanism control succeeds but the real runtime study fails, the
supported contribution is narrower: preserving decision directions helps in
this specified geometry; a broad practical speed claim remains unsupported.
