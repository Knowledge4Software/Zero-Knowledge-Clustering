# Project, paper and server review — 19 October 2026

The completed 64-row proof workloads support the changing-center cost claim. Learned head+energy is about **6.89× cheaper than cached exact answers on MNIST and 7.76× on CIFAR-10/ResNet50 at ten center sets**, under the recorded accounting below. Its mean agreement on the nine changed-center sets is **98.26% and 99.48%**. PCA+energy obtains nearly identical reuse costs. The contribution supported here is reusable authenticated statistics with a measured approximation trade-off; the runs do not establish a general JL advantage or predict journal acceptance.

## Evidence and running state

I connected through WSL SSH to `cloudfpsmobin` and captured the active campaign without restarting or changing it. The [snapshot manifest](../validation/outputs/server_review_20261019/manifest.json) was observed at **15:15 Tehran time, 19 October** (11:45 UTC). Files were read during collection rather than as an atomic filesystem snapshot.

| Campaign | Observed state | Interpretation |
| --- | --- | --- |
| `paper_runs_20261019/reuse_mnist_n64` | Complete; queue exit 0; 83 workload rows | Six methods, ten center sets, three timed proof repetitions |
| `paper_runs_20261019/reuse_cifar_n64` | Complete; queue exit 0; 83 workload rows | Same design on ResNet50 embeddings |
| `paper_runs_20261019/reuse_mnist_n256` | Running | Learned Stage 1 finished; 256 recorded verified row samples; reuse phase active |
| `paper_runs_20261019/fidelity_5starts` | Queued after n256 | Not completed evidence in this snapshot |
| `new_center_reuse_20261018` | Previously completed, 60 jobs | Original native head+energy+JL study; no new-center proofs |
| `new_center_hybrid_20261019` | 60 local job files and completed summaries | Native learned-head/JL variants; separate from proof campaign |
| `shift_owner_reuse_20261019` | 60 downloaded server jobs; 22,260 rows | Native simulated owner availability and class shift; no multi-owner proof protocol |

The newer proof campaign is not documented by the root README or current manuscript, which still summarize the 17–18 October fixed-center evidence. Those files must not be used to infer that the new experiment lacks a reuse circuit. Conversely, the new results must not be silently attributed to the older paper tables.

The proof workload uses **ten sets total: the deployed-center control plus nine changes**, with k=5 throughout. The earlier broad native campaign has 12 changed scenarios plus its control. These are different designs.

A second live check at **15:18 Tehran** confirmed that n256 was still proving learned-energy reuse queries and writing progress. The queue had not yet reached PCA, the remaining n256 methods, or the five-start fidelity step.

## Measured cost

The following table sums recorded native statistic preparation, circuit compilation, setup, witness construction and median proving time. It charges one Stage 1 and one reuse setup to learned/PCA/span energy, and the measured center-specific preparation/setup/query sequence to cached and affine for every set. These are workload component totals, not elapsed campaign durations.

| Dataset, N=64 | Method | T=1 (s) | T=2 (s) | T=10 (s) |
| --- | --- | ---: | ---: | ---: |
| MNIST | Learned head+energy | 311.38 | 317.66 | 368.83 |
| MNIST | PCA+energy | 309.53 | 315.93 | 366.22 |
| MNIST | Span+energy | 282.27 | 287.45 | 329.14 |
| MNIST | Cached exact | 250.15 | 502.18 | 2,540.88 |
| MNIST | Affine exact | 277.91 | 552.04 | 2,758.42 |
| MNIST | Full, reconstructed | 342.26 | 684.64 | 3,473.59 |
| CIFAR/ResNet50 | Learned head+energy | 554.91 | 563.65 | 635.19 |
| CIFAR/ResNet50 | PCA+energy | 555.28 | 563.71 | 635.58 |
| CIFAR/ResNet50 | Span+energy | 515.58 | 520.93 | 563.16 |
| CIFAR/ResNet50 | Cached exact | 490.30 | 980.34 | 4,928.83 |
| CIFAR/ResNet50 | Affine exact | 517.43 | 1,029.73 | 5,099.52 |
| CIFAR/ResNet50 | Full, reconstructed | 493.63 | 984.22 | 4,781.00 |

Against cached and affine, learned energy crosses over at **T=2** on both completed datasets. Its mean witness+prove query time is **6.42 s on MNIST and 8.88 s on CIFAR**. Thus the earlier 1–5 s expectation was optimistic for the 64-row learned reuse circuit. The recorded learned Stage-1 witness+prove totals are 195.11 s and 389.99 s; the partial n256 MNIST result is 776.98 s, consistent with the anticipated roughly 13 minutes.

Full measures one subset per center set and multiplies its witness+prove cost to cover the catalog: batch 8 for MNIST and batch 2 for CIFAR. Full's whole-catalog times remain reconstructed, even though the measured batch proofs are real. The main six-to-eightfold conclusion above compares fully measured learned and cached workloads and does not depend on Full extrapolation.

Excluded costs include fitting centers/bases, center-statistic computation, verification, loading, warmups, repeated benchmark trials and process/I/O overhead. The `native` column measures statistic preparation, not all model fitting. A paper should report this accounting explicitly and avoid calling it complete deployment latency. Full lacks a reported whole-catalog fidelity measurement in `workload.csv`.

The speedup is relative to these center-specific baseline implementations. A separately designed Full circuit with public variable centers could amortize setup too; it has not been measured here. Most importantly, **PCA+energy already receives the same reuse benefit**: the learned representation is approximately tied with it on proof cost.

Full-precision exports: [cumulative costs](../validation/outputs/server_review_20261019/cumulative_costs.csv), [quality](../validation/outputs/server_review_20261019/proof_quality.csv).

## Accuracy and what the representation actually does

These means exclude the deployed-center control and use nine changed sets over the same 64 records: 576 assignment comparisons per dataset, with correlated records/scenarios. Reference labels are floating original-space nearest-center labels, not class labels. Cost error is the mean absolute relative error of the aggregate estimated minimum cost; it is not per-row distance error or a worst-case bound.

| Method | MNIST agreement | MNIST cost error | CIFAR agreement | CIFAR cost error |
| --- | ---: | ---: | ---: | ---: |
| Learned head+energy | 98.26% | 0.228% | 99.48% | 0.338% |
| PCA+energy | 96.53% | 1.536% | 98.44% | 0.441% |
| Span+energy | 83.51% | 1.977% | 96.35% | 0.757% |
| Cached/affine exact | 100% | 0.0033% | 100% | 0.0021% |

The learned representation stores 15 basis coordinates plus energy. Its basis combines the deployed center span with residual PCA directions. **It contains no JL tail.** New-center queries drop the remaining residual cross term and prove the resulting head+energy estimate. The code's method name `learned_energy` does not mean it trains a neural model. PCA+energy uses 15 PCA directions plus energy in the same reuse circuit. Span+energy uses four coordinates plus energy, so its lower cost is not an equal-storage comparison.

Cached and affine are exact for the quantized relation; their tiny floating-reference cost errors reflect quantization. Rounded head/energy implementations also need this distinction from exact identities in real arithmetic. A proved margin sum is a soft output; no original-space correctness certificate follows from a large aggregate margin alone.

The favorable two-corpus proof results are not a general native ranking. The broader hybrid study's no-JL learned control has 93.19% versus PCA+energy's 93.27% agreement at budget 16 over 720 matched new-center cells. At budget 256 it has 98.07% versus 98.13% on 504 cells. The original JL-tail study loses to PCA+energy on mean agreement and aggregate-cost accuracy at every tested budget, as recorded in its [completed interpretation](../results/new_center_reuse_20261018/README.md).

## Simulated owners and distribution shift

I averaged random maps within each matching dataset/k/start/scenario/budget cell before comparing methods. The [paired export](../validation/outputs/server_review_20261019/shift_owner_paired.csv) keeps owner and class-shift settings separate.

At budget 16 in the owner setting (360 cells), learned energy obtains **89.89% agreement versus 81.71% for locally fitted PCA+energy**, and 1.20% versus 7.28% aggregate-cost error. The learned advantage remains positive through budget 128. This is a useful conditional result when a public deployed model supplies center directions but the residual/PCA basis can use only one owner's training data. Pooled-data oracle PCA reaches 92.24% at budget 16, so restricted basis access matters.

In class shift (60 cells), learned energy obtains **74.50% agreement versus 76.33% for PCA+energy** at budget 16, with large 21.35% versus 20.50% aggregate-cost error. Pooled-data oracle PCA reaches 94.65%. Learned energy stays slightly behind local PCA through budget 128. Thus strong shift remains a failure mode; these results do not demonstrate general shift robustness.

The owner experiment fits a basis on owner 0's training records, uses a model fitted on pooled training data, and evaluates pooled held-out records. It does not implement five independently committed owners, distributed proving, or an actual federated deployment. The JL cross-term variant has 72.02% owner-setting agreement at budget 16, below local PCA's 81.71%; plain JL is lower still. The owner result supports the learned protected basis, not a general JL benefit.

## Implementation findings that affect paper claims

1. **Independent reuse verification is unfinished.** `executeReuse` calls Groth16 verification while generating every proof, so these are real generated-and-verified proofs. However, the standalone `verify` path supports ordinary query/geometry public witnesses, not the new reuse witness. The observed trust file pins `reuse_64_k5`, while the result is saved as `reuse_64.json`; the verifier would look for the former filename and its later shape checks also expect ordinary query paths. Reuse public heads/offsets must be reproducibly derived from approved public centers and the pinned old model when adding verifier support. This is a release/independent-validation gap, not evidence that the generated circuit accepted an invalid witness.
2. **Center approval is a verifier responsibility.** The reuse circuit receives public center heads/offsets and binds the old catalog/context. It does not constrain training or derive those public statistics from full original-space centers in-circuit. The host computes them during this benchmark. An independent verifier must compute/check the same values from the approved new center set; arbitrary prover-supplied statistics are insufficient.
3. **Reuse does not make JL necessary.** The deployed circuit sets the residual cross term to zero. Equal-width PCA+energy is an essential baseline and gets the same cost amortization. Generic variable-center exact baselines remain an unmeasured alternative.
4. **Do not promote queued or partial results.** The 256-row learned Stage 1 is measured, but the six-method, ten-set 256-row comparison and five-start fidelity follow-up are not complete at the captured time.
5. **Small proof catalogs remain the scale limit.** Native datasets and the larger partially authenticated catalog do not establish completed full-dataset proofs, a deployment workload, or neural feature-extraction verification.

## Paper assessment and next revision

The canonical abstract still leads with fixed-center compression, 5.09 s versus 6.30 s queries and a reconstructed four-query advantage of 3.85%. It does not describe the new variable-center relation, learned basis, six-method ten-set workload, ResNet proof case, owner-access experiment or shift failure. The title still includes sparse projections and fixed-center audits. Those are material differences from the proposed reuse paper story, not merely missing table updates.

A defensible revision would lead with **reusable authenticated sufficient statistics for changing public clustering models**, distinguish exact fixed-center identities from approximate new-center scores, and position sparse JL as an evaluated alternative/negative result. Preserve the unchanged-center control, report changed-center metrics separately, include PCA+energy prominently, and give the real cost formula and baseline limitations. Prior sufficient-statistic/projection-cost-preserving work needs explicit discussion; the current bibliography cites Cohen et al. but contains no Feldman entry. This review does not establish novelty or finish a literature audit.

The manuscript currently uses the Springer `sn-jnl` template and labels its source as a Cluster Computing manuscript. Neurocomputing is an Elsevier journal. Its [publisher scope](https://shop.elsevier.com/journals/neurocomputing/0925-2312) covers learning systems, machine learning and pattern recognition, but whether this systems contribution meets that bar is an editorial judgment. The simulated model-update/owner setting helps explain ML relevance; embeddings alone do not establish an ML contribution. Submission formatting and author/declaration placeholders also remain to be resolved.

The previous statement that major revision was more likely than rejection is not justified by these experiments. The evidence improves the systems case, while novelty, baseline strength and venue fit remain open. I would treat this as a promising result set requiring a substantive manuscript revision and independent reuse verification before submission, without assigning acceptance probabilities.

## Checks performed

- Captured 780 eligible metadata/log/source files; all transferred file SHA-256 values were rechecked locally. Full proof/key binaries and private input arrays were not transferred by this review.
- All five captured Python source files match local source after line-ending normalization. The server holds the proof binary but not the inspected Go source tree at the expected paths; its SHA-256 was recorded, without claiming a reproducible binary-to-source match.
- Reconstructed every exported Stage-1/query witness+prove timing from its saved per-sample arrays and checked its constraint count. Completed workloads contain unique method/set/phase rows and all ten expected query sets.
- Completed n64 metadata reports 1,532 verified samples per dataset, including 1,472 Stage-1 rows and 60 query samples. These are recorded generation-time verifications, not 3,064 saved proofs independently reverified in this review.
- Rechecked all 60 owner/shift jobs, finite metrics, and matched comparisons. Exported [check receipt](../validation/outputs/server_review_20261019/checks.json).
- Fresh local `go test ./circuit -run TestBoundReuse -count=1` passed, including honest reuse, in-span identity and altered cost/margin/assignment/head/offset rejection cases.
- Fresh local `pytest tests/test_new_center_hybrid.py tests/test_shift_owner_reuse.py -q`: **5 passed**.

The authored manuscript and running server campaign were left unchanged. This report and its read-only capture/analysis scripts are new review artifacts.
