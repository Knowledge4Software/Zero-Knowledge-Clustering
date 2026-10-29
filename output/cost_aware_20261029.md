# Cost-aware routing: measured laptop results

Fresh warm request proofs and public replay, eight workers, one exploratory execution per case. The same original rows, frozen centers and owner-pinned setup are used by both arms. Existing row authentication, proving keys and old Full-cost evidence are reused; native planning is charged. Copy/export time and independent oracle validation are outside the phase-wall totals.

| Case | Requests | Routed | Direct Full | Full / routed | Cert / Full routes | Corrections |
|---|---:|---:|---:|---:|---:|---:|
| text_n32 | 16 | 8.221 min | 8.049 min | 0.979x | 3 / 13 | 0 |
| fashion_n32 | 8 | 3.942 min | 9.061 min | 2.299x | 8 / 0 | 4 |

The planner computes exact integer bounds from cached private heads, residual energies and old-group moments. For an initially inconclusive bound it simulates at most16 width-ordered exact row corrections using the prover’s private rows. It selects the compact relation only when that plan is conclusive and its calibrated proof cost is at least10% below Full. It sends every other request directly to Full, before generating any compact query or correction proof. These native distances are untrusted witness planning. The verifier accepts only freshly verified proofs.

Costs were frozen from the earlier Q8 pilot before this execution. The catalogs and proposals had already been inspected: this is an engineering test, not independent confirmation on held-out data. The safety margin is a prediction rule, not a guaranteed speedup.

All24 requested decisions are correct and every certificate interval contains the independently recomputed exact change. The hybrid verifier binds the branches to the same original-catalog root and approved reference, reconstructs both tasks from independently supplied centers, and requires disjoint complete request and row coverage. Nine native/coverage tests and15 real-proof tamper tests passed. The initial smoke attempt failed in JSON export on a null sample list; it was fixed before workload proving, and its failure receipt is retained.

| Case | Native planner | Fresh request proof bytes: routed / Full | Exported proof bytes including reused evidence: routed / Full | Peak RSS |
|---|---:|---:|---:|---:|
| text_n32 | 1.871 s | 11956 / 12544 | 19012 / 13328 | 1.154 GiB |
| fashion_n32 | 1.655 s | 5488 / 6272 | 11760 / 7056 | 1.675 GiB |

Larger exported bundles can result from the reused per-row certificate evidence. Fresh request proof size and complete bundle size are different metrics.

Cold activation is implemented as an optional batch cost check: decline the compact path if estimated savings do not cover extra summary setup. The text16 batch does not cover that extra setup under the frozen cost model. No new cold runtime or measured break-even count is claimed. Do not compare these warm totals directly with the earlier27.52-minute cold text run.

This change preserves the underlying prediction representation and its assignment agreement; it improves proof scheduling, not classification accuracy. All serviced change decisions are exact in sign after public verification. The routing policy is also applicable to the PCA certificate baseline. PCA and the Halo2 ZKML primitive adapter were not retimed here; optimized-versus-unoptimized comparisons would not establish a mathematical advantage over PCA. No universal superiority over Full or other baselines follows from these two warm cases.

Private exact corrections remain in the prover. Public routing choices and refinement indices are observable outputs. No additional assignment-privacy claim is made. Existing circuit definitions, binary and paper files are unchanged.

Evidence: `code/validation/outputs/cost_aware_20261029`.
