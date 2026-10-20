# Paper tables: required runs and interpretation

Reviewed 2026-10-05 against root `manuscript.tex` and current runners. Numbers follow source order; labels are stable. No new experiment campaign was launched in this review.

**Historical review:** the defects and campaign counts below describe the earlier
implementation. Later balanced runners have separate seed/start axes,
confidence-bound selection, integer quality checks, matched ordering contrasts
and failure inventories. Use [the current audit](run_audit_dataset_runtime.md)
and [server run guide](rq2345_complete.md) for today's commands and limitations.
The manuscript still requires alignment and missing full-size proof evidence.

## Resolve before the final campaign

- **Statistics:** the paper specifies 30 projection seeds x 10 paired clustering starts, hierarchical bootstrap and Holm correction. Current native configs use ten seed rows and the same seed for projection and KMeans initialization; reports use Student-t intervals. Separate seed/start axes and implement the stated analysis, or revise the paper to the completed design. Merely extending `seed` to 30 does not implement 30x10.
- **Selection:** `select_proof_shapes.py` selects on median ARI and median objective ratio; the paper requires confidence-bound eligibility. State the intended rule explicitly: for example lower ARI CI >= 0.80 and upper objective-ratio CI <= 1.10. Preregistration's `tau_Phi=0.10` is relative slack, not a literal maximum objective ratio. Document any revised rule prospectively.
- **Workload:** native RQ1 projects full-space centers and reassigns points; it does not fit KMeans in projected space. It evaluates an original-space objective after deriving centers from those labels. The circuit proves fixed-center assignment and projected cost, not KMeans training.
- **Quantized fidelity:** RQ1 native seeds and RQ2 commitment/challenge-derived maps differ. RQ2 measures its actual quantized witness quality, but a fresh matrix at a native-selected shape does not establish CI-qualified fidelity. Check quality across declared verifier-issued challenges on the same quantized preprocessing before cost comparisons.
- **Circuit model:** implemented sparse routing visits every output row and sparse slot for every point/centroid, approximately `(n+k)*m*d*q` selector work. The paper's `n*q*d` count is an algorithmic ideal. Commitment authentication includes N leaf hashes plus N-1 internal hashes. Instrument actual compiler counts before claiming reductions.
- **Fair comparisons:** freeze dataset, centers, scale/bound, backend, hardware and resource limits. The ordering ablation must reuse the exact matrix/witness; current sparse and prepublished builders do not automatically do that.
- **Conformance:** independent verification must derive expected C_A from the recorded challenge. Labels are secret witnesses, so published labels need a public binding if claimed. Metadata/scale are not all committed as described in the threat table. Fix early commit-reveal disclosure and explicit deadline enforcement. Expected negative-test rejection is distinct from unexpected rejection of honest inputs.

## All 18 tables

| Table / label | Required runs or evidence | What it shows and what is missing |
| --- | --- | --- |
| 1 `notation` | No benchmark; audit ownership, dimensions, scale and public/secret fields against Python and Go. | Defines the exact implemented statement. |
| 2 `statements` | Scope audit and differential/negative proof tests for fixed-center assignment; mark Lloyd/DBSCAN proof extensions unsupported. | Privacy/cost scope. Native DBSCAN runs do not constitute DBSCAN proofs. |
| 3 `threats` | Transcript substitution/replay, budget/deadline, commit-reveal ordering, malformed matrix and arithmetic tests; verifier-side challenge derivation. | Which malicious behaviors are rejected and which assumptions require deployment enforcement. Honest proofs alone are insufficient. |
| 4 `related-a` | Literature and cited-artifact verification; no internal benchmark needed for the comparison taxonomy. | Novelty and scope. Cross-paper timings on different hardware do not establish speedup. |
| 5 `transforms` | Projection normalization/domain audits and golden vectors; supported-circuit checks. | Arithmetic differences between methods. PCA-public excludes training; Gaussian/Achlioptas currently lack corresponding proof variants. |
| 6 `circuit-cost` | Compile matched full/sparse/dense/coordinate cases on increasing small shapes; instrument hashes, projection, ranges, distances and argmin for the selected backend. | Actual bottlenecks/asymptotics. Correct missing leaf hashing and selector cost; Lloyd update remains theoretical. Stage wall times cannot be inferred from constraint fractions. |
| 7 `claims` | Theorem-assumption review plus distribution, differential, adversarial, distortion, certificate and precision tests. | Separates formal conditional guarantees from empirical observations. Experiments do not prove universal JL guarantees. |
| 8 `datasets-a` | Load chosen datasets; freeze split, upstream identity/checksum, preprocessing, analyzed n/d, license and synthetic parameters. | Actual dataset scope. Current real runs are subsets; Arcene supplied 100 rows in the completed campaign. Embedding corpus is unspecified: implement/select it or remove its row. |
| 9 `matrix-a` | Freeze a native frontier, one-factor scaling blocks and a separately identified feasible ZK core. | Coverage and omissions. Existing configs do not cover all advertised levels. Avoid an infeasible full Cartesian product. |
| 10 `rq-metrics-a` | Freeze epsilon, thresholds, CI selection, pairing, failure policy and analysis manifest; no standalone benchmark. | Defines success before proof results. Current median-based selection conflicts with the table. |
| 11 `metrics-a` | Audit field definitions/units; add missing end-to-end stopwatch, verification memory, retries/log costs and raw repetition statistics. | Meaning of every number. Cached native component sums are not end-to-end latency; separate compile/setup/witness/prove/verify. |
| 12 `fidelity-a` | RQ1 native frontier on declared dataset/k: full, PCA, Rademacher, sparse JL, Pisot; m={16,32,64,128,256}, extending where necessary; sparse q={1,2,4,8} if retaining paper levels. Use 30 projection seeds x 10 starts if retaining that design. Save objective ratio, ARI, distortion q95/exceedance and CIs; join matching RQ4 coverage. | Smallest fidelity-eligible m and tail geometric error. Existing 7,180 native rows use the earlier ten-row design. `make_tables.py` emits fidelity pass rate where the manuscript asks for certified coverage. |
| 13 `zkperf-a` | RQ2 matched full, PCA-public, dense Rademacher, sparse, coordinate and Pisot at quantized-fidelity-qualified points. One warm-up + 10 measured proofs per configuration; raw times, constraints, RAM, verification, bytes, ARI/objective and all failures. | Primary proof cost/fidelity comparison. Both saved full-size attempts hit the memory watchdog; successful cells remain unavailable. Start small and measure scaling. Current aggregation uses medians of per-row means, not necessarily raw repetition medians. |
| 14 `ordering-a` | Isolated RQ3 paired post-commit/prepublished with identical matrix/data/centers. Nonce, recorded beacon trace and commit-reveal; separate commitment, challenge, expansion/hash, log, retries and serialized bytes. Compile matched authentication scopes and measure total proof time. Bounded-Q grinding separately. | Marginal security-ordering cost. Existing 90 overhead, 540 grinding and 30 paired rows are native; beacon is simulated. Expansion and C_A hashing are currently combined; several table stages lack separate instrumentation. Define actual wire bytes. |
| 15 `cert-a` | RQ4 high/low-margin centroid synthetic and real cases; synthetic and real DBSCAN. Sweep m/q/seeds with declared eps/radius/min_samples. Record qualified items, realized embedding premise, conditional violations, uncertified stability, near-band denominator and graph/label stability. | Conditional certificate soundness/conservatism. Existing 265 synthetic and 100 real rows help; all 100 real centroid rows failed the global eps=0.15 event. Separate global/pointwise premises. Synthetic pair coverage is dominated by far pairs; real DBSCAN needs matching evidence. |
| 16 `sensitivity-a` | RQ5 one-factor n/d/k blocks, m/q frontier, fixed-physical-reference precision {8,12,16,24} (optionally 20), gamma grid; actual proofs for time/RAM slopes. Several feasible levels with CIs and failure counts. Repeat matched configurations on second backend/hardware if claiming them. | Scaling slopes, fidelity tradeoffs and feasibility boundary. Existing 200 scaling/80 separation/precision rows are native. Backend/hardware proof grid is missing. Keep 20-bit and 24-bit reference campaigns separate; adaptive per-precision normalization can erase the intended comparison. |
| 17 `ablation-a` | Representative and stress shape: sparse reference, dense, same-matrix prepublished, unbatched ranges and lower precision; one warm-up + 10 proofs. Recheck fidelity for changed precision/matrices. | Causal effect of one change at a time. Current variant x batching grid needs matched contrasts. In-circuit generator is unimplemented: implement/validate it or omit proof columns for that row. |
| 18 `validity-a` | Evidence synthesis: complete failure inventory, versioned configs, tests, isolated timings, seed/start handling and conditional theorem checks. | Remaining limitations. Planned mitigations must not be presented as completed. |

Labels above have prefix `tab:` in LaTeX. The manuscript's result cells are still placeholders and it does not include generated tables with `\\input`. `make_tables.py` also lists `precision-a`, which has no standalone table in the current manuscript. Generated outputs need column/statistic alignment before insertion.

## Execution order and existing entry points

The new [RQ2–RQ5 server campaign](rq2345_complete.md) provides quantized RQ2
eligibility, isolated matched RQ3 comparisons, conditional RQ4 diagnostics,
RQ5 precision/scaling and implemented proof ablations, with raw artifacts and
resource/failure inventories. Use its balanced 10×5 commands for future runs.
Unsupported circuits and deployment claims remain explicit limitations; the
historical observations below are not results from the new campaign.

The [RQ1 v3 comparison runner](rq1_complete.md) implements all eight cases with
separate seed/start axes, fixed subsets, resumable rows, pointwise/projected
arrays, conditional crossed-bootstrap CIs and profiling. It includes the native
projection baselines and an explicitly unverified Pisot surrogate. Its exporter
creates per-dataset figures and CSV/LaTeX tables. The cited Pisot artifact still
needs verification; v2 sparse-only runtime estimates do not cover v3.

1. Fix conformance, selection, statistics and instrumentation; freeze a new run manifest/source version.
2. Validate tiny real proofs, then measure increasing paired full/sparse/dense shapes to establish constraints/memory.
3. Run the chosen native frontier and certificate/precision supplements; qualify operating points on the quantized challenge-derived workload.
4. Run selected paired RQ2 configurations with explicit resource limits and all failure attempts retained.
5. Run isolated RQ3, proof scaling and matched ablations, then generate and check final tables.

From repository root, these launch the **existing design**, not automatically the paper's 30x10 design:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\code\campaigns\stages.ps1 -Stage FullNative -GrindingWorkers 8
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\code\campaigns\supplements.ps1 -Stage All -Workers 6
```

From `code/`, an engineering pilot with small shapes and three measured repetitions:

```powershell
python -m experiments.rq2_proof_cost --config experiments/configs/core_light.yaml --out results/rq2/runs/pilot_new/rq2_proof_cost.parquet
```

From repository root, **after selection/instrumentation fixes**:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\code\campaigns\stages.ps1 -Stage RepresentativeProof -MaxProofShapes 2 -NativeMode Parallel
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\code\campaigns\stages.ps1 -Stage ProofAblation -MaxProofShapes 2 -NativeMode Parallel
# Requires successful proof inputs and final analysis semantics.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\code\campaigns\stages.ps1 -Stage FinalReport -NativeMode Parallel
```

Use distinct run IDs for new source/config versions. Small pilots are engineering evidence, not substitutes for the declared paper sample sizes.

## Cleanup record

Removed from active project: superseded `code/results/archive/paper_analysis_v1/` derived documents/charts, byte-identical `output/pdf/commitjl_supervisor_summary_revised.pdf`, root `.DS_Store`, and empty `tmp/`. Recovery copies are under ignored `code/.cache/cleanup_20261005/`. Permanent deletion was blocked by execution policy; reversible moves completed cleanup. The generator now writes only `commitjl_supervisor_summary.pdf`.

Preserved raw runs/logs/manifests, current reports, educational sources, LaTeX figures and historical validation evidence. Raw archive copies are required by `check_layout.py`; `supervisor_report.py` reads historical proof inputs. Historical audit snapshots remain explicitly historical, not current audits.
