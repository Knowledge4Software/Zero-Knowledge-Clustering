# Experiment readiness review

Review completed locally on Windows, September 28–29, 2026. No commits, pushes, or GitHub changes were made. This report supersedes earlier broad readiness claims in this workspace.

## Decision

**The corrected implementation passes the tested regression and small end-to-end experiment cases. The final-paper campaign is not ready to launch unchanged.** There are campaign coverage/statistics gaps and unvalidated resource requirements for the large proof grid. All eight configured dataset entries now load, while independent upstream data provenance remains outside the completed checks. Passing tests is not a formal security proof or a guarantee that no bugs remain.

The review covered numerical helpers, projections, commitments/randomness/protocol, the Python/Go bridge, Go constraints and CLI, data loaders, RQ1–RQ5/ablation runners, result persistence, and table/figure generation. The full native and core grids were not executed during this review.

## Corrections made

| Area | Problem and correction |
|---|---|
| Quantization | Reject nonfinite and out-of-range values before conversion to int64. Preserve subunit distances. Keep the no-wrap check active under Python optimization. |
| Distances | Exact Python integer fallback for unsafe int64 arithmetic, including object/uint64 inputs. Bound difference-array workspace and calculate assigned-center cost without repeating the entire centroid search. |
| Certificates | Handle coincident-centroid ties correctly. Keep exactly one normalization by q. Adapt pair chunks to feature dimension instead of using a fixed chunk that could allocate gigabytes. |
| RQ4 dataset identity | Stop replacing the requested dataset with synthetic DBSCAN data. Record the actual dimensions/hash and effective projection dimension. Cache reusable certificate inputs. |
| Proof fidelity metrics | Measure ARI/distortion/cost on the actual quantized witness and authenticated projection. Previously these could refer to an unrelated float-space projection. Preserve exact integer cost separately from the numeric reporting column. |
| Transcript binding | Independently verify the seed derived from commitments and challenge, then regenerate/check the matrix. Reject reused external challenges and malformed input coordinates. |
| Circuit constraints | Constrain witnessed dense Rademacher coefficients and coordinate-selection structure. Reject the unimplemented generator variant. Add compiled-R1CS tests with recomputed commitments so malformed witnesses cannot fail merely because their hashes are stale. |
| Go interface | Validate dimensions/witness shapes, backend names, and signed constant coefficients. Retain individual proof/verification timing samples. Build and test natively on Windows. |
| Proof process | Rebuild stale default binaries, use a Windows executable name, require successful verification, clean up children on timeout/interruption, and enforce a monitored process-RSS budget. |
| Repeated work | Bound intermediate caches, reuse PCA decomposition across target dimensions, avoid dense identity multiplication, and remove duplicate RQ5 assignment work. |
| RQ3 | Distinct commit-reveal contributions; explicitly label the local beacon simulation. Limit grinding workers/BLAS threads, honor pair-set size, reject Q larger than available trials, and distinguish empirical plug-in bounds from theoretical guarantees. |
| Result files | Atomic writes, exclusive writer locks, source/config provenance, explicit rejection of stale/corrupt files, retry failed rows without duplicate keys, and nonzero runner exit status on failures. Shards get separate default paths. |
| Reporting | Correct native/proving labels, avoid combining unlike circuit shapes and range-check variants in proof/ablation tables, and reject stale data in generated reports. |
| Data loading | Atomic temporary downloads, nested cache directories, UCI training-shape checks, finite-value/label-count validation, and separate manifest entries per sample hash. |

## Verification evidence

| Check | Observed result |
|---|---|
| Existing complete Python suite during review | 66 passed, including 10 Go-backed proof cases, in 704.30 seconds. |
| Final Python suite excluding expensive `test_zk.py` | 84 passed in 12.10 seconds, including the new regressions. |
| Native Windows Go build | `go build ./...` succeeded. |
| Native Windows Go tests | `go test ./...` succeeded; also rerun by the audit with cache disabled. |
| Fresh seven-runner smoke campaign | 49 `ok` rows, 2 explicitly `native_only` rows, no failures or duplicate hashes. About 177.7 seconds across runner commands. |
| RQ2 smoke proof | n=12, d=8, m=4; 185,612 measured constraints; successful proof and verification. |
| Ablation smoke | 12 verified proof rows; 2 native-only rows for the unimplemented in-circuit generator. |
| Reporting smoke | All table generators, all figure generators, and summary generation completed. These are smoke artifacts, not final paper tables. |
| Standalone audit on fresh smoke files | **16 PASS, 0 FAIL, 0 PARTIAL, 1 UNVERIFIED.** Real-data provenance is not independently established by the synthetic smoke run. Audit exits nonzero rather than silently treating this as all-pass. |

The 66-test run preceded the last host/reporting changes; it is not presented as a single run of the final 94-test collection. The final host tests, fresh proof campaign, and audit exercise those later changes. Honest Groth16 and PLONK cases passed in the complete run. Fresh smoke proofs use Groth16.

Evidence: [smoke command outcomes](smoke_v2/smoke_commands.json), [row summary](smoke_v2/results_summary.json), [audit](audit_results.json), [regression tests](../../../tests/test_review_regressions.py), and [compiled circuit tests](../../../zk/circuit/relation_regression_test.go).

## Measured memory and setup reuse

Each RQ4 case evaluated all 499,500 unordered pairs for 1,000 points in a separate worker process. RSS includes that worker's Python/runtime/data allocations; it is not total machine RAM use.

| Features | Peak worker RSS | Evaluation time | Under 3,000,000,000 bytes |
|---:|---:|---:|---|
| 1,024 | 230,027,264 bytes | 5.83 seconds | Yes |
| 10,000 | 401,203,200 bytes | 59.99 seconds | Yes |

This verifies two substantial cases, not an unlimited guarantee for every possible n/d or concurrent job count. Pair outputs and input data still consume memory, and large pair sets can be explicitly subsampled.

Five proofs of the same small Groth16 instance took 8.67 seconds with one setup, versus 30.94 seconds for five separate invocations: **3.57× observed speedup**. All ten proofs verified. This is a diagnostic measurement, with other verification work on the machine, not publication timing evidence. It does not establish the previously requested 5–7× speedup.

Setup reuse is currently **within one runner invocation/row**. There is no persistent proving-key cache shared across grid rows. A future cache must include backend, variant, range-check flags/width, dimensions and the complete constant matrix where applicable; shape alone is not a safe cache key.

Evidence: [memory measurements](rq4_memory.json), [setup measurements and individual samples](setup_reuse.json), [machine](machine.json).

## Dataset verification and limits

The initial dataset check completed six of eight configured entries. Fashion-MNIST failed with `RemoteDisconnected`; Gisette failed with `IncompleteRead`. **Both downloads succeeded on retry. All eight configured entries now pass the loading checks.** The retry preserves the original failures and records its own outcome in [dataset_retries.json](dataset_retries.json).

| Configured data | Verified loaded sample after retries |
|---|---|
| Gaussian mixture configuration 1 | 2,000 × 256 |
| Gaussian mixture configuration 2 | 2,000 × 512 |
| DBSCAN margin synthetic | 1,000 × 128 |
| MNIST | 5,000 × 784, from 70,000 source rows |
| Fashion-MNIST | 5,000 x 784, from 70,000 source rows; passed on retry |
| Gisette | 3,000 x 5,000, from 6,000 UCI training rows; passed on retry |
| Arcene | 100 × 10,000, the UCI training split |
| 20 Newsgroups TF-IDF | 3,000 × 2,000, from 11,314 source documents |

Checks establish loaded shapes, finite features, label alignment, and locally calculated hashes. They do not independently authenticate every original download against an upstream signed checksum. The current DBSCAN hash reproduces the currently frozen generator; this does not recover or validate the older inconsistent benchmark hash.

The configurations use dataset samples/training splits. They do not process every available row of every dataset. In particular, requesting `n_max=200` for the 100-row Arcene training split still yields 100 rows, now explicitly reported.

Evidence: [initial dataset checks](datasets.json), [retry checks](dataset_retries.json), [combined final dataset status](datasets_final.json).

## Outstanding final-paper requirements

1. **Finalize the experiment matrix.** `native.yaml` expands to 5,760 rows and `core.yaml` to 540, but precision is fixed at 16 bits and only three grid seeds are present. Independent n/d/gamma/imbalance/precision sensitivity studies and the intended sample sizes are not fully specified by these grids. The RQ4 phase sweep only varies gamma/m on a selected synthetic configuration.
2. **Implement the preregistered statistical execution.** `n_reps_native: 10` is not currently executed as ten paired clustering initializations per row. The automatic smallest fidelity-eligible m* selection described in `prereg.yaml`/`core.yaml` is not connected to the proof runner. Confidence intervals and paired significance analysis are not integrated into the final table pipeline.
3. **Resolve metric scope before publication.** The current native path measures assignment to projected full-space centroids, then evaluates reconstructed centers; it does not rerun full projected-space KMeans/Lloyd iterations. Rows now label this explicitly. Native timings reuse measured cached components and are not independent cold end-to-end timing repetitions. Adaptive normalization depends on B/S, so a naive precision sweep can rescale away the intended precision effect.
4. **Calibrate proof sizes on this machine.** The machine has about 13.86 GiB physical RAM. Full core cases include n=500,d=256; n=1000,d=1024; and MNIST n=500,d=784. The circuit commits every input coordinate regardless of the reduced dimension. These are substantially larger than verified smoke/test cases and have not been shown to fit or finish locally. The RSS watchdog records failure; it does not make a too-large circuit feasible. No reliable full-campaign completion time has been measured.
5. **Keep research limitations visible.** The beacon row is a local trace simulation, not an authenticated live public beacon. `incircuit-gen` remains unimplemented. The Pisot construction is an explicitly documented local approximation, not a verified reproduction of the cited method. PLONK uses gnark's testing SRS. Host-side transcript regeneration is part of the current verification flow; the circuit itself does not prove seed expansion.
6. **Rerun affected results and complete the analysis layer.** Earlier proof-quality and mislabeled RQ4 measurements are invalidated by these corrections. Existing aggregate headline/figure summaries are exploratory; final sensitivity comparisons must hold other factors fixed and report uncertainty. Do not combine old rows with the fresh campaign.

These are implementation, methodology, resource-validation or external-data limitations. They should not be converted into PASS labels by shrinking the final grid or silently substituting datasets/methods.

## Reproduce the completed checks

From the repository root in PowerShell:

```powershell
Set-Location code
python -m pytest tests --ignore=tests/test_zk.py -q
Set-Location zk
go build ./...
go test ./...
Set-Location ../..
python code/validation/archive/review_20260928/run_smoke.py
python code/validation/archive/review_20260928/resource_checks.py
python code/validation/audit.py --results-dir code/validation/archive/review_20260928/smoke_v2 --proof-config code/experiments/configs/smoke.yaml --out code/validation/archive/review_20260928/audit_results.json
```

The standalone audit intentionally exits 1 while any check is unverified. After code changes, use a new result directory; the result writer refuses to resume files from different source versions. Windows PowerShell was used for these measurements; no WSL performance comparison was conducted.
