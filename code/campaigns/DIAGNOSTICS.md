# Local interpretable diagnostics

From any PowerShell directory:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\PyProjects\zero-knowledge\code\campaigns\diagnostic.ps1" -Workers 3
```

This is an explicitly bounded exploratory campaign, not a replacement for the
complete paper design. It adds new results without changing previous runs.
The normal plan uses ten seeds, k=5, seven dataset families. Real-data subsets
contain 1,000 rows (Arcene: the 100-row loaded training pool). Gaussian and
DBSCAN synthetic instances contain 1,000 points. Source hashes and actual
dimensions are recorded. Full and PCA controls share the same points and
KMeans seeds. PCA dimensions never exceed available components.

## Questions the campaign can answer

- RQ1: how fidelity changes with m=16/32/64/128/256 and sparse column q=2/8,
  compared with PCA and the full reference; whether rounding or projection
  accounts for losses. Float and integer paths use the same sparse map and
  original fixed centroids. Native seed fixtures are not transcript proofs.
- RQ3: serial protocol timings at n=64/128/256, d=128, m=32, q=2, three
  randomness sources, ten repetitions. Nonce also has an order-alternated
  prepublished/postcommit paired comparison. Beacon is a simulated local
  trace. Timings exclude dataset loading and KMeans fitting. Small overhead
  is a valid measurement, not a reason to add artificial delay.
  Source order rotates between repetitions to reduce ordering bias.
- Grinding: fixed maximum sampled-pair distortion thresholds 0.15/0.3/0.5/0.75
  plus a diagnostic threshold frozen at the 70th percentile of 32 independent
  pilot draws. Evaluate 256 NEW draws at m=64/256, q=2/8 on Gaussian and MNIST.
  Q=1/2/4/8 uses disjoint independent batches with Wilson 95% intervals.
  The fixed dataset/pair set is shared between pilot and evaluation; only
  projection draws are independent. Saturation remains possible and is
  reported. This is not a malicious-proof acceptance experiment or a bound
  on rare attacks; the calibrated threshold is not a selected paper epsilon.
- RQ4: original margin versus realized pointwise premise, conditional
  eligible counts, actual changes, and empty denominators. Sweep
  eps=0.15/0.3/0.5 and m=32/128/256. For synthetic DBSCAN, check ALL 499,500
  pairs at m=32/128, q=8 and report near-boundary stability separately.
  Repeated points across parameter cells are not independent observations.
- RQ5: fixed-centroid integer precision 8/12/16/20/24 bits under a shared
  24-bit physical reference and a separate range-utilization control;
  report zero coordinates and quantization-only versus projection loss.
  Gamma=0.1/0.25/0.5/1/2 at m=16/64/128 isolates separation on a fixed synthetic
  generator draw. q denotes matrix sparsity, not precision bits. No symbolic
  numbers are presented as compiled constraints.

## Results and restart

Outputs: `code/results/diagnostics/local_v1/`.

- `plan.json`: complete plan and implementation fingerprint.
- `checkpoints/*.json`: one atomically persisted successful measurement each.
- `measurements.parquet`, `measurements.csv`: combined results.
- `analysis/interpretation.md`: automatically populated English findings.
- `analysis/rq1_summary.csv` through `rq5_summary.csv` (RQ2 excluded):
  mean, standard deviation, sample count and Student-t 95% intervals.
- `analysis/rq3_grinding.csv`: batch sizes, rates and Wilson intervals.
- `analysis/rq1.png`, `rq3.png`, `rq4.png`, `rq5.png`: static plots.
- `analysis/rq3_timing.png`, `rq5_separation.png`: timing and gamma controls.
- `campaign.log` and `completion.json`.

Repeat the same command after a crash or Ctrl+C. Completed rows are skipped;
unfinished rows are rerun. A plan/code/worker-count mismatch is rejected rather
than mixed. For a deliberately changed campaign use Python's `--out` argument
with a new directory. Do not run two launchers into the same directory at once.
Stop other benchmark jobs before starting if RQ3 latency is to be interpreted.

Three processes run independent dataset jobs; NumPy/BLAS uses one thread per
process. RQ3 timing runs alone after these jobs. Dataset/PCA/KMeans work is
cached within each dataset job. Memory is bounded by three 512 MiB caches plus
temporary arrays and library workspace, not a measured hard RSS guarantee.
No full proof is requested. CIs are conditional on the fixed dataset subsets,
not population-wide confidence intervals. Ten seeds do not establish full
paper coverage, and a statistically unresolved difference remains unresolved.

## Verification

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\PyProjects\zero-knowledge\code\campaigns\diagnostic.ps1" -Smoke -Workers 2
```

Smoke uses synthetic n=64, two seeds and shorter grinding, plus tiny native
protocol inputs. Its purpose is execution verification, not scientific results.
Do not merge smoke measurements with the normal campaign.

A measured single-seed Gaussian quality job produced 44 records in about
2.7 seconds; an isolated n=128, d=128 nonce row including its separate paired
comparison took about 60 seconds on this laptop. The complete campaign is
expected to take roughly 1-3 hours, depending on SVD, CPU load and throttling;
this is an estimate, not a measured complete-campaign runtime.
