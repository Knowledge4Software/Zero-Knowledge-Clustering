# Completed review controls

All 14 Mobin cases completed on 20 October 2026. Four exact public-center Full controls cover MNIST N=64/256/1024 and CIFAR/ResNet N=64. Ten reuse cases cover learned/PCA plain and certified audits on these catalogs and the two predeclared rank-63/T=16384 MNIST N=256 pilots.

`generated_completed_20261020/` is the canonical measured export. Its manifest checks 208 transferred files, all requested rows, matching original integer data and centers, exact Full outputs and independent Python certificate outputs. Worker public-only replay accepts all 6,848 proofs. Every proved certificate is consistent with the exact quantized optimum; none of the 90 updated method/set cases meets the illustrative 99% coverage / 1% width policy.

`generated_native_20261020/` retains the earlier native preflight and training-only representation-budget grid. The rank-15 native cells overlap the proof campaign and must not be counted as independent observations. Rank-31/63/127 results at N=1024 are native calculations, except that rank-63 proof costs are measured separately at N=256.

Re-export from the repository root into a fresh directory:

```powershell
python code/analysis/export_review_matched.py --full-capture code/validation/outputs/review_full_capture_followup_20261020 --reuse-capture code/validation/outputs/review_final_capture_20261020_1907Z --out tmp/reproduced_review_controls
```

The exporter accepts captures containing both exact and reuse cases. It preserves component-cost boundaries, emits separate cost/quality/pilot tables and a measured N=1024 figure, and checks the saved phase outputs rather than inferring proof success from native fidelity.

See the [implementation/reproduction guide](../../code/docs/review_controls_20261020.md) and [completed result report](../../output/mobin_completed_results_20261020.md). Intermediate early-morning captures and duplicate tables are retired to the external cleanup archive. Raw measurement files are unchanged.
