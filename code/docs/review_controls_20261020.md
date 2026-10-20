# Completed reviewer controls - 20 October 2026

## Finished campaign

All 14 cases completed successfully. The small exact queue finished at 06:17 Tehran; the final 1,024-row exact control finished at 20:16:39 Tehran. There are no queued cases in this campaign.

| Remote results root | Completed cases |
| --- | --- |
| `/root/commitjl/results/review_full_controls_20261019T224939Z` | MNIST N=64/256 and CIFAR/ResNet N=64 exact public-center Full |
| `/root/commitjl/results/review_certified_controls_20261019T234210Z` | Learned/PCA plain+certified N=64/256/1024 MNIST and N=64 CIFAR; rank-63/T=16384 N=256 pilots; exact N=1024 Full |

Four exact cases contribute 2,000 public proofs. Ten reuse cases contribute 4,848 Stage-1/plain/certified proofs. All 6,848 pass worker public-only replay. All rows and ten sets are actually proved; no catalog extrapolation is used.

The [canonical manifest](../../paper_results/review_controls/generated_completed_20261020/manifest.json) checks 208 transferred metadata hashes, original root/catalog/center matching, exact integer Full labels and costs, and all certificate outputs against Python. There are 100 proved certificate method/set outputs and 33,280 record/set evaluations, with no incorrect certified assignment or missed optimum interval. None of the 90 updated cases meets the illustrative 99% coverage / 1% width policy.

## Reproduce numerical exports

From the repository root, choose a NEW output directory:

```powershell
python code/analysis/export_review_matched.py --full-capture code/validation/outputs/review_full_capture_followup_20261020 --reuse-capture code/validation/outputs/review_final_capture_20261020_1907Z --out tmp/reproduced_review_controls
```

The two immutable capture manifests describe exactly which files were transferred. They do not imply that every proof/key artifact is included. Full diagnostics and public-only replay receipts are retained; one selected public Full bundle is included for standalone replay. Bulk proof artifacts remain in the remote/local raw stores.

Every alternative pays native preparation, common Stage 1 where applicable, one query compile/setup, native answers, witnesses and median proving once. The measured exploratory plain phase is not charged to certified deployment. Verification, warmups, fitting, I/O and public center preparation are excluded. The first frozen Full CSV omitted native answer time; exports reconstruct it from raw samples without changing saved results.

Main reuse batch is 64; resource-guarded Full batches are eight for MNIST and two for CIFAR. Higher-rank pilots use batch 16. All runs use four CPUs on one host, approximately 7.8 GiB RAM, one measured repetition per sample and one initial warmup. Configuration ratios do not establish an optimal exact implementation, equal-fidelity services or timing uncertainty across hosts.

## Implemented relations

- `zk/circuit/bound_full_reuse.go`: exact original-space integer nearest-center relation with public centers and a reused setup.
- `zk/circuit/bound_reuse_certificate.go`: optional rounded-basis certificate on top of the authenticated reuse relation.
- `zk/circuit/bound_verify.go`, `zk/cmd/commitjl-bound/reuse_verify.go`: public-only builders, approved-task verification and coverage checks.
- `commitjl/reuse_certificates.py`: independent Python integer reference.
- `experiments/review_controls.py`: new-output-only, all-row worker with resource guards and separate public replay.
- `validation/verify_certificate_policy.py`: cryptographic replay followed by explicit quality acceptance thresholds.

The canonical generator and input commitments are independent of whether a learned or PCA basis is approved. No proof of basis fitting, k-means training or neural feature extraction is claimed.

## Rounded-basis guarantee

Let `H(x)=T B(x-c0)`, `R(x)=(T^2 I-B^t B)(x-c0)`, `E(x)=||R(x)||^2`, and `g=max_i sum_j |(BB^t-T^2 I)_ij|`. For estimated distance `D_j=||H(x)-H(c_j)||^2+E(x)+E(c_j)` and original quantized distance `Q_j=T^4 ||x-c_j||^2`, the checked envelope is

`T^2 |D_j-Q_j| <= 2 T^2 ceil(sqrt(E(x))) ceil(sqrt(E(c_j))) + g ||H(x)-H(c_j)||^2`.

The Gram term accounts for rounding-induced nonorthogonality. The circuit constrains radii, private assignment flags and public count/cost endpoints. It certifies only unique stable winners. The public count is a lower bound, not the maximal possible count. Cost endpoints bound the exact quantized optimum at scale `T^6`; decoding to original units also divides by input scale squared.

A cryptographically valid approximation can fail the auditor's quality policy. Exact Full is the implemented fallback. Selective exact refinement of uncertified records is not implemented.

## Standalone verifier and provenance

The current verifier rejects every approved batch size lacking a pinned query key. Targeted tests and the negative mixed-size public task pass. The frozen Mobin v1/v2 workers used homogeneous, fully covered tasks; their proof relations and outputs were not changed. A selected complete 80-proof Full bundle was also replayed locally using the current verifier without private witnesses.

Compile binaries from `code/zk/`:

```powershell
go build -o bin/commitjl-bound.exe ./cmd/commitjl-bound
go test ./... -p 1
```

Frozen source archives and binary hashes are retained in selected evidence manifests. Locally compiled executables are ignored. Old per-stage `progress` snapshots are not authoritative campaign status; completion requires the final worker status and successful public replay.

The [current paper delivery](../../output/README.md) includes all finished results. Historical three-version sources and supervisor bytes remain preserved. Author metadata, declarations and scientific approval remain pending before submission to Neurocomputing.
