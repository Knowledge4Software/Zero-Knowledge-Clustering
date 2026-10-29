# Exact authenticated known-family affine baseline

The baseline is implemented and measured. All 288 proofs passed a fresh public-only replay, and every cost and private assignment matched the independent quantized Full reference.

| Method | Measured case time | Full / method | Center contract |
|---|---:|---:|---|
| Exact public-center Full | 196.80 minutes | 1.00 | Previously unknown future centers allowed |
| Head rank63, previous relative certificate | 43.46 minutes | 4.53 | Previously unknown future centers allowed |
| Exact authenticated family affine | 40.25 minutes | 4.89 | Entire bank approved before preprocessing |

These are serial laptop runs on the same frozen MNIST catalog of128 records, k5, and all20 distinct center sets. The family bank contains85 distinct centers. Head/family use16-row query batches and Full uses8, under the same1.6-million-constraint guard. One model fit and one measured repetition do not establish a general performance ranking. The enclosing uninterrupted case times include setup, preparation, generation and fresh public replay; projection fitting is charged to the head.

The family baseline is slightly faster than the earlier rank63 head run and is exact in this setting. We must report that result. Its limitation is that new centers outside the preapproved bank are rejected and require new preparation. The head's defensible deployment distinction is support for unknown future centers, rather than superiority over this closed-family exact baseline. The current85-center bank is a workload size, not a fundamental limit of the baseline.

The public-only smoke uses eight fresh proofs. Altered aggregate costs and an independently approved out-of-bank center are rejected. The larger benchmark's independent checker matches all20 sets to Full, not only a selected subset.

Source exports: `code/validation/outputs/known_family_affine_n128_20261024/checked_comparison.json`, `protocol.json`, `family_affine/public_bundle/` and `family_affine/owner_pin.json`. Smoke receipt: `code/validation/outputs/family_affine_smoke_20261024/checks.json`. The separate frozen executable is `code/zk/bin/commitjl-bound-family.exe`; its SHA256 is `37c89bca09a3167a478f87c31714dd7cda9991b317a3068986b6d20d2e6f48fe`.

This addresses the request to implement and measure the authenticated known-family candidate on one matched workload. Additional fits, scales and repeated/interleaved timings remain necessary for broader claims. It does not recover the original Mobin exports or establish the speed of the new group certificates.
